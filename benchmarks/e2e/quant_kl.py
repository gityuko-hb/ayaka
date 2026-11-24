"""First-token quality of an FP8 checkpoint against its 16-bit original, through Ayaka's models.

For each of :data:`PROMPTS` (32 fixed prompts, rendered with the checkpoint's chat template),
both checkpoints are loaded by Ayaka in turn (one at a time, so a pair fits a 24 GB card), the
prompt is prefilled once and the distribution of the first generated token is read from the
logits of the last position, in float32. The result is, per prompt and averaged:

* ``KL(16 bit || FP8)`` in nats, ``sum p16 * (log p16 - log p8)`` over the whole vocabulary;
* whether the two argmax tokens agree (top-1 agreement);
* the entropy of the 16-bit distribution, so that a prompt whose first token is all but certain
  (and whose KL says little) shows.

The threshold fixed in #112 before any run: mean KL at most 0.05 nats.

    uv run python benchmarks/e2e/quant_kl.py --reference /path/to/Llama-3.1-8B-Instruct \
        --quantized /path/to/Llama-3.1-8B-Instruct-FP8 --dtype bf16 \
        --out benchmarks/results/<date>-<machine>/quant-kl-llama-3.1-8b.json
"""

from __future__ import annotations

import argparse
import gc
import statistics
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from _machine import machine, write

from ayaka.configs.device import DeviceConfig
from ayaka.configs.model_source import ModelSourceConfig
from ayaka.core.types import ForwardMode
from ayaka.kvcache.storage.pool import KVPool
from ayaka.layers.context import ForwardContext
from ayaka.model_loader.loader import load_model
from ayaka.utils.dtypes import DType

THRESHOLD_NATS = 0.05
#: Without it, Qwen3's template lets the model open with ``<think>`` at a probability of nearly 1,
#: which makes every first-token distribution the same and the KL meaningless. Templates without
#: the switch ignore it.
TEMPLATE_KWARGS: dict[str, Any] = {"enable_thinking": False}
BLOCK_SIZE = 16
PROMPTS = (
    "What is the capital of France?",
    "Write a haiku about the sea.",
    "Explain what a hash table is in two sentences.",
    "Translate 'good morning' into Spanish.",
    "What is 17 multiplied by 23?",
    "Name three primary colors.",
    "Write a Python function that reverses a string.",
    "Why is the sky blue?",
    "Summarize the plot of Romeo and Juliet in one sentence.",
    "What does HTTP stand for?",
    "Give me a synonym for 'happy'.",
    "How many continents are there?",
    "Write a SQL query that counts the rows of a table named users.",
    "What is the boiling point of water at sea level in Celsius?",
    "List the planets of the solar system in order.",
    "What is a prime number?",
    "Describe a cat to someone who has never seen one.",
    "What year did the Second World War end?",
    "Write a short apology for being late to a meeting.",
    "What is the difference between a list and a tuple in Python?",
    "Suggest a name for a coffee shop.",
    "How do vaccines work?",
    "What is the square root of 144?",
    "Write the first line of a mystery novel.",
    "Explain recursion to a ten-year-old.",
    "What is the largest ocean on Earth?",
    "Convert 5 kilometres to miles.",
    "What is photosynthesis?",
    "Write a bash command that lists files sorted by size.",
    "Who wrote 'Pride and Prejudice'?",
    "What are the three states of matter?",
    "Give one tip for writing clear documentation.",
)


def _prompt_ids(directory: Path) -> list[list[int]]:
    from transformers import AutoTokenizer

    tokenizer: Any = AutoTokenizer.from_pretrained(str(directory), local_files_only=True)
    rendered: list[list[int]] = []
    for prompt in PROMPTS:
        messages = [{"role": "user", "content": prompt}]
        ids = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True, **TEMPLATE_KWARGS
        )
        if not isinstance(ids, list):  # a BatchEncoding in some versions
            ids = ids["input_ids"]
        rendered.append([int(token) for token in ids])
    return rendered


def _first_token_logprobs(
    directory: Path, prompts: list[list[int]], dtype: DType, backends: tuple[str, ...]
) -> tuple[torch.Tensor, float]:
    """``[P, vocab]`` float32 log-probabilities of the first generated token, and load seconds."""
    started = time.perf_counter()
    model = load_model(
        ModelSourceConfig(str(directory), offline=True),
        DeviceConfig(device="cuda", dtype=dtype, backends=backends),
    )
    loaded = time.perf_counter() - started
    ops = model.ops
    [group] = model.kv_specs(block_size=BLOCK_SIZE, layout=ops.kvcache.layout())
    longest = max(len(ids) for ids in prompts)
    blocks = -(-longest // BLOCK_SIZE)
    pool = KVPool(group, blocks + 1, device="cuda")
    buffers = {layer: pool.buffers(layer) for layer in group.layer_ids}
    rows: list[torch.Tensor] = []
    with torch.inference_mode():
        for ids in prompts:
            count = -(-len(ids) // BLOCK_SIZE)
            # The step's tensors live on the device, as the runner gives them to ``plan``.
            meta = ops.attention.plan(
                ForwardMode.MIXED,
                torch.tensor([len(ids)], dtype=torch.int32, device="cuda"),
                torch.tensor([len(ids)], dtype=torch.int32, device="cuda"),
                torch.tensor([list(range(1, count + 1))], dtype=torch.int32, device="cuda"),
                BLOCK_SIZE,
            )
            tokens = torch.tensor(ids, dtype=torch.int64, device="cuda")
            hidden = model(tokens, ForwardContext(attention=meta, kv_buffers=buffers))
            logits = model.compute_logits(hidden[-1:]).to(torch.float32)
            rows.append(torch.log_softmax(logits, dim=-1)[0].cpu())
    del model, pool, buffers
    gc.collect()
    torch.cuda.empty_cache()
    return torch.stack(rows), loaded


def main() -> None:
    parser = argparse.ArgumentParser(description="First-token KL of FP8 against 16 bit.")
    parser.add_argument("--reference", required=True, help="the 16-bit checkpoint directory")
    parser.add_argument("--quantized", required=True, help="the FP8 checkpoint directory")
    parser.add_argument("--dtype", choices=("bf16", "fp16"), default="bf16")
    parser.add_argument("--backends", default="torch")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    dtype = DType.from_str(args.dtype)
    backends = tuple(args.backends.split(","))
    reference_dir, quantized_dir = Path(args.reference), Path(args.quantized)
    prompts = _prompt_ids(reference_dir)
    if prompts != _prompt_ids(quantized_dir):
        raise SystemExit("the two checkpoints render the prompts to different tokens")

    reference, reference_load = _first_token_logprobs(reference_dir, prompts, dtype, backends)
    quantized, quantized_load = _first_token_logprobs(quantized_dir, prompts, dtype, backends)
    if reference.shape != quantized.shape:
        raise SystemExit(f"vocabularies differ: {reference.shape} and {quantized.shape}")
    kl = (reference.exp() * (reference - quantized)).sum(dim=-1)
    entropy = -(reference.exp() * reference).sum(dim=-1)
    agree = reference.argmax(dim=-1) == quantized.argmax(dim=-1)
    per_prompt = [
        {
            "prompt": text,
            "tokens": len(ids),
            "kl_nats": float(value),
            "reference_entropy_nats": float(spread),
            "top1_agrees": bool(same),
            "top1_reference": int(ref),
            "top1_quantized": int(quant),
        }
        for text, ids, value, spread, same, ref, quant in zip(
            PROMPTS,
            prompts,
            kl.tolist(),
            entropy.tolist(),
            agree.tolist(),
            reference.argmax(dim=-1).tolist(),
            quantized.argmax(dim=-1).tolist(),
            strict=True,
        )
    ]
    mean = statistics.fmean(kl.tolist())
    result = {
        "benchmark": "quant_kl",
        "machine": machine(),
        "reference": str(reference_dir),
        "quantized": str(quantized_dir),
        "dtype": dtype.label,
        "backends": list(backends),
        "prompts": len(PROMPTS),
        "threshold_nats": THRESHOLD_NATS,
        "mean_kl_nats": mean,
        "max_kl_nats": max(kl.tolist()),
        "top1_agreement": sum(agree.tolist()) / len(PROMPTS),
        "mean_reference_entropy_nats": statistics.fmean(entropy.tolist()),
        "chat_template_kwargs": TEMPLATE_KWARGS,
        "passes": mean <= THRESHOLD_NATS,
        "load_seconds": {"reference": reference_load, "quantized": quantized_load},
        "per_prompt": per_prompt,
    }
    write(result, args.out)
    print(
        f"mean KL {mean:.5f} nats (max {result['max_kl_nats']:.5f}), top-1 agreement "
        f"{result['top1_agreement']:.3f}: {'passes' if result['passes'] else 'fails'} "
        f"the {THRESHOLD_NATS} threshold"
    )


if __name__ == "__main__":
    main()
