"""Where greedy ids of an arm first differ from plain decoding's, and how close that call was.

Every arm (plain decoding on other code paths, lookup speculation) is compared with plain
decoding of one request at a time with CUDA graphs. At the first differing position, a float32
``transformers`` forward pass over the shared prefix gives the logit margin between the token
the reference chose and the token the arm chose: a margin of a few hundredths of a nat is a near
tie that the kernels' reduction order can flip, not an error of the acceptance rule.

``PROBE_FP32=1`` runs the engines in float32 with the torch kernels instead of bf16 with
flashinfer; there every arm should match the reference exactly.

Usage::

    uv run python benchmarks/e2e/spec_divergence_probe.py Qwen/Qwen3-8B out.json
"""

from __future__ import annotations

import gc
import json
import os
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import torch

from ayaka.configs.cache import CacheConfig
from ayaka.configs.device import DeviceConfig
from ayaka.configs.engine import EngineConfig
from ayaka.configs.model_source import ModelSourceConfig
from ayaka.configs.scheduler import SchedulerConfig
from ayaka.configs.speculative import SpeculativeConfig
from ayaka.configs.tokenizer import TokenizerConfig
from ayaka.core.params import SamplingParams
from ayaka.engine.llm import LLM
from ayaka.model_loader.source import resolve_source
from ayaka.utils.dtypes import DType
from tokenizers import Tokenizer

MODEL = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen3-8B"
OUT = Path(sys.argv[2] if len(sys.argv) > 2 else "probe.json")
MAX_TOKENS = 256
DTYPE = DType.FP32 if os.environ.get("PROBE_FP32") else DType.BF16
BACKENDS = ("torch",) if os.environ.get("PROBE_FP32") else ("flashinfer", "torch")

PROMPTS = {
    "code_edit": "Add type annotations to this function and keep the logic identical:\n```python\n"
    "def merge(a, b):\n    out = []\n    i = j = 0\n    while i < len(a) and j < len(b):\n"
    "        if a[i] <= b[j]:\n            out.append(a[i]); i += 1\n        else:\n"
    "            out.append(b[j]); j += 1\n    out.extend(a[i:]); out.extend(b[j:])\n"
    "    return out\n```\n",
    "prose": "Write a short story about a lighthouse keeper who finds a glass compass.",
    "json_tools": 'Return a JSON object with keys "name", "args" and "reason" for a call to a '
    'tool named read_file that reads "src/main.py" lines 10 to 40.',
    "random": "qz vex plinth 83 orbweaver lacquer +=? manifold kestrel 0x7f umbra drizzle tenet",
}


def engine_config(snap: Path, spec: SpeculativeConfig, *, graphs: bool, rows: int) -> EngineConfig:
    return EngineConfig(
        source=ModelSourceConfig(MODEL, offline=True),
        tokenizer=TokenizerConfig(tokenizer=str(snap), offline=True, skip_tokenizer_init=True),
        device=DeviceConfig(device="cuda", dtype=DTYPE, backends=BACKENDS, cuda_graph=graphs),
        cache=CacheConfig(block_size=16, num_blocks=4096, kv_reserve_tokens=8192),
        scheduler=SchedulerConfig(max_running=rows, max_tokens_per_step=1024),
        speculative=spec,
    )


def run(
    snap: Path,
    prompts: dict[str, tuple[int, ...]],
    spec: SpeculativeConfig,
    *,
    graphs: bool,
    batch: bool,
    logprobs: int | None,
) -> dict[str, list[int]]:
    params = SamplingParams(
        temperature=0.0, max_tokens=MAX_TOKENS, ignore_eos=True, logprobs=logprobs
    )
    names = list(prompts)
    with LLM.from_config(engine_config(snap, spec, graphs=graphs, rows=len(names))) as llm:
        if batch:
            outs = llm.generate([prompts[n] for n in names], params)
            result = {n: list(o.outputs[0].token_ids) for n, o in zip(names, outs, strict=True)}
        else:
            result = {n: list(llm.generate(prompts[n], params).outputs[0].token_ids) for n in names}
    gc.collect()
    torch.cuda.empty_cache()
    return result


def main() -> None:
    snap = resolve_source(ModelSourceConfig(MODEL, offline=True))
    tok = Tokenizer.from_file(str(snap / "tokenizer.json"))
    prompts = {k: tuple(tok.encode(v).ids) for k, v in PROMPTS.items()}
    off = SpeculativeConfig(enabled=False)
    lookup_fixed = SpeculativeConfig(
        enabled=True, method="lookup", controller="fixed", fixed_k=4, max_draft_tokens=4
    )
    arms: dict[str, dict[str, Any]] = {
        "plain_graph_b1": {"spec": off, "graphs": True, "batch": False, "logprobs": None},
        "plain_graph_b1_repeat": {"spec": off, "graphs": True, "batch": False, "logprobs": None},
        "plain_eager_b1": {"spec": off, "graphs": False, "batch": False, "logprobs": None},
        "plain_graph_b1_logprobs": {"spec": off, "graphs": True, "batch": False, "logprobs": 2},
        "plain_graph_batch4": {"spec": off, "graphs": True, "batch": True, "logprobs": None},
        "lookup_fixed4_b1": {
            "spec": lookup_fixed,
            "graphs": True,
            "batch": False,
            "logprobs": None,
        },
        "lookup_fixed8_b1": {
            "spec": replace(lookup_fixed, fixed_k=8, max_draft_tokens=8),
            "graphs": True,
            "batch": False,
            "logprobs": None,
        },
        "lookup_cost_batch4": {
            "spec": SpeculativeConfig(enabled=True, method="lookup"),
            "graphs": True,
            "batch": True,
            "logprobs": None,
        },
    }
    outputs = {name: run(snap, prompts, **arm) for name, arm in arms.items()}
    reference = outputs["plain_graph_b1"]

    divergences: list[dict[str, Any]] = []
    for arm, result in outputs.items():
        for name, ids in result.items():
            ref = reference[name]
            pos = next((i for i, (a, b) in enumerate(zip(ref, ids, strict=True)) if a != b), None)
            divergences.append(
                {
                    "arm": arm,
                    "prompt": name,
                    "first_difference": pos,
                    "reference_token": None if pos is None else ref[pos],
                    "arm_token": None if pos is None else ids[pos],
                }
            )

    from transformers import AutoModelForCausalLM

    hf = (
        AutoModelForCausalLM.from_pretrained(str(snap), dtype=torch.float32, local_files_only=True)
        .cuda()
        .eval()
    )
    for d in divergences:
        if d["first_difference"] is None:
            continue
        prefix = list(prompts[d["prompt"]]) + reference[d["prompt"]][: d["first_difference"]]
        with torch.no_grad():
            logits = hf(torch.tensor([prefix], device="cuda")).logits[0, -1].float()
        top2 = torch.topk(logits, 2)
        d["fp32_reference_minus_arm"] = round(
            float(logits[d["reference_token"]] - logits[d["arm_token"]]), 5
        )
        d["fp32_top2_margin"] = round(float(top2.values[0] - top2.values[1]), 5)
        d["fp32_argmax"] = int(top2.indices[0])
    OUT.write_text(
        json.dumps({"model": MODEL, "max_tokens": MAX_TOKENS, "divergences": divergences}, indent=1)
    )
    for d in divergences:
        print(d)


if __name__ == "__main__":
    main()
