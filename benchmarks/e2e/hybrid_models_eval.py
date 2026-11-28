"""Evaluate hybrid, sliding/sink, and MXFP4 checkpoints on A100 against Hugging Face transformers.

Measures:
1. 16-token greedy token-ID parity against Hugging Face transformers on the same cached
   checkpoint and compute dtype (bfloat16) for ``Qwen/Qwen3.5-9B``, ``google/gemma-4-E2B-it``,
   and ``openai/gpt-oss-20b``, with CUDA graphs enabled.
2. Chunked prefill invariance and warm state-prefix-cache exact token parity on ``Qwen/Qwen3.5-9B``.
3. State checkpoint stride sweep (``1, 2, 4, 8``) trading host RAM for suffix recompute time.
"""

from __future__ import annotations

import argparse
import gc
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from _machine import machine, write

from ayaka.configs.cache import CacheConfig
from ayaka.configs.device import DeviceConfig
from ayaka.configs.engine import EngineConfig
from ayaka.configs.model_source import ModelSourceConfig
from ayaka.configs.scheduler import SchedulerConfig
from ayaka.core.params import SamplingParams
from ayaka.engine.llm import LLM
from ayaka.utils.dtypes import DType

PROMPT_TEXT = "Explain in two clear sentences how paged key-value caching reduces LLM memory waste:"
DEFAULT_STRIDES = (1, 2, 4, 8)


def _clear_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def _hf_greedy_tokens(
    model_id: str, prompt_text: str, gen_tokens: int
) -> tuple[list[int], list[int]]:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from transformers.utils.quantization_config import Mxfp4Config

    tokenizer: Any = AutoTokenizer.from_pretrained(model_id, local_files_only=True)
    prompt_ids = [int(t) for t in tokenizer.encode(prompt_text, add_special_tokens=True)]
    loader_cls: Any = AutoModelForCausalLM
    extra: dict[str, Any] = {}
    if "Qwen3.5" in model_id or "gemma-4" in model_id:
        from transformers import AutoModelForImageTextToText

        loader_cls = AutoModelForImageTextToText
    if "gpt-oss" in model_id:
        extra["quantization_config"] = Mxfp4Config(dequantize=True)
    hf_model: Any = loader_cls.from_pretrained(
        model_id,
        local_files_only=True,
        dtype=torch.bfloat16,
        **extra,
    ).to("cuda")
    hf_model.eval()
    input_ids = torch.tensor([prompt_ids], dtype=torch.int64, device="cuda")
    attention_mask = torch.ones_like(input_ids)
    with torch.inference_mode():
        seq = hf_model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=gen_tokens,
            do_sample=False,
        )
    out_ids = [int(t) for t in seq[0, len(prompt_ids) :].tolist()]
    del hf_model
    _clear_cuda()
    return prompt_ids, out_ids


def _engine_config(
    model_id: str,
    *,
    max_running: int = 4,
    max_tokens_per_step: int = 512,
    prefix_cache: bool = True,
    state_checkpoint_stride: int = 1,
    kv_reserve_tokens: int = 2048,
) -> EngineConfig:
    return EngineConfig(
        source=ModelSourceConfig(model_id, offline=True),
        device=DeviceConfig(
            device="cuda",
            dtype=DType.BF16,
            backends=("flashinfer", "torch"),
            cuda_graph=True,
        ),
        cache=CacheConfig(
            block_size=16,
            kv_reserve_tokens=kv_reserve_tokens,
            prefix_cache=prefix_cache,
            state_checkpoint_stride=state_checkpoint_stride,
            state_checkpoint_max_bytes=4 * 2**30,
        ),
        scheduler=SchedulerConfig(
            max_running=max_running,
            max_tokens_per_step=max_tokens_per_step,
        ),
    )


def _ayaka_generate(
    llm: LLM, prompt_ids: tuple[int, ...], max_tokens: int
) -> tuple[list[int], float]:
    params = SamplingParams(temperature=0.0, max_tokens=max_tokens, ignore_eos=True)
    started = time.perf_counter()
    output = llm.generate(prompt_ids, params)
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    if isinstance(output, list):
        output = output[0]
    if output.error:
        raise RuntimeError(f"generation error: {output.error}")
    return [int(t) for t in output.outputs[0].token_ids], elapsed_ms


def _eval_model_parity(model_id: str, gen_tokens: int) -> dict[str, Any]:
    prompt_ids, hf_tokens = _hf_greedy_tokens(model_id, PROMPT_TEXT, gen_tokens)
    with LLM.from_config(_engine_config(model_id)) as llm:
        # Warm up graphs & measure greedy generation
        warmup_tokens, _ = _ayaka_generate(llm, tuple(prompt_ids), gen_tokens)
        ayaka_tokens, gen_ms = _ayaka_generate(llm, tuple(prompt_ids), gen_tokens)
        startup = llm.engine.startup
        plan_groups = len(llm.engine.kv.groups)
        weights_bytes = startup.profile.weights_bytes if startup is not None else 0
        graph_buckets = [item.bucket for item in startup.graphs] if startup is not None else []
        quant = llm.engine.runner.model.quant_format
    _clear_cuda()
    exact_match = (ayaka_tokens == hf_tokens) and (warmup_tokens == hf_tokens)
    return {
        "model": model_id,
        "prompt_len": len(prompt_ids),
        "gen_tokens": gen_tokens,
        "hf_tokens": hf_tokens,
        "ayaka_tokens": ayaka_tokens,
        "exact_greedy_match": exact_match,
        "kv_groups": plan_groups,
        "cuda_graph_buckets": graph_buckets,
        "weights_bytes": weights_bytes,
        "quant_format": None if quant is None else quant.name,
        "warm_generate_ms": gen_ms,
    }


def _run_cold_or_chunked(
    model_id: str,
    prompt_ids: tuple[int, ...],
    gen_tokens: int,
    *,
    max_tokens_per_step: int,
) -> list[int]:
    with LLM.from_config(
        _engine_config(model_id, max_tokens_per_step=max_tokens_per_step, prefix_cache=False)
    ) as llm:
        tokens, _ = _ayaka_generate(llm, prompt_ids, gen_tokens)
    _clear_cuda()
    return tokens


def _run_stride_trial(
    model_id: str,
    prompt_a: tuple[int, ...],
    prompt_b: tuple[int, ...],
    warmup_ids: tuple[int, ...],
    stride: int,
    gen_tokens: int,
    single_b: list[int],
) -> dict[str, Any]:
    with LLM.from_config(
        _engine_config(
            model_id,
            max_tokens_per_step=512,
            prefix_cache=True,
            state_checkpoint_stride=stride,
        )
    ) as llm:
        store = llm.engine.kv.state_checkpoints
        if store is None:
            raise RuntimeError("expected active state checkpoint store")
        # Warm CUDA runtime on unrelated short prompt
        _ayaka_generate(llm, warmup_ids, 2)
        pre_bytes = store.current_bytes
        _, ttft_a_1 = _ayaka_generate(llm, prompt_a, 1)
        ckpts_after_a = store.count
        bytes_after_a = store.current_bytes
        hits_before = store.hits
        _, ms_b_first = _ayaka_generate(llm, prompt_b, 1)
        hits_after = store.hits
        warm_b_tokens, ms_b_full = _ayaka_generate(llm, prompt_b, gen_tokens)
        decode_ms = (ms_b_full - ms_b_first) / max(gen_tokens - 1, 1)
    _clear_cuda()
    return {
        "stride_blocks": stride,
        "checkpoint_count": ckpts_after_a,
        "host_checkpoint_bytes": bytes_after_a,
        "host_checkpoint_delta_bytes": bytes_after_a - pre_bytes,
        "checkpoint_hits": hits_after - hits_before,
        "cold_prefill_plus1_ms": ttft_a_1,
        "warm_suffix_plus1_ms": ms_b_first,
        "decode_ms_per_token": decode_ms,
        "exact_match_vs_cold": warm_b_tokens == single_b,
    }


def _eval_qwen35_stride_sweep(
    model_id: str,
    strides: tuple[int, ...],
    gen_tokens: int,
) -> dict[str, Any]:
    from transformers import AutoTokenizer

    tok: Any = AutoTokenizer.from_pretrained(model_id, local_files_only=True)
    system_text = "You are a systems engineer optimizing LLM serving kernels. " * 32
    system_ids = [int(t) for t in tok.encode(system_text, add_special_tokens=True)][:256]
    if len(system_ids) != 256:
        raise RuntimeError(f"expected 256 system tokens, got {len(system_ids)}")
    suffix_a = [
        int(t)
        for t in tok.encode(
            " Summarize sliding-window KV caching briefly." * 8, add_special_tokens=False
        )
    ][:48]
    suffix_b = [
        int(t)
        for t in tok.encode(
            " List three tradeoffs for CUDA graph bucketing." * 8, add_special_tokens=False
        )
    ][:48]
    if len(suffix_a) != 48 or len(suffix_b) != 48:
        raise RuntimeError(f"expected 48 suffix tokens, got {len(suffix_a)} / {len(suffix_b)}")
    prompt_a = (*system_ids, *suffix_a)
    prompt_b = (*system_ids, *suffix_b)

    single_b = _run_cold_or_chunked(model_id, prompt_b, gen_tokens, max_tokens_per_step=512)
    chunked_b = _run_cold_or_chunked(model_id, prompt_b, gen_tokens, max_tokens_per_step=32)
    print(
        f"[stride] {model_id}: chunk_invariance={single_b == chunked_b}",
        flush=True,
    )

    sweep_rows: list[dict[str, Any]] = []
    warmup_ids = tuple(system_ids[:16])
    for stride in strides:
        row = _run_stride_trial(
            model_id,
            prompt_a,
            prompt_b,
            warmup_ids,
            stride,
            gen_tokens,
            single_b,
        )
        print(
            f"[stride] stride={stride} ckpts={row['checkpoint_count']} "
            f"bytes={row['host_checkpoint_bytes']} hits={row['checkpoint_hits']} "
            f"warm_ms={row['warm_suffix_plus1_ms']:.2f} match={row['exact_match_vs_cold']}",
            flush=True,
        )
        sweep_rows.append(row)

    return {
        "model": model_id,
        "shared_prefix_tokens": len(system_ids),
        "suffix_tokens": len(suffix_b),
        "single_chunk_tokens": single_b,
        "multi_chunk_tokens": chunked_b,
        "chunk_invariance_match": single_b == chunked_b,
        "stride_sweep": sweep_rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid model A100 parity and stride benchmark.")
    parser.add_argument("--gen-tokens", type=int, default=16)
    parser.add_argument(
        "--out",
        default="benchmarks/results/2026-11-27-nvidia-a100-sxm4-80gb/hybrid-models-parity-and-stride.json",
    )
    args = parser.parse_args()

    models = [
        "Qwen/Qwen3.5-9B",
        "google/gemma-4-E2B-it",
        "openai/gpt-oss-20b",
    ]
    parity: list[dict[str, Any]] = []
    for m in models:
        row = _eval_model_parity(m, args.gen_tokens)
        print(
            f"[parity] {m}: match={row['exact_greedy_match']} "
            f"kv_groups={row['kv_groups']} graphs={row['cuda_graph_buckets']} "
            f"gen_ms={row['warm_generate_ms']:.2f}",
            flush=True,
        )
        parity.append(row)
    qwen35_stride = _eval_qwen35_stride_sweep("Qwen/Qwen3.5-9B", DEFAULT_STRIDES, args.gen_tokens)
    payload = {
        "benchmark": "hybrid_models_parity_and_stride",
        "machine": machine(),
        "parity": parity,
        "qwen3_5_state_prefix_and_stride": qwen35_stride,
    }
    write(payload, args.out)


if __name__ == "__main__":
    main()
