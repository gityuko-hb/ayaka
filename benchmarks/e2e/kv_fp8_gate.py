"""FP8 KV cache opt-in gate on Qwen/Qwen3-8B: capacity, KL quality, needle retrieval, and latency.

Measures the four opt-in requirements before recommending ``--kv-dtype fp8``:
1. KV capacity: usable blocks and max context tokens against ``--kv-dtype auto``.
2. Distribution fidelity: next-token KL(bf16 KV || fp8 KV) <= 0.05 nats across 32 prompts
   evaluated after reading from the paged KV cache.
3. Long-context needle retrieval: exact passkey recall at multiple depths (4K-8K tokens).
4. Decode latency at batch=1 and batch=8 in A -> B -> B -> A alternation order.
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
from e2e.quant_kl import PROMPTS, TEMPLATE_KWARGS, THRESHOLD_NATS

from ayaka.configs.cache import CacheConfig
from ayaka.configs.device import DeviceConfig
from ayaka.configs.engine import EngineConfig
from ayaka.configs.model_source import ModelSourceConfig
from ayaka.configs.scheduler import SchedulerConfig
from ayaka.core.params import SamplingParams
from ayaka.core.types import ForwardMode
from ayaka.engine.build import plan_engine
from ayaka.engine.llm import LLM
from ayaka.kvcache.storage.pool import KVPool
from ayaka.kvcache.storage.spec import KVGroup
from ayaka.layers.context import ForwardContext
from ayaka.model_loader.loader import load_model
from ayaka.ops.kvcache import FP8_E4M3_CODEC, RAW_CODEC
from ayaka.utils.dtypes import DType

BLOCK_SIZE = 16


def _render_prompts(model_id: str) -> tuple[Any, list[list[int]]]:
    from transformers import AutoTokenizer

    tokenizer: Any = AutoTokenizer.from_pretrained(model_id, local_files_only=True)
    rendered: list[list[int]] = []
    for prompt in PROMPTS:
        messages = [{"role": "user", "content": prompt}]
        ids = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True, **TEMPLATE_KWARGS
        )
        if not isinstance(ids, list):
            ids = ids["input_ids"]
        rendered.append([int(tok) for tok in ids])
    return tokenizer, rendered


def _run_kl_pass(
    model: Any,
    ops: Any,
    blocks: int,
    prompts: list[list[int]],
    codec: str,
    scale: float,
) -> torch.Tensor:
    groups = model.kv_specs(
        block_size=BLOCK_SIZE,
        layout=ops.kvcache.layout(),
        codec=codec,
        kv_scale=scale,
    )
    pools = [KVPool(g, blocks, device="cuda") for g in groups if isinstance(g, KVGroup)]
    buffers = {lid: pool.buffers(lid) for pool in pools for lid in pool.group.layer_ids}
    logprobs: list[torch.Tensor] = []
    with torch.inference_mode():
        for ids in prompts:
            # Prefill N-1 tokens into paged KV, then decode final token over paged KV!
            split = max(len(ids) - 1, 1)
            c1 = -(-split // BLOCK_SIZE)
            tbl1 = torch.tensor([list(range(1, c1 + 1))], dtype=torch.int32, device="cuda")
            meta1 = ops.attention.plan(
                ForwardMode.PREFILL,
                torch.tensor([split], dtype=torch.int32, device="cuda"),
                torch.tensor([split], dtype=torch.int32, device="cuda"),
                tbl1,
                BLOCK_SIZE,
            )
            hidden = model(
                torch.tensor(ids[:split], dtype=torch.int64, device="cuda"),
                ForwardContext(attention=meta1, kv_buffers=buffers),
            )
            if split < len(ids):
                c2 = -(-len(ids) // BLOCK_SIZE)
                tbl2 = torch.tensor([list(range(1, c2 + 1))], dtype=torch.int32, device="cuda")
                meta2 = ops.attention.plan(
                    ForwardMode.DECODE,
                    torch.tensor([1], dtype=torch.int32, device="cuda"),
                    torch.tensor([len(ids)], dtype=torch.int32, device="cuda"),
                    tbl2,
                    BLOCK_SIZE,
                )
                hidden = model(
                    torch.tensor(ids[split:], dtype=torch.int64, device="cuda"),
                    ForwardContext(attention=meta2, kv_buffers=buffers),
                )
            logits = model.compute_logits(hidden[-1:]).to(torch.float32)
            logprobs.append(torch.log_softmax(logits, dim=-1)[0].cpu())
    return torch.stack(logprobs)


def _eval_kl(
    model_id: str,
    prompts: list[list[int]],
    dtype: DType,
    backends: tuple[str, ...],
    kv_scale: float,
) -> dict[str, Any]:
    """Run one loaded model over all 32 prompts on both raw and FP8 KV pools."""
    model = load_model(
        ModelSourceConfig(model_id, offline=True),
        DeviceConfig(device="cuda", dtype=dtype, backends=backends),
    )
    ops = model.ops
    longest = max(len(ids) for ids in prompts)
    blocks = -(-longest // BLOCK_SIZE) + 1

    ref_lp = _run_kl_pass(model, ops, blocks, prompts, RAW_CODEC, 1.0)
    fp8_lp = _run_kl_pass(model, ops, blocks, prompts, FP8_E4M3_CODEC, kv_scale)
    kl = (ref_lp.exp() * (ref_lp - fp8_lp)).sum(dim=-1)
    entropy = -(ref_lp.exp() * ref_lp).sum(dim=-1)
    agree = ref_lp.argmax(dim=-1) == fp8_lp.argmax(dim=-1)
    mean_kl = statistics.fmean(kl.tolist())

    del model
    gc.collect()
    torch.cuda.empty_cache()
    return {
        "threshold_nats": THRESHOLD_NATS,
        "mean_kl_nats": mean_kl,
        "max_kl_nats": max(kl.tolist()),
        "top1_agreement": sum(agree.tolist()) / len(prompts),
        "mean_reference_entropy_nats": statistics.fmean(entropy.tolist()),
        "passes": mean_kl <= THRESHOLD_NATS,
    }


def _needle_prompt(tokenizer: Any, depth_frac: float, passkey: str, target_tokens: int) -> str:
    filler = "The telemetry log records steady operating pressure across every auxiliary manifold. "
    needle = f"\nImportant verification passkey: {passkey}.\n"
    question = (
        "\nWhat is the exact verification passkey mentioned in the log above? "
        "Reply with ONLY the digits of the passkey."
    )
    approx_chars = target_tokens * 3
    before_len = int(approx_chars * depth_frac)
    after_len = max(0, approx_chars - before_len)
    unit_repeats_before = max(1, before_len // len(filler))
    unit_repeats_after = max(1, after_len // len(filler))
    body = (filler * unit_repeats_before) + needle + (filler * unit_repeats_after) + question
    messages = [{"role": "user", "content": body}]
    rendered = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False, **TEMPLATE_KWARGS
    )
    return str(rendered)


def _engine_config(
    model_id: str,
    dtype: DType,
    backends: tuple[str, ...],
    kv_dtype: str,
    kv_scale: float,
    *,
    cuda_graph: bool,
    max_running: int | None,
    kv_reserve: int,
) -> EngineConfig:
    return EngineConfig(
        source=ModelSourceConfig(model_id, offline=True),
        device=DeviceConfig(device="cuda", dtype=dtype, backends=backends, cuda_graph=cuda_graph),
        cache=CacheConfig(
            block_size=BLOCK_SIZE,
            kv_reserve_tokens=kv_reserve,
            kv_dtype=kv_dtype,  # type: ignore[arg-type]
            kv_fp8_scale=kv_scale,
            prefix_cache=False,
        ),
        scheduler=SchedulerConfig(max_running=max_running, max_tokens_per_step=1024),
    )


def _measure_decode_ms(llm: LLM, prompt: str, batch: int, tokens: int) -> float:
    params_1 = SamplingParams(temperature=0.0, max_tokens=1, ignore_eos=True)
    params_n = SamplingParams(temperature=0.0, max_tokens=tokens, ignore_eos=True)
    # Warm-up
    llm.generate([prompt] * batch, params_n)
    torch.cuda.synchronize()
    t1_start = time.perf_counter()
    llm.generate([prompt] * batch, params_1)
    torch.cuda.synchronize()
    t1 = time.perf_counter() - t1_start

    tn_start = time.perf_counter()
    outs = llm.generate([prompt] * batch, params_n)
    torch.cuda.synchronize()
    tn = time.perf_counter() - tn_start
    for o in outs if isinstance(outs, list) else [outs]:
        if o.error is not None:
            raise RuntimeError(f"generation failed: {o.error.message}")
    return max(tn - t1, 0.0) / max(tokens - 1, 1) * 1e3


def main() -> None:
    parser = argparse.ArgumentParser(description="Opt-in FP8 KV cache quality and speed gate.")
    parser.add_argument("--model", default="Qwen/Qwen3-8B")
    parser.add_argument("--dtype", choices=("bf16", "fp16"), default="bf16")
    parser.add_argument("--backends", default="flashinfer,torch")
    parser.add_argument("--kv-scale", type=float, default=1.0)
    parser.add_argument("--needle-tokens", type=int, default=4096)
    parser.add_argument("--decode-tokens", type=int, default=65)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    dtype = DType.from_str(args.dtype)
    backends = tuple(b.strip() for b in args.backends.split(",") if b.strip())

    # 1. Capacity plan comparison
    plan_auto = plan_engine(
        _engine_config(
            args.model,
            dtype,
            backends,
            "auto",
            1.0,
            cuda_graph=True,
            max_running=None,
            kv_reserve=4096,
        )
    )
    plan_fp8 = plan_engine(
        _engine_config(
            args.model,
            dtype,
            backends,
            "fp8",
            args.kv_scale,
            cuda_graph=True,
            max_running=None,
            kv_reserve=4096,
        )
    )

    # 2. KL divergence across 32 prompts
    tokenizer, rendered_prompts = _render_prompts(args.model)
    kl_report = _eval_kl(args.model, rendered_prompts, dtype, backends, args.kv_scale)

    # 3 & 4. Needle retrieval + A -> B -> B -> A decode latency
    needles = [
        (0.25, "749201"),
        (0.50, "381649"),
        (0.75, "905312"),
    ]
    needle_prompts = [
        (depth, code, _needle_prompt(tokenizer, depth, code, args.needle_tokens))
        for depth, code in needles
    ]
    lat_prompt = (
        "Summarize the architectural benefits of paged attention for concurrent decoding: " * 24
    )

    def run_variant(kv_dtype: str) -> dict[str, Any]:
        cfg = _engine_config(
            args.model,
            dtype,
            backends,
            kv_dtype,
            args.kv_scale,
            cuda_graph=True,
            max_running=8,
            kv_reserve=8192,
        )
        with LLM.from_config(cfg) as llm:
            needle_hits = []
            for depth, code, prompt_txt in needle_prompts:
                out = llm.generate(
                    prompt_txt, SamplingParams(temperature=0.0, max_tokens=24, ignore_eos=False)
                )
                first = out[0] if isinstance(out, list) else out
                text = first.outputs[0].text
                needle_hits.append(
                    {"depth": depth, "passkey": code, "hit": code in text, "text": text.strip()}
                )
            b1_ms = _measure_decode_ms(llm, lat_prompt, 1, args.decode_tokens)
            b8_ms = _measure_decode_ms(llm, lat_prompt, 8, args.decode_tokens)
        gc.collect()
        torch.cuda.empty_cache()
        return {"needle": needle_hits, "batch1_ms_per_step": b1_ms, "batch8_ms_per_step": b8_ms}

    # Strict A -> B -> B -> A schedule
    runs = [
        ("auto", run_variant("auto")),
        ("fp8", run_variant("fp8")),
        ("fp8", run_variant("fp8")),
        ("auto", run_variant("auto")),
    ]
    auto_b1 = statistics.fmean(r["batch1_ms_per_step"] for k, r in runs if k == "auto")
    fp8_b1 = statistics.fmean(r["batch1_ms_per_step"] for k, r in runs if k == "fp8")
    auto_b8 = statistics.fmean(r["batch8_ms_per_step"] for k, r in runs if k == "auto")
    fp8_b8 = statistics.fmean(r["batch8_ms_per_step"] for k, r in runs if k == "fp8")
    needle_auto = runs[0][1]["needle"]
    needle_fp8 = runs[1][1]["needle"]

    payload = {
        "benchmark": "kv_fp8_gate",
        "machine": machine(),
        "model": args.model,
        "dtype": dtype.label,
        "backends": list(backends),
        "kv_fp8_scale": args.kv_scale,
        "capacity": {
            "auto_bytes_per_token": plan_auto.group.spec.bytes_per_token(),
            "fp8_bytes_per_token": plan_fp8.group.spec.bytes_per_token(),
            "auto_kv_blocks": plan_auto.kv_blocks,
            "fp8_kv_blocks": plan_fp8.kv_blocks,
            "auto_max_running_at_4k": plan_auto.max_running,
            "fp8_max_running_at_4k": plan_fp8.max_running,
            "capacity_ratio": plan_fp8.kv_blocks / max(plan_auto.kv_blocks, 1),
        },
        "quality_kl": kl_report,
        "needle": {
            "target_tokens": args.needle_tokens,
            "auto": needle_auto,
            "fp8": needle_fp8,
            "auto_all_pass": all(item["hit"] for item in needle_auto),
            "fp8_all_pass": all(item["hit"] for item in needle_fp8),
        },
        "latency_abba": {
            "decode_tokens": args.decode_tokens,
            "raw_runs": [{"kv_dtype": k, **v} for k, v in runs],
            "auto_batch1_ms": auto_b1,
            "fp8_batch1_ms": fp8_b1,
            "auto_batch8_ms": auto_b8,
            "fp8_batch8_ms": fp8_b8,
            "batch1_speedup": auto_b1 / max(fp8_b1, 1e-9),
            "batch8_speedup": auto_b8 / max(fp8_b8, 1e-9),
        },
    }
    write(payload, args.out)
    print(
        f"KL={kl_report['mean_kl_nats']:.5f} nats (passes={kl_report['passes']}); "
        f"capacity ratio={payload['capacity']['capacity_ratio']:.2f}x; "
        f"needle fp8={payload['needle']['fp8_all_pass']}; "
        f"b1 {auto_b1:.2f}->{fp8_b1:.2f} ms, b8 {auto_b8:.2f}->{fp8_b8:.2f} ms"
    )


if __name__ == "__main__":
    main()
