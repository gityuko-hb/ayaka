"""Measure cold prefill vs host-RAM KV tier promotion on a tight device KV pool."""

from __future__ import annotations

import argparse
import dataclasses
import json
import tempfile
import time
from pathlib import Path

from ayaka.configs.cache import CacheConfig
from ayaka.configs.device import DeviceConfig
from ayaka.configs.engine import EngineConfig
from ayaka.configs.host_memory import HostMemoryConfig
from ayaka.configs.model_source import ModelSourceConfig
from ayaka.configs.scheduler import SchedulerConfig
from ayaka.core.params import SamplingParams
from ayaka.core.request import GenerateRequest
from ayaka.core.types import RequestId
from ayaka.engine.llm import LLM
from ayaka.utils.dtypes import COMPUTE_DTYPES

_DTYPES = {d.label: d for d in COMPUTE_DTYPES}


def _measure_run(
    config: EngineConfig,
    prompts: list[tuple[int, ...]],
    filler: tuple[int, ...],
    max_tokens: int,
) -> dict[str, object]:
    with LLM.from_config(config) as llm:
        warm_prompt = prompts[0]
        # 1. Cold run of the shared prompt.
        t0 = time.perf_counter()
        cold_out = llm.generate(
            GenerateRequest(
                request_id=RequestId("cold"),
                prompt=warm_prompt,
                sampling=SamplingParams(temperature=0.0, max_tokens=max_tokens),
            )
        )
        cold_ms = (time.perf_counter() - t0) * 1000.0
        if isinstance(cold_out, list):
            raise TypeError("expected single RequestOutput for cold prompt")

        # 2. Flood cache with filler to evict the warm prompt from device KV.
        for idx in range(4):
            llm.generate(
                GenerateRequest(
                    request_id=RequestId(f"evict-{idx}"),
                    prompt=tuple(x + idx for x in filler),
                    sampling=SamplingParams(temperature=0.0, max_tokens=1),
                )
            )
        if llm.engine.kv.host_store is not None:
            llm.engine.kv.host_store.flush()

        # 3. Re-run the warm prompt and compare latency + tokens + host promotions.
        t1 = time.perf_counter()
        warm_out = llm.generate(
            GenerateRequest(
                request_id=RequestId("warm"),
                prompt=warm_prompt,
                sampling=SamplingParams(temperature=0.0, max_tokens=max_tokens),
            )
        )
        warm_ms = (time.perf_counter() - t1) * 1000.0
        if isinstance(warm_out, list):
            raise TypeError("expected single RequestOutput for warm prompt")
        hs = llm.engine.kv.host_store.stats() if llm.engine.kv.host_store is not None else None
        return {
            "device": config.device.device,
            "host_kv_enabled": config.host_memory.enabled,
            "cold_ms": round(cold_ms, 2),
            "warm_ms": round(warm_ms, 2),
            "speedup": round(cold_ms / max(warm_ms, 1e-6), 3),
            "cached_tokens": warm_out.usage.cached_tokens if warm_out.usage else 0,
            "parity": warm_out.outputs[0].token_ids == cold_out.outputs[0].token_ids,
            "offloads": hs.offloads if hs else 0,
            "promotions": hs.promotions if hs else 0,
        }


def main(argv: list[str] | None = None) -> int:
    """Run the cold vs host-promoted KV benchmark against a local model directory."""
    parser = argparse.ArgumentParser(description="Benchmark host-RAM KV eviction and promotion")
    parser.add_argument("--model", required=True, help="Local model directory or repo id")
    parser.add_argument("--device", default="cpu", help="cpu or cuda")
    parser.add_argument("--dtype", choices=sorted(_DTYPES), default="fp32")
    parser.add_argument("--prompt-tokens", type=int, default=64)
    parser.add_argument("--max-tokens", type=int, default=8)
    parser.add_argument("--num-blocks", type=int, default=16)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    base = EngineConfig(
        source=ModelSourceConfig(args.model),
        device=DeviceConfig(device=args.device, dtype=_DTYPES[args.dtype]),
        cache=CacheConfig(block_size=16, num_blocks=args.num_blocks, prefix_cache=True),
        scheduler=SchedulerConfig(max_running=1, max_tokens_per_step=args.prompt_tokens),
    )
    prompt = tuple((i % 200) + 10 for i in range(args.prompt_tokens))
    filler = tuple((i % 200) + 300 for i in range(args.prompt_tokens))

    baseline = _measure_run(base, [prompt], filler, args.max_tokens)
    with_host = _measure_run(
        dataclasses.replace(
            base,
            host_memory=HostMemoryConfig(
                enabled=True,
                ram_keep_fraction=0.20,
                max_host_kv_mb=512,
                allow_pageable=(args.device == "cpu"),
            ),
        ),
        [prompt],
        filler,
        args.max_tokens,
    )
    report = {
        "baseline": baseline,
        "host_kv": with_host,
        "temp_root": str(Path(tempfile.gettempdir())),
    }
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print("Baseline :", baseline)
        print("Host KV  :", with_host)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
