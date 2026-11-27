"""Returning to an evicted conversation: promoting its KV from host RAM against recomputing it.

Two engines with the same model and the same small device KV pool, one with the host-RAM tier
(``--host-kv``), one without. In each, a conversation of ``--prompt-tokens`` tokens runs, other
prompts of the same length push its blocks out of the device pool, and the conversation comes
back. Without the tier its prefix must be computed again (the cold prefill of the gate); with it
the evicted blocks are copied back from host RAM. Measured on the return, per arm:

    ttft_ms        the time of a one-token request: prefill (or promotion) plus one step
    cached_tokens  prompt tokens the engine did not compute
    promotions     blocks the host tier copied back (0 without the tier)

and, once per arm, whether ``--parity-tokens`` greedy tokens after the return equal those of the
first visit (the exactness gate). Before anything is timed, each engine serves a warm-up request
of the same length, so the first-use costs of the kernels are not counted. Rounds alternate the
two engines; each number is the median over ``--rounds``. The gate reads ``ratio`` =
``ttft_ms(host tier) / ttft_ms(recompute)``, which must stay below 0.5 for prompts of 4K tokens
and more.

    uv run python benchmarks/e2e/bench_host_kv_tier.py --model Qwen/Qwen3-8B --device cuda \\
        --dtype bf16 --prompt-tokens 4096 8192 --out host_kv_tier.json
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import statistics
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

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

if TYPE_CHECKING:
    from ayaka.core.outputs import RequestOutput

_DTYPES = {d.label: d for d in COMPUTE_DTYPES}
_BLOCK = 16


def _prompt(seed: int, length: int) -> tuple[int, ...]:
    # Distinct ids per seed, so two prompts never share a cached block.
    return tuple(1000 + (seed * 7919 + i * 104729) % 50000 for i in range(length))


def _run(llm: LLM, request_id: str, prompt: tuple[int, ...], tokens: int) -> RequestOutput:
    output = llm.generate(
        GenerateRequest(
            request_id=RequestId(request_id),
            prompt=prompt,
            sampling=SamplingParams(temperature=0.0, max_tokens=tokens, ignore_eos=True),
        )
    )
    if isinstance(output, list):
        raise TypeError("expected one output")
    return output


def _visit(
    llm: LLM, tag: str, conversation: tuple[int, ...], others: int, seed: int
) -> dict[str, Any]:
    """Run the conversation, evict it with ``others`` prompts of its length, time the return."""
    length = len(conversation)
    _run(llm, f"{tag}-first", conversation, 1)
    for index in range(others):
        _run(llm, f"{tag}-evict-{index}", _prompt(seed + index, length), 1)
    store = llm.engine.kv.host_store
    if store is not None:
        store.flush()  # the write-behind copies end before the return, as between two turns
    before = store.stats().promotions if store is not None else 0
    start = time.perf_counter()
    back = _run(llm, f"{tag}-return", conversation, 1)
    ttft = (time.perf_counter() - start) * 1000.0
    after = store.stats().promotions if store is not None else 0
    return {
        "ttft_ms": ttft,
        "cached_tokens": back.usage.cached_tokens if back.usage else 0,
        "promotions": after - before,
    }


def _parity(llm: LLM, tag: str, conversation: tuple[int, ...], others: int, tokens: int) -> bool:
    first = _run(llm, f"{tag}-parity-first", conversation, tokens)
    for index in range(others):
        _run(llm, f"{tag}-parity-evict-{index}", _prompt(30_000 + index, len(conversation)), 1)
    store = llm.engine.kv.host_store
    if store is not None:
        store.flush()
    back = _run(llm, f"{tag}-parity-return", conversation, tokens)
    return back.outputs[0].token_ids == first.outputs[0].token_ids


def _engine(args: argparse.Namespace, length: int, host: bool, *, blocks: int | None = None) -> LLM:
    # Room for two conversations of this length: the third prompt evicts the first.
    pool_blocks = blocks if blocks is not None else 2 * (length // _BLOCK + 2) + 1
    config = EngineConfig(
        source=ModelSourceConfig(args.model, offline=True),
        device=DeviceConfig(
            device=args.device,
            dtype=_DTYPES[args.dtype],
            backends=tuple(args.backends),
            cuda_graph=args.cuda_graph,
        ),
        cache=CacheConfig(
            block_size=_BLOCK,
            num_blocks=pool_blocks,
            prefix_cache=True,
            kv_reserve_tokens=length + 64,
        ),
        scheduler=SchedulerConfig(max_running=1, max_tokens_per_step=args.chunk),
    )
    if host:
        config = dataclasses.replace(
            config,
            host_memory=HostMemoryConfig(
                enabled=True,
                ram_keep_fraction=0.20,
                max_host_kv_mb=args.host_kv_mb,
                allow_pageable=args.device == "cpu",
            ),
        )
    return LLM.from_config(config)


def _write_behind_round(
    llm: LLM, tag: str, prompt_len: int, decode_len: int, count: int, seed_base: int
) -> dict[str, Any]:
    store = llm.engine.kv.host_store
    offloads_before = store.stats().offloads if store is not None else 0
    prefills: list[float] = []
    decodes: list[float] = []
    tpots: list[float] = []
    t0 = time.perf_counter()
    for idx in range(count):
        out = _run(llm, f"{tag}-{idx}", _prompt(seed_base + idx, prompt_len), decode_len)
        if out.metrics is not None:
            prefills.append(out.metrics.prefill_time * 1000.0)
            decodes.append(out.metrics.decode_time * 1000.0)
            tpots.append(out.metrics.tpot * 1000.0)
    wall_ms = (time.perf_counter() - t0) * 1000.0
    if store is not None:
        store.flush()
    offloads_after = store.stats().offloads if store is not None else 0
    return {
        "wall_ms": wall_ms,
        "prefill_ms": statistics.median(prefills) if prefills else 0.0,
        "decode_ms": statistics.median(decodes) if decodes else 0.0,
        "tpot_ms": statistics.median(tpots) if tpots else 0.0,
        "offloads": offloads_after - offloads_before,
        "pinned_memory": store.stats().pinned_memory if store is not None else False,
    }


def _write_behind_overhead(args: argparse.Namespace) -> dict[str, Any]:
    total_len = args.wb_prompt + args.wb_decode
    blocks = 2 * (total_len // _BLOCK + 2) + 1
    engines = {
        "baseline": _engine(args, total_len, False, blocks=blocks),
        "host tier": _engine(args, total_len, True, blocks=blocks),
    }
    try:
        for name, llm in engines.items():
            _write_behind_round(llm, f"warm-{name}", args.wb_prompt, args.wb_decode, 3, 50_000)
        rows: dict[str, list[dict[str, Any]]] = {name: [] for name in engines}
        for index in range(args.rounds):
            for name, llm in engines.items():
                seed_base = 60_000 + index * 100 + (0 if name == "baseline" else 50)
                rows[name].append(
                    _write_behind_round(
                        llm,
                        f"wb-r{index}-{name}",
                        args.wb_prompt,
                        args.wb_decode,
                        args.wb_requests,
                        seed_base,
                    )
                )
        summary: dict[str, Any] = {
            "prompt_tokens": args.wb_prompt,
            "decode_tokens": args.wb_decode,
            "requests_per_round": args.wb_requests,
            "cuda_graph": args.cuda_graph,
            "rounds": args.rounds,
        }
        for name, values in rows.items():
            summary[name] = {
                "wall_ms_median": round(statistics.median(v["wall_ms"] for v in values), 2),
                "prefill_ms_median": round(statistics.median(v["prefill_ms"] for v in values), 2),
                "decode_ms_median": round(statistics.median(v["decode_ms"] for v in values), 2),
                "tpot_ms_median": round(statistics.median(v["tpot_ms"] for v in values), 3),
                "offloads_median": statistics.median(v["offloads"] for v in values),
                "pinned_memory": all(v["pinned_memory"] for v in values),
                "raw_rounds": values,
            }
        base_wall = summary["baseline"]["wall_ms_median"]
        host_wall = summary["host tier"]["wall_ms_median"]
        base_dec = summary["baseline"]["decode_ms_median"]
        host_dec = summary["host tier"]["decode_ms_median"]
        summary["wall_overhead_pct"] = round((host_wall - base_wall) / base_wall * 100.0, 2)
        summary["decode_overhead_pct"] = (
            round((host_dec - base_dec) / base_dec * 100.0, 2) if base_dec > 0 else 0.0
        )
        return summary
    finally:
        for llm in engines.values():
            llm.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--model", required=True, help="a local model directory or repo id")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=sorted(_DTYPES), default="bf16")
    parser.add_argument("--backends", nargs="+", default=["flashinfer", "torch"])
    parser.add_argument("--cuda-graph", action="store_true")
    parser.add_argument("--prompt-tokens", type=int, nargs="+", default=[4096, 8192])
    parser.add_argument("--chunk", type=int, default=2048, help="prefill tokens per step")
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--parity-tokens", type=int, default=32)
    parser.add_argument("--host-kv-mb", type=int, default=8192)
    parser.add_argument(
        "--write-behind-overhead",
        action="store_true",
        help="also measure steady-state async write-behind overhead under eviction",
    )
    parser.add_argument(
        "--only-write-behind",
        action="store_true",
        help="skip the promotion sweep and run only write-behind overhead",
    )
    parser.add_argument("--wb-prompt", type=int, default=1024)
    parser.add_argument("--wb-decode", type=int, default=128)
    parser.add_argument("--wb-requests", type=int, default=4)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)

    report: dict[str, Any] = {"model": args.model, "rounds": args.rounds, "lengths": {}}
    if not args.only_write_behind:
        for length in args.prompt_tokens:
            engines = {
                "recompute": _engine(args, length, False),
                "host tier": _engine(args, length, True),
            }
            try:
                for name, llm in engines.items():  # warm-up: first-use kernel costs, not measured
                    _visit(llm, f"warm-{name}", _prompt(1, length), 2, 10_000)
                rows: dict[str, list[dict[str, Any]]] = {name: [] for name in engines}
                for index in range(args.rounds):
                    for name, llm in engines.items():
                        seed = 20_000 + 10 * index
                        rows[name].append(
                            _visit(llm, f"r{index}-{name}", _prompt(100 + index, length), 2, seed)
                        )
                result: dict[str, Any] = {}
                for name, values in rows.items():
                    result[name] = {
                        "ttft_ms": round(statistics.median(v["ttft_ms"] for v in values), 2),
                        "cached_tokens": statistics.median(v["cached_tokens"] for v in values),
                        "promotions": statistics.median(v["promotions"] for v in values),
                        "token_parity": _parity(
                            engines[name], name, _prompt(900, length), 2, args.parity_tokens
                        ),
                    }
                result["ratio"] = round(
                    result["host tier"]["ttft_ms"] / result["recompute"]["ttft_ms"], 3
                )
            finally:
                for llm in engines.values():
                    llm.close()
            report["lengths"][str(length)] = result
            print(f"{length} tokens: {json.dumps(result)}", flush=True)
    if args.write_behind_overhead or args.only_write_behind:
        wb = _write_behind_overhead(args)
        report["write_behind"] = wb
        print(f"write-behind: {json.dumps(wb)}", flush=True)
    if args.out is not None:
        args.out.write_text(json.dumps(report, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
