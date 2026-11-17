"""Decode speed end to end: eager against CUDA graphs, fp16 against bf16, per backend.

One engine per configuration loads the model from ``--model`` (a checkpoint directory) and
generates greedily from one prompt per request, ``--tokens`` tokens each. The decode time per step
is the difference between a run of ``--tokens`` tokens and a run of one token (which is the
prefill), divided by ``--tokens - 1``; each configuration is measured ``--repeat`` times after a
warm-up run and the individual runs are kept. The result is JSON with the machine and the versions.

    uv run python benchmarks/e2e/decode.py --model /path/to/SmolLM2-135M \
        --out benchmarks/results/<date>-<machine>/decode.json
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

from _machine import machine, write

from ayaka.configs.cache import CacheConfig
from ayaka.configs.device import DeviceConfig
from ayaka.configs.engine import EngineConfig
from ayaka.configs.model_source import ModelSourceConfig
from ayaka.configs.scheduler import SchedulerConfig
from ayaka.core.params import SamplingParams
from ayaka.engine.llm import LLM
from ayaka.utils.dtypes import DType

PROMPT = "The history of the printing press begins"
CONFIGS = [
    (("torch",), "fp16", False),
    (("torch",), "fp16", True),
    (("flashinfer", "torch"), "fp16", False),
    (("flashinfer", "torch"), "fp16", True),
    (("flashinfer", "torch"), "bf16", False),
    (("flashinfer", "torch"), "bf16", True),
]


def _generate(llm: LLM, batch: int, tokens: int) -> float:
    params = SamplingParams(temperature=0.0, max_tokens=tokens, min_tokens=tokens)
    started = time.perf_counter()
    outs = llm.generate([PROMPT] * batch, params)
    elapsed = time.perf_counter() - started
    failed = [out.error for out in (outs if isinstance(outs, list) else [outs]) if out.error]
    if failed:
        raise RuntimeError(f"a request failed: {failed[0]}")
    return elapsed


def main() -> None:
    parser = argparse.ArgumentParser(description="Decode speed end to end.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--tokens", type=int, default=129)
    parser.add_argument("--batch", type=int, nargs="+", default=[1, 4])
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--out")
    args = parser.parse_args()
    rows: list[dict[str, Any]] = []
    for backends, dtype_name, graphs in CONFIGS:
        dtype = DType.from_str(dtype_name)
        config = EngineConfig(
            source=ModelSourceConfig(args.model, offline=True),
            device=DeviceConfig(device="cuda", dtype=dtype, backends=backends, cuda_graph=graphs),
            cache=CacheConfig(block_size=16, kv_reserve_tokens=2048),
            scheduler=SchedulerConfig(max_running=max(args.batch), max_tokens_per_step=512),
        )
        try:
            llm = LLM.from_config(config)
        except Exception as exc:  # a configuration that cannot start is recorded
            rows.append(
                {
                    "backends": backends,
                    "dtype": dtype_name,
                    "graphs": graphs,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            continue
        with llm:
            for batch in args.batch:
                _measure(llm, rows, backends, dtype_name, graphs, batch, args)
        # The next engine reads its baseline with this one gone.
        del llm
        gc.collect()
    write(
        {
            "benchmark": "e2e/decode",
            "model": args.model,
            "tokens": args.tokens,
            "machine": machine(),
            "results": rows,
        },
        args.out,
    )


def _measure(
    llm: LLM,
    rows: list[dict[str, Any]],
    backends: tuple[str, ...],
    dtype_name: str,
    graphs: bool,
    batch: int,
    args: argparse.Namespace,
) -> None:
    """Time one batch size on one engine and append the row."""
    _generate(llm, batch, args.tokens)  # warm-up
    per_step: list[float] = []
    for _ in range(args.repeat):
        prefill = _generate(llm, batch, 1)
        total = _generate(llm, batch, args.tokens)
        per_step.append((total - prefill) / (args.tokens - 1) * 1e3)
    rows.append(
        {
            "backends": backends,
            "dtype": dtype_name,
            "graphs": graphs,
            "batch": batch,
            "decode_ms_per_step": [round(x, 3) for x in per_step],
            "median_ms": round(statistics.median(per_step), 3),
            "tokens_per_s": round(batch * 1e3 / statistics.median(per_step), 1),
        }
    )


if __name__ == "__main__":
    main()
