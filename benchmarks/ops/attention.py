"""Attention per backend on fixed shapes: which backend should serve ``attention`` here.

Every backend that loads runs the same steps on the same paged cache: decode steps at several
batch sizes and context lengths, and a prefill chunk. Each step is planned once and attended
``--repeat`` times; the time is the median of CUDA-event timings after ``--warmup`` runs. The
result is JSON with the machine and the versions (``benchmarks/_machine.py``).

Per-op numbers choose a backend for a family on a device; they are not Ayaka's performance, which
is the end-to-end number of ``benchmarks/e2e``.

    uv run python benchmarks/ops/attention.py --out benchmarks/results/<run>/attention.json
"""

from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from _machine import machine, write

from ayaka.core.types import ForwardMode, KVLayoutKind
from ayaka.ops.backend import load_backend
from ayaka.ops.kvcache import RAW_CODEC, KVBuffers
from ayaka.utils.dtypes import DType

BLOCK = 16
#: (query heads, key/value heads, head width): TinyLlama-1.1B and SmolLM2-135M.
SHAPES = {"tinyllama": (32, 4, 64), "smollm2-135m": (9, 3, 64)}
DECODE = [(1, 512), (1, 2048), (1, 8192), (4, 2048), (8, 2048)]
PREFILL = [(512, 512), (256, 2048)]  # (rows, context): one request


def _time(fn: Any, *, warmup: int, repeat: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(repeat):
        start, stop = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        stop.record()
        stop.synchronize()
        times.append(start.elapsed_time(stop))
    return statistics.median(times)


def _step(ops: Any, rows: list[int], lengths: list[int], heads: tuple[int, int, int], dtype: Any):
    q_heads, kv_heads, dim = heads
    blocks = sum(-(-n // BLOCK) for n in lengths) + 1
    k = torch.randn(blocks, BLOCK, kv_heads, dim, dtype=dtype, device="cuda")
    v = torch.randn_like(k)
    buffers = KVBuffers(k, v, KVLayoutKind.NHD, RAW_CODEC)
    tables, first = [], 1
    for n in lengths:
        need = -(-n // BLOCK)
        tables.append(list(range(first, first + need)))
        first += need
    width = max(len(t) for t in tables)
    table = torch.tensor([t + [0] * (width - len(t)) for t in tables], dtype=torch.int32)
    meta = ops.attention.plan(
        ForwardMode.DECODE if set(rows) == {1} else ForwardMode.PREFILL,
        torch.tensor(rows, dtype=torch.int32, device="cuda"),
        torch.tensor(lengths, dtype=torch.int32, device="cuda"),
        table.cuda(),
        BLOCK,
    )
    q = torch.randn(sum(rows), q_heads, dim, dtype=dtype, device="cuda")
    out = torch.empty_like(q)
    return lambda: ops.attention.attend(q, buffers, meta, out, scale=dim**-0.5)


def main() -> None:
    parser = argparse.ArgumentParser(description="Attention per backend.")
    parser.add_argument("--dtype", default="fp16", choices=["fp16", "bf16"])
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeat", type=int, default=50)
    parser.add_argument("--out")
    args = parser.parse_args()
    dtype = DType.from_str(args.dtype)
    torch_dtype = dtype.torch_dtype
    rows: list[dict[str, Any]] = []
    for backend in ("torch", "flashinfer"):
        try:
            ops = load_backend(
                [backend, "torch"] if backend != "torch" else ["torch"], device="cuda", dtype=dtype
            )
        except Exception as exc:  # a backend that does not load is recorded
            rows.append({"backend": backend, "error": f"{type(exc).__name__}: {exc}"})
            continue
        if ops.providers["attention"] != backend:
            rows.append({"backend": backend, "error": "not chosen for attention here"})
            continue
        for shape, heads in SHAPES.items():
            for batch, context in DECODE:
                fn = _step(ops, [1] * batch, [context] * batch, heads, torch_dtype)
                ms = _time(fn, warmup=args.warmup, repeat=args.repeat)
                rows.append(
                    {
                        "backend": backend,
                        "shape": shape,
                        "kind": "decode",
                        "batch": batch,
                        "context": context,
                        "ms": round(ms, 4),
                    }
                )
            for count, context in PREFILL:
                fn = _step(ops, [count], [context], heads, torch_dtype)
                ms = _time(fn, warmup=args.warmup, repeat=args.repeat)
                rows.append(
                    {
                        "backend": backend,
                        "shape": shape,
                        "kind": "prefill",
                        "rows": count,
                        "context": context,
                        "ms": round(ms, 4),
                    }
                )
    write(
        {
            "benchmark": "ops/attention",
            "dtype": dtype.label,
            "machine": machine(),
            "warmup": args.warmup,
            "repeat": args.repeat,
            "results": rows,
        },
        args.out,
    )


if __name__ == "__main__":
    main()
