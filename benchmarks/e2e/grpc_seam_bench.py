"""What the gRPC engine seam adds to streamed tokens: one HTTP client, two servers, same GPU.

Arm ``in-process`` is ``ayaka serve`` with the engine in its own process. Arm ``grpc`` is
``ayaka serve --engine-grpc ADDR``: the HTTP gateway in one process and ``ayaka engine-server`` in
another, every token crossing the seam (a Unix socket or loopback TCP). Both engines load the same
model on the same GPU, each with part of its memory, and the rounds alternate the arms so only one
engine works at a time::

    ayaka engine-server Qwen/Qwen3-8B --listen unix:/tmp/ayaka-engine.sock --memory-ratio 0.45 ...
    ayaka serve Qwen/Qwen3-8B --engine-grpc unix:/tmp/ayaka-engine.sock --port 8201
    ayaka serve Qwen/Qwen3-8B --memory-ratio 0.45 --port 8200 ...

    uv run python benchmarks/e2e/grpc_seam_bench.py --in-process http://127.0.0.1:8200 \\
        --grpc http://127.0.0.1:8201 --out grpc_seam.json

Per arm, round and concurrency it streams ``--concurrency`` Chat Completions at once, greedy with
``ignore_eos`` so every request writes exactly ``--tokens`` tokens, and records:

    gap_ms        the time between two content chunks of a stream: per-token latency
    ttft_ms       request sent to first content chunk
    tok_s         output tokens of all streams over the wall time of the round

The result keeps the median of each over the rounds, and the differences ``grpc - in-process``
(``gap_added_ms_p50``) and ``grpc / in-process`` (``tok_s_ratio``): the two numbers the seam's
pre-registered gate reads (at most 0.3 ms added per token at the median, at least 0.98 of the
throughput). Set ``--sse-coalesce-ms 0`` on both servers (the default), so a chunk is a token.
"""

from __future__ import annotations

import argparse
import asyncio
import itertools
import json
import statistics
import time
from pathlib import Path
from typing import Any

import aiohttp


async def _stream(
    session: aiohttp.ClientSession, base: str, model: str, tokens: int, salt: int
) -> tuple[float, list[float], int]:
    body = {
        "model": model,
        # The salt keeps two streams from sharing a cached prefix beyond the template.
        "messages": [{"role": "user", "content": f"Count upward from {salt}, one number a line."}],
        "max_tokens": tokens,
        "temperature": 0,
        "ignore_eos": True,
        "stream": True,
        "reasoning_effort": "none",
    }
    start = time.perf_counter()
    first = 0.0
    stamps: list[float] = []
    async with session.post(f"{base}/v1/chat/completions", json=body) as response:
        response.raise_for_status()
        async for raw in response.content:
            line = raw.decode().strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            chunk = json.loads(line[6:])
            for choice in chunk.get("choices", ()):
                if choice.get("delta", {}).get("content"):
                    now = time.perf_counter()
                    if not stamps:
                        first = now - start
                    stamps.append(now)
    gaps = [(b - a) * 1000.0 for a, b in itertools.pairwise(stamps)]
    return first * 1000.0, gaps, len(stamps)


async def _round(base: str, model: str, concurrency: int, tokens: int, salt: int) -> dict[str, Any]:
    timeout = aiohttp.ClientTimeout(total=600)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        start = time.perf_counter()
        results = await asyncio.gather(
            *(_stream(session, base, model, tokens, salt + i) for i in range(concurrency))
        )
        wall = time.perf_counter() - start
    gaps = [gap for _, row, _ in results for gap in row]
    chunks = sum(count for _, _, count in results)
    return {
        "gap_ms_p50": statistics.median(gaps),
        "gap_ms_p90": statistics.quantiles(gaps, n=10)[-1],
        "ttft_ms_p50": statistics.median(first for first, _, _ in results),
        "tok_s": chunks / wall,
        "chunks": chunks,
    }


async def _measure_arm(args: argparse.Namespace, base: str, *, salt_offset: int) -> dict[str, Any]:
    await _round(base, args.model, 2, 16, 0)  # warm: graphs replayed, caches filled
    concurrency_out: dict[str, Any] = {}
    for concurrency in args.concurrency:
        rounds_data: list[dict[str, Any]] = []
        for index in range(args.rounds):
            salt = 1000 * (index + 1) + salt_offset
            rounds_data.append(await _round(base, args.model, concurrency, args.tokens, salt))
        medians = {
            key: round(statistics.median(row[key] for row in rounds_data), 4)
            for key in ("gap_ms_p50", "gap_ms_p90", "ttft_ms_p50", "tok_s")
        }
        concurrency_out[str(concurrency)] = {"medians": medians, "rounds": rounds_data}
        print(f"{args.arm or 'arm'} C={concurrency}: {json.dumps(medians)}", flush=True)
    return {
        "arm": args.arm,
        "url": base,
        "model": args.model,
        "tokens": args.tokens,
        "rounds": args.rounds,
        "concurrency": concurrency_out,
    }


def _compare(in_process_path: Path, grpc_path: Path) -> dict[str, Any]:
    in_proc = json.loads(in_process_path.read_text(encoding="utf-8"))
    grpc = json.loads(grpc_path.read_text(encoding="utf-8"))
    out: dict[str, Any] = {
        "model": in_proc.get("model"),
        "tokens": in_proc.get("tokens"),
        "rounds": in_proc.get("rounds"),
        "concurrency": {},
    }
    for key in sorted(in_proc["concurrency"], key=int):
        if key not in grpc["concurrency"]:
            continue
        m_in = in_proc["concurrency"][key]["medians"]
        m_gr = grpc["concurrency"][key]["medians"]
        entry = {
            "in-process": m_in,
            "grpc": m_gr,
            "gap_added_ms_p50": round(m_gr["gap_ms_p50"] - m_in["gap_ms_p50"], 4),
            "tok_s_ratio": round(m_gr["tok_s"] / m_in["tok_s"], 4),
            "in_process_rounds": in_proc["concurrency"][key]["rounds"],
            "grpc_rounds": grpc["concurrency"][key]["rounds"],
        }
        out["concurrency"][key] = entry
        print(
            f"compare C={key}: gap_added_ms_p50={entry['gap_added_ms_p50']} ms, "
            f"tok_s_ratio={entry['tok_s_ratio']}",
            flush=True,
        )
    return out


async def _main(args: argparse.Namespace) -> dict[str, Any]:
    arms = {"in-process": args.in_process, "grpc": args.grpc}
    out: dict[str, Any] = {"tokens": args.tokens, "rounds": args.rounds, "concurrency": {}}
    for base in arms.values():  # warm both: graphs replayed, caches filled
        await _round(base, args.model, 2, 16, 0)
    for concurrency in args.concurrency:
        rows: dict[str, list[dict[str, Any]]] = {name: [] for name in arms}
        for index in range(args.rounds):
            for name, base in arms.items():
                salt = 1000 * (index + 1) + (0 if name == "in-process" else 500)
                rows[name].append(await _round(base, args.model, concurrency, args.tokens, salt))
        medians = {
            name: {
                key: round(statistics.median(row[key] for row in values), 4)
                for key in ("gap_ms_p50", "gap_ms_p90", "ttft_ms_p50", "tok_s")
            }
            for name, values in rows.items()
        }
        medians["gap_added_ms_p50"] = round(
            medians["grpc"]["gap_ms_p50"] - medians["in-process"]["gap_ms_p50"], 4
        )
        medians["tok_s_ratio"] = round(medians["grpc"]["tok_s"] / medians["in-process"]["tok_s"], 4)
        out["concurrency"][str(concurrency)] = medians
        print(f"concurrency {concurrency}: {json.dumps(medians)}", flush=True)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--in-process", help="base URL of the in-process server")
    parser.add_argument("--grpc", help="base URL of the --engine-grpc gateway")
    parser.add_argument(
        "--arm",
        choices=("in-process", "grpc"),
        help="measure one arm sequentially on a single-GPU machine",
    )
    parser.add_argument("--url", help="base URL of the server when using --arm")
    parser.add_argument(
        "--compare",
        type=Path,
        nargs=2,
        metavar=("IN_PROCESS_JSON", "GRPC_JSON"),
        help="compare two single-arm JSON outputs",
    )
    parser.add_argument("--model", default="Qwen/Qwen3-8B")
    parser.add_argument("--tokens", type=int, default=256)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--concurrency", type=int, nargs="+", default=[1, 8])
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    if args.compare is not None:
        result = _compare(args.compare[0], args.compare[1])
    elif args.arm is not None:
        if not args.url:
            parser.error("--arm requires --url")
        offset = 0 if args.arm == "in-process" else 500
        result = asyncio.run(_measure_arm(args, args.url, salt_offset=offset))
    else:
        if not args.in_process or not args.grpc:
            parser.error("pass either --arm/--url, --compare, or both --in-process and --grpc")
        result = asyncio.run(_main(args))
    if args.out is not None:
        args.out.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
