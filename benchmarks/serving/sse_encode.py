"""Micro-benchmark for SSE text-frame encoding cost across all three wire protocols.

Measures the hot per-token ``TextDelta`` encode path of Chat Completions, Responses and Messages
on short 1-8 character strings (including quotes, backslashes, JSON text and multi-byte UTF-8).

    uv run python benchmarks/serving/sse_encode.py --rounds 5 --out out.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import timeit
from pathlib import Path
from typing import Any, Final

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ayaka.entrypoints.anthropic.messages import MessagesStreamEncoder
from ayaka.entrypoints.conversation import TextDelta
from ayaka.entrypoints.openai.chat import ChatStreamEncoder, _Meta
from ayaka.entrypoints.openai.responses import ResponsesStreamEncoder, _Snapshot

SCHEMA: Final = "ayaka.sse-encode/1"

_DELTAS: Final = (
    TextDelta("a"),
    TextDelta("hi"),
    TextDelta("token "),
    TextDelta('quote"'),
    TextDelta("a\\b"),
    TextDelta("xin ch\u00e0o"),
    TextDelta('{"k":1}'),
    TextDelta("12345678"),
)


def _encode_chat(iterations: int) -> float:
    encoder = ChatStreamEncoder(_Meta("id", 0, "m"), include_usage=True)
    deltas = _DELTAS
    n = len(deltas)
    start = timeit.default_timer()
    for i in range(iterations):
        encoder.encode(deltas[i % n])
    return (timeit.default_timer() - start) * 1e6 / iterations


def _encode_responses(iterations: int) -> float:
    encoder = ResponsesStreamEncoder(_Snapshot("resp", 0, "m", {}), "pub")
    encoder.start()
    deltas = _DELTAS
    n = len(deltas)
    start = timeit.default_timer()
    for i in range(iterations):
        encoder.encode(deltas[i % n])
    return (timeit.default_timer() - start) * 1e6 / iterations


def _encode_messages(iterations: int) -> float:
    encoder = MessagesStreamEncoder("msg", "m", prompt_tokens=16)
    encoder.start()
    deltas = _DELTAS
    n = len(deltas)
    start = timeit.default_timer()
    for i in range(iterations):
        encoder.encode(deltas[i % n])
    return (timeit.default_timer() - start) * 1e6 / iterations


def run_benchmark(*, rounds: int, iterations: int) -> dict[str, Any]:
    """Run ``rounds`` passes across all three stream encoders and return summary metrics."""
    from _machine import machine

    benches = {
        "chat": _encode_chat,
        "responses": _encode_responses,
        "messages": _encode_messages,
    }
    raw: dict[str, list[float]] = {name: [] for name in benches}
    # Warm up before timed rounds.
    for fn in benches.values():
        fn(min(1000, iterations))
    order = tuple(benches)
    for r in range(rounds):
        rotated = order[r % len(order) :] + order[: r % len(order)]
        for name in rotated:
            raw[name].append(benches[name](iterations))

    protocols = {
        name: {
            "rounds_us_per_frame": values,
            "median_us_per_frame": statistics.median(values),
        }
        for name, values in raw.items()
    }
    return {
        "schema": SCHEMA,
        "machine": machine(),
        "settings": {"rounds": rounds, "iterations": iterations, "delta_samples": len(_DELTAS)},
        "protocols": protocols,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="SSE text-frame encoding micro-benchmark.")
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=10_000)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)

    result = run_benchmark(rounds=args.rounds, iterations=args.iterations)
    for name, row in result["protocols"].items():
        print(f"{name:10s}: median {row['median_us_per_frame']:.2f} us/frame")
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
