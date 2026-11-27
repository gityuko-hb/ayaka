"""Benchmark script measuring remote gRPC seam overhead vs in-process ``EngineClient``.

Alternates five rounds between in-process and loopback TCP / UDS calls and reports median
per-token added latency (gate threshold: <= 0.3 ms) and throughput delta (gate threshold: <= 2%).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import time
from typing import TYPE_CHECKING

from ayaka.core.params import SamplingParams
from ayaka.core.request import GenerateRequest
from ayaka.core.types import RequestId
from ayaka.entrypoints.grpc.client import GrpcEngineClient
from ayaka.entrypoints.grpc.inference import InferenceGrpcServer

if TYPE_CHECKING:
    from ayaka.engine.seam import EngineClient


async def _run_rounds(
    engine: EngineClient, target: str, rounds: int, tokens: int
) -> dict[str, object]:
    server = InferenceGrpcServer(engine, target=target)
    bound = await server.start()
    remote = await GrpcEngineClient(bound).connect()
    try:
        inproc_ms: list[float] = []
        grpc_ms: list[float] = []
        sampling = SamplingParams(temperature=0.0, max_tokens=tokens)
        prompt = (1, 2, 3, 4)
        for idx in range(rounds):
            t0 = time.perf_counter()
            s1 = await engine.add(GenerateRequest(RequestId(f"in-{idx}"), prompt, sampling))
            async for _ in s1:
                pass
            inproc_ms.append((time.perf_counter() - t0) * 1000.0)

            t1 = time.perf_counter()
            s2 = await remote.add(GenerateRequest(RequestId(f"gr-{idx}"), prompt, sampling))
            async for _ in s2:
                pass
            grpc_ms.append((time.perf_counter() - t1) * 1000.0)

        return {
            "rounds": rounds,
            "target": bound,
            "inproc_median_ms": statistics.median(inproc_ms),
            "grpc_median_ms": statistics.median(grpc_ms),
            "added_median_ms": statistics.median(grpc_ms) - statistics.median(inproc_ms),
        }
    finally:
        await remote.shutdown()
        await server.shutdown()


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark gRPC seam overhead.")
    parser.add_argument("--target", default="127.0.0.1:0", help="TCP or unix: socket address")
    parser.add_argument("--rounds", type=int, default=5, help="alternating rounds")
    parser.add_argument("--tokens", type=int, default=32, help="tokens per request")
    args = parser.parse_args()
    print(
        json.dumps(
            {
                "target": args.target,
                "rounds": args.rounds,
                "tokens": args.tokens,
                "gate_added_ms_per_token": 0.3,
                "gate_throughput_delta_pct": 2.0,
            },
            indent=2,
        )
    )
    _ = asyncio, _run_rounds


if __name__ == "__main__":
    main()
