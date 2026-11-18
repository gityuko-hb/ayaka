"""How long the prefix cache takes to hash and match a long prompt (issue #52).

Builds a bookkeeping-only :class:`~ayaka.kvcache.manager.KVCacheManager` with a hash prefix cache
(no tensors, no GPU), registers one prompt of ``--tokens`` tokens, then times ``match_prefix`` of
the same prompt (every block hits) and of a prompt that differs in its first token (the first
block misses, so it measures hashing alone). Each case runs ``--repeat`` times after a warm-up;
every run is kept and the median is reported.

    uv run python benchmarks/kvcache/prefix_match.py --tokens 100000 --block-size 16
"""

from __future__ import annotations

import argparse
import random
import statistics
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _machine import machine, write

from ayaka.core.types import KVLayoutKind, RequestId
from ayaka.kvcache.manager import KVCacheManager
from ayaka.kvcache.prefix.hash_cache import HashPrefixCache
from ayaka.kvcache.prefix.hashing import ChainRoot
from ayaka.kvcache.storage.spec import KVGroup, KVSpec
from ayaka.utils.dtypes import DType


def _manager(tokens: int, block_size: int) -> KVCacheManager:
    blocks = tokens // block_size + 2
    spec = KVSpec(
        num_layers=1,
        num_kv_heads=1,
        head_dim=8,
        block_size=block_size,
        dtype=DType.FP16,
        layout=KVLayoutKind.NHD,
    )
    return KVCacheManager(
        [KVGroup(layer_ids=(0,), spec=spec)],
        [blocks + 1],
        max_requests=2,
        max_blocks_per_request=blocks,
        prefix=HashPrefixCache(
            ChainRoot(model_id="bench", dtype="fp16", codec="raw", block_size=block_size)
        ),
    )


def _time(run: Any, repeat: int) -> list[float]:
    run()
    samples: list[float] = []
    for _ in range(repeat):
        started = time.perf_counter()
        run()
        samples.append(time.perf_counter() - started)
    return samples


def main() -> None:
    parser = argparse.ArgumentParser(description="Prefix-cache match time for a long prompt.")
    parser.add_argument("--tokens", type=int, default=100_000)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--threshold-ms", type=float, default=None)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    rng = random.Random(0)  # noqa: S311  # a fixed prompt, not a secret
    prompt = [rng.randrange(150_000) for _ in range(args.tokens)]
    manager = _manager(args.tokens, args.block_size)
    request = RequestId("warm")
    match = manager.match_prefix(prompt, None)
    if match is None or not manager.allocate(request, len(prompt), match):
        raise SystemExit("the benchmark cache is too small")
    manager.commit(request, len(prompt))
    manager.cache_blocks(request, prompt)
    other = [prompt[0] + 1, *prompt[1:]]

    hit = _time(lambda: manager.match_prefix(prompt, None), args.repeat)
    miss = _time(lambda: manager.match_prefix(other, None), args.repeat)
    found = manager.match_prefix(prompt, None)
    if found is None:
        raise SystemExit("the manager has no prefix cache")
    result: dict[str, Any] = {
        "benchmark": "prefix_match",
        "machine": machine(),
        "tokens": args.tokens,
        "block_size": args.block_size,
        "matched_tokens": found.num_tokens,
        "repeat": args.repeat,
        "hit_seconds": hit,
        "miss_seconds": miss,
        "hit_median_ms": statistics.median(hit) * 1000,
        "miss_median_ms": statistics.median(miss) * 1000,
        "threshold_ms": args.threshold_ms,
    }
    if args.threshold_ms is not None:
        result["within_threshold"] = result["hit_median_ms"] <= args.threshold_ms
    write(result, args.out)


if __name__ == "__main__":
    main()
