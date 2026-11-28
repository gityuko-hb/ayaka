"""Prefix cache hashing and admission with non-zero lookahead (MTP layer KV)."""

from __future__ import annotations

from dataclasses import replace

import pytest

from ayaka.configs.scheduler import SchedulerConfig
from ayaka.core.params import SamplingParams
from ayaka.core.request import TokenizedRequest
from ayaka.core.types import KVLayoutKind, RequestId
from ayaka.kvcache.blocks import BlockPool
from ayaka.kvcache.manager import KVCacheManager
from ayaka.kvcache.prefix.hash_cache import HashPrefixCache
from ayaka.kvcache.prefix.hashing import ChainRoot, chain_digests, root_digest
from ayaka.kvcache.storage.spec import KVGroup, KVSpec
from ayaka.scheduler.scheduler import Scheduler
from ayaka.scheduler.step import StepOutput
from ayaka.utils.dtypes import DType

pytestmark = pytest.mark.unit


def _manager(*, block_size: int = 4, lookahead: int = 1) -> tuple[KVCacheManager, HashPrefixCache]:
    spec = KVSpec(
        num_layers=2,
        num_kv_heads=2,
        head_dim=8,
        block_size=block_size,
        dtype=DType.FP16,
        layout=KVLayoutKind.NHD,
    )
    group = KVGroup(layer_ids=(0, 1), spec=spec)
    prefix = HashPrefixCache(
        ChainRoot(
            model_id="mimo/test",
            dtype="fp16",
            codec="raw",
            block_size=block_size,
            lookahead=lookahead,
        )
    )
    mgr = KVCacheManager([group], [16], max_requests=4, max_blocks_per_request=8, prefix=prefix)
    return mgr, prefix


def test_lookahead_digests_include_trailing_tokens() -> None:
    root = root_digest(ChainRoot("m", "fp16", "raw", 4, lookahead=1), None)
    assert chain_digests(root, [1, 2, 3, 4], 4, lookahead=1) == []

    a = chain_digests(root, [1, 2, 3, 4, 10], 4, lookahead=1)
    b = chain_digests(root, [1, 2, 3, 4, 99], 4, lookahead=1)
    assert len(a) == 1
    assert a[0] != b[0]

    # Separate lookahead=0 chain ignores the 5th token.
    root0 = root_digest(replace(ChainRoot("m", "fp16", "raw", 4, lookahead=0)), None)
    assert chain_digests(root0, [1, 2, 3, 4, 10], 4, lookahead=0) == chain_digests(
        root0, [1, 2, 3, 4, 99], 4, lookahead=0
    )


def test_prefix_reuse_requires_matching_lookahead_token() -> None:
    pool = BlockPool(16)
    cache = HashPrefixCache(ChainRoot("m", "fp16", "raw", 4, lookahead=1))
    mgr = KVCacheManager(
        [
            KVGroup(
                layer_ids=(0,),
                spec=KVSpec(1, 1, 4, block_size=4, dtype=DType.FP16, layout=KVLayoutKind.NHD),
            )
        ],
        [16],
        max_requests=4,
        max_blocks_per_request=4,
        prefix=cache,
    )
    del pool

    prompt_a = (10, 11, 12, 13, 20, 21)
    mgr.allocate(RequestId("r1"), len(prompt_a), mgr.match_prefix(prompt_a, None))
    mgr.commit(RequestId("r1"), len(prompt_a))
    assert mgr.cacheable_blocks(RequestId("r1")) == (0, 1)
    assert mgr.cache_blocks(RequestId("r1"), prompt_a[:5]) == 1

    # Matching prefix + matching lookahead token (20) hits block 0.
    hit = mgr.match_prefix((10, 11, 12, 13, 20, 99), None)
    assert hit is not None
    assert hit.num_blocks == 1

    # Different lookahead token (77 != 20) misses block 0.
    miss = mgr.match_prefix((10, 11, 12, 13, 77, 99), None)
    assert miss is not None
    assert miss.num_blocks == 0


def test_scheduler_waits_for_lookahead_token_before_caching_block() -> None:
    mgr, _ = _manager(block_size=4, lookahead=1)
    sched = Scheduler(
        SchedulerConfig(max_running=2, max_tokens_per_step=16),
        mgr,
        max_model_len=32,
        max_chunk_tokens=16,
    )
    req = TokenizedRequest(
        request_id=RequestId("s1"),
        prompt_token_ids=(1, 2, 3, 4),
        sampling=SamplingParams(max_tokens=3),
    )
    sched.add_request(req)
    prefill = sched.schedule()
    assert prefill is not None
    sched.update(
        StepOutput(
            step_id=prefill.step_id,
            request_ids=prefill.request_ids,
            tokens=((5,),),
        )
    )
    # Only the 4 prompt tokens are committed after prefill; token 5 is not yet in KV.
    not_yet = mgr.match_prefix((1, 2, 3, 4, 5, 6), None)
    assert not_yet is not None
    assert not_yet.num_blocks == 0

    decode = sched.schedule()
    assert decode is not None
    sched.update(
        StepOutput(
            step_id=decode.step_id,
            request_ids=decode.request_ids,
            tokens=((6,),),
        )
    )
    # Prompt length 4 + committed decode token 5 = 5 -> block 0 (0..3) with lookahead 5 is cached!
    hit = mgr.match_prefix((1, 2, 3, 4, 5, 6), None)
    assert hit is not None
    assert hit.num_blocks == 1
    miss = mgr.match_prefix((1, 2, 3, 4, 9, 6), None)
    assert miss is not None
    assert miss.num_blocks == 0
