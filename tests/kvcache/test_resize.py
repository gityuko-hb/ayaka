"""Unit tests for atomic runtime KV pool grow, shrink compaction, and OOM rollback."""

from __future__ import annotations

import dataclasses
from typing import Any

import pytest

from ayaka.core.errors import CapacityError, OutOfMemoryError
from ayaka.core.params import SamplingParams
from ayaka.core.request import GenerateRequest
from ayaka.core.types import FinishReason, KVLayoutKind, RequestId
from ayaka.kvcache.host_store import HostKVStore
from ayaka.kvcache.manager import KVCacheManager
from ayaka.kvcache.prefix.hash_cache import HashPrefixCache
from ayaka.kvcache.prefix.hashing import ChainRoot
from ayaka.kvcache.storage.pool import KVPool
from ayaka.kvcache.storage.spec import KVGroup, KVSpec
from ayaka.utils.dtypes import DType

torch: Any = pytest.importorskip("torch")

from ayaka.engine.build import build_engine  # noqa: E402
from tiny_model import (  # noqa: E402
    TinyCheckpoint,
    assert_cache_released,
    greedy_reference,
    tiny_engine_config,
    tiny_tokenizer,
    write_tiny_gpt2,
)
from tokenizer_fakes import BOS  # noqa: E402

pytestmark = pytest.mark.unit


def _group(block_size: int = 4) -> KVGroup:
    spec = KVSpec(
        num_layers=2,
        num_kv_heads=2,
        head_dim=4,
        block_size=block_size,
        dtype=DType.FP32,
        layout=KVLayoutKind.NHD,
    )
    return KVGroup(layer_ids=(0, 1), spec=spec)


def _root(block_size: int = 4) -> ChainRoot:
    return ChainRoot(model_id="tiny", dtype="fp32", codec="raw", block_size=block_size)


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory: pytest.TempPathFactory) -> TinyCheckpoint:
    return write_tiny_gpt2(tmp_path_factory.mktemp("kv-resize"))


def test_grow_and_shrink_compact_preserve_active_and_cached_blocks() -> None:
    group = _group(4)
    pools = [KVPool(group, 6, device="cpu")]  # 5 usable blocks
    host = HostKVStore([group], capacity_blocks=8, pinned=False, allow_pageable=True)
    kv = KVCacheManager(
        [group],
        [6],
        max_requests=4,
        max_blocks_per_request=8,
        prefix=HashPrefixCache(_root(4)),
        host_store=host,
        pools=pools,
    )
    try:
        # Allocate 3 blocks for r_active and cache 2 MRU/LRU blocks.
        cached_old = [1, 2, 3, 4, 9]
        assert kv.allocate(RequestId("old"), 5, kv.match_prefix(cached_old, None))
        b_old = kv.blocks(RequestId("old"))[0][0]
        kv.storage_pools[0]._k[:, b_old].fill_(7.0)  # type: ignore[index]
        kv.commit(RequestId("old"), 5)
        kv.cache_blocks(RequestId("old"), cached_old[:4])
        kv.free(RequestId("old"))

        assert kv.allocate(
            RequestId("active"), 8, kv.match_prefix([20, 21, 22, 23, 24, 25, 26, 27], None)
        )
        b_active = kv.blocks(RequestId("active"))[0]
        kv.storage_pools[0]._k[:, b_active[0]].fill_(11.0)  # type: ignore[index]
        kv.storage_pools[0]._k[:, b_active[1]].fill_(13.0)  # type: ignore[index]
        kv.commit(RequestId("active"), 8)

        # Grow to 10 total blocks (9 usable).
        kv.resize([10])
        assert kv.usable_counts == (9,)
        assert kv.free_blocks == 7

        # Shrink down to 4 total blocks (3 usable: 2 active + 1 cached).
        kv.resize([4])
        assert kv.usable_counts == (3,)
        new_active = kv.blocks(RequestId("active"))[0]
        assert kv.storage_pools is not None
        assert float(kv.storage_pools[0]._k[:, new_active[0]].mean().item()) == pytest.approx(11.0)
        assert float(kv.storage_pools[0]._k[:, new_active[1]].mean().item()) == pytest.approx(13.0)
        warm = kv.match_prefix(cached_old, None)
        assert warm is not None
        assert warm.num_blocks == 1

        # Shrink to 3 total (2 usable) drops the cached block and offloads it to host RAM!
        kv.resize([3])
        assert kv.usable_counts == (2,)
        assert host.has(warm.digests[0])

        with pytest.raises(CapacityError):
            kv.resize([2])  # only 1 usable, needs 2 for "active"
        assert kv.usable_counts == (2,)
        kv.free(RequestId("active"))
        kv.check_integrity()
    finally:
        host.close()


def test_oom_during_resize_rolls_back_with_prior_cache_intact(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    group = _group(4)
    pools = [KVPool(group, 6, device="cpu")]
    kv = KVCacheManager(
        [group],
        [6],
        max_requests=4,
        max_blocks_per_request=8,
        prefix=HashPrefixCache(_root(4)),
        pools=pools,
    )
    prompt = [10, 11, 12, 13, 14]
    assert kv.allocate(RequestId("r1"), 5, kv.match_prefix(prompt, None))
    b0 = kv.blocks(RequestId("r1"))[0][0]
    pools[0]._k[:, b0].fill_(42.0)
    kv.commit(RequestId("r1"), 5)
    kv.cache_blocks(RequestId("r1"), prompt[:4])

    def oom_reallocate(self: KVPool, new_num_blocks: int, remap: Any) -> KVPool:
        raise torch.OutOfMemoryError("synthetic OOM while reallocating")

    monkeypatch.setattr(KVPool, "reallocate", oom_reallocate)
    with pytest.raises(OutOfMemoryError):
        kv.resize([12])

    assert kv.usable_counts == (5,)
    assert kv.storage_pools is not None
    assert kv.storage_pools[0] is pools[0]
    assert float(pools[0]._k[:, b0].mean().item()) == pytest.approx(42.0)
    kv.free(RequestId("r1"))
    kv.check_integrity()


def test_engine_resize_kv_pool_grows_and_preserves_tokens_end_to_end(
    checkpoint: TinyCheckpoint,
) -> None:
    cfg = dataclasses.replace(
        tiny_engine_config(checkpoint.directory),
        cache=dataclasses.replace(tiny_engine_config(checkpoint.directory).cache, block_size=4),
    )
    engine = build_engine(cfg, tokenizer=tiny_tokenizer())
    try:
        old_usable = engine.kv.usable_counts[0]
        new_counts = engine.resize_kv_pool(old_usable + 1 + 4)
        assert new_counts == (old_usable + 4,)

        prompt = "runtime pool resize check"
        expected = greedy_reference(checkpoint.state, checkpoint.config, [BOS, *prompt.encode()], 5)
        engine.add_request(
            GenerateRequest(
                request_id=RequestId("resized"),
                prompt=prompt,
                sampling=SamplingParams(temperature=0.0, max_tokens=5),
            )
        )
        outs = []
        tokens: list[int] = []
        while engine.has_unfinished:
            for out in engine.step():
                outs.append(out)
                if out.outputs and out.outputs[0].token_ids:
                    tokens.extend(out.outputs[0].token_ids)
        assert outs[-1].outputs[0].finish_reason is FinishReason.LENGTH
        assert tokens == expected
        engine.kv.check_integrity()
        assert_cache_released(engine)
    finally:
        engine.shutdown()
