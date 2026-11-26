"""Unit tests for shared-prefix pinning, HostKVStore async write-behind, and promotion."""

from __future__ import annotations

import threading
from typing import Any

import pytest

from ayaka.core.errors import EngineError
from ayaka.core.types import KVLayoutKind, RequestId
from ayaka.kvcache.blocks import BlockPool
from ayaka.kvcache.host_store import HostKVStore
from ayaka.kvcache.manager import KVCacheManager
from ayaka.kvcache.prefix.hash_cache import HashPrefixCache
from ayaka.kvcache.prefix.hashing import ChainRoot
from ayaka.kvcache.storage.pool import KVPool
from ayaka.kvcache.storage.spec import KVGroup, KVSpec
from ayaka.utils.dtypes import DType

torch: Any = pytest.importorskip("torch")

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
    return ChainRoot(
        model_id="tiny",
        dtype="fp32",
        codec="raw",
        block_size=block_size,
    )


def test_block_pool_pin_keeps_blocks_out_of_eviction_queue_and_integrity_holds() -> None:
    pool = BlockPool(4)  # blocks 1, 2, 3 usable
    b1 = pool.allocate()
    assert pool.register(b1, b"d1")
    pool.pin(b1)
    pool.free(b1)
    assert not pool.is_free(b1)
    assert pool.is_pinned(b1)
    assert pool.pinned_count == 1
    assert pool.free_count == 2
    pool.check_integrity()

    # Evicting/allocating remaining free blocks never touches pinned b1.
    b2 = pool.allocate()
    b3 = pool.allocate()
    assert {b2, b3} == {2, 3}
    assert pool.lookup(b"d1") == b1

    # Can still share a pinned unreferenced block via touch().
    pool.touch(b1)
    assert pool.ref_count(b1) == 1
    pool.free(b1)
    pool.unpin(b1)
    assert pool.is_free(b1)
    assert pool.pinned_count == 0
    pool.check_integrity()

    with pytest.raises(EngineError):
        pool.unpin(b1)


def test_evict_to_host_and_promote_restores_exact_kv_tensors() -> None:
    group = _group(block_size=4)
    pools = [KVPool(group, 4, device="cpu")]  # 3 usable blocks
    host = HostKVStore([group], capacity_blocks=8, pinned=False, allow_pageable=True)
    kv = KVCacheManager(
        [group],
        [4],
        max_requests=4,
        max_blocks_per_request=4,
        prefix=HashPrefixCache(_root(4)),
        host_store=host,
        pools=pools,
    )
    try:
        tokens = [10, 11, 12, 13, 14, 15, 16, 17, 99]  # 2 full blocks + 1 tail token
        m1 = kv.match_and_promote_prefix(tokens, None)
        assert m1 is not None
        assert m1.num_blocks == 0
        assert kv.allocate(RequestId("r1"), 9, m1)
        b_ids = kv.blocks(RequestId("r1"))[0]
        # Stamp deterministic distinguishable tensors into blocks of r1.
        pools[0]._k[:, b_ids[0]].fill_(1.25)
        pools[0]._v[:, b_ids[0]].fill_(2.50)
        pools[0]._k[:, b_ids[1]].fill_(3.75)
        pools[0]._v[:, b_ids[1]].fill_(4.50)
        kv.commit(RequestId("r1"), 9)
        kv.cache_blocks(RequestId("r1"), tokens[:8])
        kv.free(RequestId("r1"))

        # Evict every block in VRAM by writing unrelated blocks for r2.
        m2 = kv.match_and_promote_prefix([50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61], None)
        assert kv.allocate(RequestId("r2"), 12, m2)
        kv.commit(RequestId("r2"), 12)
        kv.free(RequestId("r2"))
        host.flush()
        assert host.stats().offloads >= 2

        # Read-only match_prefix misses because VRAM was evicted.
        cold = kv.match_prefix(tokens, None)
        assert cold is not None
        assert cold.num_blocks == 0

        # match_and_promote_prefix restores both blocks from host RAM cleanly.
        promoted = kv.match_and_promote_prefix(tokens, None)
        assert promoted is not None
        assert promoted.num_blocks == 2
        assert promoted.num_tokens == 8
        assert host.stats().promotions == 2

        restored_ids = promoted.blocks[0]
        assert torch.allclose(
            pools[0]._k[:, restored_ids[0]], torch.full_like(pools[0]._k[:, 0], 1.25)
        )
        assert torch.allclose(
            pools[0]._v[:, restored_ids[0]], torch.full_like(pools[0]._v[:, 0], 2.50)
        )
        assert torch.allclose(
            pools[0]._k[:, restored_ids[1]], torch.full_like(pools[0]._k[:, 0], 3.75)
        )
        assert torch.allclose(
            pools[0]._v[:, restored_ids[1]], torch.full_like(pools[0]._v[:, 0], 4.50)
        )
        kv.check_integrity()
    finally:
        host.close()


def test_pin_prefix_protects_shared_blocks_from_eviction() -> None:
    group = _group(block_size=4)
    pools = [KVPool(group, 4, device="cpu")]  # 3 usable blocks
    kv = KVCacheManager(
        [group],
        [4],
        max_requests=4,
        max_blocks_per_request=4,
        prefix=HashPrefixCache(_root(4)),
        pools=pools,
    )
    sys_prompt = [1, 2, 3, 4, 5]  # 1 full block + 1 tail token
    m = kv.match_prefix(sys_prompt, None)
    assert kv.allocate(RequestId("seed"), 5, m)
    kv.commit(RequestId("seed"), 5)
    kv.cache_blocks(RequestId("seed"), sys_prompt[:4])
    kv.free(RequestId("seed"))

    assert kv.pin_prefix(sys_prompt, None) == 1
    assert kv.free_blocks == 2

    # Unrelated request takes remaining 2 free blocks; pinned prefix remains intact.
    assert kv.allocate(
        RequestId("other"), 8, kv.match_prefix([20, 21, 22, 23, 24, 25, 26, 27], None)
    )
    kv.commit(RequestId("other"), 8)
    kv.free(RequestId("other"))

    warm = kv.match_prefix(sys_prompt, None)
    assert warm is not None
    assert warm.num_blocks == 1
    assert kv.unpin_prefix(sys_prompt, None) == 1
    kv.check_integrity()


def test_host_kv_store_concurrent_offload_and_promote_race_free() -> None:
    group = _group(block_size=4)
    pool = KVPool(group, 6, device="cpu")
    host = HostKVStore(
        [group], capacity_blocks=4, pinned=False, allow_pageable=True, max_inflight_offloads=2
    )
    errors: list[BaseException] = []
    try:
        for i in range(1, 5):
            pool._k[:, i].fill_(float(i))
            pool._v[:, i].fill_(float(i * 10))
            assert host.offload(0, i, f"d{i}".encode(), pool)

        # Immediately promote concurrently from another thread while write-behind may still run.
        def worker() -> None:
            try:
                dst = KVPool(group, 6, device="cpu")
                for i in range(1, 5):
                    assert host.promote(f"d{i}".encode(), [5], [dst])
                    assert float(dst._k[:, 5].mean().item()) == pytest.approx(float(i))
                    assert float(dst._v[:, 5].mean().item()) == pytest.approx(float(i * 10))
            except BaseException as exc:
                errors.append(exc)

        t = threading.Thread(target=worker)
        t.start()
        t.join(timeout=5.0)
        assert not errors
    finally:
        host.close()
