"""Unit tests for FP8 KV, MLA latent KV, and recurrent state storage, slots, and checkpoints."""

from __future__ import annotations

from typing import Any

import pytest

from ayaka.configs.cache import CacheConfig
from ayaka.core.types import KVLayoutKind, RequestId
from ayaka.kvcache.manager import KVCacheManager
from ayaka.kvcache.prefix.hash_cache import HashPrefixCache
from ayaka.kvcache.prefix.hashing import ChainRoot
from ayaka.kvcache.state_slots import PADDING_STATE_SLOT, StateCheckpointStore, StateSlotPool
from ayaka.kvcache.storage.pool import KVPool, LatentKVPool, RecurrentStatePool
from ayaka.kvcache.storage.spec import (
    KVGroup,
    KVSpec,
    LatentKVGroup,
    LatentKVSpec,
    RecurrentStateGroup,
    RecurrentStateSpec,
)
from ayaka.memory.budget import BudgetInputs, KVCost, StateCost, plan_budget
from ayaka.ops.kvcache import FP8_E4M3_CODEC, RAW_CODEC
from ayaka.utils.dtypes import DType

torch: Any = pytest.importorskip("torch")

pytestmark = pytest.mark.unit


def test_fp8_kv_spec_halves_bytes_per_token() -> None:
    raw = KVSpec(2, 4, 128, 16, DType.BF16, KVLayoutKind.NHD, codec=RAW_CODEC)
    fp8 = KVSpec(
        2, 4, 128, 16, DType.BF16, KVLayoutKind.NHD, codec=FP8_E4M3_CODEC, kv_scale=0.5, window=128
    )

    assert raw.storage_dtype is DType.BF16
    assert fp8.storage_dtype is DType.FP8_E4M3
    assert fp8.bytes_per_token() * 2 == raw.bytes_per_token()
    assert fp8.kv_scale == 0.5
    assert fp8.window == 128


def test_latent_kv_spec_and_pool() -> None:
    spec = LatentKVSpec(
        num_layers=2,
        kv_lora_rank=512,
        qk_rope_head_dim=64,
        block_size=16,
        dtype=DType.BF16,
    )
    assert spec.compression_ratio_vs_kv(num_kv_heads=128, head_dim=128) < 0.05
    assert spec.compression_ratio_vs_kv(num_kv_heads=8, head_dim=128) < 0.5

    group = LatentKVGroup(layer_ids=(0, 1), spec=spec)
    pool = LatentKVPool(group, num_blocks=5, device="cpu")
    buf = pool.buffers(1)
    assert buf.ckv.shape == (5, 16, 512)
    assert buf.kpe.shape == (5, 16, 64)
    assert buf.codec == RAW_CODEC
    assert pool.usable_blocks == 4


def test_recurrent_state_slot_pool_lifecycle() -> None:
    pool = StateSlotPool(4)
    assert pool.usable_slots == 3
    assert pool.free_count == 3

    s1 = pool.allocate(RequestId("r1"))
    s2 = pool.allocate(RequestId("r2"))
    s3 = pool.allocate(RequestId("r3"))
    assert s1 != PADDING_STATE_SLOT
    assert s2 != PADDING_STATE_SLOT
    assert s3 != PADDING_STATE_SLOT
    assert len({s1, s2, s3}) == 3
    assert pool.free_count == 0
    assert pool.allocate(RequestId("r4")) is None
    assert pool.allocate(RequestId("r1")) == s1

    assert pool.release(RequestId("r2")) == s2
    assert pool.free_count == 1
    assert pool.allocate(RequestId("r4")) == s2
    pool.check_integrity()


def test_recurrent_state_pool_snapshot_and_restore() -> None:
    spec = RecurrentStateSpec(
        num_layers=2,
        num_v_heads=4,
        key_head_dim=8,
        value_head_dim=8,
        conv_dim=16,
        conv_kernel_size=4,
        conv_dtype=DType.FP32,
    )
    group = RecurrentStateGroup(layer_ids=(0, 2), spec=spec)
    pool = RecurrentStatePool(group, max_slots=3, device="cpu")
    assert pool.nbytes == 3 * spec.bytes_per_slot()

    buf0 = pool.buffers(0)
    buf2 = pool.buffers(2)
    buf0.ssm_state[1].fill_(3.25)
    buf2.conv_state[1].fill_(1.5)
    snap = pool.snapshot_slot(1)

    pool.zero_slot(1)
    assert buf0.ssm_state[1].abs().sum() == 0
    assert buf2.conv_state[1].abs().sum() == 0

    pool.restore_slot(1, snap)
    assert torch.allclose(buf0.ssm_state[1], torch.full_like(buf0.ssm_state[1], 3.25))
    assert torch.allclose(buf2.conv_state[1], torch.full_like(buf2.conv_state[1], 1.5))


def test_state_checkpoint_store_lru_and_stride() -> None:
    store = StateCheckpointStore(max_bytes=200, stride_blocks=2)
    assert [store.should_checkpoint(i) for i in range(6)] == [False, True, False, True, False, True]

    assert not store.put(b"too_big", object(), 201)
    assert store.put(b"a", "va", 100)
    assert store.put(b"b", "vb", 100)
    assert store.get(b"a") == "va"  # promote "a", so "b" becomes LRU
    assert store.put(b"c", "vc", 100)
    assert not store.contains(b"b")
    assert store.contains(b"a")
    assert store.contains(b"c")
    assert store.evictions == 1
    assert store.hits == 1
    assert store.get(b"missing") is None
    assert store.misses == 1
    store.check_integrity()


def test_manager_aligns_prefix_hits_to_state_checkpoints() -> None:
    spec = KVSpec(2, 2, 4, block_size=4, dtype=DType.FP32, layout=KVLayoutKind.NHD)
    group = KVGroup(layer_ids=(3, 7), spec=spec)
    prefix = HashPrefixCache(ChainRoot(model_id="hybrid", dtype="fp32", codec="raw", block_size=4))
    slots = StateSlotPool(4)
    ckpts = StateCheckpointStore(max_bytes=1024, stride_blocks=2)
    mgr = KVCacheManager(
        [group],
        [10],
        max_requests=3,
        max_blocks_per_request=4,
        prefix=prefix,
        state_slots=slots,
        state_checkpoints=ckpts,
    )

    tokens = tuple(range(16))  # 4 blocks: indices 0, 1, 2, 3
    r1 = RequestId("r1")
    match1 = mgr.match_prefix(tokens, None)
    assert match1 is not None
    assert match1.num_tokens == 0
    assert mgr.allocate(r1, 16, match1)
    targets = mgr.preview_state_checkpoints(r1, tokens, 0, 16)
    # stride=2 -> blocks index 1 (offset 8) and 3 (offset 16)
    assert [off for off, _ in targets] == [8, 16]
    mgr.commit(r1, 16)
    mgr.cache_blocks(r1, tokens[:12])  # cache first 3 blocks (indices 0, 1, 2)
    mgr.free(r1)

    # Without putting any state checkpoint yet, prefix match must miss!
    miss_match = mgr.match_prefix(tokens, None)
    assert miss_match is not None
    assert miss_match.num_tokens == 0

    # Store checkpoint ONLY at block 1 (8 tokens); block 2 is cached in KV but has no state ckpt.
    digests = prefix.digests(prefix.root(None), tokens[:12])
    ckpts.put(digests[1], "ckpt_at_8", 16)
    match2 = mgr.match_prefix(tokens, None)
    assert match2 is not None
    assert match2.num_tokens == 8
    assert match2.checkpoint_digest == digests[1]
    mgr.check_integrity()


def test_kv_pool_and_budget_with_fp8_and_state() -> None:
    cfg = CacheConfig(kv_dtype="fp8", kv_fp8_scale=0.5, state_checkpoint_stride=2)
    assert cfg.kv_dtype == "fp8"
    assert cfg.state_checkpoint_stride == 2

    fp8_spec = KVSpec(
        1, 2, 16, block_size=4, dtype=DType.FP16, layout=KVLayoutKind.NHD, codec=FP8_E4M3_CODEC
    )
    fp8_pool = KVPool(KVGroup((0,), fp8_spec), num_blocks=4, device="cpu")
    assert fp8_pool.buffers(0).k.dtype == torch.float8_e4m3fn

    plan = plan_budget(
        BudgetInputs(
            baseline_bytes=10_000_000,
            memory_ratio=0.9,
            weights_bytes=1_000_000,
            kv=(KVCost(bytes_per_block=4096, block_size=4),),
            kv_reserve_tokens=16,
            state=StateCost(bytes_per_slot=65_536, max_slots=5),
        )
    )
    assert plan.state_bytes == 65_536 * 5
    assert plan.max_running <= 4


def test_state_slot_exhaustion_leaves_kv_untouched() -> None:
    group = KVGroup((0,), KVSpec(1, 2, 4, 4, DType.FP32, KVLayoutKind.NHD))
    slots = StateSlotPool(3)
    slots.allocate(RequestId("other"))  # 1 usable slot remains while table allows 2 requests
    mgr = KVCacheManager([group], [16], max_requests=2, max_blocks_per_request=4, state_slots=slots)
    assert mgr.allocate(RequestId("r1"), 4)
    assert not mgr.allocate(RequestId("r2"), 4)
    assert not mgr.has(RequestId("r2"))
    slots.release(RequestId("other"))
    mgr.check_integrity()
    with pytest.raises(KeyError):
        mgr.table.row_of(RequestId("r2"))
