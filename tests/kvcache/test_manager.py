"""The cache manager: all-or-nothing allocation, look-ahead rows, commit, free and integrity."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

import pytest

from ayaka.core.errors import EngineError
from ayaka.core.types import KVLayoutKind, RequestId
from ayaka.kvcache.manager import KVCacheManager, KVCacheStats
from ayaka.kvcache.storage.pool import KVPool
from ayaka.kvcache.storage.spec import KVGroup, KVSpec
from ayaka.utils.dtypes import DType

torch: Any = pytest.importorskip("torch")

pytestmark = pytest.mark.unit

BLOCK = 4


def _rid(name: str | int) -> RequestId:
    return RequestId(str(name))


def _group(layers: tuple[int, ...] = (0, 1), *, block_size: int = BLOCK) -> KVGroup:
    spec = KVSpec(
        num_layers=len(layers),
        num_kv_heads=2,
        head_dim=4,
        block_size=block_size,
        dtype=DType.FP32,
        layout=KVLayoutKind.NHD,
    )
    return KVGroup(layer_ids=layers, spec=spec)


def _manager(
    num_blocks: tuple[int, ...] = (9,),
    *,
    max_requests: int = 4,
    max_blocks: int = 6,
) -> KVCacheManager:
    groups = [_group((2 * i, 2 * i + 1)) for i in range(len(num_blocks))]
    return KVCacheManager(
        groups, num_blocks, max_requests=max_requests, max_blocks_per_request=max_blocks
    )


@dataclass
class Fingerprint:
    """Everything observable about a manager, to prove a refused call changed nothing."""

    queues: list[list[int]]
    rows: list[list[int]]
    lengths: list[list[int]]
    committed: dict[RequestId, int]
    stats: KVCacheStats

    @classmethod
    def of(cls, manager: KVCacheManager, requests: list[RequestId]) -> Fingerprint:
        return cls(
            queues=[pool.free_queue() for pool in manager._pools],
            rows=manager.table.rows.tolist(),
            lengths=manager.table.lengths.tolist(),
            committed={r: manager.num_committed(r) for r in requests if manager.has(r)},
            stats=manager.stats(),
        )


# -- allocate -----


def test_a_new_request_gets_the_blocks_its_tokens_need_in_every_group() -> None:
    manager = _manager((9, 9))

    assert manager.allocate(_rid("a"), 10)

    first, second = manager.blocks(_rid("a"))
    assert len(first) == len(second) == 3  # ceil(10 / 4)
    assert first == (1, 2, 3)
    assert second == (1, 2, 3)  # each group has its own pool, so ids coincide by coincidence
    assert manager.table.row_of(_rid("a")) == 0
    assert manager.num_committed(_rid("a")) == 0
    assert manager.stats().free_blocks == (5, 5)


def test_block_zero_is_never_given_to_a_request() -> None:
    manager = _manager((5,), max_blocks=8)
    assert manager.allocate(_rid("a"), 16)

    assert 0 not in manager.blocks(_rid("a"))[0]
    assert manager.free_blocks == 0


def test_lookahead_rows_count_toward_the_room_a_step_needs() -> None:
    manager = _manager((9,))
    assert manager.allocate(_rid("a"), 8)  # prefill: exactly two blocks
    manager.commit(_rid("a"), 8)

    # A decode step with 3 drafts writes 1 + 3 rows, which needs a third block.
    assert manager.blocks_needed(_rid("a"), 1 + 3) == 1
    assert manager.allocate(_rid("a"), 1 + 3)

    assert len(manager.blocks(_rid("a"))[0]) == 3


def test_room_is_measured_from_the_committed_length_so_asking_twice_is_free() -> None:
    manager = _manager((9,))
    manager.allocate(_rid("a"), 5)
    before = Fingerprint.of(manager, [_rid("a")])

    assert manager.allocate(_rid("a"), 5)

    assert Fingerprint.of(manager, [_rid("a")]) == before


def test_rejected_drafts_do_not_grow_the_cache_on_the_next_step() -> None:
    manager = _manager((9,))
    manager.allocate(_rid("a"), 6)
    manager.commit(_rid("a"), 6)
    manager.allocate(_rid("a"), 1 + 3)  # rows for positions 6..9
    held = len(manager.blocks(_rid("a"))[0])

    manager.commit(_rid("a"), 1)  # none accepted: only the base token stuck
    manager.allocate(_rid("a"), 1 + 3)  # next step: positions 7..10, same blocks

    assert len(manager.blocks(_rid("a"))[0]) == held == 3


def test_a_request_grows_one_block_at_a_time_while_decoding() -> None:
    manager = _manager((9,))
    manager.allocate(_rid("a"), 6)
    manager.commit(_rid("a"), 6)
    sizes = []

    for _ in range(7):
        assert manager.allocate(_rid("a"), 1)
        manager.commit(_rid("a"), 1)
        sizes.append(len(manager.blocks(_rid("a"))[0]))

    assert sizes == [2, 2, 3, 3, 3, 3, 4]
    assert manager.num_committed(_rid("a")) == 13


def test_a_zero_token_allocation_still_admits_the_request() -> None:
    manager = _manager()

    assert manager.allocate(_rid("a"), 0)

    assert manager.has(_rid("a"))
    assert manager.blocks(_rid("a")) == ((),)


# -- refusing: False, never a raise, never a change -----


def test_a_shortage_returns_false_and_changes_nothing() -> None:
    manager = _manager((5,), max_blocks=8)  # 4 usable blocks
    ids = [_rid("a"), _rid("b")]
    assert manager.allocate(ids[0], 12)  # 3 blocks
    before = Fingerprint.of(manager, ids)

    assert manager.allocate(ids[1], 9) is False  # needs 3, 1 free

    assert Fingerprint.of(manager, ids) == before
    assert not manager.has(ids[1])
    manager.check_integrity()


def test_a_known_request_that_cannot_grow_is_left_as_it_was() -> None:
    manager = _manager((4,), max_blocks=8)  # 3 usable blocks
    manager.allocate(_rid("a"), 8)
    manager.commit(_rid("a"), 8)
    before = Fingerprint.of(manager, [_rid("a")])

    assert manager.allocate(_rid("a"), 9) is False  # needs 2 more, 1 free

    assert Fingerprint.of(manager, [_rid("a")]) == before


def test_allocation_is_atomic_across_groups() -> None:
    # Group 0 has room for 5 blocks, group 1 for only 2.
    manager = _manager((6, 3), max_blocks=8)
    ids = [_rid("a")]
    before = Fingerprint.of(manager, ids)

    assert manager.allocate(_rid("a"), 12) is False  # 3 blocks: fits group 0, not group 1

    assert Fingerprint.of(manager, ids) == before
    assert manager.stats().free_blocks == (5, 2)
    assert manager.allocate(_rid("a"), 8)  # 2 blocks fit both
    assert manager.stats().free_blocks == (3, 0)
    manager.check_integrity()


def test_a_second_group_running_dry_midway_gives_the_first_group_nothing() -> None:
    manager = _manager((6, 4), max_blocks=8)
    assert manager.allocate(_rid("a"), 8)
    manager.commit(_rid("a"), 8)
    before = Fingerprint.of(manager, [_rid("a")])

    assert manager.allocate(_rid("a"), 8) is False  # +2 blocks; group 1 has 1 free

    assert Fingerprint.of(manager, [_rid("a")]) == before
    assert manager.stats().free_blocks == (3, 1)


def test_a_full_block_table_refuses_a_new_request_but_not_a_known_one() -> None:
    manager = _manager((20,), max_requests=2)
    manager.allocate(_rid("a"), 4)
    manager.allocate(_rid("b"), 4)
    before = Fingerprint.of(manager, [_rid("a"), _rid("b"), _rid("c")])

    assert manager.allocate(_rid("c"), 1) is False

    assert Fingerprint.of(manager, [_rid("a"), _rid("b"), _rid("c")]) == before
    assert manager.allocate(_rid("a"), 4)


def test_a_request_cannot_outgrow_its_table_row() -> None:
    manager = _manager((20,), max_blocks=3)
    assert manager.max_tokens_per_request == 12

    assert manager.allocate(_rid("a"), 13) is False
    assert not manager.has(_rid("a"))
    assert manager.allocate(_rid("a"), 12)
    manager.commit(_rid("a"), 12)
    assert manager.allocate(_rid("a"), 1) is False
    assert len(manager.blocks(_rid("a"))[0]) == 3


def test_pressure_never_evicts() -> None:
    manager = _manager((3,), max_blocks=8)
    manager.allocate(_rid("a"), 8)

    assert manager.allocate(_rid("b"), 1) is False

    assert manager.stats().evictions == 0
    assert manager.has(_rid("a"))


@pytest.mark.parametrize("bad", [-1, 1.5, True])
def test_a_bad_token_count_is_a_caller_error(bad: Any) -> None:
    manager = _manager()

    with pytest.raises((TypeError, ValueError)):
        manager.allocate(_rid("a"), bad)


# -- commit -----


def test_commit_records_real_tokens_up_to_the_room_held() -> None:
    manager = _manager()
    manager.allocate(_rid("a"), 5)

    manager.commit(_rid("a"), 3)
    manager.commit(_rid("a"), 5)

    assert manager.num_committed(_rid("a")) == 8
    with pytest.raises(EngineError, match="room for 8"):
        manager.commit(_rid("a"), 1)
    assert manager.num_committed(_rid("a")) == 8


def test_commit_of_an_unknown_request_is_a_key_error() -> None:
    with pytest.raises(KeyError):
        _manager().commit(_rid("ghost"), 1)
    with pytest.raises(KeyError):
        _manager().num_committed(_rid("ghost"))


# -- free -----


def test_free_returns_blocks_last_first_so_the_chain_is_reused_from_its_root() -> None:
    manager = _manager((9,))
    manager.allocate(_rid("a"), 12)
    manager.allocate(_rid("b"), 4)
    chain = manager.blocks(_rid("a"))[0]

    manager.free(_rid("a"))

    queue = manager._pools[0].free_queue()
    assert queue[:3] == list(chain)
    assert not manager.has(_rid("a"))
    assert manager.stats().free_blocks == (7,)
    manager.check_integrity()


def test_freeing_makes_room_and_the_row_reusable() -> None:
    manager = _manager((4,), max_requests=1, max_blocks=8)
    assert manager.allocate(_rid("a"), 12)
    assert manager.allocate(_rid("b"), 4) is False

    manager.free(_rid("a"))

    assert manager.allocate(_rid("b"), 12)
    assert manager.table.row_of(_rid("b")) == 0
    assert manager.blocks(_rid("b")) == ((1, 2, 3),)


def test_freeing_an_unknown_request_is_ignored() -> None:
    manager = _manager()

    manager.free(_rid("never-admitted"))

    assert manager.stats().num_requests == 0
    manager.check_integrity()


# -- the model: plain arithmetic over random operations -----


@pytest.mark.parametrize("seed", range(10))
def test_the_manager_agrees_with_plain_arithmetic_on_random_operations(seed: int) -> None:
    rng = random.Random(seed)  # noqa: S311  # a seeded simulation, not security
    groups = rng.choice([1, 2])
    totals = tuple(rng.randint(6, 40) for _ in range(groups))
    manager = _manager(totals, max_requests=5, max_blocks=7)
    held: dict[RequestId, tuple[int, int]] = {}  # request -> (committed tokens, blocks held)
    next_id = 0

    for _ in range(1200):
        free = [t - 1 - sum(b for _, b in held.values()) for t in totals]
        action = rng.choice(["new", "grow", "commit", "free", "free"])
        if action == "new":
            next_id += 1
            request, tokens = _rid(next_id), rng.randint(0, 20)
            blocks = -(-tokens // BLOCK)
            fits = len(held) < 5 and blocks <= 7 and all(f >= blocks for f in free)
            assert manager.allocate(request, tokens) == fits
            if fits:
                held[request] = (0, blocks)
        elif held and action in ("grow", "commit"):
            request = rng.choice(list(held))
            committed, blocks = held[request]
            if action == "grow":
                rows = rng.randint(1, 6)
                wanted = max(blocks, -(-(committed + rows) // BLOCK))
                fits = wanted <= 7 and all(f >= wanted - blocks for f in free)
                assert manager.allocate(request, rows) == fits
                if fits:
                    held[request] = (committed, wanted)
            else:
                add = rng.randint(0, blocks * BLOCK - committed)
                manager.commit(request, add)
                held[request] = (committed + add, blocks)
        elif held:
            request = rng.choice(list(held))
            manager.free(request)
            del held[request]
        manager.check_integrity()
        stats = manager.stats()
        assert stats.num_requests == len(held)
        assert stats.free_blocks == tuple(t - 1 - sum(b for _, b in held.values()) for t in totals)
        for request, (committed, blocks) in held.items():
            assert manager.num_committed(request) == committed
            assert all(len(chain) == blocks for chain in manager.blocks(request))

    for request in list(held):
        manager.free(request)
    assert manager.stats().free_blocks == tuple(t - 1 for t in totals)
    manager.check_integrity()


# -- integrity -----


def test_check_integrity_passes_on_a_busy_manager() -> None:
    manager = _manager((9, 9))
    for name in "abc":
        manager.allocate(_rid(name), 7)
    manager.commit(_rid("b"), 7)
    manager.free(_rid("a"))

    manager.check_integrity()


def test_check_integrity_finds_a_leaked_block() -> None:
    manager = _manager()
    manager.allocate(_rid("a"), 4)
    manager._pools[0].allocate()  # taken but never put in the table

    with pytest.raises(EngineError, match="inconsistent"):
        manager.check_integrity()


def test_check_integrity_finds_a_block_held_by_two_requests() -> None:
    manager = _manager()
    manager.allocate(_rid("a"), 4)
    manager.allocate(_rid("b"), 4)
    shared = manager.blocks(_rid("a"))[0][0]
    manager.table.rows[0, manager.table.row_of(_rid("b")), 0] = shared

    with pytest.raises(EngineError, match="two requests"):
        manager.check_integrity()


def test_check_integrity_finds_a_wrong_reference_count() -> None:
    manager = _manager()
    manager.allocate(_rid("a"), 4)
    manager._pools[0].touch(manager.blocks(_rid("a"))[0][0])

    with pytest.raises(EngineError, match="2 references, 1 in the table"):
        manager.check_integrity()


def test_check_integrity_finds_committed_tokens_beyond_the_room() -> None:
    manager = _manager()
    manager.allocate(_rid("a"), 4)
    manager._committed[_rid("a")] = 99

    with pytest.raises(EngineError, match="more tokens than it has room for"):
        manager.check_integrity()


def test_check_integrity_finds_a_table_row_without_a_request() -> None:
    manager = _manager()
    manager.allocate(_rid("a"), 4)
    del manager._committed[_rid("a")]

    with pytest.raises(EngineError, match="table rows in use"):
        manager.check_integrity()


# -- stats -----


def test_stats_report_occupancy_per_group() -> None:
    manager = _manager((9, 5))
    manager.allocate(_rid("a"), 8)

    stats = manager.stats()

    assert stats.num_requests == 1
    assert stats.usable_blocks == (8, 4)
    assert stats.free_blocks == (6, 2)
    assert stats.used_blocks == (2, 2)
    assert stats.utilization == pytest.approx(0.5)


def test_utilization_is_zero_for_an_idle_cache() -> None:
    assert _manager().stats().utilization == 0.0


# -- construction -----


def test_construction_checks_groups_counts_and_block_sizes() -> None:
    with pytest.raises(ValueError, match="at least one"):
        KVCacheManager([], [], max_requests=1, max_blocks_per_request=1)
    with pytest.raises(ValueError, match="block counts"):
        KVCacheManager([_group()], [4, 4], max_requests=1, max_blocks_per_request=1)
    with pytest.raises(ValueError, match="same block_size"):
        KVCacheManager(
            [_group((0,), block_size=4), _group((1,), block_size=8)],
            [4, 4],
            max_requests=1,
            max_blocks_per_request=1,
        )
    with pytest.raises(ValueError, match="num_blocks"):
        KVCacheManager([_group()], [1], max_requests=1, max_blocks_per_request=1)


def test_the_manager_exposes_its_groups_and_block_size() -> None:
    manager = _manager((9, 9))

    assert manager.block_size == BLOCK
    assert [g.layer_ids for g in manager.groups] == [(0, 1), (2, 3)]


# -- with the tensors -----


def test_the_ids_the_manager_hands_out_index_the_pool_tensors_and_spare_the_padding_block() -> None:
    groups = [_group((0, 1)), _group((2,))]
    counts = (7, 7)
    manager = KVCacheManager(groups, counts, max_requests=3, max_blocks_per_request=4)
    pools = [
        KVPool(group, count, device="cpu") for group, count in zip(groups, counts, strict=True)
    ]
    manager.allocate(_rid("a"), 10)
    manager.allocate(_rid("b"), 5)

    for request, marker in ((_rid("a"), 1.0), (_rid("b"), 2.0)):
        for group, chain in enumerate(manager.blocks(request)):
            layer = groups[group].layer_ids[0]
            for block_id in chain:
                pools[group].buffers(layer).k[block_id].fill_(marker)

    for group, pool in enumerate(pools):
        layer = groups[group].layer_ids[0]
        k = pool.buffers(layer).k
        assert not k[0].any()  # the padding block was never written
        assert k[list(manager.blocks(_rid("a"))[group])].eq(1.0).all()
        assert k[list(manager.blocks(_rid("b"))[group])].eq(2.0).all()
        assert torch.count_nonzero(k.flatten(1).any(dim=1)).item() == 5  # 3 + 2 blocks written
