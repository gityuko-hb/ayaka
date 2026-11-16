"""The block table: rows per request, per group, reuse, and what changed since the last push."""

from __future__ import annotations

import numpy as np
import pytest

from ayaka.core.errors import CapacityError, EngineError
from ayaka.core.types import RequestId
from ayaka.kvcache.table import BlockTable

pytestmark = pytest.mark.unit


def _id(name: str) -> RequestId:
    return RequestId(name)


def _table(groups: int = 1, requests: int = 4, blocks: int = 5) -> BlockTable:
    return BlockTable(groups, requests, blocks)


def test_rows_are_handed_out_in_order_and_start_empty() -> None:
    table = _table()

    assert [table.acquire(_id(name)) for name in "abc"] == [0, 1, 2]
    assert table.free_rows == 1
    assert table.num_blocks(_id("a"), 0) == 0
    assert table.blocks(_id("a"), 0) == ()
    assert not table.rows.any()
    assert table.has(_id("b"))
    assert not table.has(_id("z"))
    assert table.row_of(_id("c")) == 2


def test_the_table_reports_its_shape() -> None:
    table = _table(2, 3, 7)

    assert (table.num_groups, table.max_requests, table.max_blocks_per_request) == (2, 3, 7)
    assert table.rows.shape == (2, 3, 7)
    assert table.rows.dtype == np.int32
    assert table.lengths.shape == (2, 3)
    assert table.lengths.dtype == np.int32


def test_a_full_table_and_a_double_acquire_are_refused() -> None:
    table = _table(requests=2)
    table.acquire(_id("a"))
    table.acquire(_id("b"))

    with pytest.raises(CapacityError):
        table.acquire(_id("c"))
    with pytest.raises(EngineError, match="already holds"):
        table.acquire(_id("a"))


def test_a_released_row_is_cleared_and_reused_first() -> None:
    table = _table()
    table.acquire(_id("a"))
    table.acquire(_id("b"))
    table.append(_id("a"), 0, [3, 4, 5])

    table.release(_id("a"))

    assert not table.rows[0, 0].any()
    assert table.lengths[0, 0] == 0
    assert not table.has(_id("a"))
    assert table.acquire(_id("c")) == 0
    assert table.blocks(_id("c"), 0) == ()
    with pytest.raises(KeyError):
        table.release(_id("a"))


def test_append_writes_ids_in_token_order_and_zero_fills_the_rest() -> None:
    table = _table()
    table.acquire(_id("a"))

    table.append(_id("a"), 0, [7, 2])
    table.append(_id("a"), 0, [9])

    assert table.blocks(_id("a"), 0) == (7, 2, 9)
    assert table.num_blocks(_id("a"), 0) == 3
    assert table.room(_id("a"), 0) == 2
    assert table.rows[0, 0].tolist() == [7, 2, 9, 0, 0]


def test_an_append_that_does_not_fit_writes_nothing() -> None:
    table = _table(blocks=3)
    table.acquire(_id("a"))
    table.append(_id("a"), 0, [1, 2])

    with pytest.raises(CapacityError, match="needs 4 blocks"):
        table.append(_id("a"), 0, [3, 4])

    assert table.blocks(_id("a"), 0) == (1, 2)
    assert table.rows[0, 0].tolist() == [1, 2, 0]


@pytest.mark.parametrize("bad", [0, -3])
def test_the_padding_block_cannot_be_appended(bad: int) -> None:
    table = _table()
    table.acquire(_id("a"))

    with pytest.raises(EngineError, match="padding"):
        table.append(_id("a"), 0, [1, bad])

    assert table.num_blocks(_id("a"), 0) == 0


def test_an_empty_append_does_nothing() -> None:
    table = _table()
    table.acquire(_id("a"))

    table.append(_id("a"), 0, [])

    assert table.take_dirty() == []


def test_groups_hold_independent_ids_for_the_same_request() -> None:
    table = _table(groups=2)
    table.acquire(_id("a"))

    table.append(_id("a"), 0, [1, 2])
    table.append(_id("a"), 1, [8])

    assert table.blocks(_id("a"), 0) == (1, 2)
    assert table.blocks(_id("a"), 1) == (8,)


def test_gather_returns_the_requested_rows_in_order_as_a_copy() -> None:
    table = _table()
    for name, blocks in (("a", [1, 2]), ("b", [3]), ("c", [4, 5, 6])):
        table.acquire(_id(name))
        table.append(_id(name), 0, blocks)

    rows = table.gather(0, [_id("c"), _id("a")])

    assert rows.dtype == np.int32
    assert rows.tolist() == [[4, 5, 6, 0, 0], [1, 2, 0, 0, 0]]
    rows[:] = 99
    assert table.blocks(_id("c"), 0) == (4, 5, 6)


def test_dirty_ranges_cover_exactly_what_changed_and_are_taken_once() -> None:
    table = _table(groups=2)
    table.acquire(_id("a"))
    table.append(_id("a"), 0, [1, 2])
    table.append(_id("a"), 1, [5])
    table.append(_id("a"), 0, [3])

    assert table.take_dirty() == [(0, 0, 0, 2), (1, 0, 0, 1), (0, 0, 2, 3)]
    assert table.take_dirty() == []

    table.release(_id("a"))
    assert table.take_dirty() == [(0, 0, 0, 3), (1, 0, 0, 1)]


@pytest.mark.parametrize(
    ("groups", "requests", "blocks"),
    [(0, 1, 1), (1, 0, 1), (1, 1, 0)],
)
def test_a_table_needs_a_group_a_request_and_a_block(
    groups: int, requests: int, blocks: int
) -> None:
    with pytest.raises(ValueError, match=">= 1"):
        BlockTable(groups, requests, blocks)
