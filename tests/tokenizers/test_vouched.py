"""The memory of checked id tuples: identity only, one take per vouch, bounded."""

from __future__ import annotations

import pytest

from ayaka.tokenizers.vouched import VouchedIds

pytestmark = pytest.mark.unit


def _ids(*values: int) -> tuple[int, ...]:
    return (*values,)  # a fresh tuple object every call


def test_a_vouch_is_good_for_one_take() -> None:
    memory = VouchedIds(max_entries=4, max_ids=100)
    ids = _ids(1, 2, 3)
    memory.vouch(ids)

    assert memory.take(ids)
    assert not memory.take(ids)
    assert len(memory) == 0


def test_an_equal_tuple_is_not_the_one_that_was_checked() -> None:
    memory = VouchedIds(max_entries=4, max_ids=100)
    memory.vouch(_ids(1, 2, 3))

    assert not memory.take(_ids(1, 2, 3))


def test_two_vouches_for_one_tuple_give_two_takes() -> None:
    memory = VouchedIds(max_entries=4, max_ids=100)
    ids = _ids(7, 8)
    memory.vouch(ids)
    memory.vouch(ids)

    assert memory.take(ids)
    assert memory.take(ids)
    assert not memory.take(ids)


def test_the_oldest_tuple_goes_first_beyond_the_count() -> None:
    memory = VouchedIds(max_entries=2, max_ids=100)
    first, second, third = _ids(1), _ids(2), _ids(3)
    for ids in (first, second, third):
        memory.vouch(ids)

    assert not memory.take(first)
    assert memory.take(second)
    assert memory.take(third)


def test_the_oldest_tuples_go_first_beyond_the_ids() -> None:
    memory = VouchedIds(max_entries=10, max_ids=5)
    first, second = _ids(1, 2, 3), _ids(4, 5, 6)
    memory.vouch(first)
    memory.vouch(second)

    assert not memory.take(first)
    assert memory.take(second)


def test_clear_forgets_everything() -> None:
    memory = VouchedIds(max_entries=4, max_ids=100)
    ids = _ids(1)
    memory.vouch(ids)
    memory.clear()

    assert not memory.take(ids)


def test_bounds_must_be_positive() -> None:
    with pytest.raises(ValueError, match=">= 1"):
        VouchedIds(max_entries=0, max_ids=1)
