from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

import pytest

from ayaka.utils import validation
from ayaka.utils.validation import (
    plain_token_ids,
    require_bool,
    require_float,
    require_frozen,
    require_int,
    require_nonempty_text,
    require_number,
    require_text,
    require_token_set,
    require_token_tuple,
)

pytestmark = pytest.mark.unit


class Color(Enum):
    RED = "red"


@dataclass(frozen=True)
class FrozenRecord:
    a: int
    b: tuple[int, ...]


@dataclass
class MutableRecord:
    a: int


@dataclass(frozen=True)
class HoldsList:
    items: list[int]


@dataclass(frozen=True)
class HoldsMutableRecord:
    record: MutableRecord


def test_module_exports_every_helper() -> None:
    assert set(validation.__all__) == {
        "plain_token_ids",
        "require_bool",
        "require_float",
        "require_frozen",
        "require_int",
        "require_nonempty_text",
        "require_number",
        "require_text",
        "require_token_set",
        "require_token_tuple",
    }


def test_utils_facade_reexports_every_helper() -> None:
    import ayaka.utils as utils

    for name in validation.__all__:
        assert getattr(utils, name) is getattr(validation, name)


class TestRequireBool:
    @pytest.mark.parametrize("value", [True, False])
    def test_accepts_booleans_and_returns_them(self, value: bool) -> None:
        assert require_bool(value, "enabled") is value

    @pytest.mark.parametrize("value", [0, 1, 1.0, "", "true", None, [], object()])
    def test_rejects_non_booleans(self, value: object) -> None:
        with pytest.raises(TypeError, match="enabled must be bool"):
            require_bool(value, "enabled")


class TestRequireInt:
    def test_accepts_values_at_and_above_minimum(self) -> None:
        assert require_int(0, "n") == 0
        assert require_int(5, "n", minimum=5) == 5
        assert require_int(-3, "n", minimum=-3) == -3

    def test_minimum_none_accepts_any_integer(self) -> None:
        assert require_int(-10, "n", minimum=None) == -10

    @pytest.mark.parametrize("value", [True, False, 1.0, "1", None, 1 + 0j, object()])
    def test_rejects_non_integers_and_booleans(self, value: object) -> None:
        with pytest.raises(TypeError, match="n must be an integer"):
            require_int(value, "n")

    @pytest.mark.parametrize("value", [-1, 0])
    def test_rejects_values_below_minimum(self, value: int) -> None:
        with pytest.raises(ValueError, match="n must be >= 1"):
            require_int(value, "n", minimum=1)


class TestRequireFloat:
    @pytest.mark.parametrize(("value", "expected"), [(0, 0.0), (1, 1.0), (0.5, 0.5), (-2.5, -2.5)])
    def test_accepts_finite_numbers_and_returns_a_float(
        self, value: int | float, expected: float
    ) -> None:
        result = require_float(value, "x")
        assert result == expected
        assert isinstance(result, float)

    @pytest.mark.parametrize("value", [True, False, "1", None, [], object()])
    def test_rejects_booleans_and_non_numbers(self, value: object) -> None:
        with pytest.raises(TypeError, match="x must be a number"):
            require_float(value, "x")

    @pytest.mark.parametrize("value", [math.inf, -math.inf, math.nan])
    def test_rejects_non_finite_values(self, value: float) -> None:
        with pytest.raises(ValueError, match="x must be finite"):
            require_float(value, "x")


class TestRequireNumber:
    @pytest.mark.parametrize(
        ("value", "expected"),
        [(0, 0.0), (1, 1.0), (-2.5, -2.5), (math.inf, math.inf), (-math.inf, -math.inf)],
    )
    def test_accepts_numbers_and_returns_a_float(self, value: int | float, expected: float) -> None:
        result = require_number(value, "x")
        assert result == expected
        assert isinstance(result, float)

    @pytest.mark.parametrize("value", [True, False, "1", None, [], object()])
    def test_rejects_booleans_and_non_numbers(self, value: object) -> None:
        with pytest.raises(TypeError, match="x must be a number"):
            require_number(value, "x")

    def test_rejects_nan(self) -> None:
        with pytest.raises(ValueError, match="x must not be NaN"):
            require_number(math.nan, "x")

    def test_accepts_value_at_minimum(self) -> None:
        assert require_number(2, "x", minimum=2.0) == 2.0

    def test_rejects_value_below_minimum(self) -> None:
        with pytest.raises(ValueError, match=r"x must be >= 2\.0"):
            require_number(1.999, "x", minimum=2.0)


class TestRequireText:
    @pytest.mark.parametrize("value", ["a", " a ", "0", "日本語"])
    def test_accepts_non_blank_text_without_normalizing(self, value: str) -> None:
        assert require_text(value, "name") is value

    @pytest.mark.parametrize("value", ["", "   ", "\t\n", None, 3, b"x", []])
    def test_rejects_blank_or_non_string(self, value: object) -> None:
        with pytest.raises(ValueError, match="name must be a non-empty string"):
            require_text(value, "name")


class TestRequireNonemptyText:
    @pytest.mark.parametrize("value", ["a", " ", "\t\n", "0", "日本語"])
    def test_accepts_any_non_empty_text(self, value: str) -> None:
        assert require_nonempty_text(value, "stop") is value

    @pytest.mark.parametrize("value", ["", None, 3, b"x", []])
    def test_rejects_empty_or_non_string(self, value: object) -> None:
        with pytest.raises(ValueError, match="stop must be a non-empty string"):
            require_nonempty_text(value, "stop")


class TestRequireFrozen:
    @pytest.mark.parametrize(
        "value",
        [
            None,
            "s",
            1,
            -1,
            1.5,
            math.inf,
            True,
            False,
            Color.RED,
            (),
            (1,),
            (1, ("a", None, Color.RED)),
        ],
    )
    def test_accepts_immutable_values(self, value: object) -> None:
        require_frozen(value, "v")

    def test_a_tuple_of_scalars_is_accepted_without_recursion(self) -> None:
        require_frozen((1, "a", 2.0, None, True) * 10_000, "v")

    def test_an_enum_or_a_dataclass_in_a_tuple_still_gets_the_full_check(self) -> None:
        require_frozen((1, Color.RED), "v")
        with pytest.raises(TypeError, match=r"v\[1\]"):
            require_frozen((1, MutableRecord(1)), "v")

    def test_accepts_frozen_dataclass(self) -> None:
        require_frozen(FrozenRecord(1, (2, 3)), "v")

    def test_accepts_frozen_dataclass_nested_in_tuple(self) -> None:
        require_frozen((1, FrozenRecord(2, (3, 4))), "v")

    @pytest.mark.parametrize(
        "value",
        [
            [1],
            {"a": 1},
            {1},
            bytearray(b"x"),
            MutableRecord(1),
            FrozenRecord,
            object(),
        ],
    )
    def test_rejects_mutable_values(self, value: object) -> None:
        with pytest.raises(TypeError, match="immutable host values"):
            require_frozen(value, "v")

    def test_error_reports_nested_tuple_path(self) -> None:
        with pytest.raises(TypeError, match=r"v\[1\]\[0\]"):
            require_frozen((1, ([],)), "v")

    def test_error_reports_frozen_dataclass_field_path(self) -> None:
        with pytest.raises(TypeError, match=r"v\.items"):
            require_frozen(HoldsList([1]), "v")

    def test_error_reports_nested_mutable_dataclass_path(self) -> None:
        with pytest.raises(TypeError, match=r"v\.record"):
            require_frozen(HoldsMutableRecord(MutableRecord(1)), "v")


class TestRequireTokenTuple:
    @pytest.mark.parametrize("value", [(), (0,), (0, 1, 2, 0)])
    def test_accepts_tuples_of_non_negative_ids(self, value: tuple[int, ...]) -> None:
        assert require_token_tuple(value, "token_ids") is value

    @pytest.mark.parametrize("value", [[1], {1}, "1", None, 1, object()])
    def test_rejects_non_tuples(self, value: object) -> None:
        with pytest.raises(ValueError, match="token_ids must be a tuple of token ids"):
            require_token_tuple(value, "token_ids")

    @pytest.mark.parametrize("value", [(0, -1), (-1,)])
    def test_rejects_negative_ids(self, value: tuple[int, ...]) -> None:
        with pytest.raises(ValueError, match="token_ids must be >= 0"):
            require_token_tuple(value, "token_ids")

    @pytest.mark.parametrize("value", [(True,), (1.0,), ("1",), (None,)])
    def test_rejects_non_integer_ids(self, value: tuple[object, ...]) -> None:
        with pytest.raises(TypeError, match="token_ids must be an integer"):
            require_token_tuple(value, "token_ids")

    def test_the_first_bad_id_decides_the_error_as_before(self) -> None:
        # The fast check only accepts; the item-by-item loop still finds the first bad id.
        with pytest.raises(ValueError, match="must be >= 0"):
            require_token_tuple((1, -1, True), "token_ids")
        with pytest.raises(TypeError, match="must be an integer"):
            require_token_tuple((1, True, -1), "token_ids")


class TestPlainTokenIds:
    @pytest.mark.parametrize("value", [(), (0,), (5, 0, 7), tuple(range(100_000))])
    def test_plain_non_negative_ints_pass(self, value: tuple[int, ...]) -> None:
        assert plain_token_ids(value)

    @pytest.mark.parametrize("value", [(1, -1), (True,), (0, False), (1, 2.0), (1, "2"), (None,)])
    def test_anything_else_is_left_to_the_loop(self, value: tuple[object, ...]) -> None:
        assert not plain_token_ids(value)

    def test_an_int_subclass_is_not_plain(self) -> None:
        class Id(int):
            pass

        assert not plain_token_ids((Id(3),))


class TestRequireTokenSet:
    @pytest.mark.parametrize("value", [[0, 1, 2], (0, 1, 2), {0, 1, 2}, frozenset({0, 1, 2})])
    def test_accepts_iterables_and_canonicalizes_to_frozenset(self, value: object) -> None:
        result = require_token_set(value, "token_ids", allow_empty=False)
        assert result == frozenset({0, 1, 2})
        assert isinstance(result, frozenset)

    def test_accepts_generator_and_removes_duplicates(self) -> None:
        result = require_token_set((n for n in (0, 1, 1, 2)), "token_ids", allow_empty=False)
        assert result == frozenset({0, 1, 2})

    def test_allows_empty_when_requested(self) -> None:
        assert require_token_set([], "token_ids", allow_empty=True) == frozenset()

    def test_rejects_empty_when_not_allowed(self) -> None:
        with pytest.raises(ValueError, match="token_ids must not be empty"):
            require_token_set((), "token_ids", allow_empty=False)

    @pytest.mark.parametrize("value", ["12", b"12", 12, None, object()])
    def test_rejects_strings_and_non_iterables(self, value: object) -> None:
        with pytest.raises(ValueError, match="token_ids must be an iterable of token ids"):
            require_token_set(value, "token_ids", allow_empty=True)

    @pytest.mark.parametrize("value", [[0, -1], (-1,)])
    def test_rejects_negative_ids(self, value: object) -> None:
        with pytest.raises(ValueError, match="token_ids must be >= 0"):
            require_token_set(value, "token_ids", allow_empty=True)

    @pytest.mark.parametrize("value", [[True], [1.0], ["1"], [None]])
    def test_rejects_non_integer_ids(self, value: object) -> None:
        with pytest.raises(TypeError, match="token_ids must be an integer"):
            require_token_set(value, "token_ids", allow_empty=True)
