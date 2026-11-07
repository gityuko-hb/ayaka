from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import pytest

from ayaka.utils.validation import require_frozen, require_int, require_text


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


@pytest.mark.unit
class TestRequireInt:
    def test_accepts_value_at_and_above_minimum(self) -> None:
        require_int(0, "n")
        require_int(5, "n", minimum=5)

    @pytest.mark.parametrize("value", [True, False, 1.0, "1", None])
    def test_rejects_non_integers_and_booleans(self, value: object) -> None:
        with pytest.raises(TypeError, match="n must be an integer"):
            require_int(value, "n")  # type: ignore[arg-type]  # deliberately wrong type

    def test_rejects_below_minimum(self) -> None:
        with pytest.raises(ValueError, match="n must be >= 1"):
            require_int(0, "n", minimum=1)


@pytest.mark.unit
class TestRequireText:
    def test_accepts_text_without_normalizing(self) -> None:
        require_text(" a ", "name")

    @pytest.mark.parametrize("value", ["", "   ", "\t\n", None, 3])
    def test_rejects_blank_or_non_string(self, value: object) -> None:
        with pytest.raises(ValueError, match="name must be a non-empty string"):
            require_text(value, "name")  # type: ignore[arg-type]  # deliberately wrong type


@pytest.mark.unit
class TestRequireFrozen:
    @pytest.mark.parametrize("value", [None, "s", 1, 1.5, True, Color.RED, (), (1, ("a", None))])
    def test_accepts_immutable_values(self, value: object) -> None:
        require_frozen(value, "v")

    def test_accepts_frozen_dataclass(self) -> None:
        require_frozen(FrozenRecord(1, (2, 3)), "v")

    @pytest.mark.parametrize(
        "value", [[1], {"a": 1}, {1}, MutableRecord(1), FrozenRecord, object()]
    )
    def test_rejects_mutable_values(self, value: object) -> None:
        with pytest.raises(TypeError, match="immutable host values"):
            require_frozen(value, "v")

    def test_error_reports_nested_path(self) -> None:
        with pytest.raises(TypeError, match=r"v\[1\]\[0\]"):
            require_frozen((1, ([],)), "v")

    def test_rejects_mutable_field_in_frozen_dataclass(self) -> None:
        with pytest.raises(TypeError, match=r"v\.items"):
            require_frozen(HoldsList([1]), "v")
