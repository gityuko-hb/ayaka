from __future__ import annotations

import pytest

from ayaka.utils.math_utils import (
    align_down,
    align_up,
    div_ceil,
    next_power_of_2,
    percentiles,
)

pytestmark = pytest.mark.unit


class TestDivCeil:
    @pytest.mark.parametrize(
        ("a", "b", "expected"),
        [(10, 3, 4), (9, 3, 3), (0, 3, 0), (-10, 3, -3), (-9, 3, -3), (1, 1, 1)],
    )
    def test_rounds_toward_positive_infinity(self, a: int, b: int, expected: int) -> None:
        assert div_ceil(a, b) == expected

    @pytest.mark.parametrize(("a", "b"), [(1.0, 2), (1, 2.0), (True, 2), (1, True)])
    def test_rejects_non_integers(self, a: object, b: object) -> None:
        with pytest.raises(TypeError):
            div_ceil(a, b)  # type: ignore[arg-type]  # deliberately wrong type

    def test_rejects_non_positive_divisor(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            div_ceil(1, 0)


class TestNextPowerOfTwo:
    @pytest.mark.parametrize(
        ("value", "expected"), [(0, 0), (1, 1), (2, 2), (3, 4), (5, 8), (1024, 1024)]
    )
    def test_smallest_power_at_least_value(self, value: int, expected: int) -> None:
        assert next_power_of_2(value) == expected

    def test_rejects_negative_value(self) -> None:
        with pytest.raises(ValueError, match="negative"):
            next_power_of_2(-1)

    def test_rejects_non_integer(self) -> None:
        with pytest.raises(TypeError):
            next_power_of_2(2.0)  # type: ignore[arg-type]  # deliberately wrong type


class TestAlign:
    @pytest.mark.parametrize(
        ("value", "alignment", "expected"), [(0, 4, 0), (1, 4, 4), (4, 4, 4), (5, 4, 8), (5, 3, 6)]
    )
    def test_align_up(self, value: int, alignment: int, expected: int) -> None:
        assert align_up(value, alignment) == expected

    @pytest.mark.parametrize(
        ("value", "alignment", "expected"), [(0, 4, 0), (3, 4, 0), (4, 4, 4), (7, 4, 4), (8, 3, 6)]
    )
    def test_align_down(self, value: int, alignment: int, expected: int) -> None:
        assert align_down(value, alignment) == expected

    @pytest.mark.parametrize("alignment", [0, -1])
    def test_rejects_non_positive_alignment(self, alignment: int) -> None:
        with pytest.raises(ValueError, match="alignment must be positive"):
            align_up(1, alignment)
        with pytest.raises(ValueError, match="alignment must be positive"):
            align_down(1, alignment)

    def test_rejects_negative_value(self) -> None:
        with pytest.raises(ValueError, match="must not be negative"):
            align_up(-1, 4)
        with pytest.raises(ValueError, match="must not be negative"):
            align_down(-1, 4)


class TestPercentiles:
    def test_empty_sequence_is_all_zero(self) -> None:
        assert percentiles([]) == {
            "min": 0.0,
            "mean": 0.0,
            "p50": 0.0,
            "p90": 0.0,
            "p95": 0.0,
            "p99": 0.0,
            "max": 0.0,
        }

    def test_single_sample(self) -> None:
        assert percentiles([5.0]) == {
            "min": 5.0,
            "mean": 5.0,
            "p50": 5.0,
            "p90": 5.0,
            "p95": 5.0,
            "p99": 5.0,
            "max": 5.0,
        }

    def test_linear_interpolation_matches_numpy(self) -> None:
        stats = percentiles([1.0, 2.0, 3.0, 4.0])
        assert stats["min"] == 1.0
        assert stats["mean"] == 2.5
        assert stats["p50"] == 2.5
        assert stats["p90"] == pytest.approx(3.7)
        assert stats["p95"] == pytest.approx(3.85)
        assert stats["p99"] == pytest.approx(3.97)
        assert stats["max"] == 4.0
