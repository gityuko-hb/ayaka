"""Small numeric helpers shared by planning and benchmark code.

This module is deliberately pure Python/NumPy: importing it must not import
torch. Tensor and kernel argument checks live in
:mod:`ayaka.utils.torch_utils`, and dtype limits live in
:mod:`ayaka.utils.dtypes`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Sequence

__all__ = [
    "align_down",
    "align_up",
    "div_ceil",
    "next_power_of_2",
    "percentiles",
]


def div_ceil(a: int, b: int) -> int:
    """Return ``a / b`` rounded toward positive infinity.

    Args:
        a: Dividend. May be negative; the double-negation form keeps the
            result correct for every sign combination.
        b: Divisor. Must be a positive integer.

    Returns:
        The smallest integer ``q`` with ``q * b >= a``.

    Raises:
        TypeError: if either operand is not an integer.
        ValueError: if ``b`` is not positive.
    """
    if type(a) is not int:
        raise TypeError("value must be an integer")
    if type(b) is not int:
        raise TypeError("divisor must be an integer")
    if b <= 0:
        raise ValueError("divisor must be positive")
    return -(-a // b)


def next_power_of_2(num: int) -> int:
    """Return the smallest power of two that is at least ``num``.

    ``0`` and ``1`` are returned unchanged, matching
    ``triton.next_power_of_2`` so launch-geometry helpers can be shared
    between host and kernel code.

    Raises:
        TypeError: if ``num`` is not an integer.
        ValueError: if ``num`` is negative.
    """
    if type(num) is not int:
        raise TypeError("num must be an integer")
    if num < 0:
        raise ValueError("num must not be negative")
    if num <= 1:
        return num
    return 1 << (num - 1).bit_length()


def align_up(value: int, alignment: int) -> int:
    """Round ``value`` up to the next multiple of ``alignment``.

    Kept general (``alignment`` need not be a power of two) because it only
    runs in planning code, never per step. The bitmask form
    ``(v + a - 1) & ~(a - 1)`` would be faster but would silently produce
    garbage for a non-power-of-two alignment.

    Raises:
        TypeError: if either argument is not an integer.
        ValueError: for a non-positive alignment or a negative value.
    """
    if type(value) is not int:
        raise TypeError("value must be an integer")
    if type(alignment) is not int:
        raise TypeError("alignment must be an integer")
    if alignment <= 0:
        raise ValueError("alignment must be positive")
    if value < 0:
        raise ValueError("value must not be negative")
    return ((value + alignment - 1) // alignment) * alignment


def align_down(value: int, alignment: int) -> int:
    """Round ``value`` down to the previous multiple of ``alignment``.

    Counterpart of :func:`align_up`; ``0`` stays ``0`` for every valid
    alignment.

    Raises:
        TypeError: if either argument is not an integer.
        ValueError: for a non-positive alignment or a negative value.
    """
    if type(value) is not int:
        raise TypeError("value must be an integer")
    if type(alignment) is not int:
        raise TypeError("alignment must be an integer")
    if alignment <= 0:
        raise ValueError("alignment must be positive")
    if value < 0:
        raise ValueError("value must not be negative")
    return value - value % alignment


def percentiles(values: Sequence[float]) -> dict[str, float]:
    """Return min/mean/p50/p90/p95/p99/max for ``values``.

    Uses NumPy's linear-interpolation percentiles. The cheap nearest-rank
    helper used on the request settlement path lives in
    :meth:`ayaka.utils.timing.RequestTiming.inter_token_percentiles_ns`; the
    two estimators intentionally answer different questions and are documented
    where they are used.

    An empty sequence yields an all-zero mapping so benchmark reports do not
    need a special case.
    """
    if not values:
        return {"min": 0.0, "mean": 0.0, "p50": 0.0, "p90": 0.0, "p95": 0.0, "p99": 0.0, "max": 0.0}
    arr = np.array(values, dtype=np.float64)
    return {
        "min": float(np.min(arr)),
        "mean": float(np.mean(arr)),
        "p50": float(np.percentile(arr, 50)),
        "p90": float(np.percentile(arr, 90)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
        "max": float(np.max(arr)),
    }
