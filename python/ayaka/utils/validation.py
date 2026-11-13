"""Host-value validation helpers shared by ``core`` and ``utils``.

This module imports only the standard library, so the lower-level
:mod:`ayaka.core` DTOs can use it without pulling in torch, NVML or any other
optional dependency. It is the single implementation of the basic host-value
checks.

The helpers deliberately preserve the project's established exception types:
bad primitive types raise ``TypeError``, bad values raise ``ValueError``. Core
DTOs raise :class:`ayaka.core.errors.InvalidRequestError` (a ``ValueError``) for
semantic request failures such as conflicting limits.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import fields, is_dataclass
from enum import Enum
from typing import cast

__all__ = [
    "require_bool",
    "require_float",
    "require_frozen",
    "require_int",
    "require_nonempty_text",
    "require_number",
    "require_text",
    "require_token_set",
    "require_token_tuple",
]


def require_bool(value: object, name: str) -> bool:
    """Reject truthy substitutes at public boundaries."""

    if type(value) is not bool:
        raise TypeError(f"{name} must be bool")
    return value


def require_int(value: object, name: str, *, minimum: int | None = 0) -> int:
    """Reject booleans, non-integers and values below the inclusive minimum."""

    if type(value) is not int:
        raise TypeError(f"{name} must be an integer")
    result = value
    if minimum is not None and result < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return result


def require_float(value: object, name: str) -> float:
    """Require a finite floating-point value."""

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def require_number(value: object, name: str, *, minimum: float | None = None) -> float:
    """Require a number that is not NaN and, when given, above a minimum."""

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a number")
    result = float(value)
    if math.isnan(result):
        raise ValueError(f"{name} must not be NaN")
    if minimum is not None and result < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return result


def require_text(value: object, name: str) -> str:
    """Require a non-empty, non-blank string without normalizing its value."""

    candidate = value
    if not isinstance(candidate, str) or not candidate.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return candidate


def require_nonempty_text(value: object, name: str) -> str:
    """Require a non-empty string, allowing whitespace-only stop strings."""

    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string")
    return value


def require_frozen(value: object, name: str) -> None:
    """Reject mutable objects hidden inside a host contract.

    Scalars, enums, tuples and recursively frozen dataclasses are accepted.
    Lists, mappings, tensors and live resource owners are refused. Device
    metadata belongs to a runtime binding, never to a host value.
    """

    if value is None or isinstance(value, (str, int, float, bool, Enum)):
        return
    if isinstance(value, tuple):
        for index, item in enumerate(cast("tuple[object, ...]", value)):
            require_frozen(item, f"{name}[{index}]")
        return
    if (
        is_dataclass(value) and not isinstance(value, type) and value.__dataclass_params__.frozen  # type: ignore[attr-defined]  # not in typeshed's DataclassInstance
    ):
        for item in fields(value):
            require_frozen(getattr(value, item.name), f"{name}.{item.name}")
        return
    raise TypeError(f"{name} must contain only immutable host values")


def require_token_tuple(value: object, name: str) -> tuple[int, ...]:
    """Require an immutable tuple of non-negative token ids."""

    if not isinstance(value, tuple):
        raise ValueError(f"{name} must be a tuple of token ids")
    items = cast("tuple[object, ...]", value)
    for item in items:
        require_int(item, name, minimum=0)
    return cast("tuple[int, ...]", items)


def require_token_set(value: object, name: str, *, allow_empty: bool) -> frozenset[int]:
    """Require an iterable of non-negative token ids and canonicalize to a set."""

    if isinstance(value, (str, bytes)) or not isinstance(value, Iterable):
        raise ValueError(f"{name} must be an iterable of token ids")
    items = tuple(cast("Iterable[object]", value))
    result = frozenset(require_int(item, name, minimum=0) for item in items)
    if not allow_empty and not result:
        raise ValueError(f"{name} must not be empty")
    return result
