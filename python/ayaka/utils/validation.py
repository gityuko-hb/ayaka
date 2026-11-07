from __future__ import annotations

from dataclasses import fields, is_dataclass
from enum import Enum
from typing import cast

__all__ = ["require_bool", "require_frozen", "require_int", "require_text"]


def require_bool(value: bool, name: str) -> None:
    """Reject truthy substitutes at public boundaries."""
    if type(value) is not bool:
        raise TypeError(f"{name} must be bool")


def require_int(value: int, name: str, *, minimum: int = 0) -> None:
    """Reject booleans, non-integers and values below the inclusive minimum."""
    if type(value) is not int:
        raise TypeError(f"{name} must be an integer")
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}")


def require_text(value: str, name: str) -> None:
    """Require a non-empty string identity without normalizing its value."""
    candidate = cast("object", value)
    if not isinstance(candidate, str) or not candidate.strip():
        raise ValueError(f"{name} must be a non-empty string")


def require_frozen(value: object, name: str) -> None:
    """Reject mutable objects hidden inside a host contract.

    Scalars, enums, tuples and recursively frozen dataclasses are accepted.
    Lists, mappings, tensors and live resource owners are refused. Device
    metadata belongs to a runtime binding, never to this host value.
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
