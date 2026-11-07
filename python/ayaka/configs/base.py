from __future__ import annotations

import enum
import hashlib
import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, fields, is_dataclass
from pathlib import Path
from typing import Any, cast

from ayaka.utils.validation import require_int


class ConfigError(ValueError):
    """A validation failure with a stable code and a machine-readable path."""

    def __init__(
        self,
        path: str,
        code: str,
        message: str,
        *,
        context: Mapping[str, Any] | None = None,
    ) -> None:
        self.path = path
        self.code = code
        self.message = message
        self.context = dict(context or {})
        super().__init__(f"{path}: {message} [{code}]")


class FrozenMapping(tuple[tuple[str, Any], ...]):
    """An immutable frozen mapping represented as sorted ((key, value), ...) pairs."""


def freeze_mapping(
    value: Mapping[str, Any] | tuple[tuple[str, Any], ...] | None, *, path: str = "extra"
) -> FrozenMapping:
    """Return a deterministic immutable representation of a string-keyed map."""

    if value is None:
        return FrozenMapping()
    items = tuple(value.items()) if isinstance(value, Mapping) else tuple(value)
    keys: list[object] = [key for key, _ in items]
    if any(not isinstance(key, str) or not key for key in keys):
        raise ConfigError(path, "MAP_KEY_INVALID", "mapping keys must be non-empty strings")
    if len(keys) != len(set(keys)):
        raise ConfigError(path, "MAP_KEY_DUPLICATE", "mapping contains duplicate keys")
    return FrozenMapping(sorted(items, key=lambda item: item[0]))


def _canonical(value: Any) -> Any:
    """Return a canonical (stable, deterministic) representation of a Python value."""

    if isinstance(value, enum.Enum):
        return value.value
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value) and not isinstance(value, type):
        return {
            field.name: _canonical(getattr(value, field.name))
            for field in fields(value)
            if field.repr
        }
    if isinstance(value, Mapping):
        mapping = cast("Mapping[Any, Any]", value)
        return {str(key): _canonical(item) for key, item in sorted(mapping.items())}
    if isinstance(value, FrozenMapping):
        return {str(key): _canonical(item) for key, item in value}
    if isinstance(value, tuple):
        elements = cast("tuple[Any, ...]", value)
        if elements and all(
            isinstance(item, tuple)
            and len(cast("tuple[Any, ...]", item)) == 2
            and type(cast("tuple[Any, ...]", item)[0]) is str
            for item in elements
        ):
            pairs = cast("tuple[tuple[str, Any], ...]", elements)
            return {key: _canonical(item) for key, item in pairs}
        return [_canonical(item) for item in elements]
    if isinstance(value, (list, set, frozenset)):
        return [_canonical(item) for item in cast("Iterable[Any]", value)]
    return value


@dataclass(frozen=True, slots=True)
class ConfigMixin:
    """Canonical serialization and fingerprints for immutable config records."""

    def to_dict(self) -> dict[str, Any]:
        payload = _canonical(self)
        if not isinstance(payload, dict):  # pragma: no cover - defensive
            raise TypeError("a dataclass config must serialize to an object")
        return cast("dict[str, Any]", payload)

    def canonical_json(self) -> str:
        return json.dumps(
            self.to_dict(),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )

    @property
    def fingerprint(self) -> str:
        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()


def validate_buckets(buckets: tuple[int, ...]) -> None:
    """Reject coercion, duplicates and reordering of an operator's choices."""
    candidate = cast("object", buckets)
    if not isinstance(candidate, tuple) or not buckets:
        raise ValueError("buckets must be a non-empty tuple")
    for bucket in buckets:
        require_int(bucket, "bucket", minimum=1)
    if tuple(sorted(set(buckets))) != buckets:
        raise ValueError("buckets must be ascending and deduplicated")
