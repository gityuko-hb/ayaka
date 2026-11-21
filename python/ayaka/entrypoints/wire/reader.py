"""Reading a request body field by field, without the HTTP server.

:class:`JsonObject` wraps one JSON object of a request body and reads each field with its type
checked, naming the field's path in every error (``messages[2].content``). A reader collects
the sampling fields in :class:`SamplingFields`, which becomes one ``SamplingParams`` once.

Nothing here imports aiohttp, so the protocol readers that use it can be imported and tested
without the ``serve`` extra; the server reads the body (``ayaka.entrypoints.http.read_body``) and
hands it here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, cast

from ayaka.core.errors import InvalidRequestError
from ayaka.core.params import SamplingParams

if TYPE_CHECKING:
    from collections.abc import Mapping

__all__ = ["JsonObject", "SamplingFields", "sampling_params"]


class JsonObject:
    """A JSON object from a request body, read field by field with the field's path in errors.

    Args:
        data: The object.
        path: Where it is in the body, for messages (``messages[2]``); empty at the top.
    """

    def __init__(self, data: dict[str, object], path: str = "") -> None:
        self.data = data
        self.path = path

    @classmethod
    def of(cls, value: object, path: str) -> JsonObject:
        """``value`` as an object, or an error naming ``path``."""
        if not isinstance(value, dict):
            raise _invalid(path, "must be an object")
        return cls(cast("dict[str, object]", value), path)

    def where(self, key: str) -> str:
        """The path of ``key`` in the body."""
        return f"{self.path}.{key}" if self.path else key

    def has(self, key: str) -> bool:
        """Whether the field is present and not null."""
        return self.data.get(key) is not None

    def raw(self, key: str) -> object:
        """The field as it is, ``None`` when absent."""
        return self.data.get(key)

    def string(self, key: str, *, required: bool = False) -> str | None:
        """An optional string field."""
        value = self._get(key, required)
        if value is not None and not isinstance(value, str):
            raise _invalid(self.where(key), "must be a string")
        return value

    def text(self, key: str) -> str:
        """A required string field."""
        value = self.string(key, required=True)
        assert value is not None  # noqa: S101  # required
        return value

    def integer(self, key: str, *, required: bool = False) -> int | None:
        """An optional integer field."""
        value = self._get(key, required)
        if value is not None and (not isinstance(value, int) or isinstance(value, bool)):
            raise _invalid(self.where(key), "must be an integer")
        return value

    def number(self, key: str) -> float | None:
        """An optional number field."""
        value = self.data.get(key)
        if value is None:
            return None
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise _invalid(self.where(key), "must be a number")
        return float(value)

    def flag(self, key: str, default: bool = False) -> bool:
        """An optional boolean field."""
        value = self.data.get(key)
        if value is None:
            return default
        if not isinstance(value, bool):
            raise _invalid(self.where(key), "must be true or false")
        return value

    def child(self, key: str) -> JsonObject | None:
        """An optional object field."""
        value = self.data.get(key)
        return None if value is None else JsonObject.of(value, self.where(key))

    def array(self, key: str, *, required: bool = False) -> list[object] | None:
        """An optional array field."""
        value = self._get(key, required)
        if value is not None and not isinstance(value, list):
            raise _invalid(self.where(key), "must be an array")
        return cast("list[object] | None", value)

    def objects(self, key: str) -> list[JsonObject]:
        """An optional array of objects; empty when absent."""
        items = self.array(key) or []
        return [JsonObject.of(item, f"{self.where(key)}[{i}]") for i, item in enumerate(items)]

    def unknown(self, known: frozenset[str]) -> list[str]:
        """Fields this server does not know, for a debug log line."""
        return sorted(set(self.data) - known)

    def _get(self, key: str, required: bool) -> object:
        value = self.data.get(key)
        if value is None and required:
            raise _invalid(self.where(key), "is required")
        return value


def _invalid(path: str, message: str) -> InvalidRequestError:
    return InvalidRequestError(f"{path}: {message}", context={"param": path})


@dataclass(frozen=True, slots=True)
class SamplingFields:
    """The sampling fields a request set, type-checked, before they become ``SamplingParams``.

    A reader collects them; the server builds the ``SamplingParams`` once, with :meth:`build`,
    after it knows how much room the prompt leaves for ``max_tokens``.

    Attributes:
        values: Field name to value; ``None`` keeps the default.
    """

    values: Mapping[str, object] = field(default_factory=dict[str, object])

    def build(self, max_tokens: int | None) -> SamplingParams:
        """The ``SamplingParams`` of these fields with ``max_tokens``.

        Raises:
            InvalidRequestError: If a value is out of range.
        """
        return sampling_params(**self.values, max_tokens=max_tokens)


def sampling_params(**fields: object) -> SamplingParams:
    """``SamplingParams`` from request fields; a bad value is an invalid request.

    Fields that are ``None`` keep the defaults.
    """
    values = {key: value for key, value in fields.items() if value is not None}
    try:
        return SamplingParams(**values)  # type: ignore[arg-type]  # SamplingParams checks each field
    except (TypeError, ValueError) as exc:
        if isinstance(exc, InvalidRequestError):
            raise
        raise InvalidRequestError(str(exc)) from exc
