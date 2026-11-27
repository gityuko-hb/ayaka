"""Shared protocol and whitespace buffering for incremental tool-call parsers."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from collections.abc import Iterable

    from ayaka.entrypoints.conversation import OutputEvent

__all__ = ["RunBuffer", "ToolCallParserLike"]


class ToolCallParserLike(Protocol):
    """Common incremental interface implemented by JSON and XML/tag tool-call parsers."""

    @property
    def num_calls(self) -> int:
        """Calls reported so far."""
        ...

    @property
    def opened_calls(self) -> int:
        """Call bodies entered so far, whether or not a valid name completed."""
        ...

    @property
    def in_call(self) -> bool:
        """Whether the parser is currently inside an open tool-call body."""
        ...

    def note_tokens(self, token_ids: Iterable[int]) -> None:
        """Record token ids belonging to the next decoded text piece."""
        ...

    def feed(self, text: str) -> list[OutputEvent]:
        """Consume the next answer text piece and return completed events."""
        ...

    def finish(self) -> list[OutputEvent]:
        """Flush buffered state at stream end and close any incomplete call."""
        ...


class RunBuffer:
    """Text gathered a piece at a time and joined once on read."""

    __slots__ = ("_parts",)

    def __init__(self) -> None:
        self._parts: list[str] = []

    def add(self, text: str) -> None:
        """Append ``text`` when non-empty."""
        if text:
            self._parts.append(text)

    def take(self) -> str:
        """Return buffered text and reset buffer."""
        text = "".join(self._parts)
        self._parts.clear()
        return text
