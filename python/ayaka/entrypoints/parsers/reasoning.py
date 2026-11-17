"""Separating the model's reasoning block from its answer, as the text streams.

A reasoning model writes ``<think> ... </think>`` before its answer (or, with templates that open
the block in the prompt, starts inside it and only writes the closing marker). The parser is an
incremental state machine over the decoded text: it emits ``(True, text)`` for reasoning and
``(False, text)`` for answer text, holding back only what could still be part of a marker.

Whitespace at the edges of the block is layout, not content: the whitespace right after the
opening marker, right before the closing one, and right after the closing one is dropped. It is
decided character by character, so where the stream is cut never changes the result.

Formats are chosen by name when the server starts (``--reasoning-parser``), never guessed from the
output.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

from ayaka.core.errors import ConfigError
from ayaka.entrypoints.parsers.markers import Marker, MarkerScanner, resolve_marker

if TYPE_CHECKING:
    from collections.abc import Iterable

    from ayaka.tokenizers.ports import TokenizerLike

__all__ = ["REASONING_FORMATS", "ReasoningFormat", "ReasoningParser", "reasoning_format"]


@dataclass(frozen=True, slots=True)
class ReasoningFormat:
    """How a model family marks its reasoning.

    Attributes:
        name: The name the server option uses.
        open: The marker that opens the block.
        close: The marker that closes it.
        starts_inside: Whether the chat template already opened the block in the prompt, so the
            output starts inside it.
    """

    name: str
    open: str
    close: str
    starts_inside: bool = False

    def markers(self, tokenizer: TokenizerLike | None = None) -> tuple[Marker, Marker]:
        """The opening and closing markers as ``tokenizer`` spells them (by text without one).

        Raises:
            ConfigError: If a marker is a special token the detokenizer drops.
        """
        return resolve_marker(self.open, tokenizer), resolve_marker(self.close, tokenizer)


REASONING_FORMATS: Final[dict[str, ReasoningFormat]] = {
    # Qwen3, QwQ, SmolLM3 and others: the model writes both markers.
    "think": ReasoningFormat("think", "<think>", "</think>"),
    # DeepSeek-R1 and its distillations: the template ends the prompt with "<think>\n".
    "deepseek_r1": ReasoningFormat("deepseek_r1", "<think>", "</think>", starts_inside=True),
}


def reasoning_format(name: str) -> ReasoningFormat:
    """The format called ``name``.

    Raises:
        ConfigError: If there is no such format.
    """
    try:
        return REASONING_FORMATS[name]
    except KeyError:
        raise ConfigError(
            "server.reasoning_parser",
            "INVALID_CHOICE",
            f"unknown reasoning parser {name!r}; known: {sorted(REASONING_FORMATS)}",
        ) from None


class ReasoningParser:
    """Splits one request's output into reasoning and answer text.

    Args:
        fmt: The model's reasoning format.
        markers: Its markers from :meth:`ReasoningFormat.markers`, resolved once per server;
            ``None`` matches them by text.
    """

    def __init__(self, fmt: ReasoningFormat, markers: tuple[Marker, Marker] | None = None) -> None:
        self._open, self._close = markers if markers is not None else fmt.markers()
        self._scanner = MarkerScanner()
        self._inside = fmt.starts_inside
        self._strip_leading = fmt.starts_inside
        self._trailing = ""  # whitespace at the end of the reasoning seen so far

    @property
    def inside(self) -> bool:
        """Whether the output is inside the reasoning block now."""
        return self._inside

    def note_tokens(self, token_ids: Iterable[int]) -> None:
        """Record the token ids of the next piece of text, before feeding the text."""
        self._scanner.note_tokens(token_ids)

    def feed(self, text: str) -> list[tuple[bool, str]]:
        """Take the next piece of text; return ``(is_reasoning, text)`` pieces."""
        pieces: list[tuple[bool, str]] = []
        while True:
            active = (self._close,) if self._inside else (self._open,)
            before, marker = self._scanner.scan(text, active)
            text = ""
            self._emit(before, pieces)
            if marker is None:
                return pieces
            if self._inside:
                self._trailing = ""  # whitespace before the closing marker is layout
            self._inside = not self._inside
            self._strip_leading = True

    def finish(self) -> list[tuple[bool, str]]:
        """Release everything held back at the end of the output."""
        pieces: list[tuple[bool, str]] = []
        self._emit(self._scanner.flush(), pieces)
        if self._inside and self._trailing:
            pieces.append((True, self._trailing))
            self._trailing = ""
        return pieces

    def _emit(self, text: str, pieces: list[tuple[bool, str]]) -> None:
        if self._strip_leading:
            stripped = text.lstrip()
            if stripped:
                self._strip_leading = False
            text = stripped
        if not text:
            return
        if not self._inside:
            pieces.append((False, text))
            return
        text = self._trailing + text
        body = text.rstrip()
        self._trailing = text[len(body) :]
        if body:
            pieces.append((True, body))
