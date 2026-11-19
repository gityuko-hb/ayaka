"""Separating the model's reasoning block from its answer, as the text streams.

A reasoning model writes ``<think> ... </think>`` before its answer (or, with templates that open
the block in the prompt, starts inside it and only writes the closing marker). The parser is an
incremental state machine over the decoded text: it emits ``(True, text)`` for reasoning and
``(False, text)`` for answer text, holding back only what could still be part of a marker.

Whitespace at the edges of the block is layout, not content: the whitespace right after the
opening marker, right before the closing one, and right after the closing one is dropped. It is
decided character by character, so where the stream is cut never changes the result.

Formats (:mod:`ayaka.entrypoints.parsers.formats`) are chosen by name when the server starts
(``--reasoning-parser``), never guessed from the output. With a format the model opens itself
(``think``), only the first thing it writes may open the block, as the output constraint assumes:
an opening marker after answer text is answer text.

When both markers are one token each, the parser also says which token ids of a fragment belong to
the answer (:attr:`ReasoningParser.answer_tokens`), by following the same states over the ids. The
tool-call parser gets only those, so a marker token the model wrote inside its reasoning is never
credited to the answer.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ayaka.entrypoints.parsers.markers import MarkerScanner

if TYPE_CHECKING:
    from collections.abc import Iterable

    from ayaka.entrypoints.parsers.formats import ReasoningFormat
    from ayaka.entrypoints.parsers.markers import Marker

__all__ = ["ReasoningParser"]


class ReasoningParser:
    """Splits one request's output into reasoning and answer text.

    Args:
        fmt: The model's reasoning format.
        markers: Its markers from :meth:`ReasoningFormat.markers`, resolved once per server;
            ``None`` matches them by text.
    """

    def __init__(self, fmt: ReasoningFormat, markers: tuple[Marker, Marker] | None = None) -> None:
        self._open, self._close = markers if markers is not None else fmt.markers()
        self._scanner = MarkerScanner((self._open, self._close))
        self._inside = fmt.starts_inside
        # The model may open the block only with the first thing it writes.
        self._may_open = not fmt.starts_inside
        self._strip_leading = fmt.starts_inside
        self._trailing: list[str] = []  # whitespace at the end of the reasoning seen so far
        self._answer_tokens: tuple[int, ...] = ()

    @property
    def inside(self) -> bool:
        """Whether the output is inside the reasoning block now."""
        return self._inside

    @property
    def answer_tokens(self) -> tuple[int, ...]:
        """The ids of the last noted fragment that are answer tokens, markers excluded.

        With markers matched by text the ids cannot be told apart, and all of them are returned.
        """
        return self._answer_tokens

    def note_tokens(self, token_ids: Iterable[int]) -> None:
        """Record the token ids of the next piece of text, before feeding the text."""
        ids = tuple(token_ids)
        self._scanner.note_tokens(ids)
        self._answer_tokens = self._split_answer(ids)

    def _split_answer(self, ids: tuple[int, ...]) -> tuple[int, ...]:
        opening, closing = self._open.token_id, self._close.token_id
        if opening is None or closing is None:
            return ids
        inside, may_open = self._inside, self._may_open
        answer: list[int] = []
        for token in ids:
            if inside:
                if token == closing:
                    inside, may_open = False, False
            elif may_open and token == opening:
                inside, may_open = True, False
            else:
                may_open = False
                answer.append(token)
        return tuple(answer)

    def feed(self, text: str) -> list[tuple[bool, str]]:
        """Take the next piece of text; return ``(is_reasoning, text)`` pieces."""
        pieces: list[tuple[bool, str]] = []
        while True:
            if self._inside:
                active: tuple[Marker, ...] = (self._close,)
            else:
                active = (self._open,) if self._may_open else ()
            before, marker = self._scanner.scan(text, active)
            text = ""
            self._emit(before, pieces)
            if marker is None:
                return pieces
            if self._inside:
                self._trailing.clear()  # whitespace before the closing marker is layout
            self._may_open = False
            self._inside = not self._inside
            self._strip_leading = True

    def finish(self) -> list[tuple[bool, str]]:
        """Release everything held back at the end of the output."""
        pieces: list[tuple[bool, str]] = []
        self._emit(self._scanner.flush(), pieces)
        if self._inside and self._trailing:
            pieces.append((True, "".join(self._trailing)))
        self._trailing.clear()
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
            self._may_open = False  # answer text came first: no block opens after it
            pieces.append((False, text))
            return
        body = text.rstrip()
        if not body:
            self._trailing.append(text)  # a run of whitespace costs its length, not its square
            return
        pieces.append((True, "".join(self._trailing) + body))
        self._trailing.clear()
        if len(body) < len(text):
            self._trailing.append(text[len(body) :])
