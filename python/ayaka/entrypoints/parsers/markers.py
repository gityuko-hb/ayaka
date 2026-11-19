"""Finding marker strings (``<think>``, ``<tool_call>``...) in text that arrives in pieces.

A marker can be split across two pieces of streamed text, so the scanner holds back the end of a
piece while it could still be the start of a marker, and decides once the next piece arrives.
Nothing is ever rescanned: text before a possible marker is released as soon as it is known.

A marker that is one token of the vocabulary is matched **by token id**: its text counts only when
the model emitted that token. The same characters spelled with ordinary tokens (a model writing
about ``<tool_call>``) stay plain text. A marker that is not one token is matched by its text.

An emitted marker token is a *credit*, and a credit belongs to the fragment that carried it: it
can be spent on that fragment's text only, and :meth:`MarkerScanner.scan` drops what is left once
the text is scanned. A fragment's text arrives whole (a one-token marker is never split over two
fragments), so no real marker is lost. Without the rule, a real ``<tool_call>`` token that no parser
looked for (inside reasoning, say) would leave a credit behind, and the same string spelled with
ordinary tokens much later in the answer would match as a real marker. A credit is also spent when
its marker's text passes as plain text (a real ``<think>`` token inside the block, where only
``</think>`` counts), so it cannot be saved for a spelled ``<think>`` later in the same fragment.

Within one fragment the scanner knows which ids it got but not where each one sits in the text: a
fragment that holds a real marker token and the same characters spelled out is matched by the count
of real tokens, first occurrence first. Telling them apart needs the text of each token, which
fragments do not carry.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import TYPE_CHECKING

from ayaka.core.errors import ConfigError

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from ayaka.tokenizers.ports import TokenizerLike

__all__ = ["Marker", "MarkerScanner", "resolve_marker"]


@dataclass(frozen=True, slots=True)
class Marker:
    """A marker string and, when it is one token, that token's id."""

    text: str
    token_id: int | None = None

    def __post_init__(self) -> None:
        if not self.text:
            raise ValueError("a marker needs text")


def resolve_marker(text: str, tokenizer: TokenizerLike | None) -> Marker:
    """The marker ``text`` as the loaded tokenizer spells it.

    Raises:
        ConfigError: If the marker is a special token, which the detokenizer drops from the
            output text, so it could never be seen.
    """
    if tokenizer is None:
        return Marker(text)
    ids = tokenizer.encode(text, add_special_tokens=False)
    if len(ids) != 1:
        return Marker(text)
    [token_id] = ids
    if token_id in tokenizer.all_special_ids:
        raise ConfigError(
            "parser.marker",
            "MARKER_IS_SPECIAL",
            f"{text!r} is a special token of this tokenizer (id {token_id}); it is removed from "
            "the decoded text, so this output format cannot be parsed with it",
        )
    return Marker(text, token_id)


class MarkerScanner:
    """Splits streamed text at markers.

    Feed it token ids with :meth:`note_tokens` and text with :meth:`scan`. Which markers count is
    chosen per call, so a parser can look for different markers in different states.

    Args:
        known: Every marker the parser may look for. A credit of one of them that passes as plain
            text is spent there, first occurrence first.
    """

    def __init__(self, known: Sequence[Marker] = ()) -> None:
        self._held = ""
        self._credits: Counter[int] = Counter()
        self._known = tuple(marker for marker in known if marker.token_id is not None)

    def note_tokens(self, token_ids: Iterable[int]) -> None:
        """Record the ids of the next fragment's text; a marker matched by id needs its id here.

        The credits last until that text has been scanned: :meth:`scan` drops what is left of them
        when it returns no marker.
        """
        self._credits.update(token_ids)

    @property
    def held(self) -> str:
        """Text held back because it may be the start of a marker."""
        return self._held

    def scan(self, text: str, active: Sequence[Marker]) -> tuple[str, Marker | None]:
        """Take ``text`` and return the plain text before the first marker, and that marker.

        When a marker is found, the text after it stays in the scanner: call ``scan("", ...)``
        again (with the markers of the new state) until it returns no marker. When none is found,
        the end that could still become a marker is held back.
        """
        buffer = self._held + text
        start = 0
        while True:
            found = self._first(buffer, start, active)
            if found is None:
                break
            position, marker = found
            if marker.token_id is None or self._credits[marker.token_id] > 0:
                if marker.token_id is not None:
                    self._credits[marker.token_id] -= 1
                self._held = buffer[position + len(marker.text) :]
                self._spend(buffer[:position])
                return buffer[:position], marker
            # The model wrote the marker's characters with ordinary tokens: plain text.
            start = position + 1
        keep = _possible_start(buffer, active, start)
        self._held = buffer[keep:]
        self._spend(buffer[:keep])
        # The fragment's text is scanned: what it held back cannot hold a whole one-token marker
        # (a held tail is a proper prefix of one), so its credits end here.
        self._credits.clear()
        return buffer[:keep], None

    def _spend(self, plain: str) -> None:
        """Spend the credits of known markers whose text passes as plain text."""
        if not plain:
            return
        left = self._credits
        for marker in self._known:
            token = marker.token_id
            assert token is not None  # noqa: S101  # only markers with an id are known
            if left[token] > 0:
                left[token] -= min(left[token], plain.count(marker.text))

    def flush(self) -> str:
        """Release whatever is held back, as plain text."""
        held, self._held = self._held, ""
        return held

    @staticmethod
    def _first(buffer: str, start: int, active: Sequence[Marker]) -> tuple[int, Marker] | None:
        best: tuple[int, Marker] | None = None
        for marker in active:
            position = buffer.find(marker.text, start)
            if position >= 0 and (best is None or position < best[0]):
                best = (position, marker)
        return best


def _possible_start(buffer: str, active: Sequence[Marker], start: int) -> int:
    """Index of the earliest suffix of ``buffer`` that is a proper prefix of an active marker."""
    keep = len(buffer)
    for marker in active:
        longest = min(len(marker.text) - 1, len(buffer) - start)
        for size in range(longest, 0, -1):
            if marker.text.startswith(buffer[len(buffer) - size :]):
                keep = min(keep, len(buffer) - size)
                break
    return keep
