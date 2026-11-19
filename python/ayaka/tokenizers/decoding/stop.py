"""Stop-string matching for incremental detokenization.

:class:`StopChecker` scans a character stream for any of a set of stop strings with ``str.find``
and decides how much text is safe to forward. It sits between the decoder and the client::

    decoder text --> StopChecker.feed --> (emit, matched) --> client
                          |
                          +-- holds back the longest tail that is the start of a stop string

The result does not depend on how the text is split into deltas: with the stops ``("abc", "b")``,
feeding ``"abc"`` once and feeding ``"ab"`` then ``"c"`` stop at the same place with the same
string. Two rules make that true:

* **The match that ends first wins.** A match is reported as soon as its last character arrives,
  so a match that ends later can never have been seen first. Among stops that end at the same
  character, the longest one (the one that starts first) is reported.
* **Only a possible start is held back.** The checker withholds the longest tail of the text that
  is a proper prefix of some stop string, and nothing else: a character that cannot start a stop
  is released at once. Released text can therefore never be the start of a later match.

Each feed searches only where a new match can be (from ``len(stop) - 1`` characters before the new
text), so a long answer costs the length of the answer, not its square.

Typical call sequence::

    checker = StopChecker(stop=("</s>", "<|eot_id|>"), include_stop_str_in_output=False)
    for delta in token_stream:
        emit, matched = checker.feed(delta)
        if emit:
            forward_to_client(emit)
        if matched:
            break
    tail = checker.flush()          # release any held characters
    if tail:
        forward_to_client(tail)
"""

from __future__ import annotations


class StopChecker:
    """Stateful stop-string detector for streaming text output.

    Attributes:
        _stops: Non-empty stop strings in the order supplied.
        _prefixes: Every proper prefix of every stop string: what a held tail may be.
        _longest: Length of the longest stop string.
        _include: When ``True``, the matched stop string is forwarded to the client as part of
            the final :meth:`feed` result.
        _buf: Characters received and not emitted: the held tail, then the new delta.
        _done: ``True`` once a stop has been matched or :meth:`flush` has been called;
            subsequent :meth:`feed` calls are no-ops.
    """

    __slots__ = ("_buf", "_done", "_include", "_longest", "_prefixes", "_stops")

    def __init__(self, stop: tuple[str, ...], include_stop_str_in_output: bool) -> None:
        """Initialise the checker.

        Args:
            stop: Candidate stop strings. Empty strings are dropped: they would match at once
                on every input and stall the stream.
            include_stop_str_in_output: When ``True``, the matched stop string is forwarded to
                the client as part of the final delta; when ``False`` it is consumed silently.
        """
        self._stops = tuple(s for s in stop if s)
        self._prefixes = frozenset(s[:k] for s in self._stops for k in range(1, len(s)))
        self._longest = max((len(s) for s in self._stops), default=0)
        self._include = include_stop_str_in_output
        self._buf = ""
        self._done = False

    @property
    def has_stops(self) -> bool:
        """``True`` when at least one non-empty stop string was provided."""
        return bool(self._stops)

    @property
    def pending(self) -> str:
        """Characters held back, not yet emitted: the possible start of a stop string."""
        return self._buf

    def feed(self, delta: str) -> tuple[str, str | None]:
        """Feed a new text fragment and return what can be safely forwarded.

        The return value is a ``(emit, matched)`` pair:

        * ``emit`` - text that is safe to forward to the client now. It may be empty when the
          new text could still be the start of a stop string.
        * ``matched`` - the stop string that was detected, or ``None`` if generation should
          continue.

        Once ``matched`` is not ``None`` the checker is done and all subsequent calls return
        ``("", None)``.

        Args:
            delta: New text from the detokenizer. An empty string is a no-op.

        Returns:
            ``(emit, matched_stop_string_or_None)``
        """
        if self._done:
            return "", None  # already stopped - swallow all further input
        if not self._stops:
            return delta, None  # no stops configured - pass through immediately
        if not delta:
            return "", None  # empty delta is a no-op

        scanned = len(self._buf)  # the held tail holds no whole stop string
        buf = self._buf + delta
        best_end = -1
        best_stop: str | None = None
        for stop in self._stops:
            # A new match ends inside the new text, so it starts after scanned - len(stop).
            found = buf.find(stop, max(0, scanned - len(stop) + 1))
            if found == -1:
                continue
            end = found + len(stop)
            if (
                best_stop is None
                or end < best_end
                or (end == best_end and len(stop) > len(best_stop))
            ):
                best_end, best_stop = end, stop

        if best_stop is not None:
            cut = best_end if self._include else best_end - len(best_stop)
            self._buf = ""  # discard everything after the stop
            self._done = True
            return buf[:cut], best_stop

        held = self._held(buf)
        self._buf = buf[len(buf) - held :]
        return buf[: len(buf) - held], None

    def _held(self, buf: str) -> int:
        """Length of the longest tail of ``buf`` that is a proper prefix of a stop string."""
        prefixes = self._prefixes
        for size in range(min(len(buf), self._longest - 1), 0, -1):
            if buf[-size:] in prefixes:
                return size
        return 0

    def flush(self) -> str:
        """Drain the held characters and mark the checker as done.

        Called at end-of-sequence (EOS or max-token limit) to release any characters that were
        withheld pending a potential stop-string match. After ``flush``, :meth:`feed` will
        always return ``("", None)``.

        Returns:
            The held text, or an empty string when the checker was already done (stop matched
            or previously flushed).
        """
        if self._done:
            return ""
        out, self._buf = self._buf, ""
        self._done = True
        return out
