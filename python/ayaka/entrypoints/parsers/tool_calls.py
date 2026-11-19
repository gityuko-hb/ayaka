"""Finding function calls in the answer text, as it streams.

The model writes a call as a JSON object between two markers, the layout Hermes, Qwen2.5, Qwen3
and many small models were trained on::

    <tool_call>
    {"name": "get_weather", "arguments": {"city": "Paris"}}
    </tool_call>

The parser is an incremental state machine over the answer text (after the reasoning parser took
the reasoning out). It reports a call as soon as its name is known, then the raw text of its
``arguments`` value piece by piece as the model writes it, then the end of the call. Arguments are
passed on exactly as written, never re-serialized: a client concatenating the pieces gets the
model's JSON.

Nothing is repaired. At the closing marker the whole body is parsed once: a body that is one JSON
object with a string ``name`` and an object ``arguments`` is a valid call; anything else is not,
and the end event says so. A body where no name could be found at all is not a call: it comes out
as plain text, markers included, so the client sees what the model wrote.

Whitespace around calls is layout: whitespace right before a call, between calls, or after the last
call at the end of the answer is dropped. Text that follows a call keeps it.

Llama 3.x writes a call with no markers at all, and calls the arguments ``parameters``::

    {"name": "get_weather", "parameters": {"city": "Paris"}}

That format (``llama3_json``) has nothing to match, so a call is an answer that starts, after
optional whitespace, with a ``{``. The braces are held back until a string ``name`` is read: from
then on the call is reported exactly as above, and the arguments are the raw text of
``parameters``. An answer that starts with ``{`` and turns out not to be a call (the object ends or
breaks before any ``name``) comes out as the plain text it was, whitespace included. An answer that
starts with anything else is text from its first character. The call ends at its closing brace.

Measured on Llama-3.2-1B-Instruct (see ``tests/entrypoints/parsers/fixtures/``): several calls are
objects separated by ``;`` (``{...}; {...}``), so a ``;`` after a call, with a ``{`` next, starts
the next call. Anything else after a call is text, a second object with no ``;`` included. And
when the prompt shows the tools wrapped as ``{"type": "function", "function": {...}}``, which is
how Ayaka renders them, the model copies that shape for the name::

    {"type": "function", "function": "get_weather", "parameters": {"city": "Paris"}}

so a string ``function`` next to ``"type": "function"`` counts as the name (``typed_name``).
"""

from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING, Final, Literal, cast

from ayaka.entrypoints.conversation import (
    TextDelta,
    ToolCallDelta,
    ToolCallEnd,
    ToolCallStart,
)
from ayaka.entrypoints.parsers.markers import MarkerScanner
from ayaka.utils.parsing import parse_json_strict

if TYPE_CHECKING:
    from collections.abc import Iterable

    from ayaka.entrypoints.conversation import OutputEvent
    from ayaka.entrypoints.parsers.formats import ToolCallFormat
    from ayaka.entrypoints.parsers.markers import Marker

__all__ = ["ToolCallParser"]


type _Phase = Literal[
    "start", "key", "in_key", "colon", "value", "in_value", "after_value", "done", "broken"
]
_WHITESPACE: Final = " \t\r\n"
#: Inside a string value: the next character that can end it or escape one.
_STRING_STOP: Final = re.compile(r'["\\]')
#: Inside an object or array value, outside strings: the next character that changes the nesting.
_CONTAINER_STOP: Final = re.compile(r'["\\{}\[\]]')


class _MemberScanner:
    """Follows the members of one JSON object, a piece at a time, in time linear in its length.

    It records where each top-level member's value lies (:attr:`spans`) and keeps the raw text of
    the few values the parser reads (``keep``: the name and its type). It only tracks structure
    (strings, nesting); the final judgement of validity is a real JSON parse.

    The only state carried from one piece to the next is the phase, the nesting depth and whether
    the scan is inside a string or right after a backslash. Inside a value it jumps from one
    structural character to the next with a regular expression, so the characters in between are
    looked at by C code, not by this loop. The pieces are kept in a list and joined once, when
    the whole text is needed.

    Args:
        keep: Keys whose finished values are kept as text in :attr:`values`.
    """

    def __init__(self, keep: frozenset[str] = frozenset()) -> None:
        self._keep = keep
        self._pieces: list[str] = []
        self._length = 0  # characters taken so far
        self.phase: _Phase = "start"
        self.key: str | None = None  # key of the member being read
        self.value_start = 0  # absolute index where the value being read starts
        self.spans: dict[str, tuple[int, int]] = {}  # finished members: key -> (start, end)
        self.values: dict[str, str] = {}  # finished members kept as text: key -> raw value
        self._key_chars: list[str] = []
        self._depth = 0
        self._in_string = False
        self._escape = False
        self._kind: Literal["string", "container", "literal"] = "literal"

    @property
    def length(self) -> int:
        """Characters taken so far."""
        return self._length

    @property
    def text(self) -> str:
        """Everything taken so far, joined (once per change)."""
        if len(self._pieces) > 1:
            self._pieces = ["".join(self._pieces)]
        return self._pieces[0] if self._pieces else ""

    def slice(self, start: int, end: int) -> str:
        """Characters ``start`` to ``end`` of what was taken; cheap inside the last piece."""
        last = self._pieces[-1] if self._pieces else ""
        base = self._length - len(last)
        if start >= base:
            return last[start - base : end - base]
        return self.text[start:end]

    def feed(self, text: str, *, until_end: bool = False) -> str:
        """Take the next piece of the object.

        With ``until_end`` the scan stops at the brace that closes the object: ``text`` is kept up
        to it and the characters after it are returned. Otherwise everything is kept (inside
        markers, what follows the object is part of the body) and ``""`` is returned.
        """
        base = self._length
        index = 0
        size = len(text)
        while index < size:
            if self.phase == "in_value" and self._kind != "literal":
                index = self._skip(text, index, base)
                continue
            self._step(text[index], base + index + 1, text)
            index += 1
            if until_end and self.phase == "done":
                self._take(text[:index])
                return text[index:]
        self._take(text)
        return ""

    def _take(self, text: str) -> None:
        if text:
            self._pieces.append(text)
            self._length += len(text)

    def _skip(self, text: str, index: int, base: int) -> int:
        """Move through a string or container value; return the index after what it consumed."""
        if self._escape:
            self._escape = False
            return index + 1
        stop = (_STRING_STOP if self._in_string else _CONTAINER_STOP).search(text, index)
        if stop is None:
            return len(text)
        position = stop.start()
        char = text[position]
        end = base + position + 1
        if self._in_string:
            if char == "\\":
                self._escape = True
            else:  # the closing quote
                self._in_string = False
                if self._kind == "string":
                    self._end_value(end, text)
        elif char == '"':
            self._in_string = True
        elif char in "{[":
            self._depth += 1
        else:
            self._depth -= 1
            if self._depth == 0:
                self._end_value(end, text)
        return position + 1

    def _step(self, char: str, end: int, text: str) -> None:
        """One character outside a string or container value; ``end`` is the index after it."""
        match self.phase:
            case "start":
                if char == "{":
                    self.phase = "key"
                elif char not in _WHITESPACE:
                    self.phase = "broken"
            case "key":
                if char == '"':
                    self.phase, self._key_chars, self._escape = "in_key", [char], False
                elif char == "}":
                    self.phase = "done"
                elif char not in _WHITESPACE + ",":
                    self.phase = "broken"
            case "in_key":
                self._key_chars.append(char)
                if self._escape:
                    self._escape = False
                elif char == "\\":
                    self._escape = True
                elif char == '"':
                    try:
                        self.key = cast("str", json.loads("".join(self._key_chars)))
                    except ValueError:
                        self.phase = "broken"
                        return
                    self.phase = "colon"
            case "colon":
                if char == ":":
                    self.phase = "value"
                elif char not in _WHITESPACE:
                    self.phase = "broken"
            case "value":
                if char in _WHITESPACE:
                    return
                self.value_start = end - 1
                self.phase = "in_value"
                self._depth, self._in_string, self._escape = 0, False, False
                if char == '"':
                    self._kind, self._in_string = "string", True
                elif char in "{[":
                    self._kind, self._depth = "container", 1
                else:
                    self._kind = "literal"
            case "in_value":  # a literal: true, false, null or a number
                if char in ",}" or char in _WHITESPACE:
                    self._end_value(end - 1, text)
                    self._step(char, end, text)
            case "after_value":
                if char == ",":
                    self.phase = "key"
                elif char == "}":
                    self.phase = "done"
                elif char not in _WHITESPACE:
                    self.phase = "broken"
            case "done" | "broken":
                if char not in _WHITESPACE:
                    self.phase = "broken"

    def _end_value(self, end: int, text: str) -> None:
        """The value being read ends at absolute ``end``; ``text`` is the piece not taken yet."""
        key = self.key
        if key is not None:
            start = self.value_start
            self.spans[key] = (start, end)
            if key in self._keep:
                base = self._length
                if start >= base:
                    self.values[key] = text[start - base : end - base]
                else:
                    self.values[key] = self.text[start:] + text[: end - base]
        self.key = None
        self.phase = "after_value"

    @property
    def reading(self) -> str | None:
        """The key whose value is being read now, if any."""
        return self.key if self.phase == "in_value" else None


class _Run:
    """Text gathered a piece at a time and read once: a list, so a long run costs its length."""

    __slots__ = ("_parts",)

    def __init__(self) -> None:
        self._parts: list[str] = []

    def add(self, text: str) -> None:
        if text:
            self._parts.append(text)

    def take(self) -> str:
        """The text gathered so far; the run is empty afterwards."""
        text = "".join(self._parts)
        self._parts.clear()
        return text


class _Call:
    """One call being parsed; it gets its index when it is reported."""

    def __init__(self, keep: frozenset[str]) -> None:
        self.index = -1
        self.members = _MemberScanner(keep)
        self.name: str | None = None
        self.started = False
        self.sent = 0  # absolute index up to which the arguments were reported


class ToolCallParser:
    """Turns one request's answer text into text and function-call events.

    Args:
        fmt: The model's call format.
        call_id_prefix: Ids are ``{prefix}_{index}``; the server derives the prefix from the
            request id, so the same request always gives the same ids.
        markers: The markers from :meth:`ToolCallFormat.markers`, resolved once per server;
            ``None`` matches them by text. A format without markers has none either way.
    """

    def __init__(
        self,
        fmt: ToolCallFormat,
        *,
        call_id_prefix: str,
        markers: tuple[Marker, Marker] | None = None,
    ) -> None:
        self._markers = markers if markers is not None else fmt.markers()
        self._arguments = fmt.arguments
        self._typed_name = fmt.typed_name
        self._separator = fmt.separator
        # The member values the parser reads as text; the arguments are only ever sliced.
        self._keep = frozenset({"name", "type"} | ({fmt.typed_name} if fmt.typed_name else set()))
        self._prefix = call_id_prefix
        self._scanner = MarkerScanner(self._markers or ())
        self._call: _Call | None = None
        self._calls = 0  # calls reported to the client; the next call's index
        self._opened = 0  # call bodies entered, named or not
        self._space = _Run()  # whitespace held back in the text state
        # Without markers: "start" (nothing but whitespace yet), "call" (inside the object that
        # may be a call), "between" (a call ended: whitespace and one ``;`` may lead to the next)
        # or "text" (the answer is text from here on).
        self._mode: Literal["start", "call", "between", "text"] = "start"
        self._pending = _Run()  # the whitespace and ``;`` read in the "between" mode
        self._semicolon = False  # whether the "between" mode read its one ``;``

    @property
    def num_calls(self) -> int:
        """Calls reported so far."""
        return self._calls

    @property
    def opened_calls(self) -> int:
        """Call bodies entered so far, with or without a name."""
        return self._opened

    @property
    def in_call(self) -> bool:
        """Whether the text being read is the body of a call (after its opening marker)."""
        return self._call is not None

    def note_tokens(self, token_ids: Iterable[int]) -> None:
        """Record the token ids of the next piece of text, before feeding the text."""
        if self._markers is not None:
            self._scanner.note_tokens(token_ids)

    def feed(self, text: str) -> list[OutputEvent]:
        """Take the next piece of answer text; return the events it completes."""
        if self._markers is None:
            return self._feed_bare(text)
        opening, closing = self._markers
        events: list[OutputEvent] = []
        while True:
            active = (closing,) if self._call is not None else (opening,)
            before, marker = self._scanner.scan(text, active)
            text = ""
            if self._call is None:
                self._text(before, events)
                if marker is None:
                    return events
                self._space.take()  # whitespace before a call is layout
                self._call = _Call(self._keep)
                self._opened += 1
            else:
                self._body(before, events)
                if marker is None:
                    return events
                self._end_call(events, closed=True)

    def finish(self) -> list[OutputEvent]:
        """Release what is held back at the end of the answer and end an unclosed call."""
        if self._markers is None:
            return self._finish_bare()
        events: list[OutputEvent] = []
        rest = self._scanner.flush()
        if self._call is not None:
            self._body(rest, events)
            self._end_call(events, closed=False)
        else:
            self._text(rest, events)
            space = self._space.take()
            if space and self._calls == 0:
                events.append(TextDelta(space))
        return events

    # -- the text state ----------------------------------------------------------------------

    def _text(self, text: str, events: list[OutputEvent]) -> None:
        if not text:
            return
        body = text.rstrip(_WHITESPACE)
        if not body:
            self._space.add(text)  # only whitespace: held until text follows or the end
            return
        events.append(TextDelta(self._space.take() + body))
        self._space.add(text[len(body) :])

    # -- a call without markers --------------------------------------------------------------

    def _feed_bare(self, text: str) -> list[OutputEvent]:
        events: list[OutputEvent] = []
        while text:
            match self._mode:
                case "start":
                    body = text.lstrip(_WHITESPACE)
                    self._space.add(text[: len(text) - len(body)])  # layout or text: not known yet
                    text = body
                    if text:
                        opens = text[0] == "{"
                        self._call = _Call(self._keep) if opens else None
                        self._mode = "call" if opens else "text"
                case "call":
                    text = self._object(text, events)
                case "between":
                    text = self._between(text, events)
                case "text":
                    self._text(text, events)
                    break
        return events

    def _between(self, text: str, events: list[OutputEvent]) -> str:
        """After a call: whitespace, then maybe ``;`` and a ``{`` that starts the next call."""
        char = text[0]
        if char in _WHITESPACE:
            rest = text.lstrip(_WHITESPACE)
            self._pending.add(text[: len(text) - len(rest)])
            return rest
        if char == self._separator and not self._semicolon:
            self._pending.add(char)
            self._semicolon = True
            return text[1:]
        separator = self._pending.take()
        if char == "{" and self._semicolon:
            self._space.take()
            self._space.add(separator)  # dropped once the call has a name
            self._semicolon = False
            self._call = _Call(self._keep)
            self._mode = "call"
            return text  # the brace belongs to the call
        self._semicolon = False
        self._mode = "text"
        self._text(separator, events)
        return text

    def _object(self, text: str, events: list[OutputEvent]) -> str:
        """Feed the object that may be a call; return the text after its closing brace."""
        rest = self._body(text, events, until_end=True)
        call = self._call
        assert call is not None  # noqa: S101  # only called in the call state
        if call.started:
            self._space.take()  # whitespace before a call is layout
        phase = call.members.phase
        if phase == "done" or (phase == "broken" and not call.started):
            # A broken call keeps swallowing the answer, as one inside markers does.
            self._end_call(events, closed=phase == "done")
            self._mode = "between" if phase == "done" and call.started else "text"
        return rest

    def _finish_bare(self) -> list[OutputEvent]:
        events: list[OutputEvent] = []
        separator = self._pending.take()
        if self._call is not None:
            self._end_call(events, closed=False)
        elif self._mode == "between" and self._semicolon:
            self._text(separator, events)  # a separator with no call after it is text
        self._semicolon = False
        space = self._space.take()
        if space and self._calls == 0:
            events.append(TextDelta(space))
        return events

    # -- the call state ----------------------------------------------------------------------

    def _body(self, text: str, events: list[OutputEvent], *, until_end: bool = False) -> str:
        call = self._call
        assert call is not None  # noqa: S101  # only called in the call state
        rest = call.members.feed(text, until_end=until_end)
        if call.name is None:
            call.name = self._name_of(call.members.values)
        if call.name is not None and not call.started:
            call.started = True
            call.index = self._calls
            self._calls += 1
            events.append(ToolCallStart(call.index, f"{self._prefix}_{call.index}", call.name))
        if call.started:
            self._send_arguments(call, events)
        return rest

    def _name_of(self, values: dict[str, str]) -> str | None:
        """The function name among the raw member values read so far, if it is known."""
        if "name" in values:
            return _string(values["name"])
        typed = self._typed_name
        if typed is not None and typed in values and _string(values.get("type", "")) == "function":
            return _string(values[typed])
        return None

    def _send_arguments(self, call: _Call, events: list[OutputEvent]) -> None:
        """Report the arguments written since the last report: a slice of the new piece.

        Only the first report of a call whose arguments came before its name reads earlier
        pieces back; every other report is a slice of the piece just taken.
        """
        members = call.members
        span = members.spans.get(self._arguments)
        if span is not None:
            start, end = span
        elif members.reading == self._arguments:
            start, end = members.value_start, members.length
        else:
            return
        start = max(start, call.sent)
        if end > start:
            events.append(ToolCallDelta(call.index, members.slice(start, end)))
            call.sent = end

    def _end_call(self, events: list[OutputEvent], *, closed: bool) -> None:
        call = self._call
        assert call is not None  # noqa: S101  # only called in the call state
        self._call = None
        if not call.started:
            # No name, so not a call: show the model's text as it was written.
            raw = call.members.text
            if self._markers is not None:
                opening, closing = self._markers
                raw = opening.text + raw + (closing.text if closed else "")
            self._text(raw, events)
            return
        valid = closed and self._is_valid_call(call.members.text, call.name)
        events.append(ToolCallEnd(call.index, valid))

    def _is_valid_call(self, body: str, name: str | None) -> bool:
        try:
            value = parse_json_strict(body)
        except ValueError:
            return False
        if not isinstance(value, dict):
            return False
        call = cast("dict[str, object]", value)
        typed = self._typed_name
        if "name" in call:
            written = call["name"]
        elif typed is not None and call.get("type") == "function":
            written = call.get(typed)
        else:
            written = None
        return written == name and isinstance(call.get(self._arguments), dict)


def _string(raw: str) -> str | None:
    try:
        value = json.loads(raw)
    except ValueError:
        return None
    return value if isinstance(value, str) else None
