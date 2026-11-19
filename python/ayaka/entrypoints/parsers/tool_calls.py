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
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final, Literal, cast

from ayaka.core.errors import ConfigError
from ayaka.entrypoints.conversation import (
    TextDelta,
    ToolCallDelta,
    ToolCallEnd,
    ToolCallStart,
)
from ayaka.entrypoints.parsers.markers import Marker, MarkerScanner, resolve_marker
from ayaka.utils.parsing import parse_json_strict

if TYPE_CHECKING:
    from collections.abc import Iterable

    from ayaka.entrypoints.conversation import OutputEvent
    from ayaka.tokenizers.ports import TokenizerLike

__all__ = ["TOOL_CALL_FORMATS", "ToolCallFormat", "ToolCallParser", "tool_call_format"]


@dataclass(frozen=True, slots=True)
class ToolCallFormat:
    """How a model family writes a function call.

    Attributes:
        name: The name the server option uses.
        open: The marker before the JSON body, or ``None`` for a call written without markers.
        close: The marker after it, or ``None`` with ``open``.
        arguments: The key of the arguments in the call's JSON object.
        typed_name: A second key that holds the function name when the object also says
            ``"type": "function"``; ``None`` if only ``name`` does.
    """

    name: str
    open: str | None
    close: str | None
    arguments: str = "arguments"
    typed_name: str | None = None

    def __post_init__(self) -> None:
        if (self.open is None) != (self.close is None):
            raise ValueError("a call format has both markers or none")

    def markers(self, tokenizer: TokenizerLike | None = None) -> tuple[Marker, Marker] | None:
        """The markers as ``tokenizer`` spells them (by text without one); ``None`` without any.

        Raises:
            ConfigError: If a marker is a special token the detokenizer drops.
        """
        if self.open is None or self.close is None:
            return None
        return resolve_marker(self.open, tokenizer), resolve_marker(self.close, tokenizer)


TOOL_CALL_FORMATS: Final[dict[str, ToolCallFormat]] = {
    # Hermes 2/3, Qwen2.5, Qwen3, SmolLM3 and others.
    "hermes": ToolCallFormat("hermes", "<tool_call>", "</tool_call>"),
    # Llama 3.x: a bare JSON object, its arguments under "parameters". The chat templates of
    # Llama-3.1 and Llama-3.2 both teach this layout; Llama-3.2-1B also writes the function name
    # under "function" next to "type": "function" when the tools are shown wrapped that way.
    "llama3_json": ToolCallFormat(
        "llama3_json", None, None, arguments="parameters", typed_name="function"
    ),
}


def tool_call_format(name: str) -> ToolCallFormat:
    """The format called ``name``.

    Raises:
        ConfigError: If there is no such format.
    """
    try:
        return TOOL_CALL_FORMATS[name]
    except KeyError:
        raise ConfigError(
            "server.tool_call_parser",
            "INVALID_CHOICE",
            f"unknown tool call parser {name!r}; known: {sorted(TOOL_CALL_FORMATS)}",
        ) from None


type _Phase = Literal[
    "start", "key", "in_key", "colon", "value", "in_value", "after_value", "done", "broken"
]
_WHITESPACE: Final = " \t\r\n"


class _MemberScanner:
    """Follows the members of one JSON object character by character.

    It records each top-level member's key and the raw text of its value. It only tracks
    structure (strings, nesting); the final judgement of validity is a real JSON parse.
    """

    def __init__(self) -> None:
        self.text = ""
        self.phase: _Phase = "start"
        self.key: str | None = None  # key of the member being read
        self.value_start = 0
        self.values: dict[str, str] = {}  # finished members: key -> raw value text
        self._key_chars: list[str] = []
        self._depth = 0
        self._in_string = False
        self._escape = False
        self._kind: Literal["string", "container", "literal"] = "literal"
        self._end = 0

    def feed(self, text: str, *, until_end: bool = False) -> str:
        """Take the next piece of the object.

        With ``until_end`` the scan stops at the brace that closes the object: ``text`` is kept up
        to it and the characters after it are returned. Otherwise everything is kept (inside
        markers, what follows the object is part of the body) and ``""`` is returned.
        """
        base = len(self.text)
        self.text += text
        for offset, char in enumerate(text):
            self._end = base + offset + 1  # index just after ``char``
            self._step(char)
            if until_end and self.phase == "done":
                self.text = self.text[: self._end]
                return text[offset + 1 :]
        return ""

    def _step(self, char: str) -> None:
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
                self.value_start = self._end - 1
                self.phase = "in_value"
                self._depth, self._in_string, self._escape = 0, False, False
                if char == '"':
                    self._kind, self._in_string = "string", True
                elif char in "{[":
                    self._kind, self._depth = "container", 1
                else:
                    self._kind = "literal"
            case "in_value":
                self._in_value(char)
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

    def _in_value(self, char: str) -> None:
        if self._kind == "literal":
            if char in ",}" or char in _WHITESPACE:
                self._end_value(self._end - 1)
                self.phase = "after_value"
                self._step(char)
            return
        if self._in_string:
            if self._escape:
                self._escape = False
            elif char == "\\":
                self._escape = True
            elif char == '"':
                self._in_string = False
                if self._kind == "string":
                    self._end_value(self._end)
            return
        if char == '"':
            self._in_string = True
        elif char in "{[":
            self._depth += 1
        elif char in "}]":
            self._depth -= 1
            if self._depth == 0:
                self._end_value(self._end)

    def _end_value(self, end: int) -> None:
        if self.key is not None:
            self.values[self.key] = self.text[self.value_start : end]
        self.key = None
        self.phase = "after_value"

    @property
    def reading(self) -> str | None:
        """The key whose value is being read now, if any."""
        return self.key if self.phase == "in_value" else None


class _Call:
    """One call being parsed; it gets its index when it is reported."""

    def __init__(self) -> None:
        self.index = -1
        self.members = _MemberScanner()
        self.name: str | None = None
        self.started = False
        self.sent = 0  # characters of the arguments value already reported


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
        self._prefix = call_id_prefix
        self._scanner = MarkerScanner()
        self._call: _Call | None = None
        self._calls = 0  # calls reported to the client; the next call's index
        self._space = ""  # whitespace held back in the text state
        # Without markers: "start" (nothing but whitespace yet), "call" (inside the object that
        # may be a call), "between" (a call ended: whitespace and one ``;`` may lead to the next)
        # or "text" (the answer is text from here on).
        self._mode: Literal["start", "call", "between", "text"] = "start"
        self._pending = ""  # the whitespace and ``;`` read in the "between" mode

    @property
    def num_calls(self) -> int:
        """Calls reported so far."""
        return self._calls

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
                self._space = ""  # whitespace before a call is layout
                self._call = _Call()
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
            if self._space and self._calls == 0:
                events.append(TextDelta(self._space))
            self._space = ""
        return events

    # -- the text state ----------------------------------------------------------------------

    def _text(self, text: str, events: list[OutputEvent]) -> None:
        if not text:
            return
        text = self._space + text
        body = text.rstrip(_WHITESPACE)
        self._space = text[len(body) :]
        if body:
            events.append(TextDelta(body))

    # -- a call without markers --------------------------------------------------------------

    def _feed_bare(self, text: str) -> list[OutputEvent]:
        events: list[OutputEvent] = []
        while text:
            match self._mode:
                case "start":
                    body = text.lstrip(_WHITESPACE)
                    self._space += text[: len(text) - len(body)]  # layout, or text: not known yet
                    text = body
                    if text:
                        opens = text[0] == "{"
                        self._call = _Call() if opens else None
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
        if char in _WHITESPACE or (char == ";" and ";" not in self._pending):
            self._pending += char
        elif char == "{" and ";" in self._pending:
            self._space, self._pending = self._pending, ""  # dropped once the call has a name
            self._call = _Call()
            self._mode = "call"
            return text  # the brace belongs to the call
        else:
            pending, self._pending = self._pending, ""
            self._mode = "text"
            self._text(pending, events)
            return text
        return text[1:]

    def _object(self, text: str, events: list[OutputEvent]) -> str:
        """Feed the object that may be a call; return the text after its closing brace."""
        rest = self._body(text, events, until_end=True)
        call = self._call
        assert call is not None  # noqa: S101  # only called in the call state
        if call.started:
            self._space = ""  # whitespace before a call is layout
        phase = call.members.phase
        if phase == "done" or (phase == "broken" and not call.started):
            # A broken call keeps swallowing the answer, as one inside markers does.
            self._end_call(events, closed=phase == "done")
            self._mode = "between" if phase == "done" and call.started else "text"
        return rest

    def _finish_bare(self) -> list[OutputEvent]:
        events: list[OutputEvent] = []
        if self._call is not None:
            self._end_call(events, closed=False)
        elif self._mode == "between" and ";" in self._pending:
            self._text(self._pending, events)  # a separator with no call after it is text
        self._pending = ""
        if self._space and self._calls == 0:
            events.append(TextDelta(self._space))
        self._space = ""
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
        members = call.members
        if self._arguments in members.values:
            raw = members.values[self._arguments]
        elif members.reading == self._arguments:
            raw = members.text[members.value_start :]
        else:
            return
        if len(raw) > call.sent:
            events.append(ToolCallDelta(call.index, raw[call.sent :]))
            call.sent = len(raw)

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
