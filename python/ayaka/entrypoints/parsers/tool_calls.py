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
        open: The marker before the JSON body.
        close: The marker after it.
    """

    name: str
    open: str
    close: str

    def markers(self, tokenizer: TokenizerLike | None = None) -> tuple[Marker, Marker]:
        """The markers as ``tokenizer`` spells them (by text without one).

        Raises:
            ConfigError: If a marker is a special token the detokenizer drops.
        """
        return resolve_marker(self.open, tokenizer), resolve_marker(self.close, tokenizer)


TOOL_CALL_FORMATS: Final[dict[str, ToolCallFormat]] = {
    # Hermes 2/3, Qwen2.5, Qwen3, SmolLM3 and others.
    "hermes": ToolCallFormat("hermes", "<tool_call>", "</tool_call>"),
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

    def feed(self, text: str) -> None:
        base = len(self.text)
        self.text += text
        for offset, char in enumerate(text):
            self._end = base + offset + 1  # index just after ``char``
            self._step(char)

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
            ``None`` matches them by text.
    """

    def __init__(
        self,
        fmt: ToolCallFormat,
        *,
        call_id_prefix: str,
        markers: tuple[Marker, Marker] | None = None,
    ) -> None:
        self._open, self._close = markers if markers is not None else fmt.markers()
        self._prefix = call_id_prefix
        self._scanner = MarkerScanner()
        self._call: _Call | None = None
        self._calls = 0  # calls reported to the client; the next call's index
        self._space = ""  # whitespace held back in the text state

    @property
    def num_calls(self) -> int:
        """Calls reported so far."""
        return self._calls

    def note_tokens(self, token_ids: Iterable[int]) -> None:
        """Record the token ids of the next piece of text, before feeding the text."""
        self._scanner.note_tokens(token_ids)

    def feed(self, text: str) -> list[OutputEvent]:
        """Take the next piece of answer text; return the events it completes."""
        events: list[OutputEvent] = []
        while True:
            active = (self._close,) if self._call is not None else (self._open,)
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

    # -- the call state ----------------------------------------------------------------------

    def _body(self, text: str, events: list[OutputEvent]) -> None:
        call = self._call
        assert call is not None  # noqa: S101  # only called in the call state
        call.members.feed(text)
        if call.name is None and "name" in call.members.values:
            call.name = _string(call.members.values["name"])
        if call.name is not None and not call.started:
            call.started = True
            call.index = self._calls
            self._calls += 1
            events.append(ToolCallStart(call.index, f"{self._prefix}_{call.index}", call.name))
        if call.started:
            self._send_arguments(call, events)

    def _send_arguments(self, call: _Call, events: list[OutputEvent]) -> None:
        members = call.members
        if "arguments" in members.values:
            raw = members.values["arguments"]
        elif members.reading == "arguments":
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
            raw = self._open.text + call.members.text + (self._close.text if closed else "")
            self._text(raw, events)
            return
        valid = closed and _is_valid_call(call.members.text, call.name)
        events.append(ToolCallEnd(call.index, valid))


def _string(raw: str) -> str | None:
    try:
        value = json.loads(raw)
    except ValueError:
        return None
    return value if isinstance(value, str) else None


def _is_valid_call(body: str, name: str | None) -> bool:
    try:
        value = parse_json_strict(body)
    except ValueError:
        return False
    if not isinstance(value, dict):
        return False
    call = cast("dict[str, object]", value)
    return call.get("name") == name and isinstance(call.get("arguments"), dict)
