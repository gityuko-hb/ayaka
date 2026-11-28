"""Incremental JSON tool-call parser (Hermes, Qwen3-JSON, DeepSeek-JSON, Llama-3.x)."""

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
from ayaka.entrypoints.parsers.tool_calls.base import RunBuffer
from ayaka.utils.parsing import parse_json_strict

if TYPE_CHECKING:
    from collections.abc import Iterable

    from ayaka.entrypoints.conversation import OutputEvent
    from ayaka.entrypoints.parsers.formats import ToolCallFormat
    from ayaka.entrypoints.parsers.markers import Marker
    from ayaka.entrypoints.parsers.parameters import ToolSchemas

__all__ = ["ToolCallParser"]


type _Phase = Literal[
    "start", "key", "in_key", "colon", "value", "in_value", "after_value", "done", "broken"
]
_WHITESPACE: Final = " \t\r\n"
_STRING_STOP: Final = re.compile(r'["\\]')
_CONTAINER_STOP: Final = re.compile(r'["\\{}\[\]]')
_DEEPSEEK_WRAPPER_OPEN: Final = "<|tool_calls_begin|>"
_DEEPSEEK_WRAPPER_CLOSE: Final = "<|tool_calls_end|>"


class _MemberScanner:
    """Follows top-level JSON object members incrementally in linear time."""

    def __init__(self, keep: frozenset[str] = frozenset()) -> None:
        self._keep = keep
        self._pieces: list[str] = []
        self._length = 0
        self.phase: _Phase = "start"
        self.key: str | None = None
        self.value_start = 0
        self.spans: dict[str, tuple[int, int]] = {}
        self.values: dict[str, str] = {}
        self._key_chars: list[str] = []
        self._depth = 0
        self._in_string = False
        self._escape = False
        self._kind: Literal["string", "container", "literal"] = "literal"

    @property
    def length(self) -> int:
        return self._length

    @property
    def text(self) -> str:
        if len(self._pieces) > 1:
            self._pieces = ["".join(self._pieces)]
        return self._pieces[0] if self._pieces else ""

    def slice(self, start: int, end: int) -> str:
        last = self._pieces[-1] if self._pieces else ""
        base = self._length - len(last)
        if start >= base:
            return last[start - base : end - base]
        return self.text[start:end]

    def feed(self, text: str, *, until_end: bool = False) -> str:
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
            else:
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
            case "in_value":
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
        return self.key if self.phase == "in_value" else None


class _Call:
    """One JSON call being parsed."""

    def __init__(self, keep: frozenset[str]) -> None:
        self.index = -1
        self.members = _MemberScanner(keep)
        self.name: str | None = None
        self.started = False
        self.sent = 0


class ToolCallParser:
    """Turns one request's answer text into text and JSON function-call events."""

    def __init__(
        self,
        fmt: ToolCallFormat,
        *,
        call_id_prefix: str,
        markers: tuple[Marker, Marker] | None = None,
        schemas: ToolSchemas | None = None,
    ) -> None:
        self._format_name = fmt.name
        self._markers = markers if markers is not None else fmt.markers()
        self._arguments = fmt.arguments
        self._typed_name = fmt.typed_name
        self._separator = fmt.separator
        self._schemas = schemas
        self._keep = frozenset({"name", "type"} | ({fmt.typed_name} if fmt.typed_name else set()))
        self._prefix = call_id_prefix
        self._scanner = MarkerScanner(self._markers or ())
        self._call: _Call | None = None
        self._calls = 0
        self._opened = 0
        self._space = RunBuffer()
        self._mode: Literal["start", "call", "between", "text"] = "start"
        self._pending = RunBuffer()
        self._semicolon = False

    @property
    def num_calls(self) -> int:
        return self._calls

    @property
    def opened_calls(self) -> int:
        return self._opened

    @property
    def in_call(self) -> bool:
        return self._call is not None

    def note_tokens(self, token_ids: Iterable[int]) -> None:
        if self._markers is not None:
            self._scanner.note_tokens(token_ids)

    def feed(self, text: str) -> list[OutputEvent]:
        if self._format_name == "deepseek_json":
            text = text.replace(_DEEPSEEK_WRAPPER_OPEN, "").replace(_DEEPSEEK_WRAPPER_CLOSE, "")
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
                self._space.take()
                self._call = _Call(self._keep)
                self._opened += 1
            else:
                self._body(before, events)
                if marker is None:
                    return events
                self._end_call(events, closed=True)

    def finish(self) -> list[OutputEvent]:
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

    def _text(self, text: str, events: list[OutputEvent]) -> None:
        if not text:
            return
        body = text.rstrip(_WHITESPACE)
        if not body:
            self._space.add(text)
            return
        events.append(TextDelta(self._space.take() + body))
        self._space.add(text[len(body) :])

    def _feed_bare(self, text: str) -> list[OutputEvent]:
        events: list[OutputEvent] = []
        while text:
            match self._mode:
                case "start":
                    body = text.lstrip(_WHITESPACE)
                    self._space.add(text[: len(text) - len(body)])
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
            self._space.add(separator)
            self._semicolon = False
            self._call = _Call(self._keep)
            self._mode = "call"
            return text
        self._semicolon = False
        self._mode = "text"
        self._text(separator, events)
        return text

    def _object(self, text: str, events: list[OutputEvent]) -> str:
        rest = self._body(text, events, until_end=True)
        call = self._call
        assert call is not None  # noqa: S101
        if call.started:
            self._space.take()
        phase = call.members.phase
        if phase == "done" or (phase == "broken" and not call.started):
            self._end_call(events, closed=phase == "done")
            self._mode = "between" if phase == "done" and call.started else "text"
        return rest

    def _finish_bare(self) -> list[OutputEvent]:
        events: list[OutputEvent] = []
        separator = self._pending.take()
        if self._call is not None:
            self._end_call(events, closed=False)
        elif self._mode == "between" and self._semicolon:
            self._text(separator, events)
        self._semicolon = False
        space = self._space.take()
        if space and self._calls == 0:
            events.append(TextDelta(space))
        return events

    def _body(self, text: str, events: list[OutputEvent], *, until_end: bool = False) -> str:
        call = self._call
        assert call is not None  # noqa: S101
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
        if "name" in values:
            return _string(values["name"])
        typed = self._typed_name
        if typed is not None and typed in values and _string(values.get("type", "")) == "function":
            return _string(values[typed])
        return None

    def _send_arguments(self, call: _Call, events: list[OutputEvent]) -> None:
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
        assert call is not None  # noqa: S101
        self._call = None
        if not call.started:
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
