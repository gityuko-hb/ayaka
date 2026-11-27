"""Incremental streaming parser for XML and tag-based tool-call families.

Handles six model layouts through one table-driven outer-tag + inner-block parser:

* ``qwen3_coder`` / ``mimo``: ``<tool_call><function=NAME><parameter=K>V...</tool_call>``
* ``glm_xml``: ``<tool_call>NAME<arg_key>K</arg_key><arg_value>V</arg_value></tool_call>``
* ``minimax_m2`` / ``minimax_m3``: ``<minimax:tool_call><invoke name=...>...</minimax:tool_call>``
* ``deepseek_dsml``: ``<\\uff5cDSML\\uff5cfunction_calls>`` with ``invoke`` / ``parameter`` tags
"""

from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING, Final, cast

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

__all__ = ["XmlToolCallParser"]

_WHITESPACE: Final = " \t\r\n"

_QWEN_FUNCTION_RE: Final = re.compile(
    r"<function\s*=\s*([^\s>]+)\s*>(.*?)</function\s*>", re.DOTALL
)
_QWEN_PARAM_RE: Final = re.compile(r"<parameter\s*=\s*([^\s>]+)\s*>(.*?)</parameter\s*>", re.DOTALL)
_GLM_KEY_RE: Final = re.compile(
    r"<arg_key>(.*?)</arg_key>\s*<arg_value>(.*?)</arg_value>", re.DOTALL
)
_MINIMAX_INVOKE_RE: Final = re.compile(
    r"""<invoke\s+name\s*=\s*(?:"([^"]+)"|'([^']+)')\s*>(.*?)</invoke\s*>""",
    re.DOTALL,
)
_MINIMAX_PARAM_RE: Final = re.compile(
    r"""<parameter\s+name\s*=\s*(?:"([^"]+)"|'([^']+)')\s*>(.*?)</parameter\s*>""",
    re.DOTALL,
)
_DSML_INVOKE_RE: Final = re.compile(
    r"""<\uff5cDSML\uff5cinvoke\s+name\s*=\s*(?:"([^"]+)"|'([^']+)')\s*>"""
    r"(.*?)</\uff5cDSML\uff5cinvoke\s*>",
    re.DOTALL,
)
_DSML_PARAM_RE: Final = re.compile(
    r"""<\uff5cDSML\uff5cparameter\s+name\s*=\s*(?:"([^"]+)"|'([^']+)')"""
    r"""(?:\s+string\s*=\s*(?:"(true|false)"|'(true|false)'))?\s*>"""
    r"(.*?)</\uff5cDSML\uff5cparameter\s*>",
    re.DOTALL,
)


def _strip_tag_value(raw: str) -> str:
    if raw.startswith("\r\n"):
        raw = raw[2:]
    elif raw.startswith("\n"):
        raw = raw[1:]
    if raw.endswith("\r\n"):
        raw = raw[:-2]
    elif raw.endswith("\n"):
        raw = raw[:-1]
    return raw


def _coerce_param(
    tool_name: str,
    key: str,
    raw_value: str,
    schemas: ToolSchemas | None,
    *,
    string_hint: str | None = None,
) -> object:
    value = _strip_tag_value(raw_value)
    if schemas is not None:
        return schemas.coerce_value(tool_name, key, value)
    if string_hint == "false":
        try:
            return json.loads(value.strip())
        except ValueError:
            return value
    return value


def _extract_calls(
    format_name: str, body: str, schemas: ToolSchemas | None
) -> list[tuple[str, dict[str, object]]]:
    calls: list[tuple[str, dict[str, object]]] = []
    if format_name in ("qwen3_coder", "mimo"):
        for match in _QWEN_FUNCTION_RE.finditer(body):
            name = match.group(1).strip()
            inner = match.group(2)
            args = {
                pm.group(1).strip(): _coerce_param(name, pm.group(1).strip(), pm.group(2), schemas)
                for pm in _QWEN_PARAM_RE.finditer(inner)
            }
            if name:
                calls.append((name, args))
        if not calls and format_name == "mimo":
            try:
                obj = parse_json_strict(body)
            except ValueError:
                obj = None
            if isinstance(obj, dict):
                record = cast("dict[str, object]", obj)
                name = record.get("name")
                raw_args = record.get("arguments")
                if isinstance(name, str) and name and isinstance(raw_args, dict):
                    typed_args = cast("dict[str, object]", raw_args)
                    calls.append(
                        (
                            name,
                            schemas.coerce_arguments(name, typed_args) if schemas else typed_args,
                        )
                    )
        return calls

    if format_name == "glm_xml":
        first_tag = body.find("<arg_key>")
        name = (body if first_tag < 0 else body[:first_tag]).strip()
        if not name:
            return calls
        args = {
            m.group(1).strip(): _coerce_param(name, m.group(1).strip(), m.group(2), schemas)
            for m in _GLM_KEY_RE.finditer(body)
        }
        calls.append((name, args))
        return calls

    if format_name in ("minimax_m2", "minimax_m3"):
        for match in _MINIMAX_INVOKE_RE.finditer(body):
            name = (match.group(1) or match.group(2) or "").strip()
            inner = match.group(3)
            args = {
                (pm.group(1) or pm.group(2) or "").strip(): _coerce_param(
                    name, (pm.group(1) or pm.group(2) or "").strip(), pm.group(3), schemas
                )
                for pm in _MINIMAX_PARAM_RE.finditer(inner)
            }
            if name:
                calls.append((name, args))
        return calls

    if format_name == "deepseek_dsml":
        for match in _DSML_INVOKE_RE.finditer(body):
            name = (match.group(1) or match.group(2) or "").strip()
            inner = match.group(3)
            args: dict[str, object] = {}
            for pm in _DSML_PARAM_RE.finditer(inner):
                key = (pm.group(1) or pm.group(2) or "").strip()
                hint = pm.group(3) or pm.group(4)
                args[key] = _coerce_param(name, key, pm.group(5), schemas, string_hint=hint)
            if name:
                calls.append((name, args))
        return calls

    return calls


class XmlToolCallParser:
    """Incremental parser for XML and tag-delimited function calls."""

    def __init__(
        self,
        fmt: ToolCallFormat,
        *,
        call_id_prefix: str,
        markers: tuple[Marker, Marker] | None = None,
        schemas: ToolSchemas | None = None,
    ) -> None:
        resolved = markers if markers is not None else fmt.markers()
        if resolved is None:
            raise ValueError(f"XML tool call format {fmt.name!r} requires opening/closing markers")
        self._format_name = fmt.name
        self._markers = resolved
        self._schemas = schemas
        self._prefix = call_id_prefix
        self._scanner = MarkerScanner(resolved)
        self._space = RunBuffer()
        self._body_parts: list[str] | None = None
        self._calls = 0
        self._opened = 0

    @property
    def num_calls(self) -> int:
        return self._calls

    @property
    def opened_calls(self) -> int:
        return self._opened

    @property
    def in_call(self) -> bool:
        return self._body_parts is not None

    def note_tokens(self, token_ids: Iterable[int]) -> None:
        self._scanner.note_tokens(token_ids)

    def feed(self, text: str) -> list[OutputEvent]:
        opening, closing = self._markers
        events: list[OutputEvent] = []
        while True:
            active = (closing,) if self._body_parts is not None else (opening,)
            before, marker = self._scanner.scan(text, active)
            text = ""
            if self._body_parts is None:
                self._text(before, events)
                if marker is None:
                    return events
                self._space.take()
                self._body_parts = []
                self._opened += 1
            else:
                if before:
                    self._body_parts.append(before)
                if marker is None:
                    return events
                self._flush_block(events, closed=True)

    def finish(self) -> list[OutputEvent]:
        events: list[OutputEvent] = []
        rest = self._scanner.flush()
        if self._body_parts is not None:
            if rest:
                self._body_parts.append(rest)
            self._flush_block(events, closed=False)
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

    def _flush_block(self, events: list[OutputEvent], *, closed: bool) -> None:
        parts = self._body_parts
        assert parts is not None  # noqa: S101
        self._body_parts = None
        raw_body = "".join(parts)
        extracted = _extract_calls(self._format_name, raw_body, self._schemas)
        if not extracted:
            opening, closing = self._markers
            self._text(opening.text + raw_body + (closing.text if closed else ""), events)
            return
        for name, args in extracted:
            index = self._calls
            self._calls += 1
            events.append(ToolCallStart(index, f"{self._prefix}_{index}", name))
            events.append(ToolCallDelta(index, json.dumps(args, ensure_ascii=False)))
            events.append(ToolCallEnd(index, valid=closed))
