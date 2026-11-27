"""Tests for XML/tag and JSON tool-call families with schema coercion and split invariance."""

from __future__ import annotations

import json
import random

import pytest

from ayaka.core.constraints import ToolSchema
from ayaka.entrypoints.conversation import (
    TextDelta,
    ToolCallDelta,
    ToolCallEnd,
    ToolCallStart,
)
from ayaka.entrypoints.parsers.formats import TOOL_CALL_FORMATS, suggest_tool_call_parser
from ayaka.entrypoints.parsers.markers import Marker
from ayaka.entrypoints.parsers.parameters import ToolSchemas
from ayaka.entrypoints.parsers.tool_calls.factory import build_tool_call_parser

pytestmark = pytest.mark.unit


def _parse_chunks(
    fmt_name: str, chunks: list[str], *, schemas: ToolSchemas | None = None
) -> tuple[str, list[tuple[str, dict[str, object], bool]]]:
    fmt = TOOL_CALL_FORMATS[fmt_name]
    markers = (
        (Marker(fmt.open, None), Marker(fmt.close, None))
        if fmt.open is not None and fmt.close is not None
        else None
    )
    parser = build_tool_call_parser(fmt, call_id_prefix="c", markers=markers, schemas=schemas)
    text_parts: list[str] = []
    calls: dict[int, tuple[str, list[str]]] = {}
    finished: dict[int, bool] = {}
    for chunk in chunks:
        for ev in parser.feed(chunk):
            if isinstance(ev, TextDelta):
                text_parts.append(ev.text)
            elif isinstance(ev, ToolCallStart):
                calls[ev.index] = (ev.name, [])
            elif isinstance(ev, ToolCallDelta):
                calls[ev.index][1].append(ev.arguments)
            elif isinstance(ev, ToolCallEnd):
                finished[ev.index] = ev.valid
    for ev in parser.finish():
        if isinstance(ev, TextDelta):
            text_parts.append(ev.text)
        elif isinstance(ev, ToolCallStart):
            calls[ev.index] = (ev.name, [])
        elif isinstance(ev, ToolCallDelta):
            calls[ev.index][1].append(ev.arguments)
        elif isinstance(ev, ToolCallEnd):
            finished[ev.index] = ev.valid
    out_calls = [
        (calls[idx][0], json.loads("".join(calls[idx][1])), finished[idx]) for idx in sorted(calls)
    ]
    return "".join(text_parts), out_calls


def _schemas() -> ToolSchemas:
    return ToolSchemas.from_tools(
        [
            ToolSchema(
                name="get_weather",
                parameters={
                    "type": "object",
                    "properties": {
                        "city": {"type": "string"},
                        "days": {"type": "integer"},
                        "metric": {"type": "boolean"},
                    },
                },
            )
        ]
    )


@pytest.mark.parametrize(
    ("fmt_name", "payload"),
    [
        (
            "qwen3_coder",
            "Checking now.\n<tool_call><function=get_weather><parameter=city>Hue</parameter>"
            "<parameter=days>3</parameter><parameter=metric>true</parameter></function></tool_call>",
        ),
        (
            "mimo",
            "<tool_call><function=get_weather><parameter=city>Hue</parameter>"
            "<parameter=days>3</parameter><parameter=metric>true</parameter></function></tool_call>",
        ),
        (
            "glm_xml",
            "<tool_call>get_weather\n<arg_key>city</arg_key><arg_value>Hue</arg_value>"
            "<arg_key>days</arg_key><arg_value>3</arg_value>"
            "<arg_key>metric</arg_key><arg_value>true</arg_value></tool_call>",
        ),
        (
            "minimax_m2",
            '<minimax:tool_call><invoke name="get_weather"><parameter name="city">Hue</parameter>'
            '<parameter name="days">3</parameter><parameter name="metric">true</parameter>'
            "</invoke></minimax:tool_call>",
        ),
        (
            "minimax_m3",
            '<minimax:tool_call><invoke name="get_weather"><parameter name="city">Hue</parameter>'
            '<parameter name="days">3</parameter><parameter name="metric">true</parameter>'
            "</invoke></minimax:tool_call>",
        ),
        (
            "deepseek_dsml",
            '<\uff5cDSML\uff5cfunction_calls><\uff5cDSML\uff5cinvoke name="get_weather">'
            '<\uff5cDSML\uff5cparameter name="city" string="true">Hue</\uff5cDSML\uff5cparameter>'
            '<\uff5cDSML\uff5cparameter name="days" string="false">3</\uff5cDSML\uff5cparameter>'
            '<\uff5cDSML\uff5cparameter name="metric" string="false">true'
            "</\uff5cDSML\uff5cparameter>"
            "</\uff5cDSML\uff5cinvoke></\uff5cDSML\uff5cfunction_calls>",
        ),
    ],
)
def test_xml_tool_call_families_streaming_coercion_and_split_invariance(
    fmt_name: str, payload: str
) -> None:
    schemas = _schemas()
    one_shot = _parse_chunks(fmt_name, [payload], schemas=schemas)
    text, calls = one_shot
    if fmt_name == "qwen3_coder":
        assert text == "Checking now."
    else:
        assert text == ""
    assert calls == [("get_weather", {"city": "Hue", "days": 3, "metric": True}, True)]

    # 1-char split invariance
    assert _parse_chunks(fmt_name, list(payload), schemas=schemas) == one_shot

    # Random chunk sizes
    rng = random.Random(7)  # noqa: S311  # deterministic chunk-boundary fuzzer in test
    chunks: list[str] = []
    pos = 0
    while pos < len(payload):
        step = rng.randint(1, 6)
        chunks.append(payload[pos : pos + step])
        pos += step
    assert _parse_chunks(fmt_name, chunks, schemas=schemas) == one_shot


def test_json_families_qwen3_and_deepseek() -> None:
    payload_q = '<tool_call>{"name": "get_weather", "arguments": {"city": "Hue"}}</tool_call>'
    _, calls_q = _parse_chunks("qwen3_json", [payload_q])
    assert calls_q == [("get_weather", {"city": "Hue"}, True)]

    payload_d = (
        "<|tool_calls_begin|><|tool_call_begin|>"
        '{"name": "get_weather", "arguments": {"city": "Hue"}}'
        "<|tool_call_end|><|tool_calls_end|>"
    )
    _, calls_d = _parse_chunks("deepseek_json", [payload_d])
    assert calls_d == [("get_weather", {"city": "Hue"}, True)]


def test_suggest_tool_call_parser_detects_new_families() -> None:
    assert suggest_tool_call_parser("<\uff5cDSML\uff5cfunction_calls>") == "deepseek_dsml"
    assert suggest_tool_call_parser("<minimax:tool_call><invoke name=") == "minimax_m2"
    assert suggest_tool_call_parser("<arg_key>a</arg_key><arg_value>b</arg_value>") == "glm_xml"
    assert suggest_tool_call_parser("<tool_call><function=f><parameter=k>") == "qwen3_coder"
    assert suggest_tool_call_parser('<tool_call>{"name":') == "hermes"
    assert suggest_tool_call_parser("<|tool_call_begin|>") == "deepseek_json"
