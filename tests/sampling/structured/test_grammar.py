"""Grammar text built from constraints; plain Python, no grammar library needed."""

from __future__ import annotations

import json

import pytest

from ayaka.core.constraints import ConstraintKind, StructuredParams, ToolCallSyntax, ToolSchema
from ayaka.sampling.structured.grammar import call_schema, grammar_text

pytestmark = pytest.mark.unit

_WEATHER = ToolSchema(
    "get_weather",
    {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
)
_TIME = ToolSchema("get_time", {})
_SYNTAX = ToolCallSyntax("<tool_call>", "</tool_call>", 260)


def _embedded(text: str, rule: str) -> dict[str, object]:
    line = next(line for line in text.splitlines() if line.startswith(f"{rule}: %json "))
    return json.loads(line.removeprefix(f"{rule}: %json "))


_LAYOUT = {"whitespace_flexible": False, "item_separator": ", ", "key_separator": ": "}


def test_a_json_schema_is_embedded_compact() -> None:
    params = StructuredParams(kind=ConstraintKind.JSON_SCHEMA, schema={"type": "integer"})

    schema = _embedded(grammar_text(params), "start")

    assert schema == {"type": "integer", "x-guidance": _LAYOUT}


def test_a_schema_with_its_own_options_keeps_them() -> None:
    own = {"type": "object", "x-guidance": {"whitespace_flexible": True}}
    params = StructuredParams(kind=ConstraintKind.JSON_SCHEMA, schema=own)

    assert _embedded(grammar_text(params), "start") == own


def test_a_json_object_is_any_object() -> None:
    params = StructuredParams(kind=ConstraintKind.JSON_OBJECT)

    assert _embedded(grammar_text(params), "start")["type"] == "object"


def test_a_call_body_without_strict_names_a_tool_and_takes_any_object() -> None:
    schema = call_schema((_WEATHER, _TIME), strict=False)

    assert schema["properties"] == {
        "name": {"enum": ["get_weather", "get_time"]},
        "arguments": {"type": "object"},
    }


def test_a_strict_call_body_follows_each_tool_schema() -> None:
    schema = call_schema((_WEATHER, _TIME), strict=True)

    variants = schema["anyOf"]
    assert isinstance(variants, list)
    assert variants[0]["properties"]["arguments"] == _WEATHER.parameters  # type: ignore[index]
    assert variants[1]["properties"]["arguments"] == {"type": "object"}  # type: ignore[index]


def test_a_named_choice_keeps_only_that_tool() -> None:
    schema = call_schema((_WEATHER, _TIME), strict=False, only="get_time")

    assert schema["properties"]["name"] == {"enum": ["get_time"]}  # type: ignore[index]


def test_a_tool_call_grammar_ends_with_the_closing_marker() -> None:
    params = StructuredParams(
        kind=ConstraintKind.TOOL_CALL, tools=(_WEATHER,), syntax=_SYNTAX, strict=True
    )

    text = grammar_text(params)

    assert text.splitlines()[0] == "start: NEWLINE? call NEWLINE? close"
    assert 'close: "</tool_call>"' in text
    assert _embedded(text, "call")["properties"]["name"] == {"const": "get_weather"}  # type: ignore[index]


def test_a_one_token_closing_marker_is_demanded_by_id() -> None:
    syntax = ToolCallSyntax("<tool_call>", "</tool_call>", 260, 261)
    params = StructuredParams(kind=ConstraintKind.TOOL_CALL, tools=(_WEATHER,), syntax=syntax)

    assert "close: <[261]>" in grammar_text(params)


_BARE = ToolCallSyntax("", "", arguments_key="parameters", separator="; ")


@pytest.mark.parametrize("parallel", [False, True])
def test_a_call_without_markers_is_the_whole_answer_with_its_format_keys(parallel: bool) -> None:
    params = StructuredParams(
        kind=ConstraintKind.TOOL_CALL,
        tools=(ToolSchema("f"),),
        syntax=_BARE,
        tool_choice="required",
        parallel=parallel,
    )

    text = grammar_text(params)

    first, call = text.splitlines()
    assert first == ('start: call ("; " call)*' if parallel else "start: call")
    assert '"required":["name","parameters"]' in call


@pytest.mark.parametrize("strict", [False, True])
def test_an_optional_call_without_markers_follows_the_tool_schemas(strict: bool) -> None:
    """It starts on the answer's `{` (the matcher decides), blanks allowed in that first token;
    the arguments are typed even when the request is not strict, leniently then."""
    params = StructuredParams(
        kind=ConstraintKind.TOOL_CALL, tools=(_WEATHER,), syntax=_BARE, strict=strict
    )

    first, call, blank = grammar_text(params).splitlines()

    assert first == 'start: BLANK? call ("; " call)*'
    assert blank == r"BLANK: /[ \t\r\n]+/"
    schema = json.loads(call.removeprefix("call: %json "))
    assert schema["properties"]["parameters"]["properties"] == {"city": {"type": "string"}}
    assert schema["x-guidance"] == (_LAYOUT if strict else {**_LAYOUT, "lenient": True})


def test_an_optional_call_with_markers_keeps_loose_arguments_unless_strict() -> None:
    params = StructuredParams(kind=ConstraintKind.TOOL_CALL, tools=(_WEATHER,), syntax=_SYNTAX)

    call = grammar_text(params).splitlines()[1]

    assert '"arguments":{"type":"object"}' in call
