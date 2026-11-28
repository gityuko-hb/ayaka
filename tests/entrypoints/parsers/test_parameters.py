"""Tests for ToolSchemas typed coercion and output grammar constraint generation."""

from __future__ import annotations

import pytest

from ayaka.core.constraints import ConstraintKind, ToolSchema
from ayaka.entrypoints.parsers.output import OutputFormat
from ayaka.entrypoints.parsers.output_grammar import (
    GrammarCoverage,
    OutputGrammarContext,
    ToolStrictLevel,
    build_tool_constraint,
)
from ayaka.entrypoints.parsers.parameters import ToolSchemas

pytestmark = pytest.mark.unit


def test_tool_schemas_coerces_typed_fields_and_preserves_strings() -> None:
    schemas = ToolSchemas.from_tools(
        [
            ToolSchema(
                name="search",
                parameters={
                    "type": "object",
                    "properties": {
                        "limit": {"type": "integer"},
                        "threshold": {"type": "number"},
                        "exact": {"type": "boolean"},
                        "zip_code": {"type": "string"},
                        "tags": {"type": "array"},
                        "meta": {"type": "object"},
                        "nullable_count": {"anyOf": [{"type": "integer"}, {"type": "null"}]},
                    },
                },
            )
        ]
    )

    coerced = schemas.coerce_arguments(
        "search",
        {
            "limit": "5",
            "threshold": "0.75",
            "exact": "true",
            "zip_code": "0123",
            "tags": '["a", "b"]',
            "meta": '{"k": 1}',
            "nullable_count": "null",
            "unknown": "42",
        },
    )
    assert coerced == {
        "limit": 5,
        "threshold": 0.75,
        "exact": True,
        "zip_code": "0123",
        "tags": ["a", "b"],
        "meta": {"k": 1},
        "nullable_count": None,
        "unknown": "42",
    }
    assert isinstance(coerced["zip_code"], str)
    # Invalid conversion falls back cleanly to the raw string.
    assert schemas.coerce_value("search", "limit", "not-an-int") == "not-an-int"


def test_output_grammar_honors_strict_level_and_xml_structural_tags() -> None:
    tools = (
        ToolSchema(
            name="get_weather",
            parameters={
                "type": "object",
                "properties": {"city": {"type": "string"}, "days": {"type": "integer"}},
                "required": ["city"],
            },
        ),
    )

    json_fmt = OutputFormat.resolve(reasoning="think", tool_calls="hermes")
    fn_only = build_tool_constraint(
        OutputGrammarContext(
            output_format=json_fmt,
            tools=tools,
            strict_request=True,
            tool_strict_level=ToolStrictLevel.FUNCTION,
            coverage=GrammarCoverage.FINAL_OUTPUT_ONLY,
        ),
        spelled_open=(10,),
    )
    assert fn_only is not None
    assert fn_only.kind is ConstraintKind.TOOL_CALL
    assert fn_only.strict is False

    param_strict = build_tool_constraint(
        OutputGrammarContext(
            output_format=json_fmt,
            tools=tools,
            strict_request=False,
            tool_strict_level=ToolStrictLevel.PARAMETER,
        ),
        spelled_open=(10,),
    )
    assert param_strict is not None
    assert param_strict.strict is True

    xml_fmt = OutputFormat.resolve(reasoning="think", tool_calls="qwen3_coder")
    xml_constraint = build_tool_constraint(
        OutputGrammarContext(
            output_format=xml_fmt,
            tools=tools,
            tool_choice="required",
            parallel_tool_calls=False,
        )
    )
    assert xml_constraint is not None
    assert xml_constraint.kind is ConstraintKind.STRUCTURAL_TAG
    assert xml_constraint.structural_tag is not None
    assert xml_constraint.structural_tag.at_least_one is True
    assert xml_constraint.structural_tag.stop_after_first is True
    assert xml_constraint.structural_tag.tags[0].begin == "<tool_call><function=get_weather>"
