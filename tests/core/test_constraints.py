"""The output-constraint contract: what a StructuredParams accepts and its canonical key."""

from __future__ import annotations

import pytest

from ayaka.core.constraints import (
    MAX_SCHEMA_BYTES,
    ConstraintKind,
    StructuredParams,
    ToolCallSyntax,
    ToolSchema,
)
from ayaka.core.errors import InvalidRequestError

pytestmark = pytest.mark.unit

_SYNTAX = ToolCallSyntax("<tool_call>", "</tool_call>", 7)
_TOOL = ToolSchema("get_weather", {"type": "object"})


def test_a_json_schema_constraint_needs_a_schema() -> None:
    with pytest.raises(InvalidRequestError, match="needs a schema"):
        StructuredParams(kind=ConstraintKind.JSON_SCHEMA)


def test_a_tool_call_needs_tools_and_the_model_syntax() -> None:
    with pytest.raises(InvalidRequestError, match="at least one tool"):
        StructuredParams(kind=ConstraintKind.TOOL_CALL, syntax=_SYNTAX)
    with pytest.raises(InvalidRequestError, match="call syntax"):
        StructuredParams(kind=ConstraintKind.TOOL_CALL, tools=(_TOOL,))


def test_a_named_choice_must_be_one_of_the_tools() -> None:
    with pytest.raises(InvalidRequestError, match="not among the tools"):
        StructuredParams(
            kind=ConstraintKind.TOOL_CALL, tools=(_TOOL,), syntax=_SYNTAX, tool_choice="other"
        )


def test_only_an_auto_tool_call_is_lazy() -> None:
    lazy = StructuredParams(kind=ConstraintKind.TOOL_CALL, tools=(_TOOL,), syntax=_SYNTAX)
    forced = StructuredParams(
        kind=ConstraintKind.TOOL_CALL, tools=(_TOOL,), syntax=_SYNTAX, tool_choice="required"
    )

    assert lazy.lazy
    assert not forced.lazy
    assert not StructuredParams(kind=ConstraintKind.JSON_OBJECT).lazy


def test_the_canonical_key_follows_the_grammar_including_property_order() -> None:
    a = StructuredParams(kind=ConstraintKind.JSON_SCHEMA, schema={"a": 1, "b": 2})
    same = StructuredParams(kind=ConstraintKind.JSON_SCHEMA, schema={"a": 1, "b": 2})
    reordered = StructuredParams(kind=ConstraintKind.JSON_SCHEMA, schema={"b": 2, "a": 1})

    assert a.canonical() == same.canonical()
    # Properties are written in the schema's order, so the order is part of the grammar.
    assert a.canonical() != reordered.canonical()


def test_an_oversized_schema_is_refused() -> None:
    big = {"description": "x" * MAX_SCHEMA_BYTES}

    with pytest.raises(InvalidRequestError, match="at most"):
        StructuredParams(kind=ConstraintKind.JSON_SCHEMA, schema=big)


def test_a_tool_needs_a_name() -> None:
    with pytest.raises(InvalidRequestError, match="needs a name"):
        ToolSchema("")
