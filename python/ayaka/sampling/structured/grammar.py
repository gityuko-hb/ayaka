"""From :class:`~ayaka.core.constraints.StructuredParams` to the text of a grammar.

The grammars are written in llguidance's Lark dialect, where ``%json { ... }`` embeds a JSON
schema as one rule. Building the text needs no grammar library, so this module is plain Python
and its output is tested as text; :mod:`ayaka.sampling.structured.compiler` hands it to llguidance.

A tool call is the body the parser of :mod:`ayaka.entrypoints.parsers.tool_calls` reads: an
optional newline, one JSON object ``{"name": ..., "arguments": {...}}``, an optional newline, the
closing marker (demanded as its token when it is one). The opening marker is not part of the
grammar: the constraint starts once the model emitted it (a lazy call) or once the matcher made it
emit it (a forced call). With ``strict`` the arguments follow each tool's schema; without it they
are any JSON object.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from ayaka.core.constraints import ConstraintKind

if TYPE_CHECKING:
    from collections.abc import Mapping

    from ayaka.core.constraints import StructuredParams, ToolSchema

__all__ = ["call_schema", "grammar_text"]


def _json(value: object) -> str:
    # Never sort keys: the grammar writes properties in the schema's order, so sorting would
    # put ``arguments`` before ``name`` and reorder every client schema.
    return json.dumps(value, separators=(",", ":"), ensure_ascii=False)


def _compact(schema: Mapping[str, object]) -> Mapping[str, object]:
    """The schema with llguidance's compact JSON layout, unless the request chose its own.

    Without layout whitespace inside the JSON, a small model cannot spend its tokens on spaces.
    """
    if "x-guidance" in schema:
        return schema
    return {**schema, "x-guidance": {"whitespace_flexible": False}}


def call_schema(
    tools: tuple[ToolSchema, ...],
    *,
    strict: bool,
    only: str | None = None,
    name_key: str = "name",
    arguments_key: str = "arguments",
) -> Mapping[str, object]:
    """The JSON schema of one call body: a tool name and its arguments.

    Args:
        tools: The functions the request offers.
        strict: Whether the arguments follow each tool's schema.
        only: The one tool a request forces, or ``None`` for any of them.
        name_key: The key of the name in the model's format.
        arguments_key: The key of the arguments in the model's format (``parameters`` for
            ``llama3_json``).
    """
    chosen = [tool for tool in tools if only is None or tool.name == only]
    if not strict:
        return {
            "type": "object",
            "properties": {
                name_key: {"enum": [tool.name for tool in chosen]},
                arguments_key: {"type": "object"},
            },
            "required": [name_key, arguments_key],
            "additionalProperties": False,
        }
    variants = [
        {
            "type": "object",
            "properties": {
                name_key: {"const": tool.name},
                arguments_key: dict(tool.parameters) or {"type": "object"},
            },
            "required": [name_key, arguments_key],
            "additionalProperties": False,
        }
        for tool in chosen
    ]
    return variants[0] if len(variants) == 1 else {"anyOf": variants}


def grammar_text(params: StructuredParams) -> str:
    """The Lark text of the grammar a constraint follows once it is on."""
    if params.kind is ConstraintKind.JSON_SCHEMA:
        assert params.schema is not None  # noqa: S101  # StructuredParams requires it
        return f"start: %json {_json(_compact(params.schema))}\n"
    if params.kind is ConstraintKind.JSON_OBJECT:
        return f"start: %json {_json(_compact({'type': 'object'}))}\n"
    syntax = params.syntax
    assert syntax is not None  # noqa: S101  # StructuredParams requires it for a tool call
    only = params.tool_choice if params.tool_choice not in ("auto", "required") else None
    schema = call_schema(
        params.tools,
        strict=params.strict,
        only=only,
        name_key=syntax.name_key,
        arguments_key=syntax.arguments_key,
    )
    # A one-token marker is demanded as that token: the parser matches it by id.
    token = syntax.close_token
    close = f"<[{token}]>" if token is not None else _json(syntax.close)
    # One optional newline around the body, as Hermes writes it; unbounded layout would let a
    # model spend its whole budget on whitespace before the call.
    return (
        "start: NEWLINE? call NEWLINE? close\n"
        f"call: %json {_json(_compact(schema))}\n"
        f"close: {close}\n"
        'NEWLINE: "\\n"\n'
    )
