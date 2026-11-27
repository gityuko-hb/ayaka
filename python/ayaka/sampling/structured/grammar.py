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

A format without markers (Llama 3.x, ``llama3_json``) is only constrained when the call is forced:
the grammar is the answer itself, one object, or with ``parallel`` several joined by the format's
separator. The key of the arguments comes from the format (``parameters`` for Llama).
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Final

from ayaka.core.constraints import ConstraintKind

if TYPE_CHECKING:
    from collections.abc import Mapping

    from ayaka.core.constraints import StructuredParams, ToolSchema

__all__ = ["call_schema", "grammar_text"]


def _json(value: object) -> str:
    # Never sort keys: the grammar writes properties in the schema's order, so sorting would
    # put ``arguments`` before ``name`` and reorder every client schema.
    return json.dumps(value, separators=(",", ":"), ensure_ascii=False)


def _compact(schema: Mapping[str, object], *, lenient: bool = False) -> Mapping[str, object]:
    """The schema with a fixed single-line JSON layout, unless the request chose its own.

    The layout is the one ``json.dumps`` writes by default: ``", "`` between items, ``": "``
    between a key and its value, and no other whitespace. It leaves the model no free whitespace
    to spend its tokens on, and it is the layout models write unconstrained (and the one the
    Hermes tool-call template shows). With no space at all after the colon (``{"name":"``), Qwen3-8B
    reads the next token as the start of a string and writes placeholders such as ``":name:"``:
    11 of 18 sampled answers did, and one never closed its string. ``lenient`` makes llguidance
    skip the keywords it does not implement (an unknown ``format``) instead of refusing the
    schema: for a constraint the request did not ask for.
    """
    if "x-guidance" in schema:
        return schema
    options: dict[str, object] = dict(_LAYOUT)
    if lenient:
        options["lenient"] = True
    return {**schema, "x-guidance": options}


#: llguidance options of the JSON layout every constraint without its own ``x-guidance`` uses.
_LAYOUT: Final[Mapping[str, object]] = {
    "whitespace_flexible": False,
    "item_separator": ", ",
    "key_separator": ": ",
}


def call_schema(
    tools: tuple[ToolSchema, ...],
    *,
    strict: bool,
    only: str | None = None,
    arguments_key: str = "arguments",
) -> Mapping[str, object]:
    """The JSON schema of one call body: a tool name and its arguments.

    Args:
        tools: The functions the request offers.
        strict: Whether the arguments follow each tool's schema.
        only: The one tool a request forces, or ``None`` for any of them.
        arguments_key: The key of the arguments in the model's format (``parameters`` for
            ``llama3_json``).
    """
    chosen = [tool for tool in tools if only is None or tool.name == only]
    if not strict:
        return {
            "type": "object",
            "properties": {
                "name": {"enum": [tool.name for tool in chosen]},
                arguments_key: {"type": "object"},
            },
            "required": ["name", arguments_key],
            "additionalProperties": False,
        }
    variants = [
        {
            "type": "object",
            "properties": {
                "name": {"const": tool.name},
                arguments_key: dict(tool.parameters) or {"type": "object"},
            },
            "required": ["name", arguments_key],
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
    # An optional call in a format without markers starts only when the answer opens with `{`
    # (the matcher decides): such an answer is nothing but calls, which are useless to a tool
    # unless their arguments follow its schema, so they do even when the request is not strict.
    optional_bare = syntax.bare and params.tool_choice == "auto"
    schema = call_schema(
        params.tools,
        strict=params.strict or optional_bare,
        only=only,
        arguments_key=syntax.arguments_key,
    )
    if syntax.bare:
        more = f" ({_json(syntax.separator)} call)*" if params.parallel and syntax.separator else ""
        body = _json(_compact(schema, lenient=optional_bare and not params.strict))
        if optional_bare:
            # The token that opens the call may carry the whitespace before it (`▁{"`).
            return f"start: BLANK? call{more}\ncall: %json {body}\nBLANK: /[ \\t\\r\\n]+/\n"
        return f"start: call{more}\ncall: %json {body}\n"
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
