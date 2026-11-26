"""Normalized tool-call grammar generation shared across JSON and XML/tag families."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, cast

from ayaka.core.constraints import (
    ConstraintKind,
    StructuralTagItem,
    StructuralTagSpec,
    StructuredParams,
    ToolSchema,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

    from ayaka.entrypoints.parsers.formats import ToolCallFormat
    from ayaka.entrypoints.parsers.output import OutputFormat

__all__ = [
    "GrammarCoverage",
    "OutputGrammarContext",
    "ToolStrictLevel",
    "build_tool_constraint",
]


class ToolStrictLevel(StrEnum):
    """Server-wide policy controlling how strictly tool arguments are constrained."""

    AUTO = "auto"
    FUNCTION = "function"
    PARAMETER = "parameter"


class GrammarCoverage(StrEnum):
    """Portion of model output governed by a tool constraint."""

    FROM_TOKEN_ZERO = "from_token_zero"  # noqa: S105  # enum identifier, not a credential
    FINAL_OUTPUT_ONLY = "final_output_only"


@dataclass(frozen=True, slots=True)
class OutputGrammarContext:
    """Request and model context needed to build a synchronized tool constraint."""

    output_format: OutputFormat
    tools: tuple[ToolSchema, ...]
    tool_choice: str = "auto"
    strict_request: bool = False
    tool_strict_level: ToolStrictLevel = ToolStrictLevel.AUTO
    parallel_tool_calls: bool = True
    start_after: int | None = None
    reasoning_open: int | None = None
    coverage: GrammarCoverage = GrammarCoverage.FINAL_OUTPUT_ONLY


def _effective_strict(level: ToolStrictLevel, request_strict: bool) -> bool:
    if level is ToolStrictLevel.PARAMETER:
        return True
    if level is ToolStrictLevel.FUNCTION:
        return False
    return request_strict


def _lark_literal(text: str) -> str:
    return json.dumps(text, ensure_ascii=False)


def _param_value_rule(prop: Mapping[str, object]) -> str:
    ptype = prop.get("type")
    if ptype == "integer":
        return r"/-?\d+/"
    if ptype == "number":
        return r"/-?\d+(\.\d+)?([eE][+-]?\d+)?/"
    if ptype == "boolean":
        return '("true" | "false")'
    return r"/[^<]*/"


def _tool_param_pairs(tool: ToolSchema) -> list[tuple[str, str, bool]]:
    props_raw = tool.parameters.get("properties")
    req_raw = tool.parameters.get("required")
    required: set[str] = (
        {r for r in cast("list[object]", req_raw) if isinstance(r, str)}
        if isinstance(req_raw, list)
        else set()
    )
    if not isinstance(props_raw, dict) or not props_raw:
        return []
    pairs: list[tuple[str, str, bool]] = []
    for name, raw_prop in cast("Mapping[str, object]", props_raw).items():
        if isinstance(raw_prop, dict):
            rule = _param_value_rule(cast("Mapping[str, object]", raw_prop))
            pairs.append((name, rule, name in required))
    return pairs


def _xml_tool_tag_item(fmt: ToolCallFormat, tool: ToolSchema, *, strict: bool) -> StructuralTagItem:
    open_outer = fmt.open or "<tool_call>"
    close_outer = fmt.close or "</tool_call>"
    params = _tool_param_pairs(tool) if strict else []

    if fmt.name in ("qwen3_coder", "mimo"):
        begin = f"{open_outer}<function={tool.name}>"
        end = f"</function>{close_outer}"
        if params:
            parts: list[str] = []
            for idx, (pname, _, req) in enumerate(params):
                safe = re.sub(r"[^a-zA-Z0-9_]", "_", pname)
                item = f"p_{idx}_{safe}"
                parts.append(item if req else f"[{item}]")
            defs = [
                f"p_{idx}_{re.sub(r'[^a-zA-Z0-9_]', '_', pname)}: "
                f'{_lark_literal(f"<parameter={pname}>")} WS? {prule} WS? "</parameter>"'
                for idx, (pname, prule, _) in enumerate(params)
            ]
            grammar = (
                f"start: WS? {' WS? '.join(parts)} WS?\n"
                + "\n".join(defs)
                + "\nWS: /[ \\t\\r\\n]+/"
            )
        else:
            grammar = (
                "start: (WS | param)*\n"
                'param: "<parameter=" NAME ">" VALUE "</parameter>"\n'
                "NAME: /[^>=\\s]+/\n"
                "VALUE: /[^<]*/\n"
                "WS: /[ \\t\\r\\n]+/"
            )
        return StructuralTagItem(begin=begin, grammar=grammar, end=end)

    if fmt.name == "glm_xml":
        begin = f"{open_outer}{tool.name}"
        end = close_outer
        if params:
            parts = [
                (
                    f"p_{idx}_{re.sub(r'[^a-zA-Z0-9_]', '_', pname)}"
                    if req
                    else f"[p_{idx}_{re.sub(r'[^a-zA-Z0-9_]', '_', pname)}]"
                )
                for idx, (pname, _, req) in enumerate(params)
            ]
            defs = [
                f"p_{idx}_{re.sub(r'[^a-zA-Z0-9_]', '_', pname)}: "
                f'"<arg_key>" {_lark_literal(pname)} "</arg_key>" WS? '
                f'"<arg_value>" {prule} "</arg_value>"'
                for idx, (pname, prule, _) in enumerate(params)
            ]
            grammar = (
                f"start: WS? {' WS? '.join(parts)} WS?\n"
                + "\n".join(defs)
                + "\nWS: /[ \\t\\r\\n]+/"
            )
        else:
            grammar = (
                "start: (WS | pair)*\n"
                'pair: "<arg_key>" TEXT "</arg_key>" WS? "<arg_value>" TEXT "</arg_value>"\n'
                "TEXT: /[^<]*/\n"
                "WS: /[ \\t\\r\\n]+/"
            )
        return StructuralTagItem(begin=begin, grammar=grammar, end=end)

    if fmt.name in ("minimax_m2", "minimax_m3"):
        begin = f'{open_outer}<invoke name="{tool.name}">'
        end = f"</invoke>{close_outer}"
        grammar = (
            "start: (WS | param)*\n"
            'param: "<parameter name=\\"" NAME "\\">" VALUE "</parameter>"\n'
            'NAME: /[^"]+/\n'
            "VALUE: /[^<]*/\n"
            "WS: /[ \\t\\r\\n]+/"
        )
        return StructuralTagItem(begin=begin, grammar=grammar, end=end)

    # deepseek_dsml
    dsml_open = "<\uff5cDSML\uff5cfunction_calls>"
    dsml_close = "</\uff5cDSML\uff5cfunction_calls>"
    begin = f'{dsml_open}<\uff5cDSML\uff5cinvoke name="{tool.name}">'
    end = f"</\uff5cDSML\uff5cinvoke>{dsml_close}"
    grammar = (
        "start: (WS | param)*\n"
        f'param: {_lark_literal("<\uff5cDSML\uff5cparameter name=")} "\\"" NAME "\\"" '
        f'(WS "string=\\"" ("true" | "false") "\\"")? ">" VALUE '
        f"{_lark_literal('</\uff5cDSML\uff5cparameter>')}\n"
        'NAME: /[^"]+/\n'
        "VALUE: /[^<]*/\n"
        "WS: /[ \\t\\r\\n]+/"
    )
    return StructuralTagItem(begin=begin, grammar=grammar, end=end)


def build_tool_constraint(
    ctx: OutputGrammarContext,
    *,
    spelled_open: tuple[int, ...] = (),
) -> StructuredParams | None:
    """Build a synchronized :class:`~ayaka.core.constraints.StructuredParams` for tools."""
    fmt = ctx.output_format.tool_calls
    if fmt is None or not ctx.tools:
        return None
    strict = _effective_strict(ctx.tool_strict_level, ctx.strict_request)
    chosen = (
        tuple(t for t in ctx.tools if t.name == ctx.tool_choice)
        if ctx.tool_choice not in ("auto", "required")
        else ctx.tools
    )
    if fmt.style == "xml":
        open_marker = fmt.open or "<tool_call>"
        items = tuple(_xml_tool_tag_item(fmt, tool, strict=strict) for tool in chosen)
        return StructuredParams(
            kind=ConstraintKind.STRUCTURAL_TAG,
            tools=ctx.tools,
            tool_choice=ctx.tool_choice,
            strict=strict,
            parallel=ctx.parallel_tool_calls,
            start_after=ctx.start_after,
            reasoning_open=ctx.reasoning_open,
            structural_tag=StructuralTagSpec(
                tags=items,
                triggers=(open_marker,),
                at_least_one=ctx.tool_choice != "auto",
                stop_after_first=not ctx.parallel_tool_calls,
            ),
        )
    return StructuredParams(
        kind=ConstraintKind.TOOL_CALL,
        tools=ctx.tools,
        syntax=ctx.output_format.call_syntax(spelled_open),
        tool_choice=ctx.tool_choice,
        strict=strict,
        parallel=ctx.parallel_tool_calls,
        start_after=ctx.start_after,
        reasoning_open=ctx.reasoning_open,
    )
