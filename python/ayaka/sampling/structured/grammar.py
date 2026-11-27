"""From :class:`~ayaka.core.constraints.StructuredParams` to the text of a Lark grammar."""

from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING, Final

from ayaka.core.constraints import ConstraintKind

if TYPE_CHECKING:
    from collections.abc import Mapping

    from ayaka.core.constraints import StructuralTagSpec, StructuredParams, ToolSchema

__all__ = [
    "JSON_SEPARATORS",
    "call_schema",
    "compact_schema",
    "grammar_text",
    "prefix_lark_grammar",
]

_RULE_DEF_RE = re.compile(r"(?m)^([a-zA-Z_][a-zA-Z0-9_]*)\s*:")


def _json(value: object) -> str:
    return json.dumps(value, separators=(",", ":"), ensure_ascii=False)


#: The separators of constrained JSON: the layout ``json.dumps`` writes by default, and the one
#: models write unconstrained. No other whitespace is allowed, so a model has none to spend its
#: budget on. With no space after a colon (``{"name":"``) Qwen3-8B reads the next token as the
#: start of a string and writes placeholders such as ``":name:"`` (see #145).
JSON_SEPARATORS: Final[tuple[str, str]] = (", ", ": ")


def compact_schema(schema: Mapping[str, object], *, lenient: bool = False) -> Mapping[str, object]:
    """Return ``schema`` with llguidance options for the single-line :data:`JSON_SEPARATORS` layout.

    A schema that carries its own ``x-guidance`` options keeps them. ``lenient`` makes llguidance
    skip the keywords it does not implement instead of refusing the schema.
    """
    if "x-guidance" in schema:
        return schema
    item, key = JSON_SEPARATORS
    options: dict[str, object] = {
        "whitespace_flexible": False,
        "item_separator": item,
        "key_separator": key,
    }
    if lenient:
        options["lenient"] = True
    return {**schema, "x-guidance": options}


def call_schema(
    tools: tuple[ToolSchema, ...],
    *,
    strict: bool,
    only: str | None = None,
    arguments_key: str = "arguments",
) -> Mapping[str, object]:
    """Build the JSON schema of a single tool call payload."""
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


def prefix_lark_grammar(text: str, prefix: str) -> tuple[str, str]:
    """Rename top-level rules/terminals in ``text`` with ``prefix``; return ``(entry, text)``."""
    names = [m.group(1) for m in _RULE_DEF_RE.finditer(text)]
    if not names:
        return "start", text
    mapping = {name: f"{prefix}{name}" for name in dict.fromkeys(names)}
    pattern = re.compile(
        r'(?<!\\)"(?:\\.|[^"\\])*"|(?<!\\)/(?:\\.|[^/\\])*/|\b('
        + "|".join(re.escape(n) for n in sorted(mapping, key=len, reverse=True))
        + r")\b"
    )

    def replace(match: re.Match[str]) -> str:
        ident = match.group(1)
        if ident is None:
            return match.group(0)
        return mapping.get(ident, ident)

    entry = mapping.get("start") or mapping.get("root") or next(iter(mapping.values()))
    return entry, pattern.sub(replace, text)


def _lark_marker(text: str, special_tokens: Mapping[str, int] | None) -> str:
    """Render ``text`` in Lark, substituting special-token markers with ``<[id]>`` references."""
    if not text:
        return ""
    if not special_tokens:
        return _json(text)
    parts: list[str] = []
    pos = 0
    ordered = sorted(special_tokens, key=len, reverse=True)
    while pos < len(text):
        matched = next((tok for tok in ordered if tok and text.startswith(tok, pos)), None)
        if matched is not None:
            parts.append(f"<[{special_tokens[matched]}]>")
            pos += len(matched)
            continue
        end = pos + 1
        while end < len(text) and not any(tok and text.startswith(tok, end) for tok in ordered):
            end += 1
        parts.append(_json(text[pos:end]))
        pos = end
    return " ".join(parts)


def _structural_tag_grammar(
    spec: StructuralTagSpec, special_tokens: Mapping[str, int] | None = None
) -> str:
    tag_rules: list[str] = []
    extra_defs: list[str] = []
    for idx, item in enumerate(spec.tags):
        begin_lit = _lark_marker(item.begin, special_tokens)
        end_lit = _lark_marker(item.end, special_tokens) if item.end else ""
        if item.grammar is not None:
            entry, body_defs = prefix_lark_grammar(item.grammar.strip(), f"st{idx}_")
            extra_defs.append(body_defs)
            body_ref = entry
        else:
            schema = item.schema if item.schema is not None else {"type": "object"}
            extra_defs.append(f"st{idx}_json: %json {_json(compact_schema(schema))}")
            body_ref = f"st{idx}_json"
        end_part = f" {end_lit}" if end_lit else ""
        tag_rules.append(f"st_tag_{idx}: {begin_lit} {body_ref}{end_part}")

    any_tag = " | ".join(f"st_tag_{idx}" for idx in range(len(spec.tags)))
    tag_union = f"st_any_tag: {any_tag}"
    if spec.at_least_one:
        start_rule = (
            "start: st_ws? st_any_tag st_ws?"
            if spec.stop_after_first
            else "start: st_ws? st_any_tag (st_ws? st_any_tag)* st_ws?"
        )
    elif spec.stop_after_first:
        start_rule = "start: st_text? (st_any_tag st_text?)?"
    else:
        start_rule = "start: st_text? (st_any_tag st_text?)*"

    lines = [
        start_rule,
        tag_union,
        *tag_rules,
        *extra_defs,
        "st_text: /[^<]+/",
        "st_ws: /[ \\t\\r\\n]+/",
    ]
    return "\n".join(lines) + "\n"


def grammar_text(
    params: StructuredParams,
    *,
    special_tokens: Mapping[str, int] | None = None,
) -> str:
    """Return the Lark grammar text corresponding to ``params``."""
    if params.kind is ConstraintKind.JSON_SCHEMA:
        assert params.schema is not None  # noqa: S101
        return f"start: %json {_json(compact_schema(params.schema))}\n"
    if params.kind is ConstraintKind.JSON_OBJECT:
        return f"start: %json {_json(compact_schema({'type': 'object'}))}\n"
    if params.kind is ConstraintKind.REGEX:
        assert params.pattern is not None  # noqa: S101
        escaped = params.pattern.replace("/", r"\/")
        return f"start: /{escaped}/\n"
    if params.kind is ConstraintKind.CHOICE:
        options = " | ".join(_json(choice) for choice in params.choices)
        return f"start: {options}\n"
    if params.kind is ConstraintKind.GRAMMAR:
        assert params.grammar is not None  # noqa: S101
        return params.grammar.rstrip() + "\n"
    if params.kind is ConstraintKind.STRUCTURAL_TAG:
        assert params.structural_tag is not None  # noqa: S101
        return _structural_tag_grammar(params.structural_tag, special_tokens=special_tokens)

    syntax = params.syntax
    assert syntax is not None  # noqa: S101
    only = params.tool_choice if params.tool_choice not in ("auto", "required") else None
    optional_bare = syntax.bare and params.tool_choice == "auto"
    schema = call_schema(
        params.tools,
        strict=params.strict or optional_bare,
        only=only,
        arguments_key=syntax.arguments_key,
    )
    if syntax.bare:
        more = f" ({_json(syntax.separator)} call)*" if params.parallel and syntax.separator else ""
        body = _json(compact_schema(schema, lenient=optional_bare and not params.strict))
        if optional_bare:
            return f"start: BLANK? call{more}\ncall: %json {body}\nBLANK: /[ \\t\\r\\n]+/\n"
        return f"start: call{more}\ncall: %json {body}\n"
    token = syntax.close_token
    close = f"<[{token}]>" if token is not None else _json(syntax.close)
    return (
        "start: NEWLINE? call NEWLINE? close\n"
        f"call: %json {_json(compact_schema(schema))}\n"
        f"close: {close}\n"
        'NEWLINE: "\\n"\n'
    )
