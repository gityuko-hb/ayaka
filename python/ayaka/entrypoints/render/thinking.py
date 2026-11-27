"""Jinja chat-template thinking policy detection, in pure Python.

Inspects a model's chat template without importing ``jinja2`` or ``torch``. Comments,
``{% raw %}...{% endraw %}`` blocks, and string literals inside Jinja tags are stripped before
scanning identifiers so documentation text or literal tag strings never cause false positives.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Final

__all__ = [
    "AlwaysThinking",
    "NativeToggle",
    "ReasoningEffortPolicy",
    "TemplateToggle",
    "ThinkingPolicy",
    "UnknownThinkingPolicy",
    "detect_thinking_policy",
]


@dataclass(frozen=True, slots=True)
class AlwaysThinking:
    """The template always opens reasoning or the model always reasons without a toggle."""


@dataclass(frozen=True, slots=True)
class TemplateToggle:
    """The template exposes a boolean flag (``enable_thinking`` or ``thinking``)."""

    key: str
    default_enabled: bool = True


@dataclass(frozen=True, slots=True)
class NativeToggle:
    """Reasoning is controlled natively by template conventions or named tool constraints."""

    default_enabled: bool = True
    named_tool_disables: bool = False


@dataclass(frozen=True, slots=True)
class ReasoningEffortPolicy:
    """The template consumes a ``reasoning_effort`` level string."""

    levels: tuple[str, ...] = ("low", "medium", "high")
    default_level: str = "medium"


@dataclass(frozen=True, slots=True)
class UnknownThinkingPolicy:
    """No recognizable thinking control was found in the template."""


type ThinkingPolicy = (
    AlwaysThinking | TemplateToggle | NativeToggle | ReasoningEffortPolicy | UnknownThinkingPolicy
)

_COMMENT_RE: Final = re.compile(r"\{#.*?#\}", re.DOTALL)
_RAW_RE: Final = re.compile(r"\{%-?\s*raw\s*-?%\}.*?\{%-?\s*endraw\s*-?%\}", re.DOTALL)
_STMT_TAG_RE: Final = re.compile(r"\{%-?.*?-?%\}", re.DOTALL)
_TAG_RE: Final = re.compile(r"(\{%-?.*?-?%\}|\{\{.*?\}\})", re.DOTALL)
_EFFORT_RE: Final = re.compile(r"(?<![.\w'\"])reasoning_effort(?![\w'\"])")
_ENABLE_THINKING_RE: Final = re.compile(r"(?<![.\w'\"])enable_thinking(?![\w'\"])")
_THINKING_VAR_RE: Final = re.compile(r"(?<![.\w'\"])thinking(?![\w'\"])")
_LOOP_BIND_RE: Final = re.compile(r"\b(?:for|set)\s+thinking\b")
_THINK_TAG_RE: Final = re.compile(r"<think>")
_CLOSE_THINK_TAG_RE: Final = re.compile(r"</think>")
_DEFAULT_FALSE_RE: Final = re.compile(
    r"enable_thinking\s*\|\s*default\(\s*false\b|"
    r"enable_thinking\s+is\s+defined\s+and\s+enable_thinking\b(?!\s+is\s+false|\s*==\s*false)"
)
_DEFAULT_TRUE_RE: Final = re.compile(
    r"enable_thinking\s*\|\s*default\(\s*true\b|"
    r"enable_thinking\s+is\s+defined\s+and\s+(?:not\s+enable_thinking|enable_thinking\s+is\s+false|enable_thinking\s*==\s*false)|"
    r"enable_thinking\s+is\s+not\s+defined"
)


def _strip_comments_and_raw(template: str) -> str:
    """Remove Jinja comments and ``raw`` blocks while preserving surrounding structure."""
    without_comments = _COMMENT_RE.sub("", template)
    return _RAW_RE.sub("", without_comments)


def _strip_string_literals(tag: str) -> str:
    """Replace string literals inside a Jinja tag so quoted words are not read as variables."""
    out: list[str] = []
    i = 0
    length = len(tag)
    while i < length:
        char = tag[i]
        if char in ("'", '"'):
            quote = char
            out.append(quote)
            i += 1
            while i < length:
                current = tag[i]
                if current == "\\":
                    i += 2
                    continue
                if current == quote:
                    out.append(quote)
                    i += 1
                    break
                i += 1
            continue
        out.append(char)
        i += 1
    return "".join(out)


def _jinja_control_source(stripped_template: str) -> str:
    """Join all Jinja statement/expression tags with their string literals emptied."""
    parts = [
        _strip_string_literals(match.group(0)) for match in _TAG_RE.finditer(stripped_template)
    ]
    return "\n".join(parts)


def _detect_enable_thinking_default(control: str) -> bool:
    """Infer whether ``enable_thinking`` defaults to enabled when omitted by the caller."""
    if _DEFAULT_TRUE_RE.search(control):
        return True
    return not _DEFAULT_FALSE_RE.search(control)


def _detect_thinking_var_default(control: str) -> bool:
    """Infer whether ``thinking`` defaults to enabled when omitted by the caller."""
    if re.search(r"thinking\s*\|\s*default\(\s*true\b", control):
        return True
    if re.search(r"thinking\s+is\s+defined\s+and\s+thinking\b(?!\s+is\s+false)", control):
        return False
    if re.search(r"thinking\s*\|\s*default\(\s*false\b", control):
        return False
    return not re.search(r"thinking\s+is\s+true\b|\bif\s+thinking\b", control)


def _detect_effort_levels(stripped_template: str) -> tuple[str, ...]:
    levels = [
        level
        for level in ("none", "minimal", "low", "medium", "high", "max")
        if re.search(rf"""['"]{level}['"]""", stripped_template)
    ]
    return tuple(levels) if levels else ("low", "medium", "high")


def _always_opens_think(stripped_template: str) -> bool:
    """Whether the generation prompt unconditionally ends with an unclosed ``<think>``."""
    # Ignore Jinja statement tags (e.g. ``{% set ... %}``) so only emitted output counts.
    emitted = _STMT_TAG_RE.sub("", stripped_template)
    last_open = -1
    for match in _THINK_TAG_RE.finditer(emitted):
        last_open = match.start()
    if last_open < 0:
        return False
    last_close = -1
    for match in _CLOSE_THINK_TAG_RE.finditer(emitted):
        last_close = match.start()
    return last_open > last_close


def detect_thinking_policy(
    chat_template: str | None, *, model_hint: str | None = None
) -> ThinkingPolicy:
    """Inspect a Jinja chat template and classify how thinking is controlled."""
    if not chat_template:
        return UnknownThinkingPolicy()

    stripped = _strip_comments_and_raw(chat_template)
    control = _jinja_control_source(stripped)

    if _EFFORT_RE.search(control):
        levels = _detect_effort_levels(stripped)
        default_level = "medium" if "medium" in levels else levels[0]
        return ReasoningEffortPolicy(levels=levels, default_level=default_level)

    if _ENABLE_THINKING_RE.search(control):
        return TemplateToggle(
            key="enable_thinking",
            default_enabled=_detect_enable_thinking_default(control),
        )

    if _THINKING_VAR_RE.search(control) and not _LOOP_BIND_RE.search(control):
        return TemplateToggle(
            key="thinking",
            default_enabled=_detect_thinking_var_default(control),
        )

    if _always_opens_think(stripped):
        return AlwaysThinking()

    hint = (model_hint or "").lower()
    if "glm" in hint and "thinking" in stripped.lower():
        return NativeToggle(default_enabled=True, named_tool_disables=True)

    return UnknownThinkingPolicy()
