"""Prompt rendering layer: Jinja thinking policy detection and request-scoped prompts."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ayaka.utils.lazy_exports import build_lazy_exports

if TYPE_CHECKING:
    from ayaka.entrypoints.render.models.hooks import ModelRenderHook, select_render_hook
    from ayaka.entrypoints.render.reasoning import (
        ReasoningEffort,
        ResolvedThinking,
        ThinkingRequest,
        parse_reasoning_effort,
        resolve_thinking,
    )
    from ayaka.entrypoints.render.selection import PromptRenderer, RenderedChat, RenderRequest
    from ayaka.entrypoints.render.thinking import (
        AlwaysThinking,
        NativeToggle,
        ReasoningEffortPolicy,
        TemplateToggle,
        ThinkingPolicy,
        UnknownThinkingPolicy,
        detect_thinking_policy,
    )

__all__ = [
    "AlwaysThinking",
    "ModelRenderHook",
    "NativeToggle",
    "PromptRenderer",
    "ReasoningEffort",
    "ReasoningEffortPolicy",
    "RenderRequest",
    "RenderedChat",
    "ResolvedThinking",
    "TemplateToggle",
    "ThinkingPolicy",
    "ThinkingRequest",
    "UnknownThinkingPolicy",
    "detect_thinking_policy",
    "parse_reasoning_effort",
    "resolve_thinking",
    "select_render_hook",
]

_EXPORTS = {
    "models.hooks": ("ModelRenderHook", "select_render_hook"),
    "reasoning": (
        "ReasoningEffort",
        "ResolvedThinking",
        "ThinkingRequest",
        "parse_reasoning_effort",
        "resolve_thinking",
    ),
    "selection": ("PromptRenderer", "RenderRequest", "RenderedChat"),
    "thinking": (
        "AlwaysThinking",
        "NativeToggle",
        "ReasoningEffortPolicy",
        "TemplateToggle",
        "ThinkingPolicy",
        "UnknownThinkingPolicy",
        "detect_thinking_policy",
    ),
}

__getattr__, __dir__ = build_lazy_exports(__name__, _EXPORTS, globals())
