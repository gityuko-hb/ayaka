"""Request-scoped reasoning resolution from wire parameters and template policy."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING

from ayaka.entrypoints.render.thinking import (
    AlwaysThinking,
    NativeToggle,
    ReasoningEffortPolicy,
    TemplateToggle,
)

if TYPE_CHECKING:
    from ayaka.entrypoints.render.thinking import ThinkingPolicy

__all__ = [
    "ReasoningEffort",
    "ResolvedThinking",
    "ThinkingRequest",
    "parse_reasoning_effort",
    "resolve_thinking",
]


class ReasoningEffort(StrEnum):
    """Normalized reasoning effort levels across OpenAI and Anthropic requests."""

    NONE = "none"
    MINIMAL = "minimal"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    MAX = "max"


def parse_reasoning_effort(value: str, *, where: str = "reasoning_effort") -> ReasoningEffort:
    """Parse a reasoning effort string into :class:`ReasoningEffort`.

    Raises:
        ValueError: If ``value`` is not a recognized effort level.
    """
    try:
        return ReasoningEffort(value)
    except ValueError:
        allowed = ", ".join(level.value for level in ReasoningEffort)
        raise ValueError(f"{where}: must be one of {allowed}") from None


@dataclass(frozen=True, slots=True)
class ThinkingRequest:
    """Protocol-neutral thinking configuration parsed from a client request.

    Attributes:
        enabled: ``True`` to force thinking on, ``False`` to disable it, ``None`` for default.
        effort: Effort level from OpenAI ``reasoning_effort`` or ``reasoning.effort``.
        budget_tokens: Optional Anthropic thinking token budget.
        stream_reasoning: Whether to expose reasoning deltas to the client.
    """

    enabled: bool | None = None
    effort: ReasoningEffort | None = None
    budget_tokens: int | None = None
    stream_reasoning: bool = True

    @property
    def explicitly_disabled(self) -> bool:
        """Whether the client explicitly asked for no reasoning."""
        return self.enabled is False or self.effort is ReasoningEffort.NONE

    @property
    def explicitly_enabled(self) -> bool:
        """Whether the client explicitly asked for reasoning."""
        if self.enabled is True:
            return True
        return self.effort is not None and self.effort is not ReasoningEffort.NONE


@dataclass(frozen=True, slots=True)
class ResolvedThinking:
    """Template and parser thinking state resolved for one request.

    Attributes:
        enabled: Effective thinking toggle (``None`` when template policy is unknown and
            the client sent no explicit override).
        template_kwargs: Extra kwargs to forward to the Jinja chat template.
        starts_inside_reasoning: Whether the prompt already opens the reasoning block.
        force_reasoning: ``False`` disables reasoning parsing completely; ``True`` starts
            inside reasoning; ``None`` uses marker-driven detection.
        stream_reasoning: Whether reasoning deltas should be surfaced.
    """

    enabled: bool | None
    template_kwargs: tuple[tuple[str, object], ...] = ()
    starts_inside_reasoning: bool = False
    force_reasoning: bool | None = None
    stream_reasoning: bool = True


def _effort_for_policy(request: ThinkingRequest, policy: ReasoningEffortPolicy) -> str:
    if request.effort is not None:
        if request.effort.value in policy.levels:
            return request.effort.value
        if request.effort is ReasoningEffort.NONE and "none" in policy.levels:
            return "none"
        if request.effort is ReasoningEffort.MAX and "high" in policy.levels:
            return "high"
        if request.effort is ReasoningEffort.MINIMAL and "low" in policy.levels:
            return "low"
        return policy.default_level
    if request.enabled is False:
        return "none" if "none" in policy.levels else policy.levels[0]
    if request.enabled is True:
        return "high" if "high" in policy.levels else policy.default_level
    return policy.default_level


def resolve_thinking(
    policy: ThinkingPolicy,
    request: ThinkingRequest | None,
    *,
    named_tool_choice: bool = False,
) -> ResolvedThinking:
    """Combine a template's :data:`ThinkingPolicy` with a request's :class:`ThinkingRequest`."""
    req = request or ThinkingRequest()
    stream = req.stream_reasoning

    if isinstance(policy, TemplateToggle):
        if req.explicitly_disabled:
            return ResolvedThinking(
                enabled=False,
                template_kwargs=((policy.key, False),),
                starts_inside_reasoning=False,
                force_reasoning=False,
                stream_reasoning=stream,
            )
        if req.explicitly_enabled:
            return ResolvedThinking(
                enabled=True,
                template_kwargs=((policy.key, True),),
                starts_inside_reasoning=False,
                force_reasoning=None,
                stream_reasoning=stream,
            )
        if named_tool_choice:
            return ResolvedThinking(
                enabled=False,
                template_kwargs=((policy.key, False),),
                starts_inside_reasoning=False,
                force_reasoning=False,
                stream_reasoning=stream,
            )
        return ResolvedThinking(
            enabled=policy.default_enabled,
            template_kwargs=(),
            starts_inside_reasoning=False,
            force_reasoning=None if policy.default_enabled else False,
            stream_reasoning=stream,
        )

    if isinstance(policy, ReasoningEffortPolicy):
        if req.explicitly_disabled:
            return ResolvedThinking(
                enabled=False,
                template_kwargs=(("reasoning_effort", _effort_for_policy(req, policy)),),
                starts_inside_reasoning=False,
                force_reasoning=False,
                stream_reasoning=stream,
            )
        if req.explicitly_enabled:
            return ResolvedThinking(
                enabled=True,
                template_kwargs=(("reasoning_effort", _effort_for_policy(req, policy)),),
                starts_inside_reasoning=False,
                force_reasoning=None,
                stream_reasoning=stream,
            )
        return ResolvedThinking(
            enabled=policy.default_level != "none",
            template_kwargs=(),
            starts_inside_reasoning=False,
            force_reasoning=None,
            stream_reasoning=stream,
        )

    if isinstance(policy, NativeToggle):
        if req.explicitly_disabled or (
            named_tool_choice and policy.named_tool_disables and not req.explicitly_enabled
        ):
            return ResolvedThinking(
                enabled=False,
                template_kwargs=(),
                starts_inside_reasoning=False,
                force_reasoning=False,
                stream_reasoning=stream,
            )
        return ResolvedThinking(
            enabled=True if req.explicitly_enabled else policy.default_enabled,
            template_kwargs=(),
            starts_inside_reasoning=False,
            force_reasoning=None,
            stream_reasoning=stream,
        )

    if isinstance(policy, AlwaysThinking):
        if req.explicitly_disabled:
            return ResolvedThinking(
                enabled=False,
                template_kwargs=(),
                starts_inside_reasoning=False,
                force_reasoning=False,
                stream_reasoning=stream,
            )
        return ResolvedThinking(
            enabled=True,
            template_kwargs=(),
            starts_inside_reasoning=True,
            force_reasoning=True,
            stream_reasoning=stream,
        )

    if req.explicitly_disabled:
        return ResolvedThinking(
            enabled=False,
            template_kwargs=(),
            starts_inside_reasoning=False,
            force_reasoning=False,
            stream_reasoning=stream,
        )
    if req.explicitly_enabled:
        return ResolvedThinking(
            enabled=True,
            template_kwargs=(),
            starts_inside_reasoning=False,
            force_reasoning=None,
            stream_reasoning=stream,
        )
    return ResolvedThinking(
        enabled=None,
        template_kwargs=(),
        starts_inside_reasoning=False,
        force_reasoning=None,
        stream_reasoning=stream,
    )
