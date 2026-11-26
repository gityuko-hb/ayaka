"""Turn-preserving conversation truncation for opt-in over-long handling.

Drops the oldest non-system turns in atomic groups (keeping assistant tool
calls paired with their tool results) while always preserving the system prompt,
the tool definitions and the final turn group.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from ayaka.core.errors import ContextLengthError
from ayaka.entrypoints.conversation import Conversation
from ayaka.tokenizers.contracts import EncodeOptions

if TYPE_CHECKING:
    from collections.abc import Callable

    from ayaka.configs.tokenizer import EncodeResult
    from ayaka.entrypoints.conversation import Turn
    from ayaka.tokenizers.service import TokenizerService

__all__ = ["encode_truncated_chat", "group_turns", "truncate_conversation"]

_LOG = logging.getLogger("ayaka.serve")


def group_turns(turns: tuple[Turn, ...]) -> list[tuple[Turn, ...]]:
    """Group turns into atomic truncation units.

    An assistant turn that made tool calls is grouped with all immediately
    following ``tool`` result turns so a tool call is never separated from its
    results.
    """
    groups: list[tuple[Turn, ...]] = []
    index = 0
    total = len(turns)
    while index < total:
        current = turns[index]
        if current.role == "assistant" and current.tool_calls:
            end = index + 1
            while end < total and turns[end].role == "tool":
                end += 1
            groups.append(turns[index:end])
            index = end
        else:
            groups.append((current,))
            index += 1
    return groups


def truncate_conversation(
    conversation: Conversation,
    fits_within: Callable[[Conversation], tuple[bool, int]],
    *,
    max_prompt_tokens: int,
) -> tuple[Conversation, int]:
    """Drop oldest atomic turn groups until ``conversation`` fits ``max_prompt_tokens``.

    ``fits_within`` takes a candidate :class:`Conversation` and returns
    ``(fits, token_count)``. Returning ``(False, token_count)`` when the prompt
    exceeds the model or KV limit triggers dropping another oldest group.

    Raises:
        ContextLengthError: If even the system prompt, tools and final turn group
            exceed ``max_prompt_tokens``.
    """
    fits, initial_tokens = fits_within(conversation)
    if fits and initial_tokens <= max_prompt_tokens:
        return conversation, 0

    groups = group_turns(conversation.turns)
    if len(groups) <= 1:
        raise ContextLengthError(
            prompt_tokens=initial_tokens,
            output_tokens=1,
            max_context_tokens=max_prompt_tokens + 1,
        )

    dropped_turns = 0
    for start in range(1, len(groups)):
        dropped_turns += len(groups[start - 1])
        kept = tuple(turn for group in groups[start:] for turn in group)
        candidate = Conversation(
            system=conversation.system,
            turns=kept,
            tools=conversation.tools,
        )
        ok, count = fits_within(candidate)
        if ok and count <= max_prompt_tokens:
            return candidate, dropped_turns

    kept = groups[-1]
    candidate = Conversation(
        system=conversation.system,
        turns=kept,
        tools=conversation.tools,
    )
    _, count = fits_within(candidate)
    raise ContextLengthError(
        prompt_tokens=count,
        output_tokens=1,
        max_context_tokens=max_prompt_tokens + 1,
    )


def encode_truncated_chat(
    service: TokenizerService,
    conversation: Conversation,
    *,
    max_prompt_tokens: int,
    vouch: bool,
) -> EncodeResult:
    """Truncate oldest turn groups if needed and encode ``conversation`` synchronously."""

    def fits(candidate: Conversation) -> tuple[bool, int]:
        try:
            res = service.encode(
                candidate.template_data(), EncodeOptions(max_output_tokens=1), vouch=False
            )
        except ContextLengthError as exc:
            return False, exc.prompt_tokens
        n_tokens = len(res.token_ids)
        return n_tokens <= max_prompt_tokens, n_tokens

    trimmed, dropped = truncate_conversation(
        conversation, fits, max_prompt_tokens=max_prompt_tokens
    )
    if dropped:
        _LOG.info("truncated %d oldest turn(s) to fit %d prompt tokens", dropped, max_prompt_tokens)
    return service.encode(trimmed.template_data(), EncodeOptions(max_output_tokens=1), vouch=vouch)
