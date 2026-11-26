"""End-to-end chat prompt rendering, thinking resolution, and render fingerprinting."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from ayaka.core.errors import ContextLengthError, InvalidRequestError
from ayaka.entrypoints.render.models.hooks import select_render_hook
from ayaka.entrypoints.render.reasoning import ThinkingRequest, resolve_thinking
from ayaka.entrypoints.render.thinking import detect_thinking_policy
from ayaka.tokenizers.contracts import EncodeOptions, RenderedInput, TemplateData

if TYPE_CHECKING:
    from collections.abc import Mapping

    from ayaka.entrypoints.conversation import Conversation
    from ayaka.entrypoints.render.models.hooks import ModelRenderHook
    from ayaka.entrypoints.render.thinking import ThinkingPolicy
    from ayaka.tokenizers.service import TokenizerService

__all__ = ["PromptRenderer", "RenderRequest", "RenderedChat"]


@dataclass(frozen=True, slots=True)
class RenderRequest:
    """Inputs to render a conversation into prompt tokens and output interpretation metadata."""

    conversation: Conversation
    thinking: ThinkingRequest | None = None
    chat_template_kwargs: Mapping[str, object] = field(default_factory=dict[str, object])
    continue_final_message: bool = False
    add_generation_prompt: bool = True
    named_tool_choice: bool = False


@dataclass(frozen=True, slots=True)
class RenderedChat:
    """The rendered prompt together with request-scoped thinking and continuation state."""

    prompt_text: str
    token_ids: tuple[int, ...]
    thinking_enabled: bool | None
    starts_inside_reasoning: bool
    force_reasoning: bool | None
    previous_content: str
    stream_reasoning: bool
    render_fingerprint: str


def _apply_explicit_kwargs_override(
    explicit: Mapping[str, object],
    thinking_enabled: bool | None,
    force_reasoning: bool | None,
) -> tuple[bool | None, bool | None]:
    for key in ("enable_thinking", "thinking"):
        if key in explicit and isinstance(explicit[key], bool):
            val = explicit[key]
            if val is False:
                return False, False
            return True, None if force_reasoning is False else force_reasoning
    effort = explicit.get("reasoning_effort")
    if isinstance(effort, str):
        if effort == "none":
            return False, False
        return True, None if force_reasoning is False else force_reasoning
    return thinking_enabled, force_reasoning


def _fingerprint(
    *,
    tokenizer_fp: str,
    thinking_enabled: bool | None,
    starts_inside_reasoning: bool,
    force_reasoning: bool | None,
    token_ids: tuple[int, ...],
) -> str:
    digest = hashlib.blake2b(digest_size=16)
    header = {
        "tokenizer": tokenizer_fp,
        "thinking_enabled": thinking_enabled,
        "starts_inside_reasoning": starts_inside_reasoning,
        "force_reasoning": force_reasoning,
    }
    digest.update(json.dumps(header, sort_keys=True).encode("utf-8"))
    digest.update(b"\0")
    for token_id in token_ids:
        digest.update(token_id.to_bytes(4, "little", signed=True))
    return digest.hexdigest()


class PromptRenderer:
    """Renders conversations into tokenized prompts and synchronized parser metadata."""

    def __init__(self, service: TokenizerService, *, model_name: str = "") -> None:
        self._service = service
        self._model_name = model_name
        self._policy = detect_thinking_policy(service.chat_template, model_hint=model_name)
        self._hook = select_render_hook(model_name, service.chat_template)
        tokenizer = service.tokenizer
        self._tokenizer_fp = tokenizer.fingerprint() if tokenizer is not None else "disabled"

    @property
    def policy(self) -> ThinkingPolicy:
        """Detected :data:`ThinkingPolicy` for the loaded chat template."""
        return self._policy

    @property
    def hook(self) -> ModelRenderHook:
        """Selected :class:`ModelRenderHook` for the model family."""
        return self._hook

    def render(
        self,
        request: RenderRequest,
        *,
        max_tokens: int | None = None,
        vouch: bool = True,
    ) -> RenderedChat:
        """Render ``request`` into prompt text, token ids, and parser/constraint metadata.

        Raises:
            ContextLengthError: If the prompt leaves no room for one output token.
            InvalidRequestError: If rendering or tokenization fails.
        """
        resolved = resolve_thinking(
            self._policy,
            request.thinking,
            named_tool_choice=request.named_tool_choice,
        )
        merged_kwargs: dict[str, object] = dict(resolved.template_kwargs)
        merged_kwargs.update(request.chat_template_kwargs)

        thinking_enabled, force_reasoning = _apply_explicit_kwargs_override(
            request.chat_template_kwargs,
            resolved.enabled,
            resolved.force_reasoning,
        )

        conversation = request.conversation
        messages = self._hook.prepare_messages(
            conversation.messages(), thinking_enabled=thinking_enabled
        )
        tools = tuple(conversation.tool_definitions())
        add_generation_prompt = (
            False if request.continue_final_message else request.add_generation_prompt
        )
        try:
            template_data = TemplateData(
                messages=messages,
                tools=tools,
                add_generation_prompt=add_generation_prompt,
                continue_final_message=request.continue_final_message,
                chat_template_kwargs=tuple(merged_kwargs.items()),
            )
            prompt_text = self._service.render_chat(template_data)
            options = EncodeOptions(max_output_tokens=1)
            encoded = self._service.encode(RenderedInput(prompt_text), options, vouch=vouch)
        except ContextLengthError as exc:
            raise ContextLengthError(
                prompt_tokens=exc.prompt_tokens,
                output_tokens=max(max_tokens or 1, 1),
                max_context_tokens=exc.max_context_tokens,
            ) from None
        except (TypeError, ValueError, RuntimeError) as exc:
            raise InvalidRequestError(f"cannot render this prompt: {exc}") from exc

        previous_content = ""
        if request.continue_final_message and conversation.turns:
            previous_content = conversation.turns[-1].text

        if thinking_enabled is False:
            starts_inside = False
            force_reasoning = False
        elif request.continue_final_message:
            starts_inside = self._hook.detect_starts_inside(
                previous_content, thinking_enabled=thinking_enabled
            )
        else:
            starts_inside = self._hook.detect_starts_inside(
                prompt_text, thinking_enabled=thinking_enabled
            )

        if starts_inside:
            force_reasoning = True

        fingerprint = _fingerprint(
            tokenizer_fp=self._tokenizer_fp,
            thinking_enabled=thinking_enabled,
            starts_inside_reasoning=starts_inside,
            force_reasoning=force_reasoning,
            token_ids=encoded.token_ids,
        )
        return RenderedChat(
            prompt_text=prompt_text,
            token_ids=encoded.token_ids,
            thinking_enabled=thinking_enabled,
            starts_inside_reasoning=starts_inside,
            force_reasoning=force_reasoning,
            previous_content=previous_content,
            stream_reasoning=resolved.stream_reasoning,
            render_fingerprint=fingerprint,
        )
