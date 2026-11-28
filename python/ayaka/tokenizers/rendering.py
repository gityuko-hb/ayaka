"""Prompt template sanitization and rendering helpers for tokenizer services."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from ayaka.tokenizers.contracts import (
    ChatInput,
    RenderedInput,
    TemplateData,
    TemplateInput,
    TextInput,
)
from ayaka.tokenizers.security import SpecialTokenGuard, plain_json

if TYPE_CHECKING:
    from ayaka.tokenizers.ports import TokenizerLike

__all__ = ["render_encode_input"]


def _template_kwargs(
    add_generation_prompt: bool,
    continue_final_message: bool,
    extra: Any,
) -> dict[str, Any]:
    kwargs: dict[str, Any] = {}
    if not add_generation_prompt:
        kwargs["add_generation_prompt"] = False
    if continue_final_message:
        kwargs["continue_final_message"] = True
    if isinstance(extra, dict):
        kwargs.update(extra)  # pyright: ignore[reportUnknownArgumentType]
    return kwargs


def _sanitize(guard: SpecialTokenGuard | None, content: Any) -> Any:
    return content if guard is None else guard.sanitize(content)


def render_encode_input(
    tokenizer: TokenizerLike | None,
    value: TextInput | RenderedInput | ChatInput | TemplateInput | TemplateData,
    guard: SpecialTokenGuard | None,
) -> tuple[str, bool, bool]:
    """Return ``(text, add_special_tokens, rendered)`` for a text-bearing input."""
    if isinstance(value, TextInput):
        return value.text, value.add_special_tokens, False
    if isinstance(value, RenderedInput):
        return value.text, False, True
    if tokenizer is None:
        raise RuntimeError("tokenizer is disabled")
    if isinstance(value, TemplateData):
        # One walk checks every string and gives the values json.loads would; no round trip.
        plain = plain_json if guard is None else guard.sanitize_json
        extra_data = plain(dict(value.chat_template_kwargs)) if value.chat_template_kwargs else None
        kwargs = _template_kwargs(
            value.add_generation_prompt, value.continue_final_message, extra_data
        )
        text = tokenizer.apply_chat_template(
            plain(value.messages), tools=plain(value.tools), **kwargs
        )
    elif isinstance(value, TemplateInput):
        # Every string in the history and tool definitions is client-supplied, role included.
        extra_json = (
            _sanitize(guard, json.loads(value.chat_template_kwargs_json))
            if value.chat_template_kwargs_json != "{}"
            else None
        )
        kwargs = _template_kwargs(
            value.add_generation_prompt, value.continue_final_message, extra_json
        )
        text = tokenizer.apply_chat_template(
            _sanitize(guard, json.loads(value.messages_json)),
            tools=_sanitize(guard, json.loads(value.tools_json)),
            **kwargs,
        )
    else:
        # The template text itself (``chat_template``) is code, not content: never sanitized.
        rendered_messages: list[dict[str, object]] = []
        for m in value.messages:
            sanitized_text = _sanitize(guard, m.content)
            if m.images:
                parts: list[dict[str, object]] = [{"type": "image"} for _ in m.images]
                if sanitized_text:
                    parts.append({"type": "text", "text": sanitized_text})
                rendered_messages.append({"role": m.role, "content": parts})
            else:
                rendered_messages.append({"role": m.role, "content": sanitized_text})
        text = tokenizer.apply_chat_template(
            rendered_messages,
            add_generation_prompt=value.add_generation_prompt,
            chat_template=value.chat_template,
            continue_final_message=value.continue_final_message,
        )
    return text, False, True
