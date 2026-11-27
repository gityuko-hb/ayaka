"""Unit tests for PromptRenderer, model hooks, chat_template_kwargs, and prefill."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

from ayaka.configs.tokenizer import TokenizerConfig
from ayaka.entrypoints.conversation import ConversationBuilder
from ayaka.entrypoints.render.reasoning import ReasoningEffort, ThinkingRequest, resolve_thinking
from ayaka.entrypoints.render.selection import PromptRenderer, RenderRequest
from ayaka.entrypoints.render.thinking import NativeToggle, TemplateToggle
from ayaka.tokenizers.service import TokenizerService
from tokenizer_fakes import MODEL_VOCAB, FakeTokenizer, loaded_for

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

pytestmark = pytest.mark.unit


class _Qwen3LikeTokenizer(FakeTokenizer):
    """Mimics Qwen3's chat template (thinking default-on, explicit disable closes <think>)."""

    chat_template = "{% if enable_thinking is defined and enable_thinking is false %}off{% endif %}"

    def apply_chat_template(
        self,
        messages: Sequence[Mapping[str, Any]],
        *,
        tools: Sequence[Mapping[str, Any]] | None = None,
        add_generation_prompt: bool = True,
        chat_template: str | None = None,
        **kwargs: Any,
    ) -> str:
        del tools, chat_template
        body = "".join(
            f"<|im_start|>{m['role']}\n"
            + (f"[R:{m['reasoning_content']}]" if "reasoning_content" in m else "")
            + f"{m['content']}<|im_end|>\n"
            for m in messages
        )
        if kwargs.get("continue_final_message"):
            return body.removesuffix("<|im_end|>\n")
        if not add_generation_prompt:
            return body
        if kwargs.get("enable_thinking") is False:
            return body + "<|im_start|>assistant\n<think>\n\n</think>\n\n"
        return body + "<|im_start|>assistant\n<think>\n"


def _service(tokenizer: FakeTokenizer) -> TokenizerService:
    return TokenizerService(
        TokenizerConfig("fake", mode="hf"),
        max_model_len=512,
        model_vocab_size=MODEL_VOCAB,
        loaded=loaded_for(tokenizer),
    )


class TestPromptRenderer:
    def test_qwen3_disable_thinking_avoids_starts_inside_reasoning(self) -> None:
        renderer = PromptRenderer(_service(_Qwen3LikeTokenizer()), model_name="Qwen3-8B")
        assert renderer.policy == TemplateToggle(key="enable_thinking", default_enabled=True)

        builder = ConversationBuilder()
        builder.user("hello")
        conv = builder.build()

        disabled = renderer.render(
            RenderRequest(
                conversation=conv,
                thinking=ThinkingRequest(effort=ReasoningEffort.NONE),
            )
        )
        enabled = renderer.render(
            RenderRequest(
                conversation=conv,
                thinking=ThinkingRequest(effort=ReasoningEffort.HIGH),
            )
        )

        assert disabled.prompt_text.endswith("<think>\n\n</think>\n\n")
        assert disabled.thinking_enabled is False
        assert disabled.starts_inside_reasoning is False
        assert disabled.force_reasoning is False

        assert enabled.prompt_text.endswith("<think>\n")
        assert enabled.thinking_enabled is True
        assert enabled.starts_inside_reasoning is True
        assert enabled.force_reasoning is True
        assert disabled.render_fingerprint != enabled.render_fingerprint

    def test_chat_template_kwargs_and_historical_reasoning_are_forwarded(self) -> None:
        renderer = PromptRenderer(_service(_Qwen3LikeTokenizer()), model_name="Qwen3-8B")
        builder = ConversationBuilder()
        builder.user("q1")
        builder.assistant("a1", reasoning="step 1")
        builder.user("q2")

        rendered = renderer.render(
            RenderRequest(
                conversation=builder.build(),
                chat_template_kwargs={"enable_thinking": False},
            )
        )

        assert rendered.thinking_enabled is False
        assert rendered.force_reasoning is False
        assert "[R:step 1]" not in rendered.prompt_text  # dropped when thinking is off

        rendered_on = renderer.render(
            RenderRequest(
                conversation=builder.build(),
                thinking=ThinkingRequest(enabled=True),
            )
        )
        assert "[R:step 1]" in rendered_on.prompt_text

    def test_continue_final_message_tracks_previous_content_and_unclosed_think(self) -> None:
        renderer = PromptRenderer(_service(_Qwen3LikeTokenizer()), model_name="Qwen3-8B")
        builder = ConversationBuilder()
        builder.user("solve")
        builder.assistant("<think>step 1")

        rendered = renderer.render(
            RenderRequest(conversation=builder.build(), continue_final_message=True)
        )

        assert rendered.previous_content == "<think>step 1"
        assert rendered.starts_inside_reasoning is True
        assert rendered.force_reasoning is True
        assert rendered.prompt_text.endswith("<think>step 1")

    def test_named_tool_choice_disables_native_toggle_unless_forced(self) -> None:
        policy = NativeToggle(default_enabled=True, named_tool_disables=True)
        off = resolve_thinking(policy, ThinkingRequest(), named_tool_choice=True)
        forced = resolve_thinking(policy, ThinkingRequest(enabled=True), named_tool_choice=True)

        assert off.enabled is False
        assert off.force_reasoning is False
        assert forced.enabled is True
