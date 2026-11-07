from __future__ import annotations

from typing import Any

import pytest

from ayaka.tokenizers.contracts import (
    ChatInput,
    ChatMessage,
    EncodeOptions,
    LoadedTokenizer,
    RenderedInput,
    TemplateInput,
    TextInput,
    TokenIdsInput,
    TokenizerCapabilities,
    encode_input_weight,
)

USER = ChatMessage("user", "hi")


@pytest.mark.unit
class TestInputs:
    def test_text_defaults_add_special_tokens(self) -> None:
        assert TextInput("x").add_special_tokens

    @pytest.mark.parametrize("value", [b"x", 1, None])
    def test_text_must_be_str(self, value: Any) -> None:
        with pytest.raises(TypeError, match="text must be str"):
            TextInput(value)
        with pytest.raises(TypeError, match="text must be str"):
            RenderedInput(value)

    def test_chat_message_rejects_other_roles(self) -> None:
        with pytest.raises(ValueError, match="TemplateInput"):
            ChatMessage("tool", "x")  # type: ignore[arg-type]  # deliberately wrong role

    def test_chat_requires_non_empty_tuple(self) -> None:
        with pytest.raises(ValueError, match="non-empty tuple"):
            ChatInput(())
        with pytest.raises(TypeError, match="immutable"):
            ChatInput([USER])  # type: ignore[arg-type]  # list, not tuple

    def test_chat_generation_prompt_and_continuation_conflict(self) -> None:
        with pytest.raises(ValueError, match="conflict"):
            ChatInput((ChatMessage("assistant", "a"),), continue_final_message=True)

    def test_continuation_needs_final_assistant(self) -> None:
        with pytest.raises(ValueError, match="final assistant"):
            ChatInput((USER,), add_generation_prompt=False, continue_final_message=True)

    def test_chat_template_must_be_non_empty(self) -> None:
        with pytest.raises(ValueError, match="chat_template"):
            ChatInput((USER,), chat_template="")

    def test_token_ids_must_be_tuple_of_ints(self) -> None:
        assert TokenIdsInput((1, 2)).token_ids == (1, 2)
        with pytest.raises(TypeError, match="tuple"):
            TokenIdsInput([1])  # type: ignore[arg-type]  # deliberately wrong type
        with pytest.raises(TypeError):
            TokenIdsInput((True,))

    @pytest.mark.parametrize(
        ("messages", "tools", "message"),
        [
            ("[]", "[]", "non-empty JSON array"),
            ('{"a": 1}', "[]", "non-empty JSON array"),
            ("[1]", "[]", "objects"),
            ('[{"role": "user"}]', "{}", "array of objects"),
            ('[{"role": "user"}]', "[1]", "array of objects"),
        ],
    )
    def test_template_input_validates_json_shape(
        self, messages: str, tools: str, message: str
    ) -> None:
        with pytest.raises(ValueError, match=message):
            TemplateInput(messages, tools)

    def test_template_input_accepts_valid_json(self) -> None:
        assert TemplateInput('[{"role": "tool", "content": "x"}]').tools_json == "[]"


@pytest.mark.unit
class TestEncodeOptions:
    def test_defaults(self) -> None:
        options = EncodeOptions()
        assert options.max_output_tokens == 128
        assert not options.truncate

    def test_rejects_non_positive_output(self) -> None:
        with pytest.raises(ValueError, match="max_output_tokens"):
            EncodeOptions(max_output_tokens=0)

    def test_rejects_long_namespace(self) -> None:
        EncodeOptions(cache_namespace="n" * 256)
        with pytest.raises(ValueError, match="256"):
            EncodeOptions(cache_namespace="n" * 257)


@pytest.mark.unit
class TestCapabilities:
    def test_disabled_cannot_advertise_capabilities(self) -> None:
        with pytest.raises(ValueError, match="disabled"):
            TokenizerCapabilities("disabled", True, False, False, False, False)

    def test_decoder_capability_needs_text_decode(self) -> None:
        with pytest.raises(ValueError, match="text_decode"):
            TokenizerCapabilities("hf", True, False, True, True, False)

    def test_accepts_tiktoken_backend(self) -> None:
        assert (
            TokenizerCapabilities("tiktoken", True, True, True, False, True).backend == "tiktoken"
        )

    def test_rejects_unknown_backend(self) -> None:
        with pytest.raises(ValueError, match="unknown"):
            TokenizerCapabilities("mistral", True, True, True, False, False)  # type: ignore[arg-type]


@pytest.mark.unit
class TestLoadedTokenizer:
    def test_adapter_and_backend_must_agree(self, fake: Any) -> None:
        caps = TokenizerCapabilities("hf", True, True, True, False, False)
        disabled = TokenizerCapabilities("disabled", False, False, False, False, False)
        with pytest.raises(ValueError, match="disagree"):
            LoadedTokenizer(None, caps)
        with pytest.raises(ValueError, match="disagree"):
            LoadedTokenizer(fake, disabled)

    def test_accepts_disabled_without_adapter(self) -> None:
        disabled = TokenizerCapabilities("disabled", False, False, False, False, False)
        assert LoadedTokenizer(None, disabled).tokenizer is None


@pytest.mark.unit
class TestWeight:
    def test_counts_utf8_bytes(self) -> None:
        assert encode_input_weight(TextInput("é")) == 2
        assert encode_input_weight(RenderedInput("abc")) == 3

    def test_chat_counts_content_role_and_template(self) -> None:
        chat = ChatInput((USER,), chat_template="tt")
        assert encode_input_weight(chat) == len("hi") + len("user") + 2

    def test_token_ids_cost_eight_bytes_each(self) -> None:
        assert encode_input_weight(TokenIdsInput((1, 2, 3))) == 24

    def test_template_counts_both_json_strings(self) -> None:
        item = TemplateInput('[{"role": "user"}]', "[]")
        assert encode_input_weight(item) == len('[{"role": "user"}]') + 2

    def test_rejects_unknown_input(self) -> None:
        with pytest.raises(TypeError, match="unsupported"):
            encode_input_weight("text")  # type: ignore[arg-type]  # deliberately wrong type
