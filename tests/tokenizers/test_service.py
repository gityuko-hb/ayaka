from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import Any

import pytest

from ayaka.configs.tokenizer import DetokenizeParams, TokenizerConfig
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
)
from ayaka.tokenizers.decoding.byte import ByteIncrementalDetokenizer
from ayaka.tokenizers.errors import TokenizerOverloaded
from ayaka.tokenizers.service import TokenizerService
from ayaka.utils.cancellation import CancellationToken, CancelledError
from tokenizer_fakes import BOS, EOS, IM_END, IM_START, FakeTokenizer, RenderedFake, loaded_for

Make = Callable[..., TokenizerService]
OPTIONS = EncodeOptions(max_output_tokens=8)
CHAT = ChatInput((ChatMessage("user", "hi"),))


@pytest.mark.unit
class TestEncode:
    def test_text_gets_bos_and_fingerprint(self, make_service: Make) -> None:
        result = make_service().encode(TextInput("hi"), OPTIONS)

        assert result.token_ids == (BOS, ord("h"), ord("i"))
        assert result.reused_prefix_len == 0
        assert result.extra == (("tokenizer_fingerprint", "fake-fp"),)

    def test_special_tokens_can_be_left_out(self, make_service: Make) -> None:
        result = make_service().encode(TextInput("hi", add_special_tokens=False), OPTIONS)

        assert result.token_ids == (ord("h"), ord("i"))

    def test_token_ids_are_validated_and_returned(self, make_service: Make) -> None:
        assert make_service().encode(TokenIdsInput((1, 2, 3)), OPTIONS).token_ids == (1, 2, 3)

    def test_ids_the_tokenizer_cannot_produce_are_rejected(self, make_service: Make) -> None:
        with pytest.raises(ValueError, match="not supported"):
            make_service().encode(TokenIdsInput((299,)), OPTIONS)

    def test_vocabulary_larger_than_the_model_is_rejected_at_startup(
        self, make_service: Make
    ) -> None:
        with pytest.raises(ValueError, match="exceeds model"):
            make_service(model_vocab_size=100)

    def test_empty_prompt_is_rejected(self, make_service: Make) -> None:
        with pytest.raises(ValueError, match="empty"):
            make_service().encode(TextInput("", add_special_tokens=False), OPTIONS)

    def test_template_input_is_rendered_then_encoded(self, make_service: Make) -> None:
        item = TemplateInput('[{"role": "user", "content": "x"}]')

        result = make_service().encode(item, OPTIONS)

        assert bytes(result.token_ids).decode().startswith("<|im_start|>user\nx<|im_end|>")

    def test_render_chat_returns_the_prompt_text(self, make_service: Make) -> None:
        assert (
            make_service().render_chat(CHAT)
            == "<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n"
        )


@pytest.mark.unit
class TestRenderedText:
    """Template output must keep its control tokens; user text must not gain any."""

    def test_adapter_with_rendered_path_turns_control_tokens_into_ids(
        self, make_service: Make
    ) -> None:
        service = make_service(RenderedFake())

        chat = service.encode(CHAT, OPTIONS).token_ids
        rendered = service.encode(RenderedInput("<|im_start|>x"), OPTIONS).token_ids

        assert chat[0] == IM_START
        assert IM_END in chat
        assert rendered[0] == IM_START

    def test_user_text_never_becomes_control_tokens(self, make_service: Make) -> None:
        service = make_service(RenderedFake())

        ids = service.encode(TextInput("<|im_start|>", add_special_tokens=False), OPTIONS).token_ids

        assert IM_START not in ids
        assert ids == tuple(b"<|im_start|>")

    def test_adapter_without_rendered_path_encodes_as_plain_text(self, make_service: Make) -> None:
        ids = make_service(FakeTokenizer()).encode(CHAT, OPTIONS).token_ids

        assert ids[0] == ord("<")

    def test_rendered_and_plain_do_not_share_cache_entries(self, make_service: Make) -> None:
        service = make_service(RenderedFake(), enable_session_cache=True)
        text = "<|im_start|>"

        plain = service.encode(TextInput(text, add_special_tokens=False), OPTIONS)
        rendered = service.encode(RenderedInput(text), OPTIONS)

        assert rendered.reused_prefix_len == 0
        assert rendered.token_ids != plain.token_ids


@pytest.mark.unit
class TestLengthLimits:
    def test_prompt_plus_output_must_fit(self, make_service: Make) -> None:
        service = make_service(max_model_len=10)
        text = TextInput("abcdefghij", add_special_tokens=False)

        with pytest.raises(ValueError, match="exceeds context"):
            service.encode(text, EncodeOptions(max_output_tokens=4))

    def test_truncate_keeps_the_end_by_default(self, make_service: Make) -> None:
        service = make_service(max_model_len=10)
        text = TextInput("abcdefghij", add_special_tokens=False)

        result = service.encode(text, EncodeOptions(max_output_tokens=4, truncate=True))

        assert bytes(result.token_ids) == b"efghij"

    def test_truncate_right_keeps_the_start(self, make_service: Make) -> None:
        service = make_service(max_model_len=10, truncation_side="right")
        text = TextInput("abcdefghij", add_special_tokens=False)

        result = service.encode(text, EncodeOptions(max_output_tokens=4, truncate=True))

        assert bytes(result.token_ids) == b"abcdef"

    def test_input_cap_applies_below_the_context_limit(self, make_service: Make) -> None:
        text = TextInput("abcdefghij", add_special_tokens=False)
        options = EncodeOptions(max_output_tokens=4, max_input_tokens=3, truncate=True)

        result = make_service().encode(text, options)

        assert len(result.token_ids) == 3
        assert result.truncated

    def test_output_reservation_must_leave_room(self, make_service: Make) -> None:
        with pytest.raises(ValueError, match="no room"):
            make_service(max_model_len=10).encode(
                TextInput("a"), EncodeOptions(max_output_tokens=10)
            )


@pytest.mark.unit
class TestSessionCache:
    def test_repeat_reuses_the_whole_prompt(self, make_service: Make) -> None:
        tokenizer = FakeTokenizer()
        service = make_service(tokenizer, enable_session_cache=True)

        first = service.encode(TextInput("hello"), OPTIONS)
        second = service.encode(TextInput("hello"), OPTIONS)

        assert first.reused_prefix_len == 0
        assert second.reused_prefix_len == len(second.token_ids)
        assert second.token_ids == first.token_ids
        assert tokenizer.encode_calls == 1

    def test_cache_is_off_by_default(self, make_service: Make) -> None:
        service = make_service()
        service.encode(TextInput("hello"), OPTIONS)

        assert service.encode(TextInput("hello"), OPTIONS).reused_prefix_len == 0

    def test_namespaces_are_isolated(self, make_service: Make) -> None:
        service = make_service(enable_session_cache=True)
        service.encode(TextInput("hello"), EncodeOptions(max_output_tokens=8, cache_namespace="a"))

        other = service.encode(
            TextInput("hello"), EncodeOptions(max_output_tokens=8, cache_namespace="b")
        )

        assert other.reused_prefix_len == 0

    def test_verification_catches_a_corrupt_entry(self, make_service: Make) -> None:
        service = make_service(enable_session_cache=True, verify_session_cache=True)
        service.encode(TextInput("hello"), OPTIONS)
        key = next(iter(service._cache))
        service._cache[key] = (1, 2, 3)

        with pytest.raises(ValueError, match="verification mismatch"):
            service.encode(TextInput("hello"), OPTIONS)

    def test_capacity_evicts_the_oldest_entry(self, make_service: Make) -> None:
        service = make_service(enable_session_cache=True, session_cache_capacity=1)
        service.encode(TextInput("one"), OPTIONS)
        service.encode(TextInput("two"), OPTIONS)

        assert service.encode(TextInput("one"), OPTIONS).reused_prefix_len == 0

    def test_oversized_entries_are_not_cached(self, make_service: Make) -> None:
        service = make_service(enable_session_cache=True, session_cache_max_bytes=10)
        service.encode(TextInput("a long enough prompt"), OPTIONS)

        assert service.encode(TextInput("a long enough prompt"), OPTIONS).reused_prefix_len == 0


@pytest.mark.unit
class TestAdmission:
    def test_byte_budget_rejects_big_inputs(self, make_service: Make) -> None:
        service = make_service(encode_max_pending_bytes=10)

        with pytest.raises(TokenizerOverloaded):
            service.submit(TextInput("x" * 11), OPTIONS)
        assert service.pending == (0, 0)

    def test_pending_returns_to_zero_after_completion(self, make_service: Make) -> None:
        service = make_service()
        service.encode(TextInput("hi"), OPTIONS)

        assert service.pending == (0, 0)

    def test_cancelled_token_is_rejected_up_front(self, make_service: Make) -> None:
        token = CancellationToken()
        token.cancel("client went away")

        with pytest.raises(CancelledError, match="client went away"):
            make_service().submit(TextInput("hi"), OPTIONS, cancellation=token)

    def test_closed_service_refuses_work(self, make_service: Make) -> None:
        service = make_service()
        service.close()

        with pytest.raises(RuntimeError, match="closed"):
            service.submit(TextInput("hi"), OPTIONS)

    def test_wrong_argument_types_are_rejected(self, make_service: Make) -> None:
        service = make_service()

        with pytest.raises(TypeError, match="EncodeOptions"):
            service.submit(TextInput("hi"), "no")  # type: ignore[arg-type]
        with pytest.raises(TypeError, match="CancellationToken"):
            service.submit(TextInput("hi"), OPTIONS, cancellation=object())  # type: ignore[arg-type]


@pytest.mark.unit
class TestCapabilityChecks:
    def disabled_service(self) -> TokenizerService:
        loaded = LoadedTokenizer(
            None, TokenizerCapabilities("disabled", False, False, False, False, False)
        )
        return TokenizerService(
            TokenizerConfig("fake", skip_tokenizer_init=True),
            max_model_len=64,
            model_vocab_size=300,
            loaded=loaded,
        )

    def test_disabled_mode_accepts_only_token_ids(self) -> None:
        with self.disabled_service() as service:
            assert service.encode(TokenIdsInput((1, 2)), OPTIONS).token_ids == (1, 2)
            assert service.eos_token_ids == ()
            with pytest.raises(ValueError, match="only token-ID"):
                service.encode(TextInput("x"), OPTIONS)
            with pytest.raises(ValueError, match="disabled"):
                service.new_detokenizer(DetokenizeParams())

    def test_skip_tokenizer_init_must_match_the_loaded_backend(self) -> None:
        with pytest.raises(ValueError, match="disagree"):
            TokenizerService(
                TokenizerConfig("fake", skip_tokenizer_init=True),
                max_model_len=64,
                model_vocab_size=300,
                loaded=loaded_for(FakeTokenizer()),
            )

    @pytest.mark.parametrize("mode", ["slow", "tiktoken"])
    def test_requested_mode_must_match_the_loaded_backend(self, mode: Any) -> None:
        with pytest.raises(ValueError, match="requested and loaded backend"):
            TokenizerService(
                TokenizerConfig("fake", mode=mode),
                max_model_len=64,
                model_vocab_size=300,
                loaded=loaded_for(FakeTokenizer()),
            )

    def test_explicit_backends_need_the_matching_capability(self) -> None:
        for backend, kwargs in (("bytes", {"byte": False}), ("stream", {"stream": False})):
            with pytest.raises(ValueError, match="cannot use"):
                TokenizerService(
                    TokenizerConfig("fake", mode="hf", detokenize_backend=backend),  # type: ignore[arg-type]
                    max_model_len=64,
                    model_vocab_size=300,
                    loaded=loaded_for(FakeTokenizer(), **kwargs),
                )

    def test_eos_ids_are_sorted(self, make_service: Make) -> None:
        assert make_service().eos_token_ids == (EOS, IM_END)


@pytest.mark.unit
class TestWorkerPool:
    def test_many_requests_complete_and_release_their_charge(self, make_service: Make) -> None:
        service = make_service(encode_pool_workers=3, encode_batch_window_ms=1.0)

        futures = [service.submit(TextInput(f"prompt {i}"), OPTIONS) for i in range(40)]
        results = [f.result(timeout=10) for f in futures]

        assert [bytes(r.token_ids[1:]).decode() for r in results] == [
            f"prompt {i}" for i in range(40)
        ]
        assert service.pending == (0, 0)

    def test_encode_async_awaits_the_result(self, make_service: Make) -> None:
        service = make_service(encode_pool_workers=1)

        result = asyncio.run(service.encode_async(TextInput("hi"), OPTIONS))

        assert result.token_ids[-1] == ord("i")

    def test_encode_async_needs_workers(self, make_service: Make) -> None:
        with pytest.raises(ValueError, match="encode_pool_workers"):
            asyncio.run(make_service().encode_async(TextInput("hi"), OPTIONS))

    def test_blocking_encode_from_a_callback_is_refused(self, make_service: Make) -> None:
        service = make_service(encode_pool_workers=1)
        seen: list[BaseException] = []

        def callback(_future: Any) -> None:
            try:
                service.encode(TextInput("again"), OPTIONS)
            except RuntimeError as exc:
                seen.append(exc)

        future = service.submit(TextInput("hi"), OPTIONS)
        future.add_done_callback(callback)
        future.result(timeout=10)
        service.close()

        assert seen
        assert "completion callback" in str(seen[0])

    def test_close_cancels_work_that_has_not_started(self, make_service: Make) -> None:
        service = make_service(encode_pool_workers=1, encode_batch_window_ms=500.0)
        future = service.submit(TextInput("hi"), OPTIONS)

        service.close()

        assert future.cancelled() or future.done()


@pytest.mark.unit
class TestDetokenizer:
    def test_uses_the_configured_backend_and_shares_the_lock(self, make_service: Make) -> None:
        tokenizer = FakeTokenizer()
        assert tokenizer.verify_byte_path()
        service = make_service(tokenizer)

        detok = service.new_detokenizer(DetokenizeParams())

        assert isinstance(detok, ByteIncrementalDetokenizer)
        assert detok._lock is service._adapter_lock

    def test_closed_service_refuses_new_detokenizers(self, make_service: Make) -> None:
        service = make_service()
        service.close()

        with pytest.raises(RuntimeError, match="closed"):
            service.new_detokenizer(DetokenizeParams())
