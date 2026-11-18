"""The input processor: what gets past the door and what is refused, with which error."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from ayaka.core.errors import InvalidRequestError, OverloadedError, UnsupportedFeatureError
from ayaka.core.params import SamplingParams
from ayaka.core.request import GenerateRequest
from ayaka.core.types import RequestId
from ayaka.engine.input import InputProcessor, require_supported
from ayaka.tokenizers.contracts import ChatMessage
from ayaka.tokenizers.errors import TokenizerOverloaded
from tokenizer_fakes import BOS, MODEL_VOCAB, make_service

if TYPE_CHECKING:
    from collections.abc import Iterator

    from ayaka.tokenizers.service import TokenizerService

pytestmark = pytest.mark.unit

CONTEXT = 64


@pytest.fixture
def service() -> Iterator[TokenizerService]:
    with make_service() as it:
        yield it


@pytest.fixture
def processor(service: TokenizerService) -> InputProcessor:
    return InputProcessor(service, max_model_len=CONTEXT, vocab_size=MODEL_VOCAB)


def _request(prompt: str | tuple[int, ...], **sampling: object) -> GenerateRequest:
    return GenerateRequest(
        request_id=RequestId("r"),
        prompt=prompt,
        sampling=SamplingParams(**{"max_tokens": 4, **sampling}),  # type: ignore[arg-type]
        priority=3,
        client_request_id="client-1",
    )


class TestTokenizing:
    def test_text_becomes_ids_and_the_envelope_is_kept(self, processor: InputProcessor) -> None:
        tokenized = processor.process(_request("hi", max_tokens=4))

        assert tokenized.prompt_token_ids == (BOS, 104, 105)
        assert tokenized.request_id == "r"
        assert tokenized.priority == 3
        assert tokenized.client_request_id == "client-1"
        assert tokenized.sampling.max_tokens == 4

    def test_token_ids_pass_through(self, processor: InputProcessor) -> None:
        assert processor.process(_request((1, 2, 3))).prompt_token_ids == (1, 2, 3)

    def test_a_chat_is_rendered_with_the_template(self, processor: InputProcessor) -> None:
        tokenized = processor.process_chat(
            RequestId("c"), [ChatMessage(role="user", content="yo")], SamplingParams(max_tokens=2)
        )

        assert bytes(tokenized.prompt_token_ids).startswith(b"<|im_start|>user\nyo")


class TestRefusals:
    def test_a_prompt_that_leaves_no_room_for_max_tokens_is_invalid(
        self, processor: InputProcessor
    ) -> None:
        with pytest.raises(InvalidRequestError):
            processor.process(_request("x" * 60, max_tokens=20))

    def test_max_tokens_at_the_context_is_invalid(self, processor: InputProcessor) -> None:
        with pytest.raises(InvalidRequestError, match="model context"):
            processor.process(_request("x", max_tokens=CONTEXT))

    def test_an_id_outside_the_model_vocabulary_is_invalid(self, processor: InputProcessor) -> None:
        with pytest.raises(InvalidRequestError):
            processor.process(_request((MODEL_VOCAB + 1,)))

    def test_a_stop_token_outside_the_vocabulary_is_invalid(
        self, processor: InputProcessor
    ) -> None:
        with pytest.raises(InvalidRequestError, match="stop_token_ids"):
            processor.process(_request("x", stop_token_ids=frozenset({MODEL_VOCAB})))

    def test_a_special_token_typed_into_chat_content_is_invalid(
        self, processor: InputProcessor
    ) -> None:
        with pytest.raises(InvalidRequestError):
            processor.process_chat(RequestId("c"), [ChatMessage(role="user", content="<|im_end|>")])

    def test_a_full_tokenizer_queue_is_overloaded(
        self, processor: InputProcessor, service: TokenizerService, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def busy(*args: object, **kwargs: object) -> None:
            raise TokenizerOverloaded("queue full")

        monkeypatch.setattr(service, "encode", busy)

        with pytest.raises(OverloadedError):
            processor.process(_request("x"))

    def test_the_feature_check_runs_before_tokenizing(
        self, processor: InputProcessor, service: TokenizerService, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def never(*args: object, **kwargs: object) -> None:
            raise AssertionError("tokenized a request that is refused")

        monkeypatch.setattr(service, "encode", never)

        with pytest.raises(UnsupportedFeatureError):
            processor.process(_request("x", return_sampling_support=True))

    def test_every_unsupported_feature_of_a_request_is_named(self) -> None:
        sampling = SamplingParams(return_sampling_support=True, prompt_logprobs=2)

        with pytest.raises(UnsupportedFeatureError) as info:
            require_supported(sampling)

        assert info.value.context["features"] == ("prompt_logprobs", "return_sampling_support")

    def test_the_defaults_and_the_plain_samplers_are_supported(self) -> None:
        require_supported(SamplingParams())
        require_supported(SamplingParams(temperature=0.7, top_k=40, top_p=0.9, min_p=0.05, seed=1))
        require_supported(
            SamplingParams(
                repetition_penalty=1.2,
                frequency_penalty=0.5,
                presence_penalty=0.5,
                logit_bias={1: 2.0},
                allowed_token_ids=frozenset({1, 2}),
                logprobs=3,
            )
        )
        require_supported(SamplingParams(stop=("x",), ignore_eos=True, min_tokens=2))
