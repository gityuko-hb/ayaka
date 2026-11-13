from __future__ import annotations

from dataclasses import FrozenInstanceError
from typing import Any, cast

import pytest

from ayaka.core.errors import InvalidRequestError
from ayaka.core.params import SamplingParams
from ayaka.core.request import MAX_CACHE_SALT_LEN, GenerateRequest, TokenizedRequest
from ayaka.core.types import RequestId

pytestmark = pytest.mark.unit

REQUEST_ID = RequestId("req-1")


def _request(**overrides: object) -> GenerateRequest:
    values: dict[str, object] = {"request_id": "req-1", "prompt": "hello"}
    values.update(overrides)
    return GenerateRequest(**cast("Any", values))


def _tokenized(**overrides: object) -> TokenizedRequest:
    values: dict[str, object] = {"request_id": "req-1", "prompt_token_ids": (1, 2)}
    values.update(overrides)
    return TokenizedRequest(**cast("Any", values))


class TestGenerateRequest:
    def test_defaults_for_text_prompt(self) -> None:
        request = GenerateRequest(request_id=REQUEST_ID, prompt="hello")

        assert request.request_id == "req-1"
        assert request.prompt == "hello"
        assert request.sampling == SamplingParams()
        assert request.client_request_id is None
        assert request.priority == 0
        assert request.cache_salt is None
        assert request.prefix_cache is True

    def test_text_prompt_properties(self) -> None:
        request = GenerateRequest(request_id=REQUEST_ID, prompt="hello")

        assert request.is_tokenized is False
        with pytest.raises(ValueError, match="prompt_length is unavailable"):
            _ = request.prompt_length

    def test_tokenized_prompt_properties(self) -> None:
        request = GenerateRequest(request_id=REQUEST_ID, prompt=(1, 2, 3))

        assert request.is_tokenized is True
        assert request.prompt_length == 3

    def test_accepts_token_id_zero(self) -> None:
        request = GenerateRequest(request_id=REQUEST_ID, prompt=(0,))

        assert request.prompt == (0,)
        assert request.prompt_length == 1

    def test_required_fields_are_keyword_only(self) -> None:
        with pytest.raises(TypeError):
            cast("Any", GenerateRequest)("req-1", "hello")

    def test_requires_prompt(self) -> None:
        with pytest.raises(TypeError):
            cast("Any", GenerateRequest)(request_id=REQUEST_ID)

    @pytest.mark.parametrize("prompt", ["", "   ", "\t\n"])
    def test_rejects_blank_text_prompt(self, prompt: str) -> None:
        with pytest.raises(ValueError, match="prompt must be a non-empty string"):
            GenerateRequest(request_id=REQUEST_ID, prompt=prompt)

    def test_rejects_non_string_non_tuple_prompt(self) -> None:
        with pytest.raises(ValueError, match="prompt token id must be a tuple of token ids"):
            _request(prompt=[1, 2])

    def test_rejects_empty_tokenized_prompt(self) -> None:
        with pytest.raises(InvalidRequestError, match="prompt token id must not be empty"):
            GenerateRequest(request_id=REQUEST_ID, prompt=())

    def test_rejects_negative_prompt_token_id(self) -> None:
        with pytest.raises(ValueError, match="prompt token id must be >= 0"):
            GenerateRequest(request_id=REQUEST_ID, prompt=(0, -1))

    @pytest.mark.parametrize("value", [(True,), (1.0,), ("1",)])
    def test_rejects_non_integer_prompt_token_id(self, value: tuple[object, ...]) -> None:
        with pytest.raises(TypeError, match="prompt token id must be an integer"):
            GenerateRequest(
                request_id=REQUEST_ID,
                prompt=cast("tuple[int, ...]", value),
            )

    @pytest.mark.parametrize("value", ["", " ", "\t\n", None, 3])
    def test_rejects_invalid_request_id(self, value: object) -> None:
        with pytest.raises(ValueError, match="request_id must be a non-empty string"):
            _request(request_id=value)

    @pytest.mark.parametrize("value", ["", " ", "\t\n", 3])
    def test_rejects_invalid_client_request_id(self, value: object) -> None:
        with pytest.raises(ValueError, match="client_request_id must be a non-empty string"):
            _request(client_request_id=value)

    def test_accepts_client_request_id(self) -> None:
        request = GenerateRequest(
            request_id=REQUEST_ID,
            prompt="hello",
            client_request_id="client-1",
        )

        assert request.client_request_id == "client-1"

    @pytest.mark.parametrize("value", [-100, -1, 0, 1, 100])
    def test_accepts_any_integer_priority(self, value: int) -> None:
        request = GenerateRequest(request_id=REQUEST_ID, prompt="hello", priority=value)

        assert request.priority == value

    @pytest.mark.parametrize("value", [True, False, 1.0, "1", None])
    def test_rejects_non_integer_priority(self, value: object) -> None:
        with pytest.raises(TypeError, match="priority must be an integer"):
            _request(priority=value)

    @pytest.mark.parametrize("value", [1, 0, None, "true", []])
    def test_rejects_non_boolean_prefix_cache(self, value: object) -> None:
        with pytest.raises(TypeError, match="prefix_cache must be bool"):
            _request(prefix_cache=value)

    @pytest.mark.parametrize("value", [True, False])
    def test_accepts_boolean_prefix_cache(self, value: bool) -> None:
        request = GenerateRequest(request_id=REQUEST_ID, prompt="hello", prefix_cache=value)

        assert request.prefix_cache is value

    @pytest.mark.parametrize("value", ["", " ", "\t\n", 3])
    def test_rejects_invalid_cache_salt(self, value: object) -> None:
        with pytest.raises(ValueError, match="cache_salt must be a non-empty string"):
            _request(cache_salt=value)

    def test_accepts_cache_salt(self) -> None:
        request = GenerateRequest(request_id=REQUEST_ID, prompt="hello", cache_salt="tenant-a")

        assert request.cache_salt == "tenant-a"

    def test_accepts_cache_salt_at_limit(self) -> None:
        salt = "x" * MAX_CACHE_SALT_LEN
        request = GenerateRequest(request_id=REQUEST_ID, prompt="hello", cache_salt=salt)

        assert request.cache_salt == salt

    def test_rejects_cache_salt_above_limit(self) -> None:
        salt = "x" * (MAX_CACHE_SALT_LEN + 1)

        with pytest.raises(InvalidRequestError, match="cache_salt must be at most 256 characters"):
            GenerateRequest(request_id=REQUEST_ID, prompt="hello", cache_salt=salt)

    def test_accepts_custom_sampling(self) -> None:
        sampling = SamplingParams(n=2)
        request = GenerateRequest(request_id=REQUEST_ID, prompt="hello", sampling=sampling)

        assert request.sampling is sampling

    def test_is_frozen(self) -> None:
        request = GenerateRequest(request_id=REQUEST_ID, prompt="hello")

        with pytest.raises(FrozenInstanceError):
            cast("Any", request).priority = 1


class TestTokenizedRequest:
    def test_defaults(self) -> None:
        request = TokenizedRequest(request_id=REQUEST_ID, prompt_token_ids=(1, 2, 3))

        assert request.request_id == "req-1"
        assert request.prompt_token_ids == (1, 2, 3)
        assert request.sampling == SamplingParams()
        assert request.client_request_id is None
        assert request.priority == 0
        assert request.cache_salt is None
        assert request.prefix_cache is True
        assert request.prompt_length == 3

    def test_accepts_token_id_zero(self) -> None:
        request = TokenizedRequest(request_id=REQUEST_ID, prompt_token_ids=(0,))

        assert request.prompt_token_ids == (0,)
        assert request.prompt_length == 1

    def test_required_fields_are_keyword_only(self) -> None:
        with pytest.raises(TypeError):
            cast("Any", TokenizedRequest)("req-1", (1, 2))

    def test_requires_prompt_token_ids(self) -> None:
        with pytest.raises(TypeError):
            cast("Any", TokenizedRequest)(request_id=REQUEST_ID)

    def test_rejects_empty_prompt_token_ids(self) -> None:
        with pytest.raises(InvalidRequestError, match="prompt_token_id must not be empty"):
            TokenizedRequest(request_id=REQUEST_ID, prompt_token_ids=())

    def test_rejects_non_tuple_prompt_token_ids(self) -> None:
        with pytest.raises(ValueError, match="prompt_token_id must be a tuple of token ids"):
            _tokenized(prompt_token_ids=[1, 2])

    def test_rejects_negative_prompt_token_id(self) -> None:
        with pytest.raises(ValueError, match="prompt_token_id must be >= 0"):
            TokenizedRequest(request_id=REQUEST_ID, prompt_token_ids=(0, -1))

    @pytest.mark.parametrize("value", [(True,), (1.0,), ("1",)])
    def test_rejects_non_integer_prompt_token_id(self, value: tuple[object, ...]) -> None:
        with pytest.raises(TypeError, match="prompt_token_id must be an integer"):
            TokenizedRequest(
                request_id=REQUEST_ID,
                prompt_token_ids=cast("tuple[int, ...]", value),
            )

    def test_max_total_length_uses_max_tokens(self) -> None:
        request = TokenizedRequest(
            request_id=REQUEST_ID,
            prompt_token_ids=(1, 2, 3),
            sampling=SamplingParams(max_tokens=5),
        )

        assert request.max_total_length == 8

    def test_max_total_length_is_none_when_unbounded(self) -> None:
        request = TokenizedRequest(
            request_id=REQUEST_ID,
            prompt_token_ids=(1, 2, 3),
            sampling=SamplingParams(max_tokens=None),
        )

        assert request.max_total_length is None

    def test_default_max_total_length(self) -> None:
        request = TokenizedRequest(request_id=REQUEST_ID, prompt_token_ids=(1, 2, 3))

        assert request.max_total_length == 3 + 128

    @pytest.mark.parametrize("value", ["", " ", 3])
    def test_rejects_invalid_request_id(self, value: object) -> None:
        with pytest.raises(ValueError, match="request_id must be a non-empty string"):
            _tokenized(request_id=value)

    @pytest.mark.parametrize("value", ["", " ", 3])
    def test_rejects_invalid_client_request_id(self, value: object) -> None:
        with pytest.raises(ValueError, match="client_request_id must be a non-empty string"):
            _tokenized(client_request_id=value)

    @pytest.mark.parametrize("value", [True, False, 1.0, "1", None])
    def test_rejects_non_integer_priority(self, value: object) -> None:
        with pytest.raises(TypeError, match="priority must be an integer"):
            _tokenized(priority=value)

    @pytest.mark.parametrize("value", [1, 0, None, "true", []])
    def test_rejects_non_boolean_prefix_cache(self, value: object) -> None:
        with pytest.raises(TypeError, match="prefix_cache must be bool"):
            _tokenized(prefix_cache=value)

    def test_rejects_cache_salt_above_limit(self) -> None:
        with pytest.raises(InvalidRequestError, match="cache_salt must be at most 256 characters"):
            _tokenized(cache_salt="x" * (MAX_CACHE_SALT_LEN + 1))

    def test_accepts_custom_sampling(self) -> None:
        sampling = SamplingParams(max_tokens=None)
        request = TokenizedRequest(
            request_id=REQUEST_ID,
            prompt_token_ids=(1, 2),
            sampling=sampling,
        )

        assert request.sampling is sampling

    def test_is_frozen(self) -> None:
        request = TokenizedRequest(request_id=REQUEST_ID, prompt_token_ids=(1, 2))

        with pytest.raises(FrozenInstanceError):
            cast("Any", request).priority = 1


def test_max_cache_salt_len_constant() -> None:
    assert MAX_CACHE_SALT_LEN == 256
