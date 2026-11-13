from __future__ import annotations

import math
from dataclasses import FrozenInstanceError, replace
from typing import Any, cast

import pytest

from ayaka.core.errors import InvalidRequestError
from ayaka.core.params import (
    LOGIT_BIAS_LIMIT,
    MAX_LOGPROB_TOKEN_IDS,
    TOP_K_DISABLED,
    LogprobMode,
    SamplingParams,
)

pytestmark = pytest.mark.unit

_MAX_SEED = (1 << 63) - 1


def _params(**overrides: object) -> SamplingParams:
    return replace(SamplingParams(), **overrides)


class _ModeLike:
    def __init__(self, value: object) -> None:
        self.value = value


def test_constants() -> None:
    assert TOP_K_DISABLED == -1
    assert LOGIT_BIAS_LIMIT == 100.0
    assert MAX_LOGPROB_TOKEN_IDS == 128


def test_logprob_mode_values() -> None:
    assert {mode.value for mode in LogprobMode} == {"raw", "sampling"}
    assert LogprobMode.RAW == "raw"


class TestDefaults:
    def test_sampling_defaults(self) -> None:
        params = SamplingParams()

        assert params.n == 1
        assert params.temperature == 1.0
        assert params.top_p == 1.0
        assert params.top_k == TOP_K_DISABLED
        assert params.min_p == 0.0
        assert params.repetition_penalty == 1.0
        assert params.frequency_penalty == 0.0
        assert params.presence_penalty == 0.0
        assert params.seed is None
        assert params.max_tokens == 128
        assert params.min_tokens == 0
        assert params.stop == ()
        assert params.stop_token_ids == frozenset()
        assert params.ignore_eos is False
        assert params.include_stop_str_in_output is False
        assert params.logprobs is None
        assert params.prompt_logprobs is None
        assert params.logprob_token_ids is None
        assert params.logprob_mode is LogprobMode.RAW
        assert params.return_sampling_support is False
        assert params.logit_bias is None
        assert params.allowed_token_ids is None
        assert params.is_greedy is False

    def test_token_ids_logprobs_alias(self) -> None:
        assert SamplingParams().token_ids_logprobs is None
        assert SamplingParams(logprob_token_ids=(1, 2)).token_ids_logprobs == (1, 2)

    def test_is_frozen(self) -> None:
        with pytest.raises(FrozenInstanceError):
            cast("Any", SamplingParams()).n = 2


class TestN:
    @pytest.mark.parametrize("value", [1, 2, 10])
    def test_accepts_positive_integers(self, value: int) -> None:
        assert SamplingParams(n=value).n == value

    @pytest.mark.parametrize("value", [0, -1])
    def test_rejects_below_one(self, value: int) -> None:
        with pytest.raises(ValueError, match="n must be >= 1"):
            SamplingParams(n=value)

    @pytest.mark.parametrize("value", [True, False, 1.0, "1", None])
    def test_rejects_non_integers(self, value: object) -> None:
        with pytest.raises(TypeError, match="n must be an integer"):
            _params(n=value)


class TestTemperature:
    @pytest.mark.parametrize("value", [0, 1e-5, 1.0, 2.0])
    def test_accepts_valid_range(self, value: float) -> None:
        params = SamplingParams(temperature=value)

        assert params.temperature == float(value)

    @pytest.mark.parametrize("value", [-0.1, 2.1, 100.0])
    def test_rejects_out_of_range(self, value: float) -> None:
        with pytest.raises(InvalidRequestError, match="temperature must be in"):
            SamplingParams(temperature=value)

    @pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
    def test_rejects_non_finite(self, value: float) -> None:
        with pytest.raises(ValueError, match="temperature must be finite"):
            SamplingParams(temperature=value)

    @pytest.mark.parametrize("value", [True, False, "1", None])
    def test_rejects_non_numbers(self, value: object) -> None:
        with pytest.raises(TypeError, match="temperature must be a number"):
            _params(temperature=value)


class TestTopP:
    @pytest.mark.parametrize("value", [0.5, 1.0])
    def test_accepts_valid_range(self, value: float) -> None:
        assert SamplingParams(top_p=value).top_p == value

    @pytest.mark.parametrize("value", [0.0, -0.1, 1.1])
    def test_rejects_out_of_range(self, value: float) -> None:
        with pytest.raises(InvalidRequestError, match="top_p must be in"):
            SamplingParams(top_p=value)

    @pytest.mark.parametrize("value", [math.nan, math.inf])
    def test_rejects_non_finite(self, value: float) -> None:
        with pytest.raises(ValueError, match="top_p must be finite"):
            SamplingParams(top_p=value)

    @pytest.mark.parametrize("value", [True, False, "1", None])
    def test_rejects_non_numbers(self, value: object) -> None:
        with pytest.raises(TypeError, match="top_p must be a number"):
            _params(top_p=value)


class TestTopK:
    @pytest.mark.parametrize("value", [-1, 1, 2, 10])
    def test_accepts_valid_values(self, value: int) -> None:
        assert SamplingParams(top_k=value).top_k == value

    def test_zero_is_normalized_to_disabled(self) -> None:
        assert SamplingParams(top_k=0).top_k == TOP_K_DISABLED

    @pytest.mark.parametrize("value", [-2, -100])
    def test_rejects_other_negative_values(self, value: int) -> None:
        with pytest.raises(InvalidRequestError, match="top_k must be -1/0"):
            SamplingParams(top_k=value)

    @pytest.mark.parametrize("value", [True, False, 1.0, "1", None])
    def test_rejects_non_integers(self, value: object) -> None:
        with pytest.raises(TypeError, match="top_k must be an integer"):
            _params(top_k=value)

    def test_top_k_one_is_greedy(self) -> None:
        assert SamplingParams(top_k=1).is_greedy is True
        assert SamplingParams(top_k=2).is_greedy is False


class TestMinP:
    @pytest.mark.parametrize("value", [0.0, 0.5, 1.0])
    def test_accepts_valid_range(self, value: float) -> None:
        assert SamplingParams(min_p=value).min_p == value

    @pytest.mark.parametrize("value", [-0.1, 1.1])
    def test_rejects_out_of_range(self, value: float) -> None:
        with pytest.raises(InvalidRequestError, match="min_p must be in"):
            SamplingParams(min_p=value)

    @pytest.mark.parametrize("value", [math.nan, math.inf])
    def test_rejects_non_finite(self, value: float) -> None:
        with pytest.raises(ValueError, match="min_p must be finite"):
            SamplingParams(min_p=value)

    @pytest.mark.parametrize("value", [True, False, "1", None])
    def test_rejects_non_numbers(self, value: object) -> None:
        with pytest.raises(TypeError, match="min_p must be a number"):
            _params(min_p=value)


class TestPenalties:
    @pytest.mark.parametrize("value", [0.5, 1.0, 2.0])
    def test_accepts_positive_repetition_penalty(self, value: float) -> None:
        assert SamplingParams(repetition_penalty=value).repetition_penalty == value

    @pytest.mark.parametrize("value", [0.0, -0.1])
    def test_rejects_non_positive_repetition_penalty(self, value: float) -> None:
        with pytest.raises(InvalidRequestError, match="repetition_penalty must be > 0"):
            SamplingParams(repetition_penalty=value)

    @pytest.mark.parametrize("value", [math.nan, math.inf])
    def test_rejects_non_finite_repetition_penalty(self, value: float) -> None:
        with pytest.raises(ValueError, match="repetition_penalty must be finite"):
            SamplingParams(repetition_penalty=value)

    @pytest.mark.parametrize("value", [True, False, "1", None])
    def test_rejects_non_number_repetition_penalty(self, value: object) -> None:
        with pytest.raises(TypeError, match="repetition_penalty must be a number"):
            _params(repetition_penalty=value)

    @pytest.mark.parametrize("name", ["frequency_penalty", "presence_penalty"])
    @pytest.mark.parametrize("value", [-2.0, 0.0, 2.0])
    def test_accepts_penalty_range(self, name: str, value: float) -> None:
        assert getattr(SamplingParams(**cast("Any", {name: value})), name) == value

    @pytest.mark.parametrize("name", ["frequency_penalty", "presence_penalty"])
    @pytest.mark.parametrize("value", [-2.1, 2.1])
    def test_rejects_out_of_range_penalty(self, name: str, value: float) -> None:
        with pytest.raises(InvalidRequestError, match=f"{name} must be in"):
            SamplingParams(**cast("Any", {name: value}))

    @pytest.mark.parametrize("name", ["frequency_penalty", "presence_penalty"])
    @pytest.mark.parametrize("value", [math.nan, math.inf])
    def test_rejects_non_finite_penalty(self, name: str, value: float) -> None:
        with pytest.raises(ValueError, match=f"{name} must be finite"):
            SamplingParams(**cast("Any", {name: value}))

    @pytest.mark.parametrize("name", ["frequency_penalty", "presence_penalty"])
    @pytest.mark.parametrize("value", [True, False, "1", None])
    def test_rejects_non_number_penalty(self, name: str, value: object) -> None:
        with pytest.raises(TypeError, match=f"{name} must be a number"):
            SamplingParams(**cast("Any", {name: value}))


class TestSeed:
    def test_none_means_engine_entropy(self) -> None:
        assert SamplingParams().seed is None
        assert SamplingParams(seed=None).seed is None

    def test_minus_one_is_normalized_to_none(self) -> None:
        assert SamplingParams(seed=-1).seed is None

    @pytest.mark.parametrize("value", [0, 1, 123, _MAX_SEED])
    def test_accepts_non_negative_seeds(self, value: int) -> None:
        assert SamplingParams(seed=value).seed == value

    def test_accepts_maximum_seed(self) -> None:
        assert SamplingParams(seed=_MAX_SEED).seed == _MAX_SEED

    def test_rejects_seed_above_maximum(self) -> None:
        with pytest.raises(InvalidRequestError, match=f"seed must be <= {_MAX_SEED}"):
            SamplingParams(seed=_MAX_SEED + 1)

    @pytest.mark.parametrize("value", [-2, -100])
    def test_rejects_other_negative_seeds(self, value: int) -> None:
        with pytest.raises(InvalidRequestError, match="seed must be -1 or >= 0"):
            SamplingParams(seed=value)

    @pytest.mark.parametrize("value", [True, False, 1.0, "1"])
    def test_rejects_non_integer_seeds(self, value: object) -> None:
        with pytest.raises(TypeError, match="seed must be an integer"):
            _params(seed=value)


class TestTokenLimits:
    @pytest.mark.parametrize("value", [1, 5, 128])
    def test_accepts_positive_max_tokens(self, value: int) -> None:
        assert SamplingParams(max_tokens=value).max_tokens == value

    def test_accepts_none_max_tokens(self) -> None:
        assert SamplingParams(max_tokens=None).max_tokens is None

    @pytest.mark.parametrize("value", [0, -1])
    def test_rejects_non_positive_max_tokens(self, value: int) -> None:
        with pytest.raises(ValueError, match="max_tokens must be >= 1"):
            SamplingParams(max_tokens=value)

    @pytest.mark.parametrize("value", [True, False, 1.0, "1"])
    def test_rejects_non_integer_max_tokens(self, value: object) -> None:
        with pytest.raises(TypeError, match="max_tokens must be an integer"):
            _params(max_tokens=value)

    @pytest.mark.parametrize("value", [0, 1, 5])
    def test_accepts_non_negative_min_tokens(self, value: int) -> None:
        assert _params(max_tokens=None, min_tokens=value).min_tokens == value

    def test_rejects_negative_min_tokens(self) -> None:
        with pytest.raises(ValueError, match="min_tokens must be >= 0"):
            SamplingParams(min_tokens=-1)

    @pytest.mark.parametrize("value", [True, False, 1.0, "1"])
    def test_rejects_non_integer_min_tokens(self, value: object) -> None:
        with pytest.raises(TypeError, match="min_tokens must be an integer"):
            _params(min_tokens=value)

    def test_accepts_min_tokens_equal_to_max_tokens(self) -> None:
        params = SamplingParams(max_tokens=5, min_tokens=5)

        assert params.min_tokens == 5

    def test_rejects_min_tokens_above_max_tokens(self) -> None:
        with pytest.raises(InvalidRequestError, match="min_tokens must be <= max_tokens"):
            SamplingParams(max_tokens=5, min_tokens=6)

    def test_allows_large_min_tokens_when_max_tokens_is_unbounded(self) -> None:
        params = SamplingParams(max_tokens=None, min_tokens=10_000)

        assert params.min_tokens == 10_000


class TestStop:
    def test_default_is_empty(self) -> None:
        assert SamplingParams().stop == ()

    def test_string_becomes_single_item_tuple(self) -> None:
        assert _params(stop="</s>").stop == ("</s>",)

    def test_whitespace_only_string_is_allowed(self) -> None:
        assert _params(stop=" ").stop == (" ",)

    def test_sequence_becomes_tuple(self) -> None:
        values = ["a", " b "]
        params = _params(stop=values)

        assert params.stop == ("a", " b ")
        assert values == ["a", " b "]

    def test_tuple_is_preserved_as_tuple(self) -> None:
        assert SamplingParams(stop=("a", "b")).stop == ("a", "b")

    def test_empty_string_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="stop must be a non-empty string"):
            _params(stop="")

    @pytest.mark.parametrize("value", ["", 3, None])
    def test_empty_or_non_string_item_is_rejected(self, value: object) -> None:
        with pytest.raises(ValueError, match="stop must be a non-empty string"):
            _params(stop=[value])

    @pytest.mark.parametrize("value", [1, 1.5, {1, 2}, object()])
    def test_invalid_container_is_rejected(self, value: object) -> None:
        with pytest.raises(InvalidRequestError, match="stop must be a string or a sequence"):
            _params(stop=value)


class TestStopTokenIds:
    def test_default_is_empty(self) -> None:
        assert SamplingParams().stop_token_ids == frozenset()

    @pytest.mark.parametrize("value", [[0, 1, 2], (0, 1, 2), {0, 1, 2}, frozenset({0, 1, 2})])
    def test_accepts_iterables(self, value: object) -> None:
        assert SamplingParams(**cast("Any", {"stop_token_ids": value})).stop_token_ids == frozenset(
            {0, 1, 2}
        )

    def test_accepts_generator_and_removes_duplicates(self) -> None:
        params = _params(stop_token_ids=(n for n in (0, 1, 1, 2)))

        assert params.stop_token_ids == frozenset({0, 1, 2})

    def test_allows_empty(self) -> None:
        assert _params(stop_token_ids=[]).stop_token_ids == frozenset()

    @pytest.mark.parametrize("value", ["12", b"12", 12, None])
    def test_rejects_non_iterables(self, value: object) -> None:
        with pytest.raises(ValueError, match="stop_token_ids must be an iterable"):
            _params(stop_token_ids=value)

    def test_rejects_negative_ids(self) -> None:
        with pytest.raises(ValueError, match="stop_token_ids must be >= 0"):
            _params(stop_token_ids={-1})

    @pytest.mark.parametrize("value", [True, 1.0, "1", None])
    def test_rejects_non_integer_ids(self, value: object) -> None:
        with pytest.raises(TypeError, match="stop_token_ids must be an integer"):
            _params(stop_token_ids=[value])


class TestBooleans:
    @pytest.mark.parametrize("value", [True, False])
    def test_accepts_ignore_eos(self, value: bool) -> None:
        assert SamplingParams(ignore_eos=value).ignore_eos is value

    @pytest.mark.parametrize("value", [1, 0, None, "true", []])
    def test_rejects_non_boolean_ignore_eos(self, value: object) -> None:
        with pytest.raises(TypeError, match="ignore_eos must be bool"):
            _params(ignore_eos=value)

    @pytest.mark.parametrize("value", [True, False])
    def test_accepts_include_stop_str_in_output(self, value: bool) -> None:
        assert SamplingParams(include_stop_str_in_output=value).include_stop_str_in_output is value

    @pytest.mark.parametrize("value", [1, 0, None, "true", []])
    def test_rejects_non_boolean_include_stop_str_in_output(self, value: object) -> None:
        with pytest.raises(TypeError, match="include_stop_str_in_output must be bool"):
            _params(include_stop_str_in_output=value)

    @pytest.mark.parametrize("value", [True, False])
    def test_accepts_return_sampling_support(self, value: bool) -> None:
        assert SamplingParams(return_sampling_support=value).return_sampling_support is value

    @pytest.mark.parametrize("value", [1, 0, None, "true", []])
    def test_rejects_non_boolean_return_sampling_support(self, value: object) -> None:
        with pytest.raises(TypeError, match="return_sampling_support must be bool"):
            _params(return_sampling_support=value)


class TestLogprobWidths:
    @pytest.mark.parametrize("name", ["logprobs", "prompt_logprobs"])
    @pytest.mark.parametrize("value", [-1, 0, 5])
    def test_accepts_supported_widths(self, name: str, value: int) -> None:
        params = SamplingParams(**cast("Any", {name: value}))

        assert getattr(params, name) == value

    @pytest.mark.parametrize("name", ["logprobs", "prompt_logprobs"])
    def test_accepts_none(self, name: str) -> None:
        assert getattr(SamplingParams(), name) is None

    @pytest.mark.parametrize("name", ["logprobs", "prompt_logprobs"])
    def test_rejects_width_below_minus_one(self, name: str) -> None:
        with pytest.raises(ValueError, match=f"{name} must be >= -1"):
            SamplingParams(**cast("Any", {name: -2}))

    @pytest.mark.parametrize("name", ["logprobs", "prompt_logprobs"])
    @pytest.mark.parametrize("value", [True, False, 1.0, "1"])
    def test_rejects_non_integer_width(self, name: str, value: object) -> None:
        with pytest.raises(TypeError, match=f"{name} must be an integer"):
            SamplingParams(**cast("Any", {name: value}))


class TestLogprobTokenIds:
    def test_none_by_default(self) -> None:
        assert SamplingParams().logprob_token_ids is None

    @pytest.mark.parametrize("value", [(1, 2), [1, 2], ()])
    def test_accepts_tuple_or_list(self, value: object) -> None:
        params = SamplingParams(**cast("Any", {"logprob_token_ids": value}))

        assert params.logprob_token_ids == tuple(cast("Any", value))

    @pytest.mark.parametrize("value", [{1, 2}, "12", 12, None])
    def test_rejects_non_sequence(self, value: object) -> None:
        if value is None:
            assert SamplingParams(logprob_token_ids=None).logprob_token_ids is None
            return
        with pytest.raises(InvalidRequestError, match="must be a sequence of token ids"):
            _params(logprob_token_ids=value)

    def test_rejects_negative_ids(self) -> None:
        with pytest.raises(ValueError, match="logprob_token_ids must be >= 0"):
            SamplingParams(logprob_token_ids=(0, -1))

    @pytest.mark.parametrize("value", [True, 1.0, "1"])
    def test_rejects_non_integer_ids(self, value: object) -> None:
        with pytest.raises(TypeError, match="logprob_token_ids must be an integer"):
            _params(logprob_token_ids=[value])

    def test_accepts_maximum_number_of_ids(self) -> None:
        ids = tuple(range(MAX_LOGPROB_TOKEN_IDS))
        params = SamplingParams(logprob_token_ids=ids)

        assert params.logprob_token_ids == ids

    def test_rejects_too_many_ids(self) -> None:
        ids = tuple(range(MAX_LOGPROB_TOKEN_IDS + 1))

        with pytest.raises(
            InvalidRequestError,
            match=f"logprob_token_ids must contain at most {MAX_LOGPROB_TOKEN_IDS} ids",
        ):
            SamplingParams(logprob_token_ids=ids)

    def test_accepts_matching_logprobs_width(self) -> None:
        params = SamplingParams(logprobs=2, logprob_token_ids=(1, 2))

        assert params.logprobs == 2
        assert params.logprob_token_ids == (1, 2)

    def test_rejects_mismatched_logprobs_width(self) -> None:
        with pytest.raises(
            InvalidRequestError,
            match="logprobs must equal len\\(logprob_token_ids\\)",
        ):
            SamplingParams(logprobs=2, logprob_token_ids=(1, 2, 3))

    def test_minus_one_logprobs_skips_equality(self) -> None:
        params = SamplingParams(logprobs=-1, logprob_token_ids=(1, 2, 3))

        assert params.logprobs == -1

    def test_none_logprobs_skips_equality(self) -> None:
        params = SamplingParams(logprob_token_ids=(1, 2, 3))

        assert params.logprobs is None

    def test_zero_logprobs_requires_empty_ids(self) -> None:
        assert SamplingParams(logprobs=0, logprob_token_ids=()).logprob_token_ids == ()

        with pytest.raises(
            InvalidRequestError,
            match="logprobs must equal len\\(logprob_token_ids\\)",
        ):
            SamplingParams(logprobs=0, logprob_token_ids=(1,))


class TestLogprobMode:
    def test_default_is_raw(self) -> None:
        assert SamplingParams().logprob_mode is LogprobMode.RAW

    @pytest.mark.parametrize("value", ["raw", "sampling"])
    def test_accepts_string_values(self, value: str) -> None:
        assert SamplingParams(logprob_mode=cast("Any", value)).logprob_mode is LogprobMode(value)

    @pytest.mark.parametrize("value", [LogprobMode.RAW, LogprobMode.SAMPLING])
    def test_accepts_enum_values(self, value: LogprobMode) -> None:
        assert SamplingParams(logprob_mode=value).logprob_mode is value

    @pytest.mark.parametrize("value", ["raw", "sampling"])
    def test_accepts_value_like_objects(self, value: str) -> None:
        params = SamplingParams(logprob_mode=cast("Any", _ModeLike(value)))

        assert params.logprob_mode is LogprobMode(value)

    @pytest.mark.parametrize("value", ["bogus", "RAW", ""])
    def test_rejects_unknown_strings(self, value: str) -> None:
        with pytest.raises(InvalidRequestError, match="logprob_mode must be 'raw' or 'sampling'"):
            SamplingParams(logprob_mode=cast("Any", value))

    @pytest.mark.parametrize("value", [None, 1, _ModeLike(1)])
    def test_rejects_values_without_a_string_value(self, value: object) -> None:
        with pytest.raises(
            InvalidRequestError,
            match="logprob_mode must be a LogprobMode or its string value",
        ):
            _params(logprob_mode=value)


class TestLogitBias:
    def test_none_by_default(self) -> None:
        assert SamplingParams().logit_bias is None

    def test_mapping_is_sorted_and_values_are_floats(self) -> None:
        params = SamplingParams(logit_bias={2: -1, 1: 0.5})

        assert params.logit_bias == ((1, 0.5), (2, -1.0))

    def test_sequence_is_sorted(self) -> None:
        params = SamplingParams(logit_bias=((2, -1.0), (1, 0.5)))

        assert params.logit_bias == ((1, 0.5), (2, -1.0))

    @pytest.mark.parametrize("value", [{}, []])
    def test_empty_container_becomes_empty_tuple(self, value: object) -> None:
        params = SamplingParams(**cast("Any", {"logit_bias": value}))

        assert params.logit_bias == ()

    @pytest.mark.parametrize("value", ["1", {1, 2}, 1, object()])
    def test_rejects_invalid_container(self, value: object) -> None:
        with pytest.raises(InvalidRequestError, match="logit_bias must be a mapping"):
            _params(logit_bias=value)

    @pytest.mark.parametrize("value", [[[1, 0.5]], ["ab"], [1]])
    def test_rejects_invalid_entries(self, value: object) -> None:
        with pytest.raises(InvalidRequestError, match="logit_bias entries must be"):
            _params(logit_bias=value)

    @pytest.mark.parametrize("entry", [(1,), (1, 0.5, 2)])
    def test_rejects_wrong_entry_length(self, entry: tuple[object, ...]) -> None:
        with pytest.raises(InvalidRequestError, match="logit_bias entries must be"):
            _params(logit_bias=[entry])

    @pytest.mark.parametrize("token", [-1, -100])
    def test_rejects_negative_token_id(self, token: int) -> None:
        with pytest.raises(ValueError, match="logit_bias token must be >= 0"):
            SamplingParams(logit_bias=((token, 0.0),))

    @pytest.mark.parametrize("token", [True, False, 1.0, "1", None])
    def test_rejects_non_integer_token_id(self, token: object) -> None:
        with pytest.raises(TypeError, match="logit_bias token must be an integer"):
            _params(logit_bias=[(token, 0.0)])

    @pytest.mark.parametrize("bias", [math.nan, math.inf, -math.inf])
    def test_rejects_non_finite_bias(self, bias: float) -> None:
        with pytest.raises(ValueError, match="logit_bias value must be finite"):
            SamplingParams(logit_bias=((1, bias),))

    @pytest.mark.parametrize("bias", [True, False, "1", None])
    def test_rejects_non_number_bias(self, bias: object) -> None:
        with pytest.raises(TypeError, match="logit_bias value must be a number"):
            _params(logit_bias=[(1, bias)])

    @pytest.mark.parametrize("bias", [-LOGIT_BIAS_LIMIT, LOGIT_BIAS_LIMIT])
    def test_accepts_bias_at_limit(self, bias: float) -> None:
        assert SamplingParams(logit_bias=((1, bias),)).logit_bias == ((1, float(bias)),)

    @pytest.mark.parametrize("bias", [-100.1, 100.1])
    def test_rejects_bias_above_limit(self, bias: float) -> None:
        with pytest.raises(InvalidRequestError, match="logit_bias value must be in"):
            SamplingParams(logit_bias=((1, bias),))

    def test_rejects_duplicate_token_ids(self) -> None:
        with pytest.raises(InvalidRequestError, match="logit_bias contains duplicate token ids"):
            SamplingParams(logit_bias=((1, 0.0), (1, 0.5)))


class TestAllowedTokenIds:
    def test_none_by_default(self) -> None:
        assert SamplingParams().allowed_token_ids is None

    @pytest.mark.parametrize("value", [[0, 1], (0, 1), {0, 1}, frozenset({0, 1})])
    def test_accepts_iterables(self, value: object) -> None:
        params = SamplingParams(**cast("Any", {"allowed_token_ids": value}))

        assert params.allowed_token_ids == frozenset({0, 1})

    def test_accepts_generator(self) -> None:
        params = _params(allowed_token_ids=(n for n in (0, 1, 1)))

        assert params.allowed_token_ids == frozenset({0, 1})

    def test_rejects_empty(self) -> None:
        with pytest.raises(ValueError, match="allowed_token_ids must not be empty"):
            _params(allowed_token_ids=[])

    @pytest.mark.parametrize("value", ["12", b"12", 12])
    def test_rejects_non_iterables(self, value: object) -> None:
        with pytest.raises(ValueError, match="allowed_token_ids must be an iterable"):
            _params(allowed_token_ids=value)

    def test_rejects_negative_ids(self) -> None:
        with pytest.raises(ValueError, match="allowed_token_ids must be >= 0"):
            _params(allowed_token_ids={-1})

    @pytest.mark.parametrize("value", [True, 1.0, "1", None])
    def test_rejects_non_integer_ids(self, value: object) -> None:
        with pytest.raises(TypeError, match="allowed_token_ids must be an integer"):
            _params(allowed_token_ids=[value])


class TestGreedy:
    def test_temperature_boundary(self) -> None:
        assert SamplingParams(temperature=0.0).is_greedy is True
        assert SamplingParams(temperature=9.999e-6).is_greedy is True
        assert SamplingParams(temperature=1e-5).is_greedy is False

    def test_top_k_one_is_greedy(self) -> None:
        assert SamplingParams(top_k=1).is_greedy is True
        assert SamplingParams(top_k=2).is_greedy is False

    def test_greedy_requires_one_sequence(self) -> None:
        with pytest.raises(InvalidRequestError, match="n must be 1 when sampling is greedy"):
            SamplingParams(n=2, temperature=0.0)

        with pytest.raises(InvalidRequestError, match="n must be 1 when sampling is greedy"):
            SamplingParams(n=2, top_k=1)

    def test_non_greedy_allows_multiple_sequences(self) -> None:
        assert SamplingParams(n=2, temperature=1e-5).n == 2


class TestValidateForModel:
    def test_accepts_valid_request(self) -> None:
        params = SamplingParams(
            stop_token_ids=frozenset({1}),
            logprob_token_ids=(2,),
            logit_bias={3: 0.5},
            allowed_token_ids=frozenset({4}),
        )

        assert (
            params.validate_for_model(
                vocab_size=5,
                prompt_tokens=3,
                max_model_len=131,
                max_logprobs=5,
            )
            is None
        )

    @pytest.mark.parametrize("value", [0, -1])
    def test_rejects_invalid_vocab_size(self, value: int) -> None:
        with pytest.raises(ValueError, match="vocab_size must be >= 1"):
            SamplingParams().validate_for_model(vocab_size=value)

    @pytest.mark.parametrize("value", [True, False, 1.0, "1", None])
    def test_rejects_non_integer_vocab_size(self, value: object) -> None:
        with pytest.raises(TypeError, match="vocab_size must be an integer"):
            _params().validate_for_model(vocab_size=cast("int", value))

    def test_rejects_negative_prompt_tokens(self) -> None:
        with pytest.raises(ValueError, match="prompt_tokens must be >= 0"):
            SamplingParams().validate_for_model(vocab_size=10, prompt_tokens=-1)

    @pytest.mark.parametrize("value", [True, False, 1.0, "1", None])
    def test_rejects_non_integer_prompt_tokens(self, value: object) -> None:
        with pytest.raises(TypeError, match="prompt_tokens must be an integer"):
            _params().validate_for_model(vocab_size=10, prompt_tokens=cast("int", value))

    def test_accepts_context_at_limit(self) -> None:
        params = SamplingParams(max_tokens=5)

        params.validate_for_model(vocab_size=10, prompt_tokens=3, max_model_len=8)

    def test_rejects_context_above_limit(self) -> None:
        params = SamplingParams(max_tokens=5)

        with pytest.raises(
            InvalidRequestError,
            match="prompt_tokens \\+ max_tokens must be <= max_model_len \\(7\\)",
        ):
            params.validate_for_model(vocab_size=10, prompt_tokens=3, max_model_len=7)

    @pytest.mark.parametrize("value", [0, -1])
    def test_rejects_invalid_max_model_len(self, value: int) -> None:
        with pytest.raises(ValueError, match="max_model_len must be >= 1"):
            SamplingParams().validate_for_model(
                vocab_size=10,
                max_model_len=value,
            )

    def test_skips_context_check_without_max_model_len(self) -> None:
        SamplingParams(max_tokens=100).validate_for_model(vocab_size=10, prompt_tokens=1000)

    def test_skips_context_check_without_max_tokens(self) -> None:
        SamplingParams(max_tokens=None).validate_for_model(
            vocab_size=1,
            prompt_tokens=1000,
            max_model_len=1,
        )

    def test_does_not_validate_unused_max_model_len_type(self) -> None:
        SamplingParams(max_tokens=None).validate_for_model(
            vocab_size=1,
            max_model_len=cast("int", "invalid"),
        )

    def test_accepts_logprobs_at_max(self) -> None:
        SamplingParams(logprobs=5).validate_for_model(vocab_size=10, max_logprobs=5)

    def test_rejects_logprobs_above_max(self) -> None:
        with pytest.raises(InvalidRequestError, match="logprobs must be <= max_logprobs \\(4\\)"):
            SamplingParams(logprobs=5).validate_for_model(vocab_size=10, max_logprobs=4)

    def test_rejects_prompt_logprobs_above_max(self) -> None:
        with pytest.raises(
            InvalidRequestError,
            match="prompt_logprobs must be <= max_logprobs \\(4\\)",
        ):
            SamplingParams(prompt_logprobs=5).validate_for_model(vocab_size=10, max_logprobs=4)

    def test_accepts_prompt_logprobs_at_max(self) -> None:
        SamplingParams(prompt_logprobs=5).validate_for_model(vocab_size=10, max_logprobs=5)

    def test_accepts_full_vocabulary_prompt_logprobs(self) -> None:
        SamplingParams(prompt_logprobs=-1).validate_for_model(vocab_size=10, max_logprobs=10)

    def test_full_vocabulary_logprobs_use_vocab_size(self) -> None:
        SamplingParams(logprobs=-1).validate_for_model(vocab_size=10, max_logprobs=10)

        with pytest.raises(InvalidRequestError, match="logprobs must be <= max_logprobs \\(9\\)"):
            SamplingParams(logprobs=-1).validate_for_model(vocab_size=10, max_logprobs=9)

    def test_max_logprobs_minus_one_uses_vocab_size(self) -> None:
        SamplingParams(logprobs=10).validate_for_model(vocab_size=10, max_logprobs=-1)

        with pytest.raises(InvalidRequestError, match="logprobs must be <= max_logprobs \\(10\\)"):
            SamplingParams(logprobs=11).validate_for_model(vocab_size=10, max_logprobs=-1)

    def test_rejects_invalid_max_logprobs(self) -> None:
        with pytest.raises(ValueError, match="max_logprobs must be >= -1"):
            SamplingParams().validate_for_model(vocab_size=10, max_logprobs=-2)

    @pytest.mark.parametrize("value", [True, False, 1.0, "1"])
    def test_rejects_non_integer_max_logprobs(self, value: object) -> None:
        with pytest.raises(TypeError, match="max_logprobs must be an integer"):
            SamplingParams().validate_for_model(
                vocab_size=10,
                max_logprobs=cast("int", value),
            )

    def test_skips_logprob_check_when_no_width_is_requested(self) -> None:
        SamplingParams().validate_for_model(vocab_size=10, max_logprobs=0)

    @pytest.mark.parametrize("field", ["stop_token_ids", "logprob_token_ids"])
    def test_rejects_token_id_equal_to_vocab_size(self, field: str) -> None:
        params = SamplingParams(**cast("Any", {field: (10,)}))

        with pytest.raises(
            InvalidRequestError,
            match=f"{field} token id 10 must be < vocab_size 10",
        ):
            params.validate_for_model(vocab_size=10)

    def test_rejects_logit_bias_token_id_equal_to_vocab_size(self) -> None:
        params = SamplingParams(logit_bias={10: 0.5})

        with pytest.raises(
            InvalidRequestError,
            match="logit_bias token id 10 must be < vocab_size 10",
        ):
            params.validate_for_model(vocab_size=10)

    def test_rejects_allowed_token_id_equal_to_vocab_size(self) -> None:
        params = SamplingParams(allowed_token_ids=frozenset({10}))

        with pytest.raises(
            InvalidRequestError,
            match="allowed_token_ids token id 10 must be < vocab_size 10",
        ):
            params.validate_for_model(vocab_size=10)

    def test_checks_logprobs_before_token_ids(self) -> None:
        params = SamplingParams(logprobs=5, stop_token_ids=frozenset({10}))

        with pytest.raises(InvalidRequestError, match="logprobs must be <= max_logprobs"):
            params.validate_for_model(vocab_size=10, max_logprobs=4)
