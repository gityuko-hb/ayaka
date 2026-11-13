from __future__ import annotations

import math
from dataclasses import FrozenInstanceError, replace
from typing import Any, cast

import pytest

from ayaka.core.errors import ErrorInfo, InvalidRequestError
from ayaka.core.outputs import (
    CompletionOutput,
    RequestMetrics,
    RequestOutput,
    TokenLogprob,
    Usage,
)
from ayaka.core.types import FinishReason, RequestId, TokenId

pytestmark = pytest.mark.unit

_METRIC_FIELDS = (
    "arrival_time",
    "queued_time",
    "prefill_time",
    "decode_time",
    "ttft",
    "tpot",
)


class TestUsage:
    def test_defaults(self) -> None:
        usage = Usage()

        assert usage.prompt_tokens == 0
        assert usage.completion_tokens == 0
        assert usage.cached_tokens == 0
        assert usage.total_tokens == 0

    @pytest.mark.parametrize(
        ("prompt_tokens", "completion_tokens", "cached_tokens", "total"),
        [(1, 2, 0, 3), (5, 0, 5, 5), (0, 4, 0, 4)],
    )
    def test_valid_counts(
        self,
        prompt_tokens: int,
        completion_tokens: int,
        cached_tokens: int,
        total: int,
    ) -> None:
        usage = Usage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cached_tokens=cached_tokens,
        )

        assert usage.total_tokens == total

    @pytest.mark.parametrize("field", ["prompt_tokens", "completion_tokens", "cached_tokens"])
    def test_rejects_negative_counts(self, field: str) -> None:
        with pytest.raises(ValueError, match=f"{field} must be >= 0"):
            replace(Usage(), **{field: -1})

    @pytest.mark.parametrize("field", ["prompt_tokens", "completion_tokens", "cached_tokens"])
    @pytest.mark.parametrize("value", [True, False, 1.0, "1", None])
    def test_rejects_non_integer_counts(self, field: str, value: object) -> None:
        with pytest.raises(TypeError, match=f"{field} must be an integer"):
            replace(Usage(), **{field: value})

    def test_rejects_cached_tokens_above_prompt_tokens(self) -> None:
        with pytest.raises(InvalidRequestError, match="cached_tokens must be <= prompt_tokens"):
            Usage(prompt_tokens=1, cached_tokens=2)

    def test_allows_cached_tokens_equal_to_prompt_tokens(self) -> None:
        usage = Usage(prompt_tokens=2, cached_tokens=2)

        assert usage.cached_tokens == 2

    def test_is_frozen(self) -> None:
        usage = Usage()

        with pytest.raises(FrozenInstanceError):
            cast("Any", usage).prompt_tokens = 1


class TestRequestMetrics:
    def test_defaults(self) -> None:
        metrics = RequestMetrics()

        for field in _METRIC_FIELDS:
            assert getattr(metrics, field) == 0.0

    def test_accepts_numbers_and_converts_to_float(self) -> None:
        metrics = RequestMetrics(
            arrival_time=1,
            queued_time=2.5,
            prefill_time=3,
            decode_time=4.5,
            ttft=5,
            tpot=6.5,
        )

        assert metrics.arrival_time == 1.0
        assert metrics.queued_time == 2.5
        assert metrics.prefill_time == 3.0
        assert metrics.decode_time == 4.5
        assert metrics.ttft == 5.0
        assert metrics.tpot == 6.5
        assert all(isinstance(getattr(metrics, field), float) for field in _METRIC_FIELDS)

    def test_accepts_infinity(self) -> None:
        metrics = RequestMetrics(ttft=math.inf)

        assert metrics.ttft == math.inf

    @pytest.mark.parametrize("field", _METRIC_FIELDS)
    def test_rejects_negative_values(self, field: str) -> None:
        with pytest.raises(ValueError, match=f"{field} must be >= 0.0"):
            replace(RequestMetrics(), **{field: -1.0})

    @pytest.mark.parametrize("field", _METRIC_FIELDS)
    @pytest.mark.parametrize("value", [True, False, "1", None, [], object()])
    def test_rejects_non_numbers(self, field: str, value: object) -> None:
        with pytest.raises(TypeError, match=f"{field} must be a number"):
            replace(RequestMetrics(), **{field: value})

    @pytest.mark.parametrize("field", _METRIC_FIELDS)
    def test_rejects_nan(self, field: str) -> None:
        with pytest.raises(ValueError, match=f"{field} must not be NaN"):
            replace(RequestMetrics(), **{field: math.nan})

    def test_is_frozen(self) -> None:
        metrics = RequestMetrics()

        with pytest.raises(FrozenInstanceError):
            cast("Any", metrics).ttft = 1.0


class TestTokenLogprob:
    def test_defaults(self) -> None:
        logprob = TokenLogprob(token_id=1, logprob=-0.5)

        assert logprob.token_id == 1
        assert logprob.logprob == -0.5
        assert logprob.rank is None
        assert logprob.decoded is None
        assert logprob.top == ()

    def test_full_value(self) -> None:
        logprob = TokenLogprob(
            token_id=2,
            logprob=-1.5,
            rank=0,
            decoded="hello",
            top=((2, -1.5), (3, -2.5)),
        )

        assert logprob.rank == 0
        assert logprob.decoded == "hello"
        assert logprob.top == ((2, -1.5), (3, -2.5))

    @pytest.mark.parametrize("value", [-1, -100])
    def test_rejects_negative_token_id(self, value: int) -> None:
        with pytest.raises(ValueError, match="token_id must be >= 0"):
            TokenLogprob(token_id=value, logprob=0.0)

    @pytest.mark.parametrize("value", [True, False, 1.0, "1", None])
    def test_rejects_non_integer_token_id(self, value: object) -> None:
        with pytest.raises(TypeError, match="token_id must be an integer"):
            replace(TokenLogprob(token_id=0, logprob=0.0), token_id=value)

    @pytest.mark.parametrize("value", [True, False, "1", None, []])
    def test_rejects_non_number_logprob(self, value: object) -> None:
        with pytest.raises(TypeError, match="logprob must be a number"):
            replace(TokenLogprob(token_id=0, logprob=0.0), logprob=value)

    def test_rejects_nan_logprob(self) -> None:
        with pytest.raises(ValueError, match="logprob must not be NaN"):
            TokenLogprob(token_id=0, logprob=math.nan)

    def test_accepts_infinite_logprob(self) -> None:
        logprob = TokenLogprob(token_id=0, logprob=-math.inf)

        assert logprob.logprob == -math.inf

    @pytest.mark.parametrize("value", [-1, -100])
    def test_rejects_negative_rank(self, value: int) -> None:
        with pytest.raises(ValueError, match="rank must be >= 0"):
            TokenLogprob(token_id=0, logprob=0.0, rank=value)

    @pytest.mark.parametrize("value", [True, False, 1.0, "1"])
    def test_rejects_non_integer_rank(self, value: object) -> None:
        with pytest.raises(TypeError, match="rank must be an integer"):
            replace(TokenLogprob(token_id=0, logprob=0.0), rank=value)

    def test_accepts_none_rank(self) -> None:
        assert TokenLogprob(token_id=0, logprob=0.0, rank=None).rank is None

    @pytest.mark.parametrize(
        ("value", "message"),
        [
            ([], "top must be a tuple of \\(token_id, logprob\\) pairs"),
            ([(1, -0.1)], "top must be a tuple of \\(token_id, logprob\\) pairs"),
            (((1,),), "top entries must be \\(token_id, logprob\\) pairs"),
            (((1, 2, 3),), "top entries must be \\(token_id, logprob\\) pairs"),
            (([1, -0.1],), "top entries must be \\(token_id, logprob\\) pairs"),
        ],
    )
    def test_rejects_malformed_top(self, value: object, message: str) -> None:
        with pytest.raises(InvalidRequestError, match=message):
            replace(TokenLogprob(token_id=0, logprob=0.0), top=value)

    def test_rejects_negative_top_token_id(self) -> None:
        with pytest.raises(ValueError, match="top token id must be >= 0"):
            TokenLogprob(token_id=0, logprob=0.0, top=((-1, -0.1),))

    @pytest.mark.parametrize("value", [True, False, 1.0, "1", None])
    def test_rejects_non_integer_top_token_id(self, value: object) -> None:
        with pytest.raises(TypeError, match="top token id must be an integer"):
            TokenLogprob(token_id=0, logprob=0.0, top=((cast("TokenId", value), -0.1),))

    @pytest.mark.parametrize("value", [True, False, "1", None, []])
    def test_rejects_non_number_top_logprob(self, value: object) -> None:
        with pytest.raises(TypeError, match="top logprob must be a number"):
            TokenLogprob(token_id=0, logprob=0.0, top=((1, cast("float", value)),))

    def test_rejects_nan_top_logprob(self) -> None:
        with pytest.raises(ValueError, match="top logprob must not be NaN"):
            TokenLogprob(token_id=0, logprob=0.0, top=((1, math.nan),))

    def test_is_frozen(self) -> None:
        logprob = TokenLogprob(token_id=0, logprob=0.0)

        with pytest.raises(FrozenInstanceError):
            cast("Any", logprob).token_id = 1


class TestCompletionOutput:
    def test_defaults(self) -> None:
        output = CompletionOutput(index=0)

        assert output.index == 0
        assert output.token_ids == ()
        assert output.text == ""
        assert output.finish_reason is None
        assert output.stop_reason is None
        assert output.cumulative_logprob is None
        assert output.logprobs is None
        assert output.token_ids_logprobs is None

    def test_full_value(self) -> None:
        logprob = TokenLogprob(token_id=1, logprob=-0.5)
        output = CompletionOutput(
            index=1,
            token_ids=(1, 2, 3),
            text="hi",
            finish_reason=FinishReason.STOP,
            stop_reason="</s>",
            cumulative_logprob=-1.5,
            logprobs=(logprob,),
            token_ids_logprobs=(logprob,),
        )

        assert output.index == 1
        assert output.token_ids == (1, 2, 3)
        assert output.text == "hi"
        assert output.finish_reason is FinishReason.STOP
        assert output.stop_reason == "</s>"
        assert output.cumulative_logprob == -1.5
        assert output.logprobs == (logprob,)
        assert output.token_ids_logprobs == (logprob,)

    @pytest.mark.parametrize("value", [-1, -100])
    def test_rejects_negative_index(self, value: int) -> None:
        with pytest.raises(ValueError, match="index must be >= 0"):
            CompletionOutput(index=value)

    @pytest.mark.parametrize("value", [True, False, 1.0, "1", None])
    def test_rejects_non_integer_index(self, value: object) -> None:
        with pytest.raises(TypeError, match="index must be an integer"):
            replace(CompletionOutput(index=0), index=value)

    @pytest.mark.parametrize("value", [[1], {1}, "1", None])
    def test_rejects_non_tuple_token_ids(self, value: object) -> None:
        with pytest.raises(ValueError, match="token_ids must be a tuple of token ids"):
            replace(CompletionOutput(index=0), token_ids=value)

    def test_rejects_negative_token_ids(self) -> None:
        with pytest.raises(ValueError, match="token_ids must be >= 0"):
            CompletionOutput(index=0, token_ids=(0, -1))

    @pytest.mark.parametrize("value", [(True,), (1.0,), ("1",)])
    def test_rejects_non_integer_token_ids(self, value: tuple[object, ...]) -> None:
        with pytest.raises(TypeError, match="token_ids must be an integer"):
            replace(CompletionOutput(index=0), token_ids=value)

    @pytest.mark.parametrize("value", [0, 1, 100])
    def test_accepts_integer_stop_reason(self, value: int) -> None:
        assert CompletionOutput(index=0, stop_reason=value).stop_reason == value

    @pytest.mark.parametrize("value", ["</s>", "stop", " a "])
    def test_accepts_string_stop_reason(self, value: str) -> None:
        assert CompletionOutput(index=0, stop_reason=value).stop_reason == value

    @pytest.mark.parametrize("value", ["", "   ", "\t\n"])
    def test_rejects_blank_string_stop_reason(self, value: str) -> None:
        with pytest.raises(ValueError, match="stop_reason must be a non-empty string"):
            CompletionOutput(index=0, stop_reason=value)

    def test_rejects_negative_integer_stop_reason(self) -> None:
        with pytest.raises(ValueError, match="stop_reason must be >= 0"):
            CompletionOutput(index=0, stop_reason=-1)

    @pytest.mark.parametrize("value", [True, False, 1.0, []])
    def test_rejects_invalid_stop_reason_type(self, value: object) -> None:
        with pytest.raises(TypeError, match="stop_reason must be an integer"):
            replace(CompletionOutput(index=0), stop_reason=value)

    @pytest.mark.parametrize("value", [True, False, "1", []])
    def test_rejects_non_number_cumulative_logprob(self, value: object) -> None:
        with pytest.raises(TypeError, match="cumulative_logprob must be a number"):
            replace(CompletionOutput(index=0), cumulative_logprob=value)

    def test_rejects_nan_cumulative_logprob(self) -> None:
        with pytest.raises(ValueError, match="cumulative_logprob must not be NaN"):
            CompletionOutput(index=0, cumulative_logprob=math.nan)

    def test_accepts_infinite_cumulative_logprob(self) -> None:
        output = CompletionOutput(index=0, cumulative_logprob=-math.inf)

        assert output.cumulative_logprob == -math.inf

    def test_is_frozen(self) -> None:
        output = CompletionOutput(index=0)

        with pytest.raises(FrozenInstanceError):
            cast("Any", output).index = 1


class TestRequestOutput:
    def test_defaults(self) -> None:
        output = RequestOutput(request_id=RequestId("req-1"))

        assert output.request_id == "req-1"
        assert output.prompt_token_ids is None
        assert output.outputs == ()
        assert output.finished is False
        assert output.usage == Usage()
        assert output.metrics is None
        assert output.error is None

    def test_full_value(self) -> None:
        completion = CompletionOutput(index=0, token_ids=(1,), finish_reason=FinishReason.STOP)
        usage = Usage(prompt_tokens=2, completion_tokens=1)
        metrics = RequestMetrics(ttft=0.5)
        error = ErrorInfo("bad", "failed")
        output = RequestOutput(
            request_id=RequestId("req-1"),
            prompt_token_ids=(1, 2),
            outputs=(completion,),
            finished=True,
            usage=usage,
            metrics=metrics,
            error=error,
        )

        assert output.prompt_token_ids == (1, 2)
        assert output.outputs == (completion,)
        assert output.finished is True
        assert output.usage is usage
        assert output.metrics is metrics
        assert output.error is error

    def test_usage_default_is_per_instance(self) -> None:
        first = RequestOutput(request_id=RequestId("req-1"))
        second = RequestOutput(request_id=RequestId("req-2"))

        assert first.usage is not second.usage

    @pytest.mark.parametrize("value", ["", " ", "\t\n", None, 3])
    def test_rejects_invalid_request_id(self, value: object) -> None:
        with pytest.raises(ValueError, match="request_id must be a non-empty string"):
            replace(RequestOutput(request_id=RequestId("req-1")), request_id=value)

    @pytest.mark.parametrize("value", [[1], {1}, "1"])
    def test_rejects_non_tuple_prompt_token_ids(self, value: object) -> None:
        with pytest.raises(ValueError, match="prompt_token_ids must be a tuple of token ids"):
            replace(RequestOutput(request_id=RequestId("req-1")), prompt_token_ids=value)

    def test_rejects_negative_prompt_token_ids(self) -> None:
        with pytest.raises(ValueError, match="prompt_token_ids must be >= 0"):
            RequestOutput(request_id=RequestId("req-1"), prompt_token_ids=(0, -1))

    @pytest.mark.parametrize("value", [(True,), (1.0,), ("1",)])
    def test_rejects_non_integer_prompt_token_ids(self, value: tuple[object, ...]) -> None:
        with pytest.raises(TypeError, match="prompt_token_ids must be an integer"):
            RequestOutput(
                request_id=RequestId("req-1"),
                prompt_token_ids=cast("tuple[TokenId, ...]", value),
            )

    @pytest.mark.parametrize("value", [1, 0, None, "true", []])
    def test_rejects_non_boolean_finished(self, value: object) -> None:
        with pytest.raises(TypeError, match="finished must be bool"):
            replace(RequestOutput(request_id=RequestId("req-1")), finished=value)

    def test_is_frozen(self) -> None:
        output = RequestOutput(request_id=RequestId("req-1"))

        with pytest.raises(FrozenInstanceError):
            cast("Any", output).finished = True
