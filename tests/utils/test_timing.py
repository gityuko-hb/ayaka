from __future__ import annotations

import pytest

from ayaka.utils.timing import (
    MAX_INTER_TOKEN_SAMPLES,
    Clock,
    RequestTiming,
    Timestamp,
)

pytestmark = pytest.mark.unit


class TestTimestamp:
    def test_conversions(self) -> None:
        stamp = Timestamp(1_000_000)
        assert stamp.as_nanos() == 1_000_000
        assert stamp.as_micros_f64() == 1000.0
        assert stamp.as_millis_f64() == 1.0

    def test_ordering(self) -> None:
        assert Timestamp(1) < Timestamp(2)


class TestClock:
    def test_now_is_a_timestamp(self) -> None:
        assert isinstance(Clock().now(), Timestamp)


class TestRequestTiming:
    def test_queue_time_and_ttft(self) -> None:
        timing = RequestTiming(arrival_ns=10)
        assert timing.queue_time_ns() is None
        assert timing.ttft_ns() is None

        timing.mark_scheduled(20)
        timing.mark_scheduled(30)
        assert timing.queue_time_ns() == 10

        timing.mark_streamed(100)
        assert timing.ttft_ns() == 90

    def test_tpot_requires_two_tokens(self) -> None:
        timing = RequestTiming(arrival_ns=10, first_token_ns=100, last_token_ns=300)
        assert timing.tpot_ns() is None
        timing.num_output_tokens = 3
        assert timing.tpot_ns() == 100

    def test_e2e_prefers_finished_time(self) -> None:
        timing = RequestTiming(arrival_ns=10)
        timing.mark_streamed(100)
        assert timing.e2e_ns() == 90
        timing.mark_finished(400)
        assert timing.e2e_ns() == 390

    def test_streaming_samples_are_bounded_and_ordered(self) -> None:
        timing = RequestTiming(arrival_ns=0)
        for value in range(MAX_INTER_TOKEN_SAMPLES + 100):
            timing.mark_streamed(1_000 + value)
        assert len(timing.inter_token_intervals_ns) == MAX_INTER_TOKEN_SAMPLES

        timing.mark_streamed(1)
        assert len(timing.inter_token_intervals_ns) == MAX_INTER_TOKEN_SAMPLES

    def test_inter_token_stats_use_linear_interpolation(self) -> None:
        timing = RequestTiming(inter_token_intervals_ns=[50, 100])
        stats = timing.inter_token_stats_ns()
        assert stats["min"] == 50.0
        assert stats["mean"] == 75.0
        assert stats["p50"] == 75.0
        assert stats["p95"] == pytest.approx(97.5)
        assert stats["max"] == 100.0

    def test_settlement_percentiles_use_nearest_rank(self) -> None:
        timing = RequestTiming(inter_token_intervals_ns=[1, 2, 3, 4])
        assert timing.inter_token_percentiles_ns() == (3, 4)

    def test_settlement_percentiles_of_empty_buffer(self) -> None:
        assert RequestTiming().inter_token_percentiles_ns() == (0, 0)
