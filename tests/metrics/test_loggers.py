from __future__ import annotations

import logging

import pytest
from prometheus_client import CollectorRegistry, generate_latest

from ayaka.metrics import IterationStats, LoggingStatLogger, PrometheusStatLogger

pytestmark = pytest.mark.unit


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def test_logging_logger_aggregates_between_intervals(
    caplog: pytest.LogCaptureFixture,
) -> None:
    clock = FakeClock()
    logger = LoggingStatLogger(clock=clock)
    with caplog.at_level(logging.INFO, logger="ayaka.engine.stats"):
        for step in range(4):
            logger.record(IterationStats(step_id=step, num_new_tokens=10, num_settled=1))
        clock.now = 5.0
        logger.log()

    line = caplog.records[-1].getMessage()
    assert "running=0 waiting=0" in line
    assert "output_tokens=40 (8.0 tok/s)" in line
    assert "settled=4" in line
    assert "over 5.0s" in line


def test_logging_logger_resets_after_emitting(caplog: pytest.LogCaptureFixture) -> None:
    clock = FakeClock()
    logger = LoggingStatLogger(clock=clock)
    with caplog.at_level(logging.INFO, logger="ayaka.engine.stats"):
        logger.record(IterationStats(step_id=0, num_new_tokens=10))
        clock.now = 5.0
        logger.log()
        logger.record(IterationStats(step_id=1, num_new_tokens=5))
        clock.now = 10.0
        logger.log()

    line = caplog.records[-1].getMessage()
    assert "output_tokens=5 (1.0 tok/s)" in line


def test_logging_logger_skips_zero_elapsed_without_resetting(
    caplog: pytest.LogCaptureFixture,
) -> None:
    clock = FakeClock()
    logger = LoggingStatLogger(clock=clock)
    logger.record(IterationStats(step_id=0, num_new_tokens=3))

    with caplog.at_level(logging.INFO, logger="ayaka.engine.stats"):
        logger.log()
        assert caplog.records == []
        clock.now = 5.0
        logger.log()

    assert "output_tokens=3" in caplog.records[-1].getMessage()


def test_prometheus_logger_exports_engine_metrics() -> None:
    logger = PrometheusStatLogger()
    logger.record(
        IterationStats(
            step_id=1,
            num_running=3,
            num_waiting=2,
            num_new_tokens=7,
            num_scheduled_tokens=11,
            num_finished=1,
            num_aborted=2,
            num_failed=3,
            num_preempted=4,
            step_seconds=0.01,
        )
    )

    body = generate_latest(logger.registry).decode()
    assert "ayaka_engine_running_requests 3" in body
    assert "ayaka_engine_waiting_requests 2" in body
    assert "ayaka_engine_output_tokens_total 7" in body
    assert "ayaka_engine_scheduled_tokens_total 11" in body
    assert "ayaka_engine_preemptions_total 4" in body
    assert 'reason="finished"' in body
    assert 'reason="aborted"' in body
    assert 'reason="failed"' in body
    assert "ayaka_engine_step_seconds_count 1" in body


def test_prometheus_logger_does_not_create_zero_terminal_series() -> None:
    logger = PrometheusStatLogger()
    logger.record(IterationStats(step_id=1))

    body = generate_latest(logger.registry).decode()
    assert "reason=" not in body


def test_prometheus_logger_accepts_an_external_registry() -> None:
    registry = CollectorRegistry()
    logger = PrometheusStatLogger(registry=registry)

    assert logger.registry is registry


def test_prometheus_logger_log_is_a_noop() -> None:
    logger = PrometheusStatLogger()

    assert logger.log() is None
