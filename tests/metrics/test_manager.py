from __future__ import annotations

import pytest

from ayaka.metrics import IterationStats, StatLoggerManager

pytestmark = pytest.mark.unit


class SpyLogger:
    def __init__(self) -> None:
        self.records: list[IterationStats] = []
        self.log_calls = 0

    def record(self, stats: IterationStats) -> None:
        self.records.append(stats)

    def log(self) -> None:
        self.log_calls += 1


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def test_records_fan_out_and_log_only_after_the_interval() -> None:
    spy, clock = SpyLogger(), FakeClock()
    manager = StatLoggerManager(loggers=[spy], log_interval=10.0, clock=clock)
    stats = IterationStats(step_id=1, num_new_tokens=4)

    manager.record(stats)
    manager.maybe_log()
    assert spy.records == [stats]
    assert spy.log_calls == 0

    clock.now = 9.999
    manager.maybe_log()
    assert spy.log_calls == 0

    clock.now = 10.0
    manager.maybe_log()
    assert spy.log_calls == 1

    manager.maybe_log()
    assert spy.log_calls == 1


def test_log_forces_emission_and_restarts_the_interval() -> None:
    spy, clock = SpyLogger(), FakeClock()
    manager = StatLoggerManager(loggers=[spy], log_interval=10.0, clock=clock)

    manager.log()
    assert spy.log_calls == 1

    clock.now = 5.0
    manager.maybe_log()
    assert spy.log_calls == 1

    clock.now = 5.0
    manager.log()
    clock.now = 14.0
    manager.maybe_log()
    assert spy.log_calls == 2


def test_loggers_property_preserves_order() -> None:
    first, second = SpyLogger(), SpyLogger()
    manager = StatLoggerManager(loggers=[first, second])

    assert manager.loggers == (first, second)


def test_log_interval_must_be_positive() -> None:
    with pytest.raises(ValueError, match="log_interval"):
        StatLoggerManager(log_interval=0)
