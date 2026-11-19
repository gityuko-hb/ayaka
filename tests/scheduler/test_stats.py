"""Scheduler statistics: exact counts on hand-built steps, taken once per update."""

from __future__ import annotations

import pytest

from ayaka.configs.scheduler import SchedulerConfig
from ayaka.core.errors import BackendError
from ayaka.core.types import FinishReason, ForwardMode
from ayaka.scheduler.scheduler import Scheduler
from ayaka.scheduler.step import StepOutput
from fake_runner import FakeRunner, StepLoop, make_kv, make_request, rid

pytestmark = pytest.mark.unit


class Clock:
    def __init__(self) -> None:
        self.now = 50.0

    def __call__(self) -> float:
        return self.now


def _loop(*, budget: int = 2048, num_blocks: int = 33) -> tuple[StepLoop, Clock]:
    kv = make_kv(num_blocks)
    clock = Clock()
    scheduler = Scheduler(SchedulerConfig(max_tokens_per_step=budget), kv, clock=clock)
    return StepLoop(scheduler, FakeRunner(kv), kv), clock


def test_no_stats_exist_before_a_step_and_each_is_taken_once() -> None:
    loop, _ = _loop()

    assert loop.scheduler.take_stats() is None
    loop.scheduler.add_request(make_request("a", 4, max_tokens=2))
    loop.step()

    assert loop.scheduler.take_stats() is not None
    assert loop.scheduler.take_stats() is None


def test_a_prefill_step_counts_rows_tokens_and_the_queue_delay() -> None:
    loop, clock = _loop()
    loop.scheduler.add_request(make_request("a", 6, max_tokens=3))
    clock.now += 2.0
    loop.scheduler.add_request(make_request("b", 4, max_tokens=1))
    clock.now += 0.5

    loop.step()
    stats = loop.scheduler.take_stats()

    assert stats is not None
    assert stats.mode is ForwardMode.PREFILL
    assert stats.step_id == 1
    assert stats.num_scheduled_tokens == 10
    assert stats.num_new_tokens == 2  # one first token each
    assert stats.num_finished == 1  # "b" had max_tokens=1
    assert (stats.num_aborted, stats.num_failed, stats.num_preempted) == (0, 0, 0)
    assert stats.num_settled == 1
    assert (stats.num_running, stats.num_waiting) == (1, 0)
    assert stats.max_queue_delay == pytest.approx(2.5)
    assert stats.prefill_streak == 1
    assert stats.kv_free_blocks == loop.kv.free_counts


def test_a_decode_step_and_the_end_of_a_request() -> None:
    loop, _ = _loop()
    loop.scheduler.add_request(make_request("a", 3, max_tokens=2))
    loop.step()
    loop.scheduler.take_stats()

    loop.step()
    stats = loop.scheduler.take_stats()

    assert stats is not None
    assert stats.mode is ForwardMode.DECODE
    assert stats.num_scheduled_tokens == 1
    assert stats.num_new_tokens == 1
    assert (stats.num_finished, stats.num_settled, stats.num_running) == (1, 1, 0)
    assert stats.max_queue_delay == 0.0
    assert stats.prefill_streak == 0


def test_a_mid_prompt_chunk_reports_zero_new_tokens_not_nothing() -> None:
    loop, _ = _loop(budget=4)
    loop.scheduler.add_request(make_request("a", 10, max_tokens=2))

    loop.step()
    stats = loop.scheduler.take_stats()

    assert stats is not None
    assert stats.num_scheduled_tokens == 4
    assert stats.num_new_tokens == 0


def test_aborts_between_steps_and_errors_are_counted_with_the_next_step() -> None:
    loop, _ = _loop()
    for name in ("a", "b", "c"):
        loop.scheduler.add_request(make_request(name, 4, max_tokens=5))
    loop.step()
    loop.scheduler.take_stats()
    loop.scheduler.abort(rid("a"))  # outside a step: ends now

    step = loop.scheduler.schedule()
    assert step is not None
    loop.scheduler.update(
        StepOutput(
            step_id=step.step_id,
            request_ids=step.request_ids,
            error=BackendError("boom").to_info(),
        )
    )
    stats = loop.scheduler.take_stats()

    assert stats is not None
    assert stats.num_aborted == 1
    assert stats.num_failed == 2
    assert stats.num_settled == 3
    assert stats.num_running == 0
    reasons = {e.request_id: e.finish_reason for e in loop.scheduler.take_events() if e.finished}
    assert reasons[rid("b")] is FinishReason.ERROR


def test_a_released_request_is_settled_when_the_runner_releases_it() -> None:
    kv = make_kv(33)
    scheduler = Scheduler(SchedulerConfig(), kv)
    loop = StepLoop(scheduler, FakeRunner(kv, lag_releases=True), kv)
    scheduler.add_request(make_request("a", 4, max_tokens=1))

    loop.step()
    first = scheduler.take_stats()
    scheduler.add_request(make_request("b", 4, max_tokens=1))
    loop.step()
    second = scheduler.take_stats()

    assert first is not None
    assert second is not None
    assert (first.num_finished, first.num_settled) == (1, 0)  # finished, KV still read
    assert second.num_settled >= 1
