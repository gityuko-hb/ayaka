"""Unit tests for KV pressure watermarks, victim selection, preemption, and timeouts."""

from __future__ import annotations

import pytest
from prometheus_client import generate_latest

from ayaka.configs.scheduler import SchedulerConfig
from ayaka.core.types import FinishReason, ForwardMode, RequestId
from ayaka.metrics.loggers import PrometheusStatLogger
from ayaka.metrics.stats import IterationStats
from ayaka.scheduler.preemption import select_victim
from ayaka.scheduler.request import Request, RequestStatus
from ayaka.scheduler.scheduler import Scheduler
from fake_runner import FakeRunner, StepLoop, expected_tokens, make_kv, make_request

pytestmark = pytest.mark.unit


class _Clock:
    def __init__(self, start: float = 100.0) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now


def test_select_victim_respects_priority_and_policy() -> None:
    r1 = Request.from_tokenized(make_request("r1", 4, max_tokens=4, priority=2), arrival=1.0)
    r2 = Request.from_tokenized(make_request("r2", 8, max_tokens=4, priority=0), arrival=2.0)
    r2.computed = 8
    r2.output.extend([1, 2, 3])
    r3 = Request.from_tokenized(make_request("r3", 4, max_tokens=4, priority=0), arrival=3.0)
    r3.computed = 4
    r3.output.extend([4])

    # Lowest priority wins first, even if newer requests have higher priority.
    assert select_victim([r2, r3, r1], "newest").request_id == RequestId("r3")
    assert select_victim([r2, r3, r1], "lowest_progress").request_id == RequestId("r3")

    # With equal priority, "lowest_progress" picks fewer emitted tokens even if arrived earlier.
    r2.priority = r3.priority = 1
    r2.arrival = 10.0  # newer arrival, more progress
    r3.arrival = 5.0  # older arrival, less progress
    assert select_victim([r3, r2], "newest").request_id == RequestId("r2")
    assert select_victim([r3, r2], "lowest_progress").request_id == RequestId("r3")


def test_pressure_watermark_pauses_admission_then_resumes() -> None:
    # 11 blocks = 10 usable blocks (block_size=4). At 0.60 high threshold, >= 6 blocks pauses.
    kv = make_kv(num_blocks=11, max_requests=4, max_blocks=10)
    clock = _Clock()
    scheduler = Scheduler(
        SchedulerConfig(max_tokens_per_step=128, pressure_high=0.60, pressure_critical=0.90),
        kv,
        clock=clock,
    )
    runner = FakeRunner(kv)
    loop = StepLoop(scheduler, runner, kv)

    first = make_request("first", 24, max_tokens=2)  # 6 blocks -> utilization 0.60 -> HIGH
    second = make_request("second", 4, max_tokens=2)
    scheduler.add_request(first)
    scheduler.add_request(second)

    step1 = loop.step()
    assert step1 is not None
    assert step1.request_ids == (first.request_id,)
    assert scheduler.num_running == 1
    assert scheduler.num_waiting == 1

    # While "first" is running at high pressure, "second" stays waiting even though 4 blocks remain.
    step2 = loop.step()
    assert step2 is not None
    assert step2.mode is ForwardMode.DECODE
    assert step2.request_ids == (first.request_id,)
    assert scheduler.num_waiting == 1
    second_live = scheduler.get(second.request_id)
    assert second_live is not None
    assert second_live.status is RequestStatus.WAITING

    # Once "first" finishes and releases KV, utilization drops to 0.0 and "second" is admitted.
    loop.run()
    assert loop.tokens(first.request_id) == expected_tokens(first)[0]
    assert loop.tokens(second.request_id) == expected_tokens(second)[0]
    kv.check_integrity()


def test_recompute_preemption_with_exact_block_multiple_and_prefix_cache() -> None:
    # 5 blocks = 4 usable blocks (16 tokens). Two requests each take 8-token prompts (2 blocks)
    # and decode 8 tokens (prompt + output = 16, exact multiple of block_size=4).
    kv = make_kv(num_blocks=5, max_requests=4, max_blocks=8, prefix=True)
    clock = _Clock()
    scheduler = Scheduler(
        SchedulerConfig(
            max_running=2,
            max_tokens_per_step=128,
            preemption_policy="lowest_progress",
            pressure_high=0.80,
            pressure_critical=0.95,
        ),
        kv,
        clock=clock,
    )
    runner = FakeRunner(kv)
    loop = StepLoop(scheduler, runner, kv)

    high = make_request("high", 8, max_tokens=8, priority=1)
    low = make_request("low", 8, max_tokens=8, priority=0)
    scheduler.add_request(high)
    clock.now += 1.0
    scheduler.add_request(low)

    loop.run()

    assert loop.tokens(high.request_id) == expected_tokens(high)[0]
    assert loop.tokens(low.request_id) == expected_tokens(low)[0]
    assert kv.stats().num_requests == 0
    kv.check_integrity()


def test_preemption_timeout_expires_starved_request() -> None:
    kv = make_kv(num_blocks=5, max_requests=4, max_blocks=8)
    clock = _Clock(start=10.0)
    scheduler = Scheduler(
        SchedulerConfig(
            max_running=2,
            max_tokens_per_step=128,
            preemption_timeout=2.0,
            pressure_high=0.90,
            pressure_critical=0.95,
        ),
        kv,
        clock=clock,
    )
    runner = FakeRunner(kv)
    loop = StepLoop(scheduler, runner, kv)

    keep = make_request("keep", 8, max_tokens=6, priority=1)
    victim = make_request("victim", 8, max_tokens=6, priority=0)
    scheduler.add_request(keep)
    clock.now += 0.1
    scheduler.add_request(victim)

    # Run until "victim" is preempted while "keep" holds the pool.
    preempted_seen = False
    for _ in range(10):
        step = loop.step()
        if step is None:
            break
        live = scheduler.get(victim.request_id)
        if live is not None and live.status is RequestStatus.PREEMPTED:
            preempted_seen = True
            break
    assert preempted_seen
    assert scheduler.num_preempted == 1

    # Advance past preemption_timeout while pool stays too tight to resume victim.
    clock.now += 3.0
    loop.step()
    victim_events = [e for e in loop.events if e.request_id == victim.request_id and e.finished]
    assert len(victim_events) == 1
    assert victim_events[0].finish_reason is FinishReason.TIMEOUT
    assert victim_events[0].error is not None
    assert victim_events[0].error.code == "preempted_timeout"

    loop.run()
    assert loop.tokens(keep.request_id) == expected_tokens(keep)[0]
    kv.check_integrity()


def test_prometheus_logger_records_preemptions_and_pressure() -> None:
    logger = PrometheusStatLogger()
    logger.record(
        IterationStats(
            step_id=1,
            num_running=2,
            num_waiting=1,
            num_preempted=3,
            num_scheduled_tokens=8,
            num_new_tokens=2,
            num_finished=0,
            num_aborted=0,
            num_failed=0,
            num_settled=0,
            step_seconds=0.01,
            pressure_level="critical",
            num_oom_recoveries=2,
        )
    )
    text = generate_latest(logger.registry).decode("utf-8")
    assert "ayaka_engine_preemptions_total 3.0" in text
    assert "ayaka_engine_kv_pressure_level 2.0" in text
    assert "ayaka_engine_oom_recoveries_total 2.0" in text
