"""The scheduler on hand-built scenarios: admission, chunks, steps, stops, aborts and limits."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import TYPE_CHECKING

import pytest

from ayaka.configs.scheduler import SchedulerConfig
from ayaka.core.errors import EngineError, InvalidRequestError, OverloadedError
from ayaka.core.types import FinishReason, ForwardMode
from ayaka.scheduler.request import RequestStatus
from ayaka.scheduler.scheduler import Scheduler
from ayaka.scheduler.step import StepInput, StepOutput
from fake_runner import (
    FakeRunner,
    StepLoop,
    expected_tokens,
    make_kv,
    make_request,
    rid,
    token_at,
)

if TYPE_CHECKING:
    from collections.abc import Collection, Mapping

    from ayaka.core.types import RequestId
    from ayaka.kvcache.manager import KVCacheManager

pytestmark = pytest.mark.unit


class Clock:
    """A clock the test moves by hand."""

    def __init__(self) -> None:
        self.now = 100.0

    def __call__(self) -> float:
        return self.now


@dataclass
class Rig:
    scheduler: Scheduler
    runner: FakeRunner
    loop: StepLoop
    kv: KVCacheManager
    clock: Clock


def rig(
    *,
    num_blocks: int = 33,
    max_requests: int = 4,
    max_blocks: int = 16,
    max_running: int | None = None,
    budget: int = 2048,
    streak: int = 2,
    policy: str = "fcfs",
    max_waiting: int = 1024,
    max_model_len: int | None = None,
    max_chunk_tokens: int | None = None,
    eos: Collection[int] = (),
    queue_timeout: float | None = None,
    draft_accuracy: float = 0.5,
    lag: bool = False,
    fail_steps: frozenset[int] = frozenset(),
    lookahead: Mapping[RequestId, int] | None = None,
) -> Rig:
    kv = make_kv(num_blocks, max_requests=max_requests, max_blocks=max_blocks)
    clock = Clock()
    config = SchedulerConfig(
        max_running=max_running,
        max_tokens_per_step=budget,
        max_waiting=max_waiting,
        policy=policy,  # pyright: ignore[reportArgumentType]
        max_consecutive_prefills=streak,
    )
    scheduler = Scheduler(
        config,
        kv,
        max_model_len=max_model_len,
        max_chunk_tokens=max_chunk_tokens,
        eos_token_ids=eos,
        queue_timeout=queue_timeout,
        clock=clock,
    )
    runner = FakeRunner(kv, draft_accuracy=draft_accuracy, lag_releases=lag, fail_steps=fail_steps)
    loop = StepLoop(scheduler, runner, kv, lookahead=lookahead)
    return Rig(scheduler, runner, loop, kv, clock)


def _modes(r: Rig) -> list[ForwardMode]:
    return [step.mode for step in r.runner.steps]


# -- admission ------------------------------------------------------------------------


def test_a_request_beyond_the_model_context_is_refused_before_it_holds_anything() -> None:
    r = rig(max_model_len=20)

    with pytest.raises(InvalidRequestError, match="model context of 20"):
        r.scheduler.add_request(make_request("a", 15, max_tokens=6))

    assert (r.scheduler.num_waiting, r.scheduler.has_unfinished) == (0, False)
    assert r.scheduler.get(rid("a")) is None


def test_a_prompt_as_long_as_the_context_is_refused_even_without_max_tokens() -> None:
    r = rig(max_model_len=10)
    request = make_request("a", 10)
    unbounded = dataclasses.replace(
        request, sampling=dataclasses.replace(request.sampling, max_tokens=None)
    )

    with pytest.raises(InvalidRequestError, match="shorter than the model context"):
        r.scheduler.add_request(unbounded)


def test_a_prompt_beyond_the_kv_of_one_request_names_the_limit_of_this_machine() -> None:
    r = rig(max_blocks=4)  # 4 blocks of 4 tokens

    with pytest.raises(InvalidRequestError, match=r"16 tokens of KV"):
        r.scheduler.add_request(make_request("a", 17))

    r.scheduler.add_request(make_request("b", 16))


def test_a_prompt_beyond_the_pool_is_refused_with_the_pool_size() -> None:
    r = rig(num_blocks=3, max_blocks=16)  # 2 usable blocks: 8 tokens

    with pytest.raises(InvalidRequestError, match=r"8 tokens of KV"):
        r.scheduler.add_request(make_request("a", 9))


def test_a_full_queue_refuses_and_leaves_the_queue_as_it_was() -> None:
    r = rig(max_waiting=2)
    r.scheduler.add_request(make_request("a", 4))
    r.scheduler.add_request(make_request("b", 4))

    with pytest.raises(OverloadedError):
        r.scheduler.add_request(make_request("c", 4))

    assert r.scheduler.num_waiting == 2
    assert r.scheduler.get(rid("c")) is None
    r.loop.run()
    assert r.loop.tokens(rid("a")) == expected_tokens(make_request("a", 4))[0]


def test_the_checks_run_in_order_so_a_bad_request_is_invalid_not_overloaded() -> None:
    r = rig(max_waiting=1, max_model_len=10)
    r.scheduler.add_request(make_request("a", 4, max_tokens=2))

    with pytest.raises(InvalidRequestError):
        r.scheduler.add_request(make_request("b", 9, max_tokens=5))


def test_an_id_already_in_use_is_refused() -> None:
    r = rig()
    r.scheduler.add_request(make_request("a", 4))

    with pytest.raises(InvalidRequestError, match="already in use"):
        r.scheduler.add_request(make_request("a", 5))

    assert r.scheduler.num_waiting == 1


# -- the shape of steps -----------------------------------------------------------------------


def test_a_single_request_prefills_then_decodes_one_row_per_step() -> None:
    r = rig()
    request = make_request("a", 6, max_tokens=4)
    r.scheduler.add_request(request)

    r.loop.run()

    assert _modes(r) == [
        ForwardMode.PREFILL,
        ForwardMode.DECODE,
        ForwardMode.DECODE,
        ForwardMode.DECODE,
    ]
    assert r.runner.steps[0].tokens == (request.prompt_token_ids,)
    assert r.runner.steps[0].need_logits == (True,)
    assert [s.start_positions for s in r.runner.steps] == [(0,), (6,), (7,), (8,)]
    assert [s.tokens for s in r.runner.steps[1:]] == [
        ((token_at(request.request_id, 6),),),
        ((token_at(request.request_id, 7),),),
        ((token_at(request.request_id, 8),),),
    ]
    assert r.loop.tokens(request.request_id) == expected_tokens(request)[0]
    assert r.loop.finish_reason(request.request_id) is FinishReason.LENGTH
    assert r.kv.stats().num_requests == 0


def test_chunks_end_on_block_boundaries_and_only_the_last_needs_logits() -> None:
    r = rig(budget=4)
    request = make_request("a", 10, max_tokens=2)
    r.scheduler.add_request(request)

    r.loop.run()

    prefill = [s for s in r.runner.steps if s.mode is ForwardMode.PREFILL]
    assert [s.rows_per_request for s in prefill] == [(4,), (4,), (2,)]
    assert [s.start_positions for s in prefill] == [(0,), (4,), (8,)]
    assert [s.need_logits for s in prefill] == [(False,), (False,), (True,)]
    assert r.loop.tokens(request.request_id) == expected_tokens(request)[0]


def test_a_budget_off_the_block_grid_still_ends_the_chunk_on_a_boundary() -> None:
    r = rig(budget=6)
    r.scheduler.add_request(make_request("a", 10, max_tokens=1))

    r.loop.run()

    assert [s.rows_per_request for s in r.runner.steps] == [(4,), (6,)]


def test_a_budget_below_one_block_still_progresses_and_realigns_on_the_next_chunk() -> None:
    r = rig(budget=3)
    r.scheduler.add_request(make_request("a", 7, max_tokens=1))

    r.loop.run()

    assert [s.rows_per_request for s in r.runner.steps] == [(3,), (1,), (3,)]


def test_the_memory_cap_lowers_the_prefill_budget() -> None:
    r = rig(budget=2048, max_chunk_tokens=4)
    r.scheduler.add_request(make_request("a", 9, max_tokens=1))

    r.loop.run()

    assert r.scheduler.token_budget == 4
    assert [s.rows_per_request for s in r.runner.steps] == [(4,), (4,), (1,)]


def test_one_prefill_step_shares_its_budget_across_requests_in_queue_order() -> None:
    r = rig(budget=8)
    a, b = make_request("a", 6, max_tokens=2), make_request("b", 6, max_tokens=2)
    r.scheduler.add_request(a)
    r.scheduler.add_request(b)

    first = r.loop.step()

    assert first is not None
    assert first.request_ids == (a.request_id, b.request_id)
    assert first.rows_per_request == (6, 2)
    assert first.need_logits == (True, False)
    r.loop.run()
    assert r.loop.tokens(a.request_id) == expected_tokens(a)[0]
    assert r.loop.tokens(b.request_id) == expected_tokens(b)[0]


def test_a_half_done_prefill_continues_before_anything_new_is_admitted() -> None:
    r = rig(budget=4)
    long, short = make_request("long", 12, max_tokens=1), make_request("short", 3, max_tokens=1)
    r.scheduler.add_request(long)
    r.loop.step()
    r.scheduler.add_request(short)

    second = r.loop.step()

    assert second is not None
    assert second.request_ids == (long.request_id,)


def test_decode_batches_every_ready_request_in_a_stable_order() -> None:
    r = rig()
    for name in "abc":
        r.scheduler.add_request(make_request(name, 5, max_tokens=4))

    r.loop.run()

    decode = [s for s in r.runner.steps if s.mode is ForwardMode.DECODE]
    assert decode[0].request_ids == (rid("a"), rid("b"), rid("c"))
    assert decode[0].batch_rows == (0, 1, 2)
    assert all(s.rows_per_request == (1,) * s.num_requests for s in decode)


def test_a_request_keeps_its_row_and_a_new_one_takes_a_freed_row() -> None:
    r = rig(max_running=2)
    r.scheduler.add_request(make_request("a", 4, max_tokens=1))
    r.scheduler.add_request(make_request("b", 4, max_tokens=6))
    r.scheduler.add_request(make_request("c", 4, max_tokens=2))

    r.loop.run()

    rows: dict[RequestId, set[int]] = {}
    for step in r.runner.steps:
        for request_id, row in zip(step.request_ids, step.batch_rows, strict=True):
            rows.setdefault(request_id, set()).add(row)
    assert rows == {rid("a"): {0}, rid("b"): {1}, rid("c"): {0}}


def test_running_never_exceeds_max_running() -> None:
    r = rig(max_running=2)
    for name in "abcde":
        r.scheduler.add_request(make_request(name, 4, max_tokens=3))

    peak = 0
    while r.loop.step() is not None:
        peak = max(peak, r.scheduler.num_running)

    assert peak == 2
    assert r.scheduler.max_running == 2


def test_the_block_table_caps_max_running_when_the_config_allows_more() -> None:
    r = rig(max_requests=3, max_running=10)

    assert r.scheduler.max_running == 3


def test_table_updates_report_each_change_once() -> None:
    r = rig()
    r.scheduler.add_request(make_request("a", 6, max_tokens=4))

    steps = [r.loop.step() for _ in range(4)]

    assert steps[0] is not None
    assert steps[0].table_updates == ((0, 0, 0, 2),)  # prompt 6 -> two blocks
    assert steps[1] is not None
    assert steps[1].table_updates == ()  # the pending token fits in block 2
    assert steps[2] is not None
    assert steps[2].table_updates == ()  # position 7 is the last slot of block 2
    assert steps[3] is not None
    assert steps[3].table_updates == ((0, 0, 2, 3),)  # position 8 opens block 3


def test_an_idle_schedule_keeps_the_table_changes_for_the_next_step() -> None:
    r = rig()
    r.scheduler.add_request(make_request("a", 6, max_tokens=4))
    r.loop.step()
    assert r.scheduler.abort(rid("a"))
    assert r.scheduler.schedule() is None

    r.scheduler.add_request(make_request("b", 4, max_tokens=1))
    step = r.scheduler.schedule()

    assert step is not None
    assert (0, 0, 0, 2) in step.table_updates  # row 0 of request a was cleared
    assert (0, 0, 0, 1) in step.table_updates  # and request b took it


# -- anti-starvation -----------------------------------------------------------


def test_decode_is_inserted_after_k_prefill_steps_while_a_request_decodes() -> None:
    r = rig(budget=4, streak=2)
    r.scheduler.add_request(make_request("a", 4, max_tokens=30))
    r.loop.step()
    r.loop.step()  # a prefilled and decoding
    r.scheduler.add_request(make_request("b", 20, max_tokens=1))
    before = len(r.runner.steps)

    for _ in range(8):
        r.loop.step()

    after = [m.value for m in _modes(r)[before:]]
    assert after[:6] == ["prefill", "prefill", "decode", "prefill", "prefill", "decode"]


def test_prefill_continues_without_a_pause_when_nothing_is_decoding() -> None:
    r = rig(budget=4, streak=1)
    r.scheduler.add_request(make_request("a", 16, max_tokens=1))

    for _ in range(3):
        r.loop.step()

    assert _modes(r) == [ForwardMode.PREFILL] * 3


# -- look-ahead rows -------------------------------------------------


def test_lookahead_reserves_several_rows_for_one_request() -> None:
    a = rid("a")
    r = rig(lookahead={a: 2}, draft_accuracy=1.0)
    request = make_request("a", 5, max_tokens=12)
    r.scheduler.add_request(request)

    r.loop.run()

    verify = [s for s in r.runner.steps if s.mode is ForwardMode.VERIFY]
    assert verify
    assert all(s.rows_per_request == (3,) and s.draft_slots == (2,) for s in verify[:-1])
    assert all(len(s.tokens[0]) == 1 for s in verify)  # only the pending token is known
    assert r.loop.tokens(a) == expected_tokens(request)[0]
    assert r.kv.stats().num_requests == 0


@pytest.mark.parametrize("accuracy", [0.0, 0.5, 1.0])
def test_the_output_does_not_depend_on_how_many_drafts_are_accepted(accuracy: float) -> None:
    a = rid("a")
    r = rig(lookahead={a: 3}, draft_accuracy=accuracy)
    request = make_request("a", 7, max_tokens=20)
    r.scheduler.add_request(request)

    r.loop.run()

    assert r.loop.tokens(a) == expected_tokens(request)[0]


def test_lookahead_is_capped_by_what_the_request_may_still_generate() -> None:
    a = rid("a")
    r = rig(lookahead={a: 5})
    request = make_request("a", 4, max_tokens=3)
    r.scheduler.add_request(request)

    r.loop.run()

    decode = r.runner.steps[1:]
    assert max(rows for s in decode for rows in s.rows_per_request) <= 2  # 2 tokens still to come
    assert r.loop.tokens(a) == expected_tokens(request)[0]


def test_lookahead_falls_back_to_one_row_when_the_extra_rows_do_not_fit() -> None:
    a = rid("a")
    r = rig(num_blocks=2, lookahead={a: 3})  # one usable block
    r.scheduler.add_request(make_request("a", 3, max_tokens=5))

    r.loop.step()
    step = r.loop.step()

    assert step is not None
    assert step.mode is ForwardMode.DECODE
    assert step.rows_per_request == (1,)


def test_a_negative_lookahead_is_refused() -> None:
    a = rid("a")
    r = rig(lookahead={a: -1})
    r.scheduler.add_request(make_request("a", 3, max_tokens=5))
    r.loop.step()

    with pytest.raises(ValueError, match="lookahead"):
        r.loop.step()


def test_lookahead_for_a_request_that_is_not_decoding_is_ignored() -> None:
    r = rig(lookahead={rid("ghost"): 4, rid("a"): 4})
    r.scheduler.add_request(make_request("a", 4, max_tokens=1))

    r.loop.run()

    assert [s.mode for s in r.runner.steps] == [ForwardMode.PREFILL]


# -- stops and limits -------------------------------------------------------------------------


def test_an_eos_token_ends_the_request_and_is_its_last_token() -> None:
    request = make_request("a", 5, max_tokens=30)
    eos = token_at(request.request_id, 5 + 3)
    r = rig(eos={eos})
    r.scheduler.add_request(request)

    r.loop.run()

    tokens = r.loop.tokens(request.request_id)
    assert tokens == expected_tokens(request, eos={eos})[0]
    assert tokens[-1] == eos
    assert r.loop.finish_reason(request.request_id) is FinishReason.STOP
    last = r.loop.events[-1]
    assert last.stop_reason == eos


def test_ignore_eos_runs_past_eos_but_a_stop_token_id_still_stops() -> None:
    a = rid("a")
    eos, stop = token_at(a, 5 + 1), token_at(a, 5 + 2)
    r = rig(eos={eos})
    ignoring = make_request("a", 5, max_tokens=12, ignore_eos=True, stop_token_ids={stop})
    r.scheduler.add_request(ignoring)

    r.loop.run()

    assert r.loop.tokens(a) == expected_tokens(ignoring, eos={eos})[0]
    assert len(r.loop.tokens(a)) == 3
    assert r.loop.finish_reason(a) is FinishReason.STOP


def test_without_stops_the_request_runs_to_max_tokens() -> None:
    a = rid("a")
    r = rig(eos={token_at(a, 5 + 1)})
    ignoring = make_request("a", 5, max_tokens=12, ignore_eos=True)
    r.scheduler.add_request(ignoring)

    r.loop.run()

    assert len(r.loop.tokens(a)) == 12
    assert r.loop.finish_reason(a) is FinishReason.LENGTH


def test_a_stop_token_before_min_tokens_does_not_stop() -> None:
    stop = token_at(rid("a"), 5 + 1)  # the second token
    request = make_request("a", 5, max_tokens=10, min_tokens=4, stop_token_ids={stop})
    r = rig()
    r.scheduler.add_request(request)

    r.loop.run()

    assert r.loop.tokens(request.request_id) == expected_tokens(request)[0]
    assert len(r.loop.tokens(request.request_id)) >= 4


def test_accepted_drafts_after_a_stop_token_are_dropped_and_not_committed() -> None:
    a = rid("a")
    probe = make_request("a", 5, max_tokens=30)
    stop = token_at(a, 5 + 2)  # the third token
    request = make_request("a", 5, max_tokens=30, stop_token_ids={stop})
    r = rig(lookahead={a: 4}, draft_accuracy=1.0)
    r.scheduler.add_request(request)

    r.loop.run()

    tokens = r.loop.tokens(a)
    assert tokens == expected_tokens(request)[0]
    assert tokens[-1] == stop
    assert len(tokens) == 3
    assert probe.prompt_token_ids == request.prompt_token_ids
    assert r.loop.finish_reason(a) is FinishReason.STOP


def test_the_model_context_ends_a_request_that_has_no_max_tokens() -> None:
    r = rig(max_model_len=12)
    request = make_request("a", 8)
    unbounded = dataclasses.replace(
        request, sampling=dataclasses.replace(request.sampling, max_tokens=None)
    )
    r.scheduler.add_request(unbounded)

    r.loop.run()

    assert r.loop.tokens(rid("a")) == expected_tokens(unbounded, max_model_len=12)[0]
    assert len(r.loop.tokens(rid("a"))) == 4
    assert r.loop.finish_reason(rid("a")) is FinishReason.LENGTH


def test_the_kv_of_one_request_ends_it_with_length_and_a_memory_note() -> None:
    r = rig(max_blocks=3)  # 12 tokens per request
    request = make_request("a", 6, max_tokens=50)
    r.scheduler.add_request(request)

    r.loop.run()

    tokens = r.loop.tokens(rid("a"))
    assert tokens == expected_tokens(request)[0][: len(tokens)]
    assert len(tokens) == 7
    assert r.loop.finish_reason(rid("a")) is FinishReason.LENGTH
    error = r.loop.events[-1].error
    assert error is not None
    assert "memory" in error.message
    assert r.kv.stats().num_requests == 0


# -- memory pressure without preemption ------------------------------------------------------


def test_a_refused_admission_changes_nothing_and_the_request_waits_its_turn() -> None:
    r = rig(num_blocks=4)  # 3 usable blocks
    big = make_request("big", 10, max_tokens=3)  # takes all three blocks
    small = make_request("small", 4, max_tokens=2)
    r.scheduler.add_request(big)
    r.loop.step()
    r.scheduler.add_request(small)
    stats_before = r.kv.stats()

    step = r.loop.step()  # tries to admit `small`, is refused, decodes `big`

    assert step is not None
    assert step.request_ids == (big.request_id,)
    assert r.kv.stats() == stats_before
    assert r.scheduler.num_waiting == 1
    head = r.scheduler.get(small.request_id)
    assert head is not None
    assert head.status is RequestStatus.WAITING
    r.loop.run()
    assert r.loop.tokens(small.request_id) == expected_tokens(small)[0]
    assert r.loop.tokens(big.request_id) == expected_tokens(big)[0]


def test_a_prefill_chunk_shrinks_to_the_free_blocks() -> None:
    r = rig(num_blocks=4, max_running=2)  # 3 usable blocks
    a, b = make_request("a", 6, max_tokens=2), make_request("b", 8, max_tokens=2)
    r.scheduler.add_request(a)
    r.scheduler.add_request(b)

    first = r.loop.step()

    assert first is not None
    assert first.rows_per_request == (6, 4)  # b gets the one block left
    assert first.need_logits == (True, False)
    r.loop.run()
    assert r.loop.tokens(a.request_id) == expected_tokens(a)[0]
    assert r.loop.tokens(b.request_id) == expected_tokens(b)[0]


def test_when_no_request_can_move_the_lowest_ranked_ends_with_length_and_a_note() -> None:
    r = rig(num_blocks=5, max_running=2)  # 4 usable blocks, 2 per request after prefill
    keep = make_request("keep", 8, max_tokens=8, priority=1)
    shed = make_request("shed", 8, max_tokens=8, priority=0)
    r.scheduler.add_request(keep)
    r.scheduler.add_request(shed)

    r.loop.run()

    assert r.loop.finish_reason(shed.request_id) is FinishReason.LENGTH
    assert r.loop.tokens(shed.request_id) == expected_tokens(shed)[0][:1]
    shed_error = next(
        e.error for e in r.loop.events if e.request_id == shed.request_id and e.finished
    )
    assert shed_error is not None
    assert "memory" in shed_error.message
    assert r.loop.tokens(keep.request_id) == expected_tokens(keep)[0]
    assert r.kv.stats().num_requests == 0


def test_among_equal_priorities_the_request_that_arrived_last_is_shed() -> None:
    r = rig(num_blocks=5, max_running=2)
    first, second = make_request("first", 8, max_tokens=8), make_request("second", 8, max_tokens=8)
    r.scheduler.add_request(first)
    r.clock.now += 1
    r.scheduler.add_request(second)

    r.loop.run()

    assert r.loop.tokens(first.request_id) == expected_tokens(first)[0]
    assert len(r.loop.tokens(second.request_id)) < 8


def test_a_single_request_that_runs_out_of_kv_ends_with_length_and_a_memory_note() -> None:
    r = rig(num_blocks=3)  # 2 usable blocks: the prompt fills them
    request = make_request("a", 8, max_tokens=3)
    r.scheduler.add_request(request)

    r.loop.run()

    assert r.loop.tokens(rid("a")) == expected_tokens(request)[0][:1]
    assert r.loop.finish_reason(rid("a")) is FinishReason.LENGTH
    error = r.loop.events[-1].error
    assert error is not None
    assert "memory, not the model context" in error.message
    assert r.kv.stats().num_requests == 0


def test_a_head_request_that_cannot_be_admitted_with_nothing_running_is_failed() -> None:
    r = rig(num_blocks=3)
    assert r.kv.allocate(rid("squatter"), 8)  # a holder outside the scheduler takes the pool
    r.scheduler.add_request(make_request("a", 4, max_tokens=3))

    assert r.scheduler.schedule() is None

    [event] = r.scheduler.take_events()
    assert (event.request_id, event.finish_reason) == (rid("a"), FinishReason.ERROR)
    assert event.error is not None
    assert event.error.code == "capacity_exceeded"


# -- abort ----------------------------------------------------------------------


def test_aborting_a_waiting_request_ends_it_at_once_and_it_never_held_kv() -> None:
    r = rig(max_running=1)
    r.scheduler.add_request(make_request("a", 4, max_tokens=5))
    r.scheduler.add_request(make_request("b", 4, max_tokens=5))
    r.loop.step()

    assert r.scheduler.abort(rid("b"))

    [event] = r.scheduler.take_events()
    assert (event.request_id, event.finish_reason) == (rid("b"), FinishReason.CANCELLED)
    assert r.scheduler.num_waiting == 0
    assert not r.kv.has(rid("b"))
    assert not r.scheduler.abort(rid("b"))  # nothing left to abort


def test_aborting_an_unknown_request_does_nothing() -> None:
    r = rig()

    assert not r.scheduler.abort(rid("nobody"))
    assert r.scheduler.take_events() == ()


def test_aborting_a_running_request_between_steps_frees_its_kv_at_once() -> None:
    r = rig()
    r.scheduler.add_request(make_request("a", 6, max_tokens=9))
    r.loop.step()

    assert r.kv.has(rid("a"))
    assert r.scheduler.abort(rid("a"))

    assert not r.kv.has(rid("a"))
    r.kv.check_integrity()
    [event] = r.scheduler.take_events()
    assert event.finish_reason is FinishReason.CANCELLED


def test_aborting_inside_a_step_waits_for_its_update_and_keeps_the_kv_until_then() -> None:
    r = rig()
    request = make_request("a", 6, max_tokens=9)
    r.scheduler.add_request(request)
    r.loop.step()
    step = r.scheduler.schedule()
    assert step is not None

    assert r.scheduler.abort(request.request_id)

    assert r.kv.has(request.request_id)
    assert r.scheduler.take_events() == ()
    assert r.scheduler.get(request.request_id) is not None
    r.scheduler.update(r.runner.execute(step))
    assert not r.kv.has(request.request_id)
    [event] = r.scheduler.take_events()
    assert event.finish_reason is FinishReason.CANCELLED
    assert event.token_ids == (token_at(request.request_id, 7),)  # the step's token still goes out
    r.kv.check_integrity()


def test_a_natural_end_in_the_step_wins_over_an_abort_that_arrived_during_it() -> None:
    request = make_request("a", 4, max_tokens=1)
    r = rig()
    r.scheduler.add_request(request)
    step = r.scheduler.schedule()
    assert step is not None
    r.scheduler.abort(request.request_id)

    r.scheduler.update(r.runner.execute(step))

    [event] = r.scheduler.take_events()
    assert event.finish_reason is FinishReason.LENGTH


def test_the_engine_can_end_a_request_with_stop_for_a_stop_string() -> None:
    r = rig()
    r.scheduler.add_request(make_request("a", 4, max_tokens=9))
    r.loop.step()

    r.scheduler.abort(rid("a"), FinishReason.STOP)

    [event] = r.scheduler.take_events()
    assert event.finish_reason is FinishReason.STOP


def test_a_second_abort_of_a_request_in_flight_keeps_the_first_reason() -> None:
    r = rig()
    r.scheduler.add_request(make_request("a", 4, max_tokens=9))
    step = r.scheduler.schedule()
    assert step is not None
    r.scheduler.abort(rid("a"), FinishReason.STOP)
    r.scheduler.abort(rid("a"), FinishReason.CANCELLED)

    r.scheduler.update(r.runner.execute(step))

    assert r.scheduler.take_events()[-1].finish_reason is FinishReason.STOP


# -- time -----------------------------------------------------------------------------


def test_a_request_that_waits_past_the_queue_timeout_ends_before_it_holds_kv() -> None:
    r = rig(max_running=1, queue_timeout=5.0)
    r.scheduler.add_request(make_request("a", 4, max_tokens=30))
    r.scheduler.add_request(make_request("b", 4, max_tokens=3))
    r.loop.step()
    r.clock.now += 6.0

    r.loop.step()

    event = next(e for e in r.loop.events if e.request_id == rid("b"))
    assert event.finish_reason is FinishReason.TIMEOUT
    assert event.error is not None
    assert event.error.code == "request_timeout"
    assert not r.kv.has(rid("b"))
    assert r.scheduler.get(rid("a")) is not None


def test_a_waiting_request_past_its_deadline_times_out() -> None:
    r = rig(max_running=1)
    r.scheduler.add_request(make_request("a", 4, max_tokens=30))
    r.scheduler.add_request(make_request("b", 4, max_tokens=3), deadline=r.clock.now + 2.0)
    r.loop.step()
    r.clock.now += 3.0

    r.loop.step()

    assert r.loop.finish_reason(rid("b")) is FinishReason.TIMEOUT


def test_a_running_request_past_its_deadline_ends_at_the_step_boundary() -> None:
    r = rig()
    r.scheduler.add_request(make_request("a", 4, max_tokens=30), deadline=r.clock.now + 2.0)
    r.loop.step()
    step = r.scheduler.schedule()
    assert step is not None
    r.clock.now += 3.0

    r.scheduler.update(r.runner.execute(step))

    [event] = r.scheduler.take_events()
    assert event.finish_reason is FinishReason.TIMEOUT
    assert event.token_ids == (token_at(rid("a"), 5),)
    assert not r.kv.has(rid("a"))


# -- order ------------------------------------------------------------------------------------


def test_the_priority_policy_admits_the_higher_priority_first() -> None:
    r = rig(max_running=1, policy="priority")
    r.scheduler.add_request(make_request("running", 4, max_tokens=2))
    r.loop.step()
    r.scheduler.add_request(make_request("low", 4, max_tokens=1, priority=0))
    r.scheduler.add_request(make_request("high", 4, max_tokens=1, priority=5))

    r.loop.run()

    order = [e.request_id for e in r.loop.events if e.finished]
    assert order == [rid("running"), rid("high"), rid("low")]


# -- errors and the update contract -----------------------------------------------------------


def test_a_step_error_ends_every_request_of_the_step_with_error_and_frees_their_kv() -> None:
    r = rig(fail_steps=frozenset({1}))
    r.scheduler.add_request(make_request("a", 4, max_tokens=5))
    r.scheduler.add_request(make_request("b", 4, max_tokens=5))

    r.loop.run()

    for name in "ab":
        assert r.loop.finish_reason(rid(name)) is FinishReason.ERROR
    errors = [e.error for e in r.loop.events if e.finished]
    assert all(e is not None and e.code == "backend_error" for e in errors)
    assert r.kv.stats().num_requests == 0


def test_update_without_a_step_in_flight_is_an_error() -> None:
    r = rig()

    with pytest.raises(EngineError, match="without a step"):
        r.scheduler.update(StepOutput(step_id=1, request_ids=(rid("a"),), tokens=((),)))


def test_schedule_twice_without_update_is_an_error() -> None:
    r = rig()
    r.scheduler.add_request(make_request("a", 4))
    assert r.scheduler.schedule() is not None
    assert r.scheduler.in_flight

    with pytest.raises(EngineError, match="in flight"):
        r.scheduler.schedule()


def test_an_answer_for_another_step_is_refused_and_the_step_stays_in_flight() -> None:
    r = rig()
    r.scheduler.add_request(make_request("a", 4))
    step = r.scheduler.schedule()
    assert step is not None

    with pytest.raises(EngineError, match="in flight"):
        r.scheduler.update(StepOutput(step_id=99, request_ids=step.request_ids, tokens=((1,),)))

    assert r.scheduler.in_flight
    r.scheduler.update(r.runner.execute(step))
    assert not r.scheduler.in_flight


def _answer(step: StepInput, tokens: tuple[tuple[int, ...], ...]) -> StepOutput:
    return StepOutput(step_id=step.step_id, request_ids=step.request_ids, tokens=tokens)


def test_a_prefill_chunk_that_does_not_end_the_prompt_must_not_emit_a_token() -> None:
    r = rig(budget=4)
    r.scheduler.add_request(make_request("a", 9))
    step = r.scheduler.schedule()
    assert step is not None

    with pytest.raises(EngineError, match="emitted tokens"):
        r.scheduler.update(_answer(step, ((5,),)))


def test_the_last_prefill_chunk_must_emit_exactly_one_token() -> None:
    r = rig()
    r.scheduler.add_request(make_request("a", 4))
    step = r.scheduler.schedule()
    assert step is not None

    with pytest.raises(EngineError, match="must emit one token"):
        r.scheduler.update(_answer(step, ((),)))


def test_a_decode_step_cannot_emit_more_tokens_than_it_has_rows() -> None:
    r = rig()
    r.scheduler.add_request(make_request("a", 4, max_tokens=9))
    r.loop.step()
    step = r.scheduler.schedule()
    assert step is not None

    with pytest.raises(EngineError, match="emitted 2 tokens"):
        r.scheduler.update(_answer(step, ((5, 6),)))


# -- release of finished requests ------------------------------------------------------


def test_a_finished_request_keeps_its_kv_until_the_runner_releases_it() -> None:
    r = rig(lag=True)
    a, b = make_request("a", 4, max_tokens=1), make_request("b", 4, max_tokens=6)
    r.scheduler.add_request(a)
    r.scheduler.add_request(b)

    r.loop.step()  # a finishes in this step; the runner does not list it yet

    assert r.loop.finish_reason(a.request_id) is FinishReason.LENGTH
    assert r.kv.has(a.request_id)
    assert r.scheduler.num_awaiting_release == 1
    with pytest.raises(InvalidRequestError, match="already in use"):
        r.scheduler.add_request(make_request("a", 4))

    r.loop.step()  # the next output lists it

    assert not r.kv.has(a.request_id)
    assert r.scheduler.num_awaiting_release == 0
    r.scheduler.add_request(make_request("a", 4, max_tokens=1))


def test_the_runner_may_release_a_finished_request_in_the_same_output() -> None:
    r = rig()
    r.scheduler.add_request(make_request("a", 4, max_tokens=1))

    r.loop.step()

    assert not r.kv.has(rid("a"))
    assert r.scheduler.num_awaiting_release == 0


def test_a_lagging_runner_still_produces_the_expected_tokens() -> None:
    r = rig(lag=True)
    requests = [make_request(name, 5, max_tokens=4) for name in "abc"]
    for request in requests:
        r.scheduler.add_request(request)

    r.loop.run()

    for request in requests:
        assert r.loop.tokens(request.request_id) == expected_tokens(request)[0]
