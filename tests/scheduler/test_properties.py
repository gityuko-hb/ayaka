"""Properties of the scheduler over random and swept workloads.

Every test drives a real :class:`~ayaka.kvcache.manager.KVCacheManager` with a small pool through a
fake runner whose tokens are a function of ``(request, position)``. After every ``schedule()`` and
every ``update()`` the KV books are checked with ``check_integrity``. Whatever the batching, the
chunking, the speculative rows, the pool pressure or the aborts, a request's tokens must be a prefix
of ``expected_tokens``, and the whole of it when the request ended on its own terms.
"""

from __future__ import annotations

import itertools
import random
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import pytest

from ayaka.configs.scheduler import SchedulerConfig
from ayaka.core.types import FinishReason, ForwardMode
from ayaka.scheduler.request import Request, RequestEvent, RequestStatus
from ayaka.scheduler.scheduler import Scheduler
from fake_runner import (
    FakeRunner,
    expected_tokens,
    make_kv,
    make_request,
    rid,
    token_at,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from ayaka.core.request import TokenizedRequest
    from ayaka.core.types import RequestId
    from ayaka.kvcache.manager import KVCacheManager
    from ayaka.scheduler.step import StepInput

pytestmark = pytest.mark.unit

SEEDS = range(40)

#: The legal status moves, written out independently of ``LEGAL_TRANSITIONS``.
SCH_B3 = {
    ("waiting", "running"),
    ("waiting", "finished"),
    ("running", "preempted"),
    ("running", "finished"),
    ("preempted", "running"),
    ("preempted", "finished"),
}


class _Clock:
    def __call__(self) -> float:
        return 0.0


@dataclass
class World:
    """One scheduler, its manager, a fake runner and the log of what happened."""

    requests: list[TokenizedRequest]
    arrive_at: dict[RequestId, int]
    scheduler: Scheduler
    runner: FakeRunner
    kv: KVCacheManager
    eos: frozenset[int]
    k_max: int
    policy: str
    streak: int
    rng: random.Random
    events: list[RequestEvent] = field(default_factory=list[RequestEvent])
    steps: list[StepInput] = field(default_factory=list["StepInput"])
    transitions: list[tuple[RequestId, str, str]] = field(
        default_factory=list[tuple[Any, str, str]]
    )
    aborted: set[RequestId] = field(default_factory=set[Any])

    def tokens(self, request_id: RequestId) -> tuple[int, ...]:
        return tuple(t for e in self.events if e.request_id == request_id for t in e.token_ids)

    def final_event(self, request_id: RequestId) -> RequestEvent:
        finals = [e for e in self.events if e.request_id == request_id and e.finished]
        assert len(finals) == 1, f"{request_id}: {len(finals)} final events"
        return finals[0]


def make_world(
    seed: int,
    *,
    policy: str | None = None,
    num_requests: int = 14,
    roomy: bool = False,
    streak: int | None = None,
    budget: int | None = None,
) -> World:
    rng = random.Random(seed)  # noqa: S311  # a reproducible workload, not a secret
    chosen_policy = policy or rng.choice(("fcfs", "priority"))
    num_blocks = 200 if roomy else rng.randint(8, 24)  # 7..23 usable blocks of 4 tokens
    max_requests = rng.choice((2, 3, 4))
    kv = make_kv(num_blocks, max_requests=max_requests, max_blocks=16)
    eos = frozenset({rng.randint(8, 400)})
    config = SchedulerConfig(
        max_running=rng.choice((None, 1, max_requests)),
        max_tokens_per_step=budget if budget is not None else rng.choice((3, 4, 6, 8, 2048)),
        policy=chosen_policy,  # pyright: ignore[reportArgumentType]
        max_consecutive_prefills=streak if streak is not None else rng.randint(1, 3),
    )
    scheduler = Scheduler(config, kv, eos_token_ids=eos, clock=_Clock())
    runner = FakeRunner(kv, draft_accuracy=rng.choice((0.0, 0.5, 1.0)), seed=seed)
    requests: list[TokenizedRequest] = []
    arrive_at: dict[RequestId, int] = {}
    for index in range(num_requests):
        name = f"r{index}"
        length, max_tokens = rng.randint(1, 24), rng.randint(1, 12)
        stop = {token_at(rid(name), length + rng.randint(0, 8))} if rng.random() < 0.3 else ()
        requests.append(
            make_request(
                name,
                length,
                max_tokens=max_tokens,
                priority=rng.randint(0, 2),
                stop_token_ids=stop,
                min_tokens=min(max_tokens, rng.choice((0, 0, 2))),
            )
        )
        arrive_at[rid(name)] = rng.randint(0, 25)
    return World(
        requests=requests,
        arrive_at=arrive_at,
        scheduler=scheduler,
        runner=runner,
        kv=kv,
        eos=eos,
        k_max=rng.choice((0, 2, 3)),
        policy=chosen_policy,
        streak=config.max_consecutive_prefills,
        rng=rng,
    )


def _live_ids(world: World) -> list[RequestId]:
    return [r.request_id for r in world.requests if world.scheduler.get(r.request_id) is not None]


def _record_transitions(monkeypatch: pytest.MonkeyPatch, world: World) -> None:
    original: Callable[[Request, RequestStatus], None] = Request.move_to

    def recording(self: Request, status: RequestStatus) -> None:
        world.transitions.append((self.request_id, str(self.status), str(status)))
        original(self, status)

    monkeypatch.setattr(Request, "move_to", recording)


def _check_books(world: World) -> None:
    """The KV books agree with the scheduler's view of every live request."""
    world.kv.check_integrity()
    for request_id in _live_ids(world):
        live = world.scheduler.get(request_id)
        assert live is not None
        if live.status is RequestStatus.WAITING:
            assert not world.kv.has(request_id), f"{request_id} waits but holds KV"
        else:
            assert world.kv.num_committed(request_id) == live.computed
            assert world.kv.table.row_of(request_id) == live.row


def _scan_for_mutables(value: object, path: str) -> None:
    """Nothing reachable from a snapshot may be a list, a dict, a set or a live ``Request``."""
    assert not isinstance(value, (list, dict, set, Request)), f"{path} is a {type(value).__name__}"
    if isinstance(value, tuple):
        for index, item in enumerate(value):  # pyright: ignore[reportUnknownVariableType,reportUnknownArgumentType]
            _scan_for_mutables(item, f"{path}[{index}]")


def _fingerprint(step: StepInput) -> tuple[int, str]:
    return hash(step), repr(step)


def _scribble_on_live_requests(world: World, step: StepInput) -> list[Callable[[], None]]:
    """Change every live request of the step; return the functions that undo the changes."""
    undo: list[Callable[[], None]] = []
    for request_id in step.request_ids:
        live = world.scheduler.get(request_id)
        assert live is not None
        saved = (live.prompt, list(live.output), live.row, live.computed, live.pending)
        live.prompt = (*live.prompt, 999_999)
        live.output.append(-1)
        live.row += 7
        live.computed += 11
        live.pending = -5

        def restore(live: Request = live, saved: tuple[Any, ...] = saved) -> None:
            live.prompt, live.row, live.computed, live.pending = (
                saved[0],
                saved[2],
                saved[3],
                saved[4],
            )
            live.output[:] = saved[1]

        undo.append(restore)
    return undo


def drive(
    world: World,
    *,
    random_aborts: bool = False,
    abort_at: tuple[int, str, RequestId] | None = None,
    check_snapshots: bool = False,
    max_iterations: int = 3000,
) -> None:
    """Run the engine loop until every request has ended, injecting aborts as asked."""
    scheduler, runner = world.scheduler, world.runner
    added: set[RequestId] = set()
    for iteration in range(max_iterations):
        for request in world.requests:
            if request.request_id not in added and world.arrive_at[request.request_id] <= iteration:
                scheduler.add_request(request)
                added.add(request.request_id)
        if abort_at is not None and abort_at[:2] == (iteration, "between"):
            if scheduler.abort(abort_at[2]):
                world.aborted.add(abort_at[2])
        elif random_aborts and world.rng.random() < 0.1:
            live = _live_ids(world)
            if live:
                victim = world.rng.choice(live)
                if scheduler.abort(victim):
                    world.aborted.add(victim)
        lookahead = {r.request_id: world.rng.randint(0, world.k_max) for r in world.requests}
        step = scheduler.schedule(lookahead)
        world.events.extend(scheduler.take_events())
        _check_books(world)
        if step is None:
            if len(added) == len(world.requests) and not scheduler.has_unfinished:
                return
            continue
        world.steps.append(step)
        before = _fingerprint(step)
        if check_snapshots:
            undo = _scribble_on_live_requests(world, step)
            assert _fingerprint(step) == before, "a StepInput changed when its Request changed"
            for restore in undo:
                restore()
        if abort_at is not None and abort_at[:2] == (iteration, "in_flight"):
            if scheduler.abort(abort_at[2]):
                world.aborted.add(abort_at[2])
        elif random_aborts and world.rng.random() < 0.1:
            victim = world.rng.choice(step.request_ids)
            if scheduler.abort(victim):
                world.aborted.add(victim)
        scheduler.update(runner.execute(step))
        world.events.extend(scheduler.take_events())
        _check_books(world)
        assert _fingerprint(step) == before
    raise AssertionError(f"still busy after {max_iterations} iterations")


def _check_outcome(world: World) -> None:
    """Every request ended once, with tokens that agree with the rules, and nothing leaked."""
    for request in world.requests:
        request_id = request.request_id
        final = world.final_event(request_id)
        want, natural = expected_tokens(request, eos=world.eos)
        got = world.tokens(request_id)
        assert got == want[: len(got)], f"{request_id}: tokens differ from the fake model"
        if final.finish_reason in (FinishReason.STOP, FinishReason.LENGTH) and final.error is None:
            assert got == want
            assert final.finish_reason is natural
        elif final.finish_reason is FinishReason.CANCELLED:
            assert request_id in world.aborted
        else:
            assert final.finish_reason is FinishReason.LENGTH
            assert final.error is not None
            assert "memory" in final.error.message
        if final.finish_reason is FinishReason.STOP and final.error is None:
            assert final.stop_reason == got[-1]
    stats = world.kv.stats()
    assert stats.num_requests == 0
    assert stats.free_blocks == stats.usable_blocks
    assert world.kv.table.free_rows == world.kv.table.max_requests
    assert not world.scheduler.has_unfinished
    assert world.scheduler.num_awaiting_release == 0


# -- every request ends --------------------------------------------------------------


@pytest.mark.parametrize("seed", SEEDS)
def test_every_request_ends_and_the_books_stay_clean_in_a_random_workload(seed: int) -> None:
    world = make_world(seed)

    drive(world)

    _check_outcome(world)
    assert not world.aborted


@pytest.mark.parametrize("seed", SEEDS)
def test_every_request_ends_when_random_aborts_hit_it_at_any_moment(seed: int) -> None:
    world = make_world(seed)

    drive(world, random_aborts=True)

    _check_outcome(world)


# -- order follows the policy --------------------------------------------------------


def _policy_key(world: World, request_id: RequestId) -> tuple[int, int]:
    request = next(r for r in world.requests if r.request_id == request_id)
    order = world.requests.index(request)
    return (-request.priority if world.policy == "priority" else 0, order)


@pytest.mark.parametrize("policy", ["fcfs", "priority"])
@pytest.mark.parametrize("seed", SEEDS)
def test_requests_are_admitted_in_the_order_the_policy_gives(policy: str, seed: int) -> None:
    world = make_world(seed, policy=policy)
    arrival_rank: dict[RequestId, int] = {}
    waiting: set[RequestId] = set()
    admitted: set[RequestId] = set()
    scheduler = world.scheduler
    added = 0

    for iteration in range(3000):
        for request in world.requests:
            if world.arrive_at[request.request_id] == iteration:
                scheduler.add_request(request)
                arrival_rank[request.request_id] = added
                added += 1
                waiting.add(request.request_id)
        step = scheduler.schedule({})
        world.events.extend(scheduler.take_events())
        if step is None:
            if added == len(world.requests) and not scheduler.has_unfinished:
                break
            continue
        for request_id, start in zip(step.request_ids, step.start_positions, strict=True):
            if request_id in admitted:
                continue
            assert start == 0
            head = min(waiting, key=lambda r: _arrival_key(world, r, arrival_rank))
            assert request_id == head, f"{request_id} admitted ahead of {head}"
            waiting.discard(request_id)
            admitted.add(request_id)
        for event in world.events:
            if event.finished:
                waiting.discard(event.request_id)
        scheduler.update(world.runner.execute(step))
        world.events.extend(scheduler.take_events())
    assert not waiting


def _arrival_key(
    world: World, request_id: RequestId, rank: dict[RequestId, int]
) -> tuple[int, int]:
    request = next(r for r in world.requests if r.request_id == request_id)
    return (-request.priority if world.policy == "priority" else 0, rank[request_id])


# -- decode is not starved -----------------------------------------------------------


@pytest.mark.parametrize("seed", SEEDS)
def test_no_ready_request_waits_more_than_k_prefill_steps_for_a_decode_step(seed: int) -> None:
    world = make_world(seed, roomy=True, num_requests=10)
    consecutive = 0
    scheduler = world.scheduler

    for iteration in range(3000):
        for request in world.requests:
            if world.arrive_at[request.request_id] == iteration:
                scheduler.add_request(request)
        ready = any(
            (live := scheduler.get(r.request_id)) is not None
            and live.status is RequestStatus.RUNNING
            and not live.is_prefilling
            for r in world.requests
        )
        step = scheduler.schedule({})
        world.events.extend(scheduler.take_events())
        if step is None:
            if iteration > 25 and not scheduler.has_unfinished:
                break
            continue
        if not ready:
            consecutive = 0
        elif step.mode is ForwardMode.PREFILL:
            consecutive += 1
            assert consecutive <= world.streak
        else:
            consecutive = 0
        scheduler.update(world.runner.execute(step))
        world.events.extend(scheduler.take_events())


# -- an abort at any moment leaves clean books ---------------------------------------


def _sweep_world() -> World:
    """Three requests, two slots, a pool tight enough that the third waits and memory is scarce."""
    kv = make_kv(10, max_requests=3, max_blocks=16)
    config = SchedulerConfig(max_running=2, max_tokens_per_step=6, max_consecutive_prefills=2)
    world = World(
        requests=[],
        arrive_at={},
        scheduler=Scheduler(config, kv, clock=_Clock()),
        runner=FakeRunner(kv, draft_accuracy=1.0),
        kv=kv,
        eos=frozenset(),
        k_max=2,
        policy="fcfs",
        streak=2,
        rng=random.Random(0),  # noqa: S311  # a reproducible workload, not a secret
    )
    for name, length, tokens in (("a", 9, 7), ("b", 5, 6), ("c", 14, 5)):
        request = make_request(name, length, max_tokens=tokens)
        world.requests.append(request)
        world.arrive_at[request.request_id] = 0
    return world


@pytest.mark.parametrize("point", ["between", "in_flight"])
@pytest.mark.parametrize("target", ["a", "b", "c"])
@pytest.mark.parametrize("iteration", range(0, 24))
def test_an_abort_injected_at_each_step_leaves_the_books_clean(
    iteration: int, target: str, point: str
) -> None:
    world = _sweep_world()

    drive(world, abort_at=(iteration, point, rid(target)))

    _check_outcome(world)
    finals = {e.request_id: e.finish_reason for e in world.events if e.finished}
    assert set(finals) == {rid("a"), rid("b"), rid("c")}


def test_the_sweep_reaches_every_phase_of_a_request() -> None:
    """The injection points cover waiting, mid-prefill and mid-decode, not just the easy ones."""
    world = _sweep_world()
    seen: set[tuple[str, str]] = set()
    scheduler = world.scheduler
    for request in world.requests:
        scheduler.add_request(request)
    for _ in range(40):
        for request in world.requests:
            live = scheduler.get(request.request_id)
            if live is None:
                continue
            phase = (
                "decoding"
                if not live.is_prefilling
                else ("mid_prefill" if live.computed else str(live.status))
            )
            seen.add((str(request.request_id), phase))
        step = scheduler.schedule({})
        if step is None:
            break
        scheduler.update(world.runner.execute(step))
    phases = {phase for _, phase in seen}
    assert {"waiting", "mid_prefill", "decoding"} <= phases


# -- rows of a step stay within the budget and the bucket ----------------------------


@pytest.mark.parametrize("seed", SEEDS)
def test_no_step_exceeds_the_token_budget_or_the_largest_bucket(seed: int) -> None:
    world = make_world(seed)

    drive(world)

    bucket = world.scheduler.max_running * (1 + world.k_max)
    assert world.steps
    for step in world.steps:
        assert step.num_rows >= step.num_requests
        if step.mode is ForwardMode.PREFILL:
            assert step.num_rows <= world.scheduler.token_budget
        else:
            assert step.num_requests <= world.scheduler.max_running
            assert step.num_rows <= bucket
            assert all(rows <= 1 + world.k_max for rows in step.rows_per_request)
        assert step.num_requests <= world.scheduler.max_running


# -- a step is a snapshot ------------------------------------------------------------


@pytest.mark.parametrize("seed", SEEDS[:15])
def test_changing_a_request_after_schedule_does_not_change_the_step(seed: int) -> None:
    world = make_world(seed)

    drive(world, check_snapshots=True)

    _check_outcome(world)


@pytest.mark.parametrize("seed", SEEDS[:15])
def test_a_step_holds_only_immutable_values(seed: int) -> None:
    world = make_world(seed)

    drive(world)

    for step in world.steps:
        for name in step.__dataclass_fields__:  # type: ignore[attr-defined]
            _scan_for_mutables(getattr(step, name), name)


# -- every move is in the table ------------------------------------------------------


@pytest.mark.parametrize("seed", SEEDS)
def test_every_status_move_is_in_the_sch_b3_table(
    seed: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    world = make_world(seed)
    _record_transitions(monkeypatch, world)

    drive(world, random_aborts=seed % 2 == 0)

    assert world.transitions
    assert {(old, new) for _, old, new in world.transitions} <= SCH_B3
    for request in world.requests:
        path = [t for t in world.transitions if t[0] == request.request_id]
        assert path[0][1] == "waiting"
        assert path[-1][2] == "finished"
        assert sum(1 for t in path if t[2] == "finished") == 1
        assert all(a[2] == b[1] for a, b in itertools.pairwise(path))
        assert "preempted" not in {t[2] for t in path}  # nothing preempts before M3


# -- refusal is harmless ------------------------------------------------------------------------


@pytest.mark.parametrize("seed", SEEDS[:20])
def test_the_scheduler_never_evicts_or_raises_for_a_shortage(seed: int) -> None:
    world = make_world(seed)  # the pool is small on purpose: refusals happen all the time

    drive(world)

    assert world.kv.stats().evictions == 0
    _check_outcome(world)


def test_stop_tokens_match_their_request_in_the_sweep_world() -> None:
    world = _sweep_world()
    request = make_request("d", 6, max_tokens=9, stop_token_ids={token_at(rid("d"), 6 + 2)})
    world.requests.append(request)
    world.arrive_at[request.request_id] = 2

    drive(world)

    _check_outcome(world)
    assert world.final_event(request.request_id).finish_reason is FinishReason.STOP
