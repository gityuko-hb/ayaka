"""A runner without a model, so the scheduler is tested with a real KV manager and no torch.

Lives in ``tests/_support`` (on ``pythonpath``) so test modules can import it.

:class:`FakeRunner` turns a ``StepInput`` into a ``StepOutput`` with tokens that are a pure function
of ``(request id, position)``. Whatever way the scheduler batches, chunks, speculates or reorders,
the tokens a request ends with must be the ones :func:`expected_tokens` predicts, which is the
property the scheduler tests rest on. :class:`StepLoop` is the engine loop in miniature.
"""

from __future__ import annotations

import random
import zlib
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from ayaka.core.errors import EngineError, ErrorInfo
from ayaka.core.params import SamplingParams
from ayaka.core.request import TokenizedRequest
from ayaka.core.types import FinishReason, ForwardMode, KVLayoutKind, RequestId
from ayaka.kvcache.manager import KVCacheManager
from ayaka.kvcache.storage.spec import KVGroup, KVSpec
from ayaka.scheduler.step import StepInput, StepOutput
from ayaka.utils.dtypes import DType

if TYPE_CHECKING:
    from collections.abc import Collection, Mapping

    from ayaka.core.types import TokenId
    from ayaka.scheduler.request import RequestEvent
    from ayaka.scheduler.scheduler import Scheduler

__all__ = [
    "BLOCK_SIZE",
    "VOCAB",
    "FakeRunner",
    "StepLoop",
    "expected_tokens",
    "make_kv",
    "make_request",
    "token_at",
]

BLOCK_SIZE = 4
VOCAB = 1000
#: Ids below this are never produced by :func:`token_at`, so a test can use them as EOS or stop ids.
RESERVED_IDS = 8


def rid(name: str | int) -> RequestId:
    """A request id from a plain label."""
    return RequestId(str(name))


def make_kv(
    num_blocks: int = 33, *, max_requests: int = 4, max_blocks: int = 16, groups: int = 1
) -> KVCacheManager:
    """A real :class:`KVCacheManager` over ``groups`` tiny groups of ``BLOCK_SIZE``-token blocks."""
    kv_groups = [
        KVGroup(
            layer_ids=(2 * i, 2 * i + 1),
            spec=KVSpec(
                num_layers=2,
                num_kv_heads=2,
                head_dim=4,
                block_size=BLOCK_SIZE,
                dtype=DType.FP32,
                layout=KVLayoutKind.NHD,
            ),
        )
        for i in range(groups)
    ]
    return KVCacheManager(
        kv_groups,
        [num_blocks] * groups,
        max_requests=max_requests,
        max_blocks_per_request=max_blocks,
    )


def make_request(
    name: str | int,
    prompt_length: int,
    *,
    max_tokens: int = 8,
    priority: int = 0,
    stop_token_ids: Collection[int] = (),
    min_tokens: int = 0,
    ignore_eos: bool = False,
) -> TokenizedRequest:
    """A greedy request whose prompt is ``prompt_length`` tokens derived from its id."""
    request_id = rid(name)
    prompt = tuple(RESERVED_IDS + (i * 7 + len(str(name))) % 50 for i in range(prompt_length))
    return TokenizedRequest(
        request_id=request_id,
        prompt_token_ids=prompt,
        sampling=SamplingParams(
            temperature=0.0,
            max_tokens=max_tokens,
            min_tokens=min_tokens,
            stop_token_ids=frozenset(stop_token_ids),
            ignore_eos=ignore_eos,
        ),
        priority=priority,
    )


def token_at(request_id: RequestId, position: int) -> TokenId:
    """The token the fake model puts at ``position`` of the sequence of a request."""
    digest = zlib.crc32(f"{request_id}:{position}".encode())
    return RESERVED_IDS + digest % (VOCAB - RESERVED_IDS)


def expected_tokens(
    request: TokenizedRequest, *, eos: Collection[int] = (), max_model_len: int | None = None
) -> tuple[tuple[TokenId, ...], FinishReason]:
    """What a request must end with, derived from the rules alone, with no scheduler involved."""
    params = request.sampling
    stops = set(params.stop_token_ids) | (set() if params.ignore_eos else set(eos))
    out: list[TokenId] = []
    while True:
        token = token_at(request.request_id, request.prompt_length + len(out))
        out.append(token)
        if token in stops and len(out) > params.min_tokens:
            return tuple(out), FinishReason.STOP
        if params.max_tokens is not None and len(out) >= params.max_tokens:
            return tuple(out), FinishReason.LENGTH
        if max_model_len is not None and request.prompt_length + len(out) >= max_model_len:
            return tuple(out), FinishReason.LENGTH


@dataclass(slots=True)
class FakeRunner:
    """Executes steps by formula and checks, as a real runner would rely on, that KV room exists.

    Args:
        kv: When given, every request's blocks and committed length are checked against the step.
        draft_accuracy: Chance that each draft row of a ``VERIFY`` step is right.
        seed: Seeds the draft guesses; the output tokens do not depend on it.
        lag_releases: Report each step's requests as releasable one step late, as an overlapped
            engine would, instead of at once.
        fail_steps: Step ids that return a step-level error.
    """

    kv: KVCacheManager | None = None
    draft_accuracy: float = 0.5
    seed: int = 0
    lag_releases: bool = False
    fail_steps: frozenset[int] = frozenset()
    steps: list[StepInput] = field(default_factory=list[StepInput])
    _previous: tuple[RequestId, ...] = ()

    def execute(self, step: StepInput) -> StepOutput:
        """Run one step and answer it."""
        self.steps.append(step)
        releasable = self._previous if self.lag_releases else None
        self._previous = step.request_ids
        if step.step_id in self.fail_steps:
            return StepOutput(
                step_id=step.step_id,
                request_ids=step.request_ids,
                releasable=releasable,
                error=ErrorInfo(code="backend_error", message="injected failure"),
            )
        tokens: list[tuple[TokenId, ...]] = []
        accepted: list[int] = []
        for index in range(step.num_requests):
            self._check_kv(step, index)
            produced = self._produce(step, index)
            tokens.append(produced)
            accepted.append(max(0, len(produced) - 1) if step.mode.is_decode_like else 0)
        return StepOutput(
            step_id=step.step_id,
            request_ids=step.request_ids,
            tokens=tuple(tokens),
            accepted=tuple(accepted),
            releasable=releasable,
        )

    def _produce(self, step: StepInput, index: int) -> tuple[TokenId, ...]:
        request_id = step.request_ids[index]
        start, rows = step.start_positions[index], step.rows_per_request[index]
        if step.mode is ForwardMode.PREFILL:
            return (token_at(request_id, start + rows),) if step.need_logits[index] else ()
        drafts = step.draft_slots[index]
        rng = random.Random(f"{self.seed}:{request_id}:{start}")  # noqa: S311  # not security
        accepted = 0
        for _ in range(drafts):
            if rng.random() >= self.draft_accuracy:
                break
            accepted += 1
        return tuple(token_at(request_id, start + 1 + t) for t in range(accepted + 1))

    def _check_kv(self, step: StepInput, index: int) -> None:
        if self.kv is None:
            return
        request_id = step.request_ids[index]
        start, rows = step.start_positions[index], step.rows_per_request[index]
        kv = self.kv
        if kv.num_committed(request_id) != start:
            raise EngineError(
                f"{request_id}: step starts at {start}, KV has committed "
                f"{kv.num_committed(request_id)}"
            )
        for group_blocks in kv.blocks(request_id):
            if len(group_blocks) * kv.block_size < start + rows:
                raise EngineError(f"{request_id}: no KV room for rows up to {start + rows}")
        if int(kv.table.row_of(request_id)) != step.batch_rows[index]:
            raise EngineError(f"{request_id}: batch row differs from its block table row")


@dataclass(slots=True)
class StepLoop:
    """The engine loop in miniature: ``schedule``, ``execute``, ``update``, then collect events.

    Args:
        scheduler: The scheduler under test.
        runner: The fake runner.
        kv: The manager the scheduler uses; checked after every step.
        lookahead: Optional ``k`` per request id, given to every ``schedule()``.
    """

    scheduler: Scheduler
    runner: FakeRunner
    kv: KVCacheManager
    lookahead: Mapping[RequestId, int] | None = None
    events: list[RequestEvent] = field(default_factory=list["RequestEvent"])

    def step(self) -> StepInput | None:
        """One full cycle; returns the step it ran, or ``None`` when the scheduler was idle."""
        step = self.scheduler.schedule(self.lookahead)
        self.events.extend(self.scheduler.take_events())
        self.kv.check_integrity()
        if step is None:
            return None
        self.scheduler.update(self.runner.execute(step))
        self.events.extend(self.scheduler.take_events())
        self.kv.check_integrity()
        return step

    def run(self, max_steps: int = 10_000) -> int:
        """Step until the scheduler is idle; returns the steps run."""
        for count in range(max_steps):
            if self.step() is None:
                return count
        raise AssertionError(f"still busy after {max_steps} steps")

    def tokens(self, request_id: RequestId) -> tuple[TokenId, ...]:
        """Every token emitted for a request, in order."""
        return tuple(t for e in self.events if e.request_id == request_id for t in e.token_ids)

    def finish_reason(self, request_id: RequestId) -> FinishReason | None:
        """The reason of the request's last event, or ``None`` if it has not ended."""
        for event in self.events:
            if event.request_id == request_id and event.finish_reason is not None:
                return event.finish_reason
        return None
