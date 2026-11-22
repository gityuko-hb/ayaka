"""Which requests run in the next step, and what happens to them afterwards.

:class:`Scheduler` owns the waiting queue, the running set and the request state machine. It asks
the :class:`~ayaka.kvcache.manager.KVCacheManager` for room and commits what the runner reports;
it never touches a tensor and never calls a runner. The engine loop is one thread::

    step = scheduler.schedule(lookahead)      # None: nothing to do
    output = runner.execute(step)
    scheduler.update(output)
    events = scheduler.take_events()

A step is a prefill of one chunk per request or a decode of every ready request, never a
mix. Memory never makes the scheduler evict or raise: ``allocate`` returns ``False`` and the
scheduler schedules less. Preemption does not exist yet, so a request that cannot get a row for a
step simply skips it; if no running request can move, the one that ranks lowest ends with
``LENGTH`` and a note that memory was the limit (a last resort), which is also what keeps every
request finishing.

With a prefix cache, a request is matched when it leaves the queue, not when it arrives, so the
blocks it attaches cannot be evicted in between: its prefill starts after the matched tokens. After
every commit the request's newly full blocks are registered so later requests can attach them.
The sequences of a request with ``n > 1`` arrive as sibling requests; a sibling may wait in the
queue until the first one has prefilled the shared prompt (``blocked_on``), the one exception to
strict queue order, so it attaches the prompt instead of computing it again.

Every :meth:`Scheduler.update` leaves a :class:`~ayaka.scheduler.stats.SchedulerStats` for
:meth:`Scheduler.take_stats`.

Every end of a request goes through one method of the roster (:mod:`ayaka.scheduler.roster`): it
sets the reason, frees the KV, releases the batch row and queues the last event.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

from ayaka.core.errors import (
    CapacityError,
    EngineError,
    InvalidRequestError,
    OverloadedError,
)
from ayaka.core.types import FinishReason, ForwardMode
from ayaka.scheduler.queue import RequestQueue, resolve_policy
from ayaka.scheduler.request import Request, RequestEvent, RequestStatus
from ayaka.scheduler.roster import InFlight, Roster, timeout_error
from ayaka.scheduler.stats import SchedulerStats
from ayaka.scheduler.step import StepInput
from ayaka.utils.validation import require_int

if TYPE_CHECKING:
    from collections.abc import Callable, Collection, Mapping

    from ayaka.configs.scheduler import SchedulerConfig
    from ayaka.core.errors import ErrorInfo
    from ayaka.core.outputs import TokenLogprob
    from ayaka.core.request import TokenizedRequest
    from ayaka.core.types import RequestId, TokenId
    from ayaka.kvcache.manager import KVCacheManager
    from ayaka.kvcache.prefix.base import PrefixMatch
    from ayaka.scheduler.step import StepOutput

__all__ = ["Scheduler"]


class Scheduler:
    """Admits requests, plans steps and applies their results.

    Args:
        config: Limits the user chose.
        kv: The cache manager. Its block table decides how many requests can run and how long one
            can get.
        max_model_len: The model's context length; ``None`` for no check.
        max_chunk_tokens: The memory-derived cap on a prefill step's tokens. The
            budget of a prefill step is ``min(config.max_tokens_per_step, max_chunk_tokens)``.
        eos_token_ids: Ids that end a request unless it sets ``ignore_eos``.
        queue_timeout: Seconds a request may wait in the queue before it times out; ``None`` for no
            limit.
        clock: A monotonic clock in seconds, injected so tests control time.

    Raises:
        TypeError: If a limit is not an ``int``.
        ValueError: If a limit is below 1.
    """

    def __init__(
        self,
        config: SchedulerConfig,
        kv: KVCacheManager,
        *,
        max_model_len: int | None = None,
        max_chunk_tokens: int | None = None,
        eos_token_ids: Collection[TokenId] = (),
        queue_timeout: float | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if max_model_len is not None:
            require_int(max_model_len, "max_model_len", minimum=1)
        budget = config.max_tokens_per_step
        if max_chunk_tokens is not None:
            budget = min(budget, require_int(max_chunk_tokens, "max_chunk_tokens", minimum=1))
        rows = kv.table.max_requests
        self._config = config
        self._kv = kv
        self._max_model_len = max_model_len
        self._budget = budget
        self._max_running = rows if config.max_running is None else min(config.max_running, rows)
        self._eos = frozenset(eos_token_ids)
        self._queue_timeout = queue_timeout
        self._clock = clock
        self._queue = RequestQueue(resolve_policy(config.policy))
        self._roster = Roster(kv, self._queue)
        # Aliases of the roster's dicts, which it never rebinds.
        self._requests = self._roster.live
        self._running = self._roster.running
        self._awaiting = self._roster.awaiting
        self._flight: InFlight | None = None
        self._steps = 0
        self._prefill_streak = 0
        self._admit_delay = 0.0
        self._stats: SchedulerStats | None = None

    # -- reading ---------------------------------------------------------------------------

    @property
    def max_running(self) -> int:
        """Most requests that run at once: the config limit, capped by the block table rows."""
        return self._max_running

    @property
    def token_budget(self) -> int:
        """Most tokens one prefill step carries."""
        return self._budget

    @property
    def num_waiting(self) -> int:
        """Requests in the queue."""
        return len(self._queue)

    @property
    def num_running(self) -> int:
        """Requests that hold KV and have not finished."""
        return len(self._running)

    @property
    def num_awaiting_release(self) -> int:
        """Finished requests whose KV waits for the runner to release them."""
        return len(self._awaiting)

    @property
    def has_unfinished(self) -> bool:
        """Whether any request is waiting or running."""
        return bool(self._requests)

    @property
    def in_flight(self) -> bool:
        """Whether a step was handed out and its ``update`` has not come yet."""
        return self._flight is not None

    def get(self, request_id: RequestId) -> Request | None:
        """The live record of a request that has not finished; do not change it."""
        return self._requests.get(request_id)

    def take_events(self) -> tuple[RequestEvent, ...]:
        """Return the events since the last call, oldest first, and forget them."""
        return self._roster.take_events()

    def take_stats(self) -> SchedulerStats | None:
        """The counts of the last :meth:`update`, once; ``None`` if no step ran since."""
        stats, self._stats = self._stats, None
        return stats

    # -- admission and abort ---------------------------------------------------------------

    def add_request(
        self,
        request: TokenizedRequest,
        *,
        deadline: float | None = None,
        blocked_on: RequestId | None = None,
    ) -> None:
        """Queue a request, after the admission checks, in order, and before it holds anything.

        Args:
            request: A validated, tokenized request.
            deadline: Absolute clock reading after which the request times out, or ``None``.
            blocked_on: A sibling queued before this request (a sequence of the same ``n > 1``
                parent); this request is not admitted while that one is waiting or prefilling,
                so it can attach the shared prompt from the prefix cache.

        Raises:
            InvalidRequestError: If the id is in use, the prompt and ``max_tokens`` exceed the
                model context, or the prompt exceeds the KV one request can get on this machine.
            OverloadedError: If the waiting queue is full.
        """
        request_id = request.request_id
        if request_id in self._requests or request_id in self._awaiting:
            raise InvalidRequestError(f"request id {request_id!r} is already in use")
        _check_context(request, self._max_model_len)
        _check_kv_capacity(request, self._kv)
        if len(self._queue) >= self._config.max_waiting:
            raise OverloadedError(
                f"the waiting queue is full ({self._config.max_waiting} requests); retry later"
            )
        live = Request.from_tokenized(
            request, arrival=self._clock(), deadline=deadline, blocked_on=blocked_on
        )
        self._requests[request_id] = live
        self._queue.push(live)

    def abort(self, request_id: RequestId, reason: FinishReason = FinishReason.CANCELLED) -> bool:
        """End a request from outside: a cancel, or a stop the engine decided.

        A request outside the step in flight ends now and frees its KV. One inside it is marked and
        ends at the ``update`` of that step, so no block is freed while a kernel may still read it.
        The engine uses ``FinishReason.STOP`` for a stop string it matched on the decoded text.

        Returns:
            ``False`` if the request is unknown or already finished.
        """
        request = self._requests.get(request_id)
        if request is None:
            return False
        if self._flight is not None and request_id in self._flight.request_ids:
            if request.abort_reason is None:
                request.abort_reason = reason
            return True
        self._roster.finish(request, reason, release=True)
        return True

    # -- schedule --------------------------------------------------------------------------

    def schedule(self, lookahead: Mapping[RequestId, int] | None = None) -> StepInput | None:
        """Plan the next step, or return ``None`` when there is nothing to run.

        The order is: expire requests that ran out of time, then continue a prefill that
        is half done, else admit waiting requests, else decode every ready request. After
        ``max_consecutive_prefills`` prefill steps in a row, one decode step goes first while
        requests are decoding. Admission is strictly in queue order: a request that does not fit
        is never skipped for a smaller one behind it.

        Args:
            lookahead: Extra decode rows per request (``k``), which the execution layer asks the
                speculative layer for. The scheduler reserves ``1 + k`` rows, falls back to 1 when
                they do not fit, and caps ``k`` by what the request may still generate.

        Raises:
            EngineError: If the previous step has not been updated.
        """
        if self._flight is not None:
            raise EngineError("schedule() called while a step is in flight; call update() first")
        self._roster.expire(self._clock(), self._queue_timeout)
        extra = lookahead or {}
        while True:
            step = self._plan(extra)
            if step is not None:
                return step
            if self._running:
                self._roster.shed_stalled()
            elif not self._roster.shed_unservable_head():
                return None

    def _plan(self, lookahead: Mapping[RequestId, int]) -> StepInput | None:
        ready = any(not r.is_prefilling for r in self._running.values())
        if ready and self._prefill_streak >= self._config.max_consecutive_prefills:
            return self._plan_decode(lookahead) or self._plan_prefill()
        return self._plan_prefill() or self._plan_decode(lookahead)

    def _plan_prefill(self) -> StepInput | None:
        budget = self._budget
        chunks: list[tuple[Request, int]] = []
        partial = [r for r in self._running.values() if r.is_prefilling]
        if partial:
            block_size = self._kv.block_size
            for request in partial:
                if budget <= 0:
                    break
                chunk = _next_chunk(request.prompt_length, request.computed, budget, block_size)
                size = _grant_chunk(self._kv, request, chunk)
                if size:
                    chunks.append((request, size))
                    budget -= size
        else:
            now = self._clock()
            while budget > 0 and len(self._running) < self._max_running:
                head = self._queue.peek()
                if head is None or self._waits_for_sibling(head):
                    break
                match = (
                    self._kv.match_prefix(head.prompt, head.cache_salt)
                    if head.prefix_cache
                    else None
                )
                start = match.num_tokens if match is not None else 0
                size = _grant_chunk(
                    self._kv,
                    head,
                    _next_chunk(head.prompt_length, start, budget, self._kv.block_size),
                    match,
                )
                if not size:
                    break
                head.computed = head.cached_tokens = start
                head.blocked_on = None
                self._admit_delay = max(self._admit_delay, now - head.arrival)
                self._queue.pop()
                head.row = self._kv.table.row_of(head.request_id)
                head.move_to(RequestStatus.RUNNING)
                self._running[head.request_id] = head
                chunks.append((head, size))
                budget -= size
        if not chunks:
            return None
        self._prefill_streak += 1
        return self._issue(
            ForwardMode.PREFILL,
            [
                (
                    r,
                    size,
                    r.prompt[r.computed : r.computed + size],
                    r.computed + size >= len(r.prompt),
                )
                for r, size in chunks
            ],
        )

    def _waits_for_sibling(self, request: Request) -> bool:
        """Whether a sibling this request waits for is still queued or prefilling."""
        if request.blocked_on is None:
            return False
        sibling = self._requests.get(request.blocked_on)
        return sibling is not None and (
            sibling.status is RequestStatus.WAITING or sibling.is_prefilling
        )

    def _plan_decode(self, lookahead: Mapping[RequestId, int]) -> StepInput | None:
        entries: list[tuple[Request, int, tuple[TokenId, ...], bool]] = []
        for request in self._running.values():
            if request.is_prefilling or request.pending is None:
                continue
            rows = _grant_rows(
                self._kv,
                request,
                lookahead.get(request.request_id, 0),
                request.tokens_left(self._max_model_len),
            )
            if rows:
                entries.append((request, rows, (request.pending,), True))
        if not entries:
            return None
        self._prefill_streak = 0
        mode = (
            ForwardMode.VERIFY if any(rows > 1 for _, rows, _, _ in entries) else ForwardMode.DECODE
        )
        return self._issue(mode, entries)

    def _issue(
        self,
        mode: ForwardMode,
        entries: list[tuple[Request, int, tuple[TokenId, ...], bool]],
    ) -> StepInput:
        """Freeze the entries into a ``StepInput`` and remember it as the step in flight."""
        self._steps += 1
        step = StepInput(
            step_id=self._steps,
            mode=mode,
            request_ids=tuple(r.request_id for r, _, _, _ in entries),
            batch_rows=tuple(r.row for r, _, _, _ in entries),
            rows_per_request=tuple(rows for _, rows, _, _ in entries),
            start_positions=tuple(r.computed for r, _, _, _ in entries),
            tokens=tuple(known for _, _, known, _ in entries),
            need_logits=tuple(logits for _, _, _, logits in entries),
            sampling_params=tuple(r.params for r, _, _, _ in entries),
            table_updates=tuple(self._kv.table.take_dirty()),
            sequence_indices=tuple(r.sequence_index for r, _, _, _ in entries),
            output_lengths=tuple(len(r.output) for r, _, _, _ in entries),
        )
        self._flight = InFlight(
            step_id=step.step_id,
            mode=mode,
            request_ids=step.request_ids,
            rows=step.rows_per_request,
            max_queue_delay=self._admit_delay,
        )
        self._admit_delay = 0.0
        return step

    # -- update ----------------------------------------------------------------------------

    def update(self, output: StepOutput) -> None:
        """Apply the runner's answer: commit tokens, emit them, end the requests that are done.

        Tokens are cut at the first stop token and at the request's length limits.
        A request ends at this call for a stop, a length limit, the KV limit of this machine, a
        deadline, a pending abort or a step error. Its KV is freed here if ``output.releasable``
        lists it, otherwise when a later output does.

        Raises:
            EngineError: If no step is in flight, the output answers another step, or the runner
                returned a token count the step cannot have produced.
        """
        flight = self._flight
        if flight is None:
            raise EngineError("update() called without a step in flight")
        if output.step_id != flight.step_id or output.request_ids != flight.request_ids:
            raise EngineError(
                f"update() got an answer for step {output.step_id} {output.request_ids}, "
                f"but step {flight.step_id} {flight.request_ids} is in flight"
            )
        self._flight = None
        if output.releasable is not None:
            self._roster.release(output.releasable)
        releasable = frozenset(
            output.request_ids if output.releasable is None else output.releasable
        )
        now = self._clock()
        emitted = 0
        for index, request_id in enumerate(flight.request_ids):
            emitted += self._apply(
                self._requests[request_id],
                flight,
                index,
                output,
                now,
                release=request_id in releasable,
            )
        finished, aborted, failed, settled = self._roster.take_counts()
        self._stats = SchedulerStats(
            step_id=flight.step_id,
            mode=flight.mode,
            num_running=len(self._running),
            num_waiting=len(self._queue),
            num_preempted=0,
            num_scheduled_tokens=sum(flight.rows),
            num_new_tokens=emitted,
            num_finished=finished,
            num_aborted=aborted,
            num_failed=failed,
            num_settled=settled,
            prefill_streak=self._prefill_streak,
            max_queue_delay=flight.max_queue_delay,
            kv_free_blocks=self._kv.free_counts,
        )

    def _apply(
        self,
        request: Request,
        flight: InFlight,
        index: int,
        output: StepOutput,
        now: float,
        *,
        release: bool,
    ) -> int:
        """Apply one request's answer; returns the tokens it emitted."""
        if output.error is not None:
            self._roster.finish(request, FinishReason.ERROR, error=output.error, release=release)
            return 0
        produced = output.tokens[index]
        rows = flight.rows[index]
        emitted: tuple[TokenId, ...] = ()
        reason: FinishReason | None = None
        stop: TokenId | None = None
        if flight.mode is ForwardMode.PREFILL:
            self._kv.commit(request.request_id, rows)
            request.computed += rows
            self._cache_full_blocks(request)
            if request.is_prefilling and produced:
                raise EngineError(f"request {request.request_id!r}: a prefill chunk emitted tokens")
            if not request.is_prefilling:
                if len(produced) != 1:
                    raise EngineError(
                        f"request {request.request_id!r}: the last prefill chunk must emit one "
                        f"token, got {len(produced)}"
                    )
                emitted, reason, stop = request.accept(produced, self._eos, self._max_model_len)
        else:
            if not 1 <= len(produced) <= rows:
                raise EngineError(
                    f"request {request.request_id!r}: a decode step with {rows} rows emitted "
                    f"{len(produced)} tokens"
                )
            emitted, reason, stop = request.accept(produced, self._eos, self._max_model_len)
            self._kv.commit(request.request_id, len(emitted))
            request.computed += len(emitted)
            self._cache_full_blocks(request)
        error: ErrorInfo | None = None
        if reason is None:
            if request.computed >= self._kv.max_tokens_per_request:
                reason = FinishReason.LENGTH
                error = CapacityError(
                    f"request {request.request_id!r} reached the {request.computed} tokens of KV "
                    "one request can hold on this machine; the limit is memory, not the model "
                    "context"
                ).to_info()
            elif request.abort_reason is not None:
                reason = request.abort_reason
            elif request.deadline is not None and now > request.deadline:
                reason, error = FinishReason.TIMEOUT, timeout_error(request)
        logprobs: tuple[TokenLogprob, ...] | None = None
        if output.logprobs and output.logprobs[index] is not None:
            logprobs = (output.logprobs[index] or ())[: len(emitted)]
        if reason is not None:
            self._roster.finish(
                request,
                reason,
                tokens=emitted,
                stop_reason=stop,
                error=error,
                logprobs=logprobs,
                release=release,
            )
        elif emitted:
            self._roster.events.append(
                RequestEvent(
                    request_id=request.request_id,
                    token_ids=emitted,
                    logprobs=logprobs,
                    cached_tokens=request.cached_tokens,
                    cache_written_tokens=request.cache_written_tokens,
                )
            )
        return len(emitted)

    def _cache_full_blocks(self, request: Request) -> None:
        """Register the blocks the last commit filled, so later requests can attach them.

        The prompt's blocks are registered in a call of their own, so the blocks this request
        added to the cache from its prompt are counted (``cache_written_tokens``); the blocks
        of its output are registered too but are not prompt tokens.
        """
        registered, full = self._kv.cacheable_blocks(request.request_id)
        if full <= registered:
            return
        size = self._kv.block_size
        prompt_end = min(max(len(request.prompt) // size, registered), full)
        if prompt_end > registered:
            added = self._kv.cache_blocks(
                request.request_id, request.tokens_between(registered * size, prompt_end * size)
            )
            request.cache_written_tokens += added * size
            registered = prompt_end
        if full > registered:
            self._kv.cache_blocks(
                request.request_id, request.tokens_between(registered * size, full * size)
            )


def _check_context(request: TokenizedRequest, limit: int | None) -> None:
    """Admission check: the prompt and ``max_tokens`` fit the model context."""
    if limit is None:
        return
    total = request.max_total_length
    if total is not None and total > limit:
        raise InvalidRequestError(
            f"prompt ({request.prompt_length}) plus max_tokens "
            f"({request.sampling.max_tokens}) must be <= the model context of {limit} tokens"
        )
    if request.prompt_length >= limit:
        raise InvalidRequestError(
            f"prompt ({request.prompt_length}) must be shorter than the model context "
            f"of {limit} tokens"
        )


def _check_kv_capacity(request: TokenizedRequest, kv: KVCacheManager) -> None:
    """Admission check: the prompt fits the KV one request can hold on this machine.

    Read from fixed sizes only: :meth:`KVCacheManager.stats` walks the free queue to count cached
    blocks, which costs milliseconds on a large pool, on the engine thread, for every request.
    """
    capacity = kv.max_context_tokens
    if request.prompt_length > capacity:
        pool_tokens = min(kv.stats().usable_blocks) * kv.block_size
        raise InvalidRequestError(
            f"prompt ({request.prompt_length} tokens) exceeds the {capacity} tokens of KV "
            f"cache this engine can give one request (block table row: "
            f"{kv.max_tokens_per_request}, pool: {pool_tokens}); shorten the prompt or "
            "give the cache more memory"
        )


def _next_chunk(prompt_length: int, computed: int, budget: int, block_size: int) -> int:
    """The next chunk of a prompt from ``computed``: ends on a block boundary unless the last."""
    remaining = prompt_length - computed
    if remaining <= budget:
        return remaining
    size = budget - ((computed + budget) % block_size)
    return size if size > 0 else budget


def _grant_chunk(
    kv: KVCacheManager, request: Request, size: int, prefix: PrefixMatch | None = None
) -> int:
    """Allocate room for a chunk, or for as much of it as the free blocks allow.

    Args:
        kv: The cache manager.
        request: The request.
        size: Tokens of the chunk.
        prefix: For a request leaving the queue, the prefix match to attach first.

    Returns:
        The tokens the request now has room for, 0 if none.
    """
    request_id = request.request_id
    if kv.allocate(request_id, size, prefix):
        return size
    held = (
        kv.table.num_blocks(request_id, 0) * kv.block_size - request.computed
        if kv.has(request_id)
        else 0
    )
    shrunk = min(size, held + kv.free_blocks * kv.block_size)
    if 1 <= shrunk < size and kv.allocate(request_id, shrunk, prefix):
        return shrunk
    return 0


def _grant_rows(kv: KVCacheManager, request: Request, drafts: int, left: int) -> int:
    """Rows to reserve for a decode step: ``1 + k`` if they fit, else 1, else 0.

    Args:
        kv: The cache manager.
        request: A request that finished its prefill.
        drafts: The look-ahead ``k`` the runner asked for.
        left: Tokens the request may still emit; ``k`` is capped so no row is wasted.
    """
    k = require_int(drafts, "lookahead", minimum=0)
    room = kv.max_tokens_per_request - request.computed
    if room < 1:
        return 0
    k = min(k, room - 1, left - 1)
    if kv.allocate(request.request_id, 1 + k):
        return 1 + k
    if k > 0 and kv.allocate(request.request_id, 1):
        return 1
    return 0
