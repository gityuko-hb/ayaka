"""The roster of a scheduler: every request that has not finished, and the one way each ends.

:class:`Roster` owns the live, running and awaiting-release sets and the pending events; every end
of a request goes through :meth:`Roster.finish`: it sets the reason, frees the KV, releases the
batch row and queues the last event. :class:`InFlight` is what the scheduler remembers about the
step it handed out, to check the answer against. Both are the scheduler's own state, not a public
contract.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from ayaka.core.errors import CapacityError, RequestTimeout
from ayaka.core.types import FinishReason, ForwardMode
from ayaka.scheduler.preemption import preempted_timeout_error
from ayaka.scheduler.request import Request, RequestEvent, RequestStatus

if TYPE_CHECKING:
    from collections.abc import Collection

    from ayaka.core.errors import ErrorInfo
    from ayaka.core.outputs import TokenLogprob
    from ayaka.core.types import RequestId, TokenId
    from ayaka.kvcache.manager import KVCacheManager
    from ayaka.scheduler.queue import RequestQueue


__all__ = ["InFlight", "Roster", "timeout_error"]


@dataclass(frozen=True, slots=True)
class InFlight:
    """What the scheduler remembers about the step it handed out, to check the answer against."""

    step_id: int
    mode: ForwardMode
    request_ids: tuple[RequestId, ...]
    rows: tuple[int, ...]
    max_queue_delay: float = 0.0


class Roster:
    """Every request that has not finished, and the one way each of them ends.

    It owns the live, running and awaiting-release sets and the pending events, and every end of a
    request goes through :meth:`finish`: reason, KV, batch row, last event. The sets are never
    rebound, so the scheduler may hold references to them.
    """

    def __init__(self, kv: KVCacheManager, queue: RequestQueue) -> None:
        self._kv = kv
        self._queue = queue
        self.live: dict[RequestId, Request] = {}
        self.running: dict[RequestId, Request] = {}
        self.preempted: list[Request] = []
        self.awaiting: dict[RequestId, Request] = {}
        self.events: list[RequestEvent] = []
        # Ends, preemptions and frees since the last stats were taken.
        self.finished = 0
        self.aborted = 0
        self.failed = 0
        self.settled = 0
        self.preempted_step = 0

    def take_counts(self) -> tuple[int, int, int, int, int]:
        """``(finished, aborted, failed, settled, preempted)`` since last call, then reset."""
        counts = (self.finished, self.aborted, self.failed, self.settled, self.preempted_step)
        self.finished = self.aborted = self.failed = self.settled = self.preempted_step = 0
        return counts

    def take_events(self) -> tuple[RequestEvent, ...]:
        events, self.events = self.events, []
        return tuple(events)

    def release(self, request_ids: Collection[RequestId]) -> None:
        """Free the KV of finished requests the runner now lists as releasable."""
        for request_id in request_ids:
            if self.awaiting.pop(request_id, None) is not None:
                self._kv.free(request_id)
                self.settled += 1

    def finish(
        self,
        request: Request,
        reason: FinishReason,
        *,
        tokens: tuple[TokenId, ...] = (),
        stop_reason: TokenId | None = None,
        error: ErrorInfo | None = None,
        logprobs: tuple[TokenLogprob, ...] | None = None,
        release: bool,
    ) -> None:
        """End a request: set the reason, free the KV and the row, queue the last event.

        Args:
            request: A waiting or running request.
            reason: Why it ended.
            tokens: Tokens emitted by the step that ended it.
            stop_reason: The stop token id, when a stop token ended it.
            error: Detail for ``ERROR``, ``TIMEOUT`` or a memory ``LENGTH``.
            logprobs: Logprobs of ``tokens``.
            release: Whether the KV may be freed now; if not it waits for the runner's release.
        """
        request_id = request.request_id
        holds_kv = request.status is RequestStatus.RUNNING
        if request.status is RequestStatus.WAITING:
            self._queue.remove(request_id)
        elif request.status is RequestStatus.PREEMPTED:
            self.preempted = [r for r in self.preempted if r.request_id != request_id]
        request.move_to(RequestStatus.FINISHED)
        request.finish_reason = reason
        del self.live[request_id]
        self.running.pop(request_id, None)
        if holds_kv:
            if release:
                self._kv.free(request_id)
                self.settled += 1
            else:
                self.awaiting[request_id] = request
        if reason is FinishReason.CANCELLED:
            self.aborted += 1
        elif reason in (FinishReason.ERROR, FinishReason.TIMEOUT):
            self.failed += 1
        else:
            self.finished += 1
        self.events.append(
            RequestEvent(
                request_id=request_id,
                token_ids=tokens,
                finish_reason=reason,
                stop_reason=stop_reason,
                error=error,
                logprobs=logprobs,
                cached_tokens=request.cached_tokens,
                cache_written_tokens=request.cache_written_tokens,
            )
        )

    def preempt(self, request: Request, *, now: float) -> None:
        """Evict a running request back to ``PREEMPTED`` and release all its KV blocks."""
        request_id = request.request_id
        self._kv.free(request_id)
        request.move_to(RequestStatus.PREEMPTED)
        self.running.pop(request_id, None)
        request.row = -1
        request.computed = 0
        request.resuming = True
        request.preempted_count += 1
        request.preempted_since = now
        self.preempted.append(request)
        self.preempted_step += 1
        self.settled += 1

    def expire(
        self,
        now: float,
        queue_timeout: float | None,
        preemption_timeout: float | None = None,
    ) -> None:
        """End waiting, running and preempted requests past their deadlines or timeouts."""
        for request in (*self._queue, *self.running.values()):
            waited = (
                request.status is RequestStatus.WAITING
                and queue_timeout is not None
                and now - request.arrival > queue_timeout
            )
            if waited or (request.deadline is not None and now > request.deadline):
                self.finish(
                    request, FinishReason.TIMEOUT, error=timeout_error(request), release=True
                )
        for request in tuple(self.preempted):
            if request.deadline is not None and now > request.deadline:
                self.finish(
                    request, FinishReason.TIMEOUT, error=timeout_error(request), release=True
                )
                continue
            if preemption_timeout is not None and request.preempted_since is not None:
                elapsed = request.preempted_seconds + max(0.0, now - request.preempted_since)
                if elapsed > preemption_timeout:
                    self.finish(
                        request,
                        FinishReason.TIMEOUT,
                        error=preempted_timeout_error(request, elapsed),
                        release=True,
                    )

    def shed_stalled(self) -> None:
        """No running request could take its next rows: end the lowest-ranked one."""
        # ``max`` keeps the first of equal keys, so walking newest-first picks the latest admitted.
        victim = max(reversed(self.running.values()), key=lambda r: (-r.priority, r.arrival))
        self.finish(
            victim,
            FinishReason.LENGTH,
            error=CapacityError(
                f"the KV cache is full: request {victim.request_id!r} stopped at "
                f"{victim.computed} tokens and no running request can free memory; "
                "the limit is memory, not the model context"
            ).to_info(),
            release=True,
        )

    def shed_unservable_head(self) -> bool:
        """With nothing running, fail a head request that still cannot be admitted.

        Returns:
            ``True`` if a request was ended, so planning should be tried again.
        """
        head = self._queue.peek()
        if head is None or self.awaiting:
            return False
        self.finish(
            head,
            FinishReason.ERROR,
            error=CapacityError(
                f"request {head.request_id!r} cannot get KV room even with nothing else running"
            ).to_info(),
            release=True,
        )
        return True

    def shed_preempted_head(self) -> bool:
        """With nothing running, end a preempted head request that still cannot fit KV."""
        if not self.preempted or self.awaiting:
            return False
        head = self.preempted[0]
        self.finish(
            head,
            FinishReason.LENGTH,
            error=CapacityError(
                f"request {head.request_id!r} cannot re-acquire KV room after preemption"
            ).to_info(),
            release=True,
        )
        return True


def timeout_error(request: Request) -> ErrorInfo:
    """The error of a request that passed its deadline."""
    return RequestTimeout(f"request {request.request_id!r} passed its deadline").to_info()
