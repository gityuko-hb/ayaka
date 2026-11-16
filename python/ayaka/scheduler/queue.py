"""The waiting queue and the policies that order it.

A policy is a pure function from a request to a sort key; smaller keys are served first and equal
keys are served in the order the requests joined. :class:`RequestQueue` owns the structure and knows
no policy by name, so a new order is one more function.
"""

from __future__ import annotations

import heapq
from typing import TYPE_CHECKING, Protocol

from ayaka.core.errors import EngineError

if TYPE_CHECKING:
    from collections.abc import Iterator

    from ayaka.configs.scheduler import SchedulingPolicyName
    from ayaka.core.types import RequestId
    from ayaka.scheduler.request import Request

__all__ = ["RequestQueue", "SchedulingPolicy", "by_priority", "fcfs", "resolve_policy"]


class SchedulingPolicy(Protocol):
    """Maps a request to its place in the queue: smaller keys go first, ties by arrival order."""

    def __call__(self, request: Request, /) -> tuple[int, ...]:
        """Return the sort key of ``request``; it must not change while the request waits."""
        ...


def fcfs(request: Request, /) -> tuple[int, ...]:
    """First come, first served: every request has the same key, so arrival order decides."""
    del request
    return ()


def by_priority(request: Request, /) -> tuple[int, ...]:
    """Higher ``priority`` first; requests of one priority go first come, first served."""
    return (-request.priority,)


_POLICIES: dict[str, SchedulingPolicy] = {"fcfs": fcfs, "priority": by_priority}


def resolve_policy(name: SchedulingPolicyName) -> SchedulingPolicy:
    """The policy a :data:`~ayaka.configs.scheduler.SchedulingPolicyName` stands for.

    Raises:
        KeyError: If the name is unknown; ``SchedulerConfig`` has already refused such a name.
    """
    return _POLICIES[name]


class RequestQueue:
    """Waiting requests in service order.

    Pushing and popping cost ``O(log n)``; removing a request is lazy and costs ``O(1)`` until the
    next ``peek`` or ``pop`` reaches its stale entry.

    Args:
        policy: How requests are ordered; first come, first served by default.
    """

    def __init__(self, policy: SchedulingPolicy = fcfs) -> None:
        self._policy = policy
        self._heap: list[tuple[tuple[int, ...], int, RequestId]] = []
        self._live: dict[RequestId, tuple[tuple[int, ...], int, Request]] = {}
        self._counter = 0

    def __len__(self) -> int:
        return len(self._live)

    def __contains__(self, request_id: object) -> bool:
        return request_id in self._live

    def __iter__(self) -> Iterator[Request]:
        """The waiting requests in the order they would be served."""
        for _key, _seq, request in sorted(self._live.values(), key=lambda entry: entry[:2]):
            yield request

    def push(self, request: Request) -> None:
        """Add a request behind the ones that rank no worse.

        Raises:
            EngineError: If a request with the same id is already queued.
        """
        if request.request_id in self._live:
            raise EngineError(f"request {request.request_id!r} is already queued")
        key = self._policy(request)
        self._counter += 1
        self._live[request.request_id] = (key, self._counter, request)
        heapq.heappush(self._heap, (key, self._counter, request.request_id))

    def peek(self) -> Request | None:
        """The request that would be served next, or ``None`` when the queue is empty."""
        self._drop_stale()
        if not self._heap:
            return None
        return self._live[self._heap[0][2]][2]

    def pop(self) -> Request:
        """Remove and return the request that is served next.

        Raises:
            IndexError: If the queue is empty.
        """
        self._drop_stale()
        if not self._heap:
            raise IndexError("pop from an empty RequestQueue")
        _key, _seq, request_id = heapq.heappop(self._heap)
        return self._live.pop(request_id)[2]

    def remove(self, request_id: RequestId) -> Request:
        """Take a request out of the queue wherever it is.

        Raises:
            KeyError: If the request is not queued.
        """
        return self._live.pop(request_id)[2]

    def _drop_stale(self) -> None:
        while self._heap:
            _key, seq, request_id = self._heap[0]
            entry = self._live.get(request_id)
            if entry is not None and entry[1] == seq:
                return
            heapq.heappop(self._heap)
