"""The running constraints of the requests in flight, keyed by request id.

The runner holds one :class:`ConstraintRuntime`. The engine adds a request's matcher when it queues
the request, feeds it the request's committed tokens before the next step, and drops it when the
request ends; the runner asks it for the masks of a step's rows. A request without a constraint is
never in it, and a constraint that has not started (a lazy tool call before the model opens one)
gives no mask, so neither costs anything per step.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable

    import numpy as np
    from numpy.typing import NDArray

    from ayaka.core.types import RequestId
    from ayaka.sampling.structured.matcher import ConstraintMatcher

__all__ = ["ConstraintRuntime"]


class ConstraintRuntime:
    """Matchers of the requests that have a constraint."""

    def __init__(self) -> None:
        self._matchers: dict[RequestId, ConstraintMatcher] = {}

    def __len__(self) -> int:
        return len(self._matchers)

    def __contains__(self, request_id: object) -> bool:
        return request_id in self._matchers

    def admit(self, request_id: RequestId, matcher: ConstraintMatcher) -> None:
        """Start tracking a request's constraint."""
        self._matchers[request_id] = matcher

    def accept(self, request_id: RequestId, token_ids: Iterable[int]) -> None:
        """Feed committed tokens; ignored for a request without a constraint."""
        matcher = self._matchers.get(request_id)
        if matcher is not None:
            matcher.accept(token_ids)

    def drop(self, request_id: RequestId) -> None:
        """Forget a request that ended."""
        self._matchers.pop(request_id, None)

    def active(self, request_id: RequestId) -> bool:
        """Whether the request's next draw is constrained."""
        matcher = self._matchers.get(request_id)
        return matcher is not None and matcher.active

    def mask(self, request_id: RequestId) -> NDArray[np.uint32] | None:
        """The request's allowed tokens as ``uint32`` words, or ``None`` when free."""
        matcher = self._matchers.get(request_id)
        return matcher.mask() if matcher is not None else None
