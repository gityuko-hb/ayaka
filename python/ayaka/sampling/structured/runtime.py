"""The running constraints of requests in flight, keyed by request id."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    import numpy as np
    from numpy.typing import NDArray

    from ayaka.core.types import RequestId
    from ayaka.sampling.structured.matcher import ConstraintMatcher

__all__ = ["ConstraintRuntime"]


class ConstraintRuntime:
    """Matchers of the requests that have an active or pending constraint."""

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

    def validate_tokens(self, request_id: RequestId, token_ids: Sequence[int]) -> bool:
        """Check whether draft ``token_ids`` form a valid constrained continuation."""
        matcher = self._matchers.get(request_id)
        return True if matcher is None else matcher.validate_tokens(token_ids)

    def rollback(self, request_id: RequestId, num_tokens: int) -> None:
        """Undo ``num_tokens`` tentative or rolled-back steps for ``request_id``."""
        matcher = self._matchers.get(request_id)
        if matcher is not None:
            matcher.rollback(num_tokens)

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

    def window_masks(
        self, request_id: RequestId, draft_tokens: Sequence[int]
    ) -> list[NDArray[np.uint32] | None] | None:
        """Per-row masks across a speculative verification window, or ``None`` when absent."""
        matcher = self._matchers.get(request_id)
        if matcher is None:
            return None
        return matcher.window_masks(draft_tokens)
