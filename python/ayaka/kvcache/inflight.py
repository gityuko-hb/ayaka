"""Track active in-flight prefix prefills so cold concurrent requests coalesce."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

    from ayaka.core.types import RequestId

__all__ = ["InFlightPrefixTracker"]


class InFlightPrefixTracker:
    """Index of full-block prefix digests currently being prefilled by running requests."""

    def __init__(self) -> None:
        self._by_first_digest: dict[bytes, list[RequestId]] = {}
        self._request_digest: dict[RequestId, bytes] = {}

    def register(self, request_id: RequestId, digests: Sequence[bytes]) -> None:
        """Record that ``request_id`` is prefilling a prompt starting with ``digests[0]``."""
        if not digests:
            return
        first = digests[0]
        if request_id in self._request_digest:
            return
        self._request_digest[request_id] = first
        leaders = self._by_first_digest.setdefault(first, [])
        if request_id not in leaders:
            leaders.append(request_id)

    def leader_for(self, digests: Sequence[bytes]) -> RequestId | None:
        """Return an active leader prefilling the same first prefix block, if any."""
        if not digests:
            return None
        leaders = self._by_first_digest.get(digests[0])
        return leaders[0] if leaders else None

    def complete(self, request_id: RequestId) -> None:
        """Stop tracking ``request_id`` once its prefill finishes, aborts, or preempts."""
        digest = self._request_digest.pop(request_id, None)
        if digest is None:
            return
        leaders = self._by_first_digest.get(digest)
        if leaders is None:
            return
        if request_id in leaders:
            leaders.remove(request_id)
        if not leaders:
            del self._by_first_digest[digest]

    def __len__(self) -> int:
        return len(self._request_digest)
