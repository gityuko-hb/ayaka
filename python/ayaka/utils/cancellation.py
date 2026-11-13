"""Cooperative cancellation for work that runs before a request exists.

One :class:`CancellationToken` is shared by whoever submits work and whoever runs
it. The submitter calls :meth:`~CancellationToken.cancel`; the worker polls
:meth:`~CancellationToken.raise_if_cancelled` between steps. Nothing is interrupted
forcibly, so native calls that are already running finish before the token is seen.

The token lives in ``utils`` and not next to ``Request`` because tokenization
happens *before* a request is created and must not depend on the request layer.
"""

from __future__ import annotations

import threading

from ayaka.core.errors import CancelledError

__all__ = ["CancellationToken"]


class CancellationToken:
    """A one-way, thread-safe cancellation flag with the reason of the first cancel.

    Cancelling is idempotent: the first reason wins and later calls change nothing.
    """

    __slots__ = ("_lock", "_reason")

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._reason: str | None = None

    @property
    def is_cancelled(self) -> bool:
        """Whether :meth:`cancel` has been called."""
        return self._reason is not None

    @property
    def reason(self) -> str | None:
        """Why the token was cancelled, or ``None`` while it is still active."""
        return self._reason

    def cancel(self, reason: str = "cancelled") -> None:
        """Mark the token cancelled; only the first call records its reason."""
        with self._lock:
            if self._reason is None:
                self._reason = reason

    def raise_if_cancelled(self) -> None:
        """Raise :class:`CancelledError` carrying the reason if already cancelled."""
        reason = self._reason
        if reason is not None:
            raise CancelledError(reason)
