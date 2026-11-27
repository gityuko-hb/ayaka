"""Victim selection and timeout helpers for scheduler preemption."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ayaka.core.errors import PreemptedTimeoutError

if TYPE_CHECKING:
    from collections.abc import Iterable

    from ayaka.configs.scheduler import PreemptionPolicyName
    from ayaka.core.errors import ErrorInfo
    from ayaka.scheduler.request import Request

__all__ = ["preempted_timeout_error", "select_victim"]


def select_victim(running: Iterable[Request], policy: PreemptionPolicyName) -> Request:
    """Choose which running request to preempt under KV pressure.

    ``"newest"`` picks the lowest-priority request, breaking ties by latest
    arrival (most recently admitted). ``"lowest_progress"`` picks the lowest
    priority, then fewest emitted tokens, then fewest computed tokens, then
    latest arrival.

    Raises:
        ValueError: If ``running`` is empty.
    """
    candidates = list(reversed(list(running)))
    if not candidates:
        raise ValueError("cannot select a preemption victim from an empty running set")
    if policy == "lowest_progress":
        return max(
            candidates,
            key=lambda req: (-req.priority, -len(req.output), -req.computed, req.arrival),
        )
    return max(candidates, key=lambda req: (-req.priority, req.arrival))


def preempted_timeout_error(request: Request, elapsed: float) -> ErrorInfo:
    """Structured error when a preempted request exceeds ``preemption_timeout``."""
    return PreemptedTimeoutError(
        f"request {request.request_id!r} remained preempted for {elapsed:.2f}s"
    ).to_info()
