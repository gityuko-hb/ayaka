"""Which request runs in each step: requests, step contracts, the waiting queue and the scheduler.

Public names are resolved from the module that defines them on first access, so importing this
package never loads torch.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ayaka.utils.lazy_exports import build_lazy_exports

if TYPE_CHECKING:
    from ayaka.scheduler.queue import RequestQueue, SchedulingPolicy
    from ayaka.scheduler.request import Request, RequestEvent, RequestStatus
    from ayaka.scheduler.scheduler import Scheduler
    from ayaka.scheduler.step import StepInput, StepOutput

__all__ = [
    "Request",
    "RequestEvent",
    "RequestQueue",
    "RequestStatus",
    "Scheduler",
    "SchedulingPolicy",
    "StepInput",
    "StepOutput",
]

_EXPORTS = {
    "queue": ("RequestQueue", "SchedulingPolicy"),
    "request": ("Request", "RequestEvent", "RequestStatus"),
    "scheduler": ("Scheduler",),
    "step": ("StepInput", "StepOutput"),
}

__getattr__, __dir__ = build_lazy_exports(__name__, _EXPORTS, globals())
