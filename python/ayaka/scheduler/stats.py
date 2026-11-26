"""What one scheduler step did, counted where it happened.

:class:`SchedulerStats` is built inside :meth:`~ayaka.scheduler.scheduler.Scheduler.update` from
the step in flight, the running set, the queue and the requests that ended, never by re-deriving
from the engine and never by reading a tensor. The engine takes it with
:meth:`~ayaka.scheduler.scheduler.Scheduler.take_stats` after each step and hands it to the stat
loggers; it carries no request id, so nothing per request reaches a metric label.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ayaka.core.types import ForwardMode

__all__ = ["SchedulerStats"]


@dataclass(frozen=True, slots=True)
class SchedulerStats:
    """The counts of one step.

    Attributes:
        step_id: The step these counts describe.
        mode: Prefill, decode or verify.
        num_running: Requests holding KV after the step.
        num_waiting: Requests in the queue after the step.
        num_preempted: Requests that gave their KV back to wait again; 0 until preemption exists.
        num_scheduled_tokens: Rows the step ran, look-ahead rows included.
        num_new_tokens: Tokens emitted after the stop and length cuts.
        num_finished: Requests that ended normally (stop or length) since the previous step.
        num_aborted: Requests cancelled since the previous step.
        num_failed: Requests that ended with an error or a timeout since the previous step.
        num_settled: Requests whose KV was freed since the previous step.
        prefill_streak: Prefill steps in a row so far; the starvation guard reads it.
        max_queue_delay: Longest wait, in seconds, of a request this step admitted; 0 if none.
        kv_free_blocks: Free blocks per KV group after the step.
    """

    step_id: int
    mode: ForwardMode
    num_running: int
    num_waiting: int
    num_preempted: int
    num_scheduled_tokens: int
    num_new_tokens: int
    num_finished: int
    num_aborted: int
    num_failed: int
    num_settled: int
    prefill_streak: int
    max_queue_delay: float
    kv_free_blocks: tuple[int, ...]
    pressure_level: str = "normal"
    num_oom_recoveries: int = 0
