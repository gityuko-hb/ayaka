"""How requests are admitted, ordered and batched into steps.

``SchedulerConfig`` is a frozen host value with the limits the scheduler itself
enforces. Limits that depend on memory (the largest prefill chunk, the number of
running requests the KV pool can feed) are derived by ``memory.plan_budget`` and
combined with these at engine build time; this record holds only what the user
chose.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, cast

from ayaka.configs.base import ConfigMixin
from ayaka.core.errors import ConfigError
from ayaka.utils.validation import require_int

__all__ = ["SchedulerConfig", "SchedulingPolicyName"]

SchedulingPolicyName = Literal[
    "fcfs",  # first come, first served
    "priority",  # higher ``priority`` of the request first; ties go first come, first served
]
"""Order of the waiting queue. The comparison itself lives in ``ayaka.scheduler``."""

_POLICIES: tuple[str, ...] = ("fcfs", "priority")


@dataclass(frozen=True, slots=True)
class SchedulerConfig(ConfigMixin):
    """Admission, ordering and step-size limits of the scheduler.

    Attributes:
        max_running: Most requests that run at once, or ``None`` to let
            ``plan_budget`` derive it from the KV memory.
        max_tokens_per_step: Token budget of one prefill step. The scheduler uses
            ``min(max_tokens_per_step, max_chunk_tokens)`` where
            ``max_chunk_tokens`` is the memory-derived cap. The default of 2048 is
            provisional.
        max_waiting: Longest waiting queue. A request that arrives when it is full
            is rejected as overloaded instead of queued.
        policy: Queue order. See :data:`SchedulingPolicyName`.
        max_consecutive_prefills: Prefill steps in a row after which one decode
            step is inserted while requests are decoding, so a stream of new
            prompts cannot starve them.

    Raises:
        TypeError: If a field has the wrong type.
        ValueError: If a number is below its minimum.
        ConfigError: ``INVALID_CHOICE`` for an unknown ``policy``.
    """

    max_running: int | None = None
    max_tokens_per_step: int = 2048
    max_waiting: int = 1024
    policy: SchedulingPolicyName = "fcfs"
    max_consecutive_prefills: int = 2

    def __post_init__(self) -> None:
        if self.max_running is not None:
            require_int(self.max_running, "scheduler.max_running", minimum=1)
        require_int(self.max_tokens_per_step, "scheduler.max_tokens_per_step", minimum=1)
        require_int(self.max_waiting, "scheduler.max_waiting", minimum=1)
        require_int(self.max_consecutive_prefills, "scheduler.max_consecutive_prefills", minimum=1)
        if cast("object", self.policy) not in _POLICIES:
            raise ConfigError("scheduler.policy", "INVALID_CHOICE", f"must be one of {_POLICIES}")
