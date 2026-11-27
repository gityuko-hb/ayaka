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
from ayaka.utils.validation import require_int, require_number

__all__ = ["PreemptionPolicyName", "SchedulerConfig", "SchedulingPolicyName"]

SchedulingPolicyName = Literal[
    "fcfs",  # first come, first served
    "priority",  # higher ``priority`` of the request first; ties go first come, first served
]
"""Order of the waiting queue. The comparison itself lives in ``ayaka.scheduler``."""

PreemptionPolicyName = Literal[
    "newest",  # lowest priority first, then most recently admitted
    "lowest_progress",  # lowest priority first, then fewest emitted/computed tokens
]
"""Policy for picking a running victim request under KV cache pressure."""

_POLICIES: tuple[str, ...] = ("fcfs", "priority")
_PREEMPTION_POLICIES: tuple[str, ...] = ("newest", "lowest_progress")


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
    preemption_policy: PreemptionPolicyName = "newest"
    pressure_high: float = 0.80
    pressure_critical: float = 0.92
    preemption_timeout: float | None = None
    prefix_wait_timeout: float | None = 5.0

    def __post_init__(self) -> None:
        if self.max_running is not None:
            require_int(self.max_running, "scheduler.max_running", minimum=1)
        require_int(self.max_tokens_per_step, "scheduler.max_tokens_per_step", minimum=1)
        require_int(self.max_waiting, "scheduler.max_waiting", minimum=1)
        require_int(self.max_consecutive_prefills, "scheduler.max_consecutive_prefills", minimum=1)
        if cast("object", self.policy) not in _POLICIES:
            raise ConfigError("scheduler.policy", "INVALID_CHOICE", f"must be one of {_POLICIES}")
        if cast("object", self.preemption_policy) not in _PREEMPTION_POLICIES:
            raise ConfigError(
                "scheduler.preemption_policy",
                "INVALID_CHOICE",
                f"must be one of {_PREEMPTION_POLICIES}",
            )
        high = require_number(self.pressure_high, "scheduler.pressure_high")
        critical = require_number(self.pressure_critical, "scheduler.pressure_critical")
        if not 0.0 < high < critical <= 1.0:
            raise ConfigError(
                "scheduler.pressure",
                "INVALID_RANGE",
                f"must satisfy 0 < pressure_high < pressure_critical <= 1, "
                f"got {high} and {critical}",
            )
        object.__setattr__(self, "pressure_high", high)
        object.__setattr__(self, "pressure_critical", critical)
        if self.preemption_timeout is not None:
            timeout = require_number(self.preemption_timeout, "scheduler.preemption_timeout")
            if timeout <= 0.0:
                raise ConfigError(
                    "scheduler.preemption_timeout",
                    "INVALID_RANGE",
                    f"must be > 0, got {timeout}",
                )
            object.__setattr__(self, "preemption_timeout", timeout)
        if self.prefix_wait_timeout is not None:
            wait = require_number(self.prefix_wait_timeout, "scheduler.prefix_wait_timeout")
            if wait <= 0.0:
                raise ConfigError(
                    "scheduler.prefix_wait_timeout",
                    "INVALID_RANGE",
                    f"must be > 0, got {wait}",
                )
            object.__setattr__(self, "prefix_wait_timeout", wait)
