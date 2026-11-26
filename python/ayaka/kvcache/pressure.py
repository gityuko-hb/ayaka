"""KV cache pressure levels and utilization thresholds."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from ayaka.core.errors import ConfigError
from ayaka.utils.validation import require_number

__all__ = ["PressureLevel", "PressureThresholds", "evaluate_pressure"]


class PressureLevel(StrEnum):
    """Occupancy state of the device KV cache pool."""

    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass(frozen=True, slots=True)
class PressureThresholds:
    """Utilization thresholds separating ``normal``, ``high`` and ``critical`` pressure.

    Attributes:
        high: Utilization in ``(0, 1)`` above which new request admission pauses.
        critical: Utilization in ``(high, 1]`` above which pressure is critical.
    """

    high: float = 0.80
    critical: float = 0.92

    def __post_init__(self) -> None:
        high = require_number(self.high, "pressure.high")
        critical = require_number(self.critical, "pressure.critical")
        if not 0.0 < high < critical <= 1.0:
            raise ConfigError(
                "scheduler.pressure",
                "INVALID_RANGE",
                f"pressure thresholds must satisfy 0 < high < critical <= 1, "
                f"got {high} and {critical}",
            )
        object.__setattr__(self, "high", high)
        object.__setattr__(self, "critical", critical)


def evaluate_pressure(utilization: float, thresholds: PressureThresholds) -> PressureLevel:
    """Map current KV pool utilization in ``[0, 1]`` to a :class:`PressureLevel`."""
    if utilization >= thresholds.critical:
        return PressureLevel.CRITICAL
    if utilization >= thresholds.high:
        return PressureLevel.HIGH
    return PressureLevel.NORMAL
