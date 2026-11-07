"""Engine-level metrics: iteration stats and stat loggers.

:class:`IterationStats` is the per-step source of truth; loggers aggregate it
and export it through the logging system or a Prometheus registry.
"""

from __future__ import annotations

from ayaka.metrics.loggers import LoggingStatLogger, PrometheusStatLogger, StatLogger
from ayaka.metrics.manager import StatLoggerManager
from ayaka.metrics.stats import IterationStats

__all__ = [
    "IterationStats",
    "LoggingStatLogger",
    "PrometheusStatLogger",
    "StatLogger",
    "StatLoggerManager",
]
