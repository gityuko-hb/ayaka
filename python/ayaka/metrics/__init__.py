"""Engine-level metrics: iteration stats, stat loggers and native histograms.

:class:`IterationStats` is the per-step source of truth; loggers aggregate it
and export it through the logging system or a Prometheus registry.
:class:`Histogram` provides the bounded native-histogram layout for
latency-style observations and maps directly onto Prometheus native histograms.
"""

from __future__ import annotations

from ayaka.metrics.histogram import Histogram
from ayaka.metrics.loggers import LoggingStatLogger, PrometheusStatLogger, StatLogger
from ayaka.metrics.manager import StatLoggerManager
from ayaka.metrics.stats import IterationStats

__all__ = [
    "Histogram",
    "IterationStats",
    "LoggingStatLogger",
    "PrometheusStatLogger",
    "StatLogger",
    "StatLoggerManager",
]
