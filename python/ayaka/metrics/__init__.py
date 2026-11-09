"""Engine-level metrics: iteration stats, stat loggers and native histograms.

:class:`IterationStats` is the per-step source of truth; loggers aggregate it
and export it through the logging system or a Prometheus registry.
:class:`Histogram` provides the bounded native-histogram layout for
latency-style observations and maps directly onto Prometheus native histograms.

Importing this package is cheap: each public name is resolved from its
submodule on first access.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ayaka.utils.lazy_exports import build_lazy_exports

if TYPE_CHECKING:
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

_EXPORTS = {
    "histogram": ("Histogram",),
    "loggers": ("LoggingStatLogger", "PrometheusStatLogger", "StatLogger"),
    "manager": ("StatLoggerManager",),
    "stats": ("IterationStats",),
}

__getattr__, __dir__ = build_lazy_exports(__name__, _EXPORTS, globals())
