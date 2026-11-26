"""Server metrics: Prometheus counters, gauges and histograms on a registry of their own.

Every label has a small, fixed set of values: routes, statuses, outcomes and normalizer rules,
never a request id or anything a client chose.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram

from ayaka.entrypoints.conversation import (
    ChoiceEvent,
    ErrorEvent,
    FinishEvent,
    ToolCallEnd,
    UsageEvent,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from ayaka.entrypoints.conversation import OutputEvent


__all__ = ["ServingMetrics"]


class ServingMetrics:
    """Server metrics on a registry of their own; every label has a small, fixed set of values."""

    def __init__(self) -> None:
        self.registry = CollectorRegistry()
        self.http = Counter(
            "ayaka_http_requests",
            "HTTP requests by route and status",
            ["route", "status"],
            registry=self.registry,
        )
        self.finished = Counter(
            "ayaka_requests_finished",
            "Generation requests by how they ended",
            ["outcome"],
            registry=self.registry,
        )
        self.memory_limit_finishes = Counter(
            "ayaka_memory_limit_finishes",
            "Generation requests stopped because a memory limit was reached",
            registry=self.registry,
        )
        self.prompt_tokens = Counter(
            "ayaka_prompt_tokens", "Prompt tokens of finished requests", registry=self.registry
        )
        self.cached_tokens = Counter(
            "ayaka_prompt_cached_tokens",
            "Prompt tokens of finished requests whose KV came from the prefix cache",
            registry=self.registry,
        )
        self.generation_tokens = Counter(
            "ayaka_generation_tokens",
            "Tokens sent to clients by finished requests",
            registry=self.registry,
        )
        self.tool_calls = Counter(
            "ayaka_tool_calls",
            "Function calls the model wrote, by whether their arguments were one JSON object",
            ["valid"],
            registry=self.registry,
        )
        self.normalized = Counter(
            "ayaka_normalized_replacements",
            "Churn replacements the prompt normalizer made, by rule and client",
            ["rule", "client"],
            registry=self.registry,
        )
        self.running = Gauge(
            "ayaka_requests_running", "Generation requests in flight", registry=self.registry
        )
        self.ready = Gauge(
            "ayaka_server_ready", "1 when the server accepts requests", registry=self.registry
        )
        self.progress_age = Gauge(
            "ayaka_engine_progress_age_seconds",
            "Seconds since the engine last made progress while it had work; 0 when idle",
            registry=self.registry,
        )
        self.duration = Histogram(
            "ayaka_request_seconds",
            "Time from a generation request's arrival to its last byte",
            buckets=(0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 60, 120, 300),
            registry=self.registry,
        )

    def observe(self, events: Sequence[OutputEvent]) -> None:
        """Count what a batch of events says."""
        for event in events:
            match event:
                case ToolCallEnd(_, valid) | ChoiceEvent(_, ToolCallEnd(_, valid)):
                    self.tool_calls.labels("true" if valid else "false").inc()
                case UsageEvent():
                    self.prompt_tokens.inc(event.prompt_tokens)
                    self.cached_tokens.inc(event.cached_tokens)
                    self.generation_tokens.inc(event.completion_tokens)
                case FinishEvent(reason, _, _, limit):
                    self.finished.labels(str(reason)).inc()
                    if limit == "memory":
                        self.memory_limit_finishes.inc()
                case ErrorEvent(code):
                    self.finished.labels(code if code in _OUTCOMES else "error").inc()
                case _:
                    pass


_OUTCOMES: Final = frozenset(
    {"cancelled", "request_timeout", "preempted_timeout", "out_of_memory", "engine_dead"}
)
