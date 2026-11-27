"""Deterministic session replay for ``ayaka.session/1`` recordings."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any, Literal, cast

from ayaka.core.params import SamplingParams
from ayaka.core.request import GenerateRequest
from ayaka.core.types import RequestId
from ayaka.engine.llm import LLM
from ayaka.engine.recorder import RecordedRequest, read_session
from ayaka.entrypoints.console import EXIT_ENGINE, EXIT_OK

if TYPE_CHECKING:
    import argparse
    from collections.abc import Callable, Mapping
    from contextlib import AbstractContextManager
    from pathlib import Path

    from ayaka.configs.engine import EngineConfig

__all__ = [
    "ReplayConfig",
    "ReplayItemResult",
    "ReplayReport",
    "replay_command",
    "replay_lines",
    "replay_session",
]


@dataclass(frozen=True, slots=True)
class ReplayConfig:
    """Options controlling one session replay run."""

    session_path: Path
    output: Literal["text", "json"] = "text"


@dataclass(frozen=True, slots=True)
class ReplayItemResult:
    """Outcome of replaying one recorded request."""

    request_id: str
    prompt_tokens: int
    expected_tokens: tuple[int, ...]
    actual_tokens: tuple[int, ...]
    parity: bool
    cached_tokens: int
    ttft_ms: float | None
    latency_ms: float


@dataclass(frozen=True, slots=True)
class ReplayReport:
    """Aggregated token parity, prefix cache hit ratio, and latency metrics."""

    session_path: str
    total_requests: int
    parity_matched: int
    parity_mismatched: int
    total_prompt_tokens: int
    total_cached_tokens: int
    cache_hit_ratio: float
    items: tuple[ReplayItemResult, ...]

    @property
    def ok(self) -> bool:
        """Whether every replayed request matched the recorded tokens."""
        return self.parity_mismatched == 0

    def as_json(self) -> dict[str, object]:
        """Serialize the report to JSON-compatible primitives."""
        return {
            "session_path": self.session_path,
            "ok": self.ok,
            "total_requests": self.total_requests,
            "parity_matched": self.parity_matched,
            "parity_mismatched": self.parity_mismatched,
            "total_prompt_tokens": self.total_prompt_tokens,
            "total_cached_tokens": self.total_cached_tokens,
            "cache_hit_ratio": self.cache_hit_ratio,
            "items": [asdict(item) for item in self.items],
        }


def _sampling_from_record(raw: Mapping[str, object]) -> SamplingParams:
    stop_raw = raw.get("stop", ())
    stop = tuple(str(s) for s in cast("list[object] | tuple[object, ...]", stop_raw))
    stop_ids_raw = raw.get("stop_token_ids", ())
    stop_ids = frozenset(int(x) for x in cast("list[int] | tuple[int, ...]", stop_ids_raw))
    bias_raw = raw.get("logit_bias", {})
    logit_bias = (
        {int(k): float(v) for k, v in cast("dict[str, float]", bias_raw).items()}
        if isinstance(bias_raw, dict)
        else {}
    )
    max_tokens_raw = raw.get("max_tokens")
    top_k_raw = raw.get("top_k", -1)
    logprobs_raw = raw.get("logprobs")
    seed_raw = raw.get("seed")
    return SamplingParams(
        n=int(cast("int", raw.get("n", 1))),
        temperature=float(cast("float", raw.get("temperature", 0.0))),
        top_p=float(cast("float", raw.get("top_p", 1.0))),
        top_k=int(cast("int", top_k_raw)),
        min_p=float(cast("float", raw.get("min_p", 0.0))),
        presence_penalty=float(cast("float", raw.get("presence_penalty", 0.0))),
        frequency_penalty=float(cast("float", raw.get("frequency_penalty", 0.0))),
        repetition_penalty=float(cast("float", raw.get("repetition_penalty", 1.0))),
        max_tokens=int(cast("int", max_tokens_raw)) if max_tokens_raw is not None else None,
        min_tokens=int(cast("int", raw.get("min_tokens", 0))),
        ignore_eos=bool(raw.get("ignore_eos", False)),
        stop=stop,
        stop_token_ids=stop_ids,
        seed=int(cast("int", seed_raw)) if seed_raw is not None else None,
        logprobs=int(cast("int", logprobs_raw)) if logprobs_raw is not None else None,
        logit_bias=logit_bias,
    )


def _expected_tokens(record: RecordedRequest) -> tuple[int, ...]:
    for seq in record.outputs:
        if int(cast("int", seq.get("index", 0))) == 0:
            raw = cast("list[int]", seq.get("token_ids", []))
            return tuple(int(tok) for tok in raw)
    return ()


def replay_session(
    session_path: Path,
    engine_config: EngineConfig,
    *,
    engine_factory: Callable[[EngineConfig], AbstractContextManager[Any]] | None = None,
) -> ReplayReport:
    """Replay all requests from ``session_path`` and verify token parity and cache hits."""
    records = read_session(session_path)
    factory = engine_factory if engine_factory is not None else LLM.from_config
    items: list[ReplayItemResult] = []
    with factory(engine_config) as llm:
        for index, rec in enumerate(records):
            request = GenerateRequest(
                request_id=RequestId(f"replay:{index}:{rec.request_id}"),
                prompt=rec.prompt_token_ids,
                sampling=_sampling_from_record(rec.sampling),
            )
            started = time.perf_counter()
            first_token_ms: float | None = None
            actual: list[int] = []
            cached = 0
            for chunk in llm.stream(request):
                for seq in chunk.outputs:
                    if seq.index == 0 and seq.token_ids:
                        if first_token_ms is None:
                            first_token_ms = (time.perf_counter() - started) * 1000.0
                        actual.extend(seq.token_ids)
                if chunk.finished and chunk.usage is not None:
                    cached = chunk.usage.cached_tokens
            latency_ms = (time.perf_counter() - started) * 1000.0
            expected = _expected_tokens(rec)
            actual_tuple = tuple(actual)
            items.append(
                ReplayItemResult(
                    request_id=rec.request_id,
                    prompt_tokens=len(rec.prompt_token_ids),
                    expected_tokens=expected,
                    actual_tokens=actual_tuple,
                    parity=(actual_tuple == expected),
                    cached_tokens=cached,
                    ttft_ms=first_token_ms,
                    latency_ms=latency_ms,
                )
            )
    matched = sum(1 for item in items if item.parity)
    mismatched = len(items) - matched
    total_prompt = sum(item.prompt_tokens for item in items)
    total_cached = sum(item.cached_tokens for item in items)
    hit_ratio = (total_cached / total_prompt) if total_prompt else 0.0
    return ReplayReport(
        session_path=str(session_path),
        total_requests=len(items),
        parity_matched=matched,
        parity_mismatched=mismatched,
        total_prompt_tokens=total_prompt,
        total_cached_tokens=total_cached,
        cache_hit_ratio=hit_ratio,
        items=tuple(items),
    )


def replay_lines(report: ReplayReport) -> list[str]:
    """Format a human-readable summary of ``report``."""
    lines = [
        f"session: {report.session_path}",
        f"requests: {report.total_requests}, parity: {report.parity_matched}/"
        f"{report.total_requests}, cache hit: {report.total_cached_tokens}/"
        f"{report.total_prompt_tokens} ({report.cache_hit_ratio * 100:.1f}%)",
    ]
    for item in report.items:
        ttft = f"{item.ttft_ms:.1f}ms" if item.ttft_ms is not None else "n/a"
        status = "MATCH" if item.parity else "MISMATCH"
        lines.append(
            f"  {item.request_id}: [{status}] prompt={item.prompt_tokens} "
            f"cached={item.cached_tokens} ttft={ttft} total={item.latency_ms:.1f}ms"
        )
    return lines


def replay_command(args: argparse.Namespace, engine_config: EngineConfig) -> int:
    """Execute ``ayaka replay`` CLI command and return exit code."""
    report = replay_session(args.session, engine_config)
    if args.output == "json":
        print(json.dumps(report.as_json(), indent=2))
    else:
        for line in replay_lines(report):
            print(line)
    return EXIT_OK if report.ok else EXIT_ENGINE
