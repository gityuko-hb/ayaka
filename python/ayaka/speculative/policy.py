"""Draft-length controllers: fixed window and cost-aware throughput maximization.

The cost-aware controller maximizes ``E[tokens committed] / T(rows)`` over ``k in 0..max_k``,
using running EMAs of conditional per-position acceptance ``a_i`` (bucketed by draft source and
match length) and outlier-resistant EMAs of round wall time ``T(rows, context_bucket)``.
It keeps ``k = 0`` unless the best drafting choice beats ``1 / T(base_rows)`` by ``min_gain``, and
periodically probes at exponential backoff intervals so estimates never permanently freeze.

This module imports no torch.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from ayaka.speculative.proposer import Proposal

__all__ = ["CostAwareController", "DraftController", "FixedController", "match_bucket"]


class DraftController(Protocol):
    """Decides how many draft tokens ``k`` to request for one request on the next round."""

    max_allowed_rows: int | None

    def choose_k(
        self,
        proposal: Proposal,
        max_k: int,
        *,
        batch_size: int,
        context_len: int,
    ) -> int:
        """Choose ``k`` in ``[0, min(len(proposal.tokens), max_k)]``."""
        ...

    def observe(
        self,
        *,
        source: str,
        confidence: int,
        k: int,
        accepted: int,
        batch_size: int,
        context_len: int,
        round_seconds: float,
    ) -> None:
        """Update acceptance and timing models from a completed step."""
        ...


def match_bucket(source: str, confidence: int) -> str:
    """Group lookup proposals into Strata-style match-length buckets, or name the draft source."""
    if source in ("mtp", "prediction", "token_recycling"):
        return source
    if confidence >= 16:
        return "lookup:16+"
    if confidence >= 8:
        return "lookup:8-15"
    if confidence >= 5:
        return "lookup:5-7"
    return "lookup:3-4"


def _context_bucket(context_len: int) -> int:
    if context_len < 1024:
        return 0
    if context_len < 4096:
        return 1
    if context_len < 16384:
        return 2
    return 3


_DEFAULT_BUCKET_ACC: dict[str, float] = {
    "lookup:3-4": 0.35,
    "lookup:5-7": 0.55,
    "lookup:8-15": 0.75,
    "lookup:16+": 0.85,
    "prediction": 0.75,
    "mtp": 0.80,
    "token_recycling": 0.35,
}


@dataclass(slots=True)
class _ControllerStats:
    rounds: int = 0
    total_drafts: int = 0
    total_accepted: int = 0

    def record(self, k: int, accepted: int) -> None:
        if k > 0:
            self.rounds += 1
            self.total_drafts += k
            self.total_accepted += accepted

    def snapshot(self) -> dict[str, int]:
        return {
            "rounds": self.rounds,
            "total_drafts": self.total_drafts,
            "total_accepted": self.total_accepted,
        }


class FixedController:
    """Always drafts ``min(fixed_k, max_k, len(proposal.tokens))`` (honoring confidence cut)."""

    def __init__(self, fixed_k: int, *, confidence_threshold: float = 0.0) -> None:
        self._fixed_k = fixed_k
        self._confidence_threshold = confidence_threshold
        self.max_allowed_rows: int | None = None
        self._stats = _ControllerStats()

    def choose_k(
        self,
        proposal: Proposal,
        max_k: int,
        *,
        batch_size: int,
        context_len: int,
    ) -> int:
        del context_len
        usable = _usable_proposal_len(proposal, max_k, self._confidence_threshold)
        if self.max_allowed_rows is not None:
            room = max(0, self.max_allowed_rows - batch_size)
            usable = min(usable, room)
        return min(usable, self._fixed_k)

    def observe(
        self,
        *,
        source: str,
        confidence: int,
        k: int,
        accepted: int,
        batch_size: int,
        context_len: int,
        round_seconds: float,
    ) -> None:
        del source, confidence, batch_size, context_len, round_seconds
        self._stats.record(k, accepted)

    def snapshot(self) -> dict[str, int]:
        """Return accumulated round, draft, and accepted token counters."""
        return self._stats.snapshot()


@dataclass(slots=True)
class _AcceptanceTracker:
    alpha: float
    prior: float
    max_k: int
    rates: list[float] = field(init=False)

    def __post_init__(self) -> None:
        # Decay slightly across later draft positions as an initial prior.
        self.rates = [max(0.05, self.prior * (0.94**i)) for i in range(self.max_k)]

    def expected_tokens(self, k: int) -> float:
        expected = 1.0
        prob = 1.0
        for pos in range(min(k, len(self.rates))):
            prob *= self.rates[pos]
            expected += prob
        return expected

    def observe(self, k: int, accepted: int) -> None:
        alpha = self.alpha
        # Positions 0 .. accepted-1 succeeded; position `accepted` (if < k) failed.
        limit = min(k, len(self.rates))
        for pos in range(limit):
            if pos < accepted:
                self.rates[pos] = (1.0 - alpha) * self.rates[pos] + alpha * 1.0
            elif pos == accepted:
                self.rates[pos] = (1.0 - alpha) * self.rates[pos]
                break


class CostAwareController:
    """Selects ``k`` maximizing ``E[tokens committed] / T(rows)`` with ``k=0`` fallback."""

    def __init__(
        self,
        *,
        max_k: int = 8,
        min_gain: float = 0.05,
        ema: float = 0.05,
        confidence_threshold: float = 0.0,
    ) -> None:
        self._max_k = max_k
        self._min_gain = min_gain
        self._ema = ema
        self._confidence_threshold = confidence_threshold
        self.max_allowed_rows: int | None = None
        self._acceptance: dict[str, _AcceptanceTracker] = {}
        # (total_rows, context_bucket) -> EMA round time in seconds
        self._step_time: dict[tuple[int, int], float] = {}
        self._step_samples: dict[tuple[int, int], int] = {}
        self._zeros_in_row: dict[str, int] = {}
        self._probe_threshold: dict[str, int] = {}
        self._stats = _ControllerStats()

    def _tracker(self, bucket: str) -> _AcceptanceTracker:
        tracker = self._acceptance.get(bucket)
        if tracker is None:
            prior = _DEFAULT_BUCKET_ACC.get(bucket, 0.4)
            tracker = _AcceptanceTracker(self._ema, prior, self._max_k)
            self._acceptance[bucket] = tracker
        return tracker

    def _estimate_time(self, rows: int, base_rows: int, ctx_bucket: int) -> float:
        exact = self._step_time.get((rows, ctx_bucket))
        if exact is not None:
            return exact
        base = self._step_time.get((base_rows, ctx_bucket))
        if base is None:
            # No timing sample yet: assume verify rows add 2% per extra row on top of 1.0 unit.
            return 1.0 + 0.02 * max(0, rows - base_rows)
        # Interpolate/extrapolate smoothly from observed points in the same context bucket.
        for delta in range(1, rows + 1):
            if (rows - delta, ctx_bucket) in self._step_time:
                prev = self._step_time[rows - delta, ctx_bucket]
                slope = (
                    max(0.0, prev - base) / max(1, (rows - delta) - base_rows)
                    if rows - delta > base_rows
                    else 0.02 * base
                )
                return prev + slope * delta
        return base * (1.0 + 0.02 * max(0, rows - base_rows))

    def choose_k(
        self,
        proposal: Proposal,
        max_k: int,
        *,
        batch_size: int,
        context_len: int,
    ) -> int:
        usable = _usable_proposal_len(proposal, min(max_k, self._max_k), self._confidence_threshold)
        if self.max_allowed_rows is not None:
            usable = min(usable, max(0, self.max_allowed_rows - batch_size))
        if usable <= 0:
            return 0

        bucket = match_bucket(proposal.source, proposal.confidence)
        tracker = self._tracker(bucket)
        ctx = _context_bucket(context_len)
        base_rows = max(1, batch_size)
        t_base = self._estimate_time(base_rows, base_rows, ctx)
        if t_base <= 0:
            return usable
        base_rate = 1.0 / t_base
        target_rate = base_rate * (1.0 + self._min_gain)

        best_k = 0
        best_rate = target_rate
        for k in range(1, usable + 1):
            t_k = self._estimate_time(base_rows + k, base_rows, ctx)
            if t_k <= 0:
                continue
            rate = tracker.expected_tokens(k) / t_k
            if rate > best_rate:
                best_rate = rate
                best_k = k

        if best_k > 0:
            self._zeros_in_row[bucket] = 0
            return best_k

        # Exponential-backoff probing when cost model suppresses drafting, so changing workload
        # phases or noisy initial timings can recover.
        zeros = self._zeros_in_row.get(bucket, 0) + 1
        threshold = self._probe_threshold.get(bucket, 8)
        if zeros >= threshold:
            self._zeros_in_row[bucket] = 0
            self._probe_threshold[bucket] = min(128, threshold * 2)
            return min(2, usable)
        self._zeros_in_row[bucket] = zeros
        return 0

    def observe(
        self,
        *,
        source: str,
        confidence: int,
        k: int,
        accepted: int,
        batch_size: int,
        context_len: int,
        round_seconds: float,
    ) -> None:
        ctx = _context_bucket(context_len)
        rows = max(1, batch_size) + max(0, k)
        if round_seconds > 0:
            key = (rows, ctx)
            samples = self._step_samples.get(key, 0)
            prev = self._step_time.get(key)
            if prev is None or samples < 3:
                self._step_time[key] = (
                    round_seconds if prev is None else 0.5 * (prev + round_seconds)
                )
            else:
                # Outlier-resistant clamp: ignore extreme GC/scheduling spikes beyond [0.5x, 2x].
                clamped = min(max(round_seconds, 0.5 * prev), 2.0 * prev)
                self._step_time[key] = (1.0 - self._ema) * prev + self._ema * clamped
            self._step_samples[key] = samples + 1

        if k > 0:
            self._stats.record(k, accepted)
            bucket = match_bucket(source, confidence)
            tracker = self._tracker(bucket)
            tracker.observe(k, accepted)
            if accepted > 0:
                self._probe_threshold[bucket] = 8

    def snapshot(self) -> dict[str, int]:
        """Return accumulated round, draft, and accepted token counters."""
        return self._stats.snapshot()

    def acceptance_rates(self, bucket: str) -> tuple[float, ...]:
        """Current EMA acceptance estimates for ``bucket``, for diagnostics and tests."""
        tracker = self._acceptance.get(bucket)
        return tuple(tracker.rates) if tracker is not None else ()


def _usable_proposal_len(proposal: Proposal, max_k: int, confidence_threshold: float) -> int:
    usable = min(len(proposal.tokens), max_k)
    if usable <= 0:
        return 0
    if confidence_threshold > 0.0 and proposal.probs:
        prod = 1.0
        cut = 0
        for idx in range(min(usable, len(proposal.probs))):
            prod *= proposal.probs[idx]
            if prod < confidence_threshold:
                break
            cut += 1
        usable = cut
    return usable
