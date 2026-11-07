"""Bounded base-2 exponential histogram for engine metrics.

Positive bucket ``i`` has upper bound ``2 ** (i * 2 ** -schema)``; counts are
noncumulative and contiguous from ``offset``, and exact zeros are counted
separately. Resolution drops as needed to keep at most :data:`MAX_BUCKETS`
buckets, preserving every observation. The layout maps one-to-one onto
Prometheus native histograms.
"""

from __future__ import annotations

import math
from bisect import bisect_left
from dataclasses import dataclass, field

__all__ = ["INITIAL_SCHEMA", "MAX_BUCKETS", "MIN_SCHEMA", "Histogram"]

INITIAL_SCHEMA = 4
MIN_SCHEMA = -4
MAX_BUCKETS = 160

# Standard Prometheus schema-4 boundaries in [0.5, 1), the same constants the
# Rust engine uses. Coarser schemas are derived by downscaling, not recomputed.
_FRACTION_BOUNDS = (
    0.5,
    0.5221368912137069,
    0.5452538663326288,
    0.5693943173783458,
    0.5946035575013605,
    0.620928906036742,
    0.6484197773255048,
    0.6771277734684463,
    0.7071067811865475,
    0.7384130729697496,
    0.7711054127039704,
    0.805245165974627,
    0.8408964152537144,
    0.8781260801866495,
    0.9170040432046711,
    0.9576032806985735,
)


@dataclass(slots=True)
class Histogram:
    """Native-histogram state for one metric; cloning yields a snapshot.

    Field names match the Rust engine so snapshots can cross a JSON boundary
    unchanged. Build one per metric, then merge shards when they are combined.
    Instances are mutable and unsynchronized; guard a shared one externally.
    """

    schema: int = INITIAL_SCHEMA
    zero_count: int = 0
    offset: int = 0
    counts: list[int] = field(default_factory=list[int])
    count: int = 0
    sum: float = 0.0

    def validate(self) -> None:
        """Validate an externally supplied snapshot; raise ``ValueError`` if invalid."""
        if not MIN_SCHEMA <= self.schema <= INITIAL_SCHEMA:
            raise ValueError("histogram schema must be in -4..=4")
        if not math.isfinite(self.sum) or self.sum < 0.0:
            raise ValueError("histogram sum must be finite and nonnegative")
        if self.zero_count < 0 or self.count < 0 or any(count < 0 for count in self.counts):
            raise ValueError("histogram counts must be nonnegative")
        if len(self.counts) > MAX_BUCKETS:
            raise ValueError("histogram exceeds 160 buckets")
        if self.zero_count + sum(self.counts) != self.count:
            raise ValueError("histogram count must equal zero_count plus bucket counts")
        scale = 2.0**self.schema
        minimum = math.ceil(-1074.0 * scale)
        maximum = math.ceil(1024.0 * scale)
        if self.counts:
            last = self.offset + len(self.counts) - 1
            if self.offset < minimum or last > maximum:
                raise ValueError("histogram bucket indices are outside the finite positive range")
        elif self.offset != 0:
            raise ValueError("histogram bucket indices are outside the finite positive range")

    def observe(self, value: float) -> None:
        """Record one observation; ``value`` must be finite and nonnegative."""
        self.observe_weighted(value, 1)

    def observe_weighted(self, value: float, weight: int) -> None:
        """Record ``weight`` equal observations without looping over them."""
        if not math.isfinite(value) or value < 0.0:
            raise ValueError("histogram observations must be finite and nonnegative")
        if weight < 0:
            raise ValueError("histogram weight must be nonnegative")
        if weight == 0:
            return
        if value == 0.0:
            self.zero_count += weight
        else:
            index = _bucket_index(value, self.schema)
            while not self._fits(index, index):
                self._downscale()
                index = _downscaled_index(index, 1)
            self._extend(index, index)
            self.counts[index - self.offset] += weight
        self.count += weight
        self.sum += value * weight

    def merge(self, other: Histogram) -> None:
        """Merge ``other``, lowering this histogram's resolution when needed."""
        if other is self:
            raise ValueError("cannot merge a histogram into itself")
        while self.schema > other.schema:
            self._downscale()
        if other.counts:
            while True:
                shift = other.schema - self.schema
                first = _downscaled_index(other.offset, shift)
                last = _downscaled_index(other._last_index(), shift)
                if self._fits(first, last):
                    self._extend(first, last)
                    for position, count in enumerate(other.counts):
                        index = _downscaled_index(other.offset + position, shift)
                        self.counts[index - self.offset] += count
                    break
                self._downscale()
        self.zero_count += other.zero_count
        self.count += other.count
        self.sum += other.sum

    def clone(self) -> Histogram:
        """Return an independent copy; mutating it leaves this histogram unchanged."""
        return Histogram(
            schema=self.schema,
            zero_count=self.zero_count,
            offset=self.offset,
            counts=list(self.counts),
            count=self.count,
            sum=self.sum,
        )

    def _last_index(self) -> int:
        return self.offset + len(self.counts) - 1

    def _fits(self, first: int, last: int) -> bool:
        if self.counts:
            first = min(first, self.offset)
            last = max(last, self._last_index())
        return last - first < MAX_BUCKETS

    def _extend(self, first: int, last: int) -> None:
        if not self.counts:
            self.offset = first
            self.counts = [0] * (last - first + 1)
            return
        if first < self.offset:
            self.counts[:0] = [0] * (self.offset - first)
            self.offset = first
        missing = last - self._last_index()
        if missing > 0:
            self.counts.extend([0] * missing)

    def _downscale(self) -> None:
        if self.schema <= MIN_SCHEMA:
            raise ValueError("histogram cannot downscale below schema -4")
        self.schema -= 1
        if not self.counts:
            return
        first = _downscaled_index(self.offset, 1)
        last = _downscaled_index(self._last_index(), 1)
        counts = [0] * (last - first + 1)
        for position, count in enumerate(self.counts):
            index = _downscaled_index(self.offset + position, 1)
            counts[index - first] += count
        self.offset = first
        self.counts = counts


def _downscaled_index(index: int, shift: int) -> int:
    """Return ``ceil(index / 2**shift)``, including negative bucket indexes."""
    return (index + (1 << shift) - 1) >> shift


def _bucket_index(value: float, schema: int) -> int:
    # frexp is exact for every finite double, including subnormals:
    # value == fraction * 2**exponent with 0.5 <= fraction < 1. That is the
    # same decomposition the Rust engine builds from the IEEE bit pattern.
    fraction, exponent = math.frexp(value)
    bucket = bisect_left(_FRACTION_BOUNDS, fraction)
    return _downscaled_index(bucket + (exponent - 1) * 16, INITIAL_SCHEMA - schema)
