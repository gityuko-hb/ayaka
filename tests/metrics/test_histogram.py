from __future__ import annotations

import math
import sys

import pytest

from ayaka.metrics.histogram import (
    _FRACTION_BOUNDS,
    INITIAL_SCHEMA,
    MAX_BUCKETS,
    MIN_SCHEMA,
    Histogram,
    _bucket_index,
)

pytestmark = pytest.mark.unit


def test_merging_different_resolutions_matches_observing_all_values() -> None:
    groups = (
        ((0.0, 3), (2.0**-80, 2), (0.25, 7)),
        ((1.0, 2), (1.01, 9), (1.25, 4), (2.0**80, 3)),
    )
    combined = Histogram()
    merged = Histogram()
    for values in groups:
        shard = Histogram()
        for value, weight in values:
            shard.observe_weighted(value, weight)
            combined.observe_weighted(value, weight)
        merged.merge(shard)

    assert merged.schema < INITIAL_SCHEMA
    assert merged == combined
    merged.validate()
    assert merged.count == 30
    assert merged.zero_count + sum(merged.counts) == 30


def _values_around_boundaries() -> list[float]:
    values = [0.0, math.ldexp(1.0, -1074), 1.0, sys.float_info.max]
    for exponent in (-100, -1, 0, 1, 100):
        for fraction in _FRACTION_BOUNDS:
            value = math.ldexp(fraction, exponent)
            values.extend([math.nextafter(value, 0.0), value, math.nextafter(value, math.inf)])
    return values


def test_boundary_values_keep_every_observation_within_the_bucket_limit() -> None:
    values = _values_around_boundaries()
    histogram = Histogram()
    for value in values:
        histogram.observe(value)
        assert len(histogram.counts) <= MAX_BUCKETS

    assert histogram.zero_count == 1
    assert histogram.zero_count + sum(histogram.counts) == len(values)
    assert histogram.schema == MIN_SCHEMA
    histogram.validate()


@pytest.mark.parametrize(
    ("value", "schema", "expected"),
    [
        (1.0, INITIAL_SCHEMA, 0),
        (0.75, INITIAL_SCHEMA, -6),
        (math.ldexp(1.0, -1074), INITIAL_SCHEMA, -17184),
        (math.ldexp(1.0, -1074), MIN_SCHEMA, -67),
        (sys.float_info.max, INITIAL_SCHEMA, 16384),
        (sys.float_info.max, MIN_SCHEMA, 64),
    ],
)
def test_bucket_index_matches_the_rust_reference(value: float, schema: int, expected: int) -> None:
    assert _bucket_index(value, schema) == expected


def test_zero_observations_are_counted_separately() -> None:
    histogram = Histogram()
    histogram.observe(0.0)
    histogram.observe(1.0)

    assert histogram.zero_count == 1
    assert histogram.counts == [1]
    assert histogram.count == 2
    assert histogram.sum == 1.0


def test_zero_weight_is_a_noop() -> None:
    histogram = Histogram()
    histogram.observe_weighted(2.5, 0)

    assert histogram == Histogram()


@pytest.mark.parametrize("value", [-1.0, math.nan, math.inf])
def test_observe_rejects_invalid_values(value: float) -> None:
    with pytest.raises(ValueError, match="observations must be finite and nonnegative"):
        Histogram().observe(value)


def test_observe_rejects_a_negative_weight() -> None:
    with pytest.raises(ValueError, match="weight must be nonnegative"):
        Histogram().observe_weighted(1.0, -1)


def test_merge_into_an_empty_histogram_copies_the_buckets() -> None:
    other = Histogram()
    other.observe(123.0)
    merged = Histogram()
    merged.merge(other)

    assert merged == other
    assert merged is not other


def test_merge_keeps_the_coarser_schema() -> None:
    coarse = Histogram(schema=0)
    coarse.observe(1.0)
    fine = Histogram()
    fine.observe(2.0)

    coarse.merge(fine)

    assert coarse.schema == 0
    assert coarse.count == 2


def test_merge_rejects_self() -> None:
    histogram = Histogram()
    with pytest.raises(ValueError, match="cannot merge a histogram into itself"):
        histogram.merge(histogram)


def test_clone_is_independent() -> None:
    original = Histogram()
    original.observe(1.0)
    copy = original.clone()
    copy.observe(2.0)

    assert original.count == 1
    assert copy.count == 2


def test_validate_accepts_observed_snapshots() -> None:
    histogram = Histogram()
    for value in (0.0, 1e-300, 0.5, 1.0, 2.0, 1e300):
        histogram.observe(value)

    histogram.validate()


@pytest.mark.parametrize(
    ("histogram", "message"),
    [
        (Histogram(schema=INITIAL_SCHEMA + 1), "schema must be in"),
        (Histogram(schema=MIN_SCHEMA - 1), "schema must be in"),
        (Histogram(sum=-1.0), "sum must be finite and nonnegative"),
        (Histogram(sum=math.nan), "sum must be finite and nonnegative"),
        (Histogram(sum=math.inf), "sum must be finite and nonnegative"),
        (Histogram(count=1), "count must equal zero_count"),
        (Histogram(count=-1), "counts must be nonnegative"),
        (Histogram(counts=[1], count=2), "count must equal zero_count"),
        (Histogram(counts=[-1], count=-1), "counts must be nonnegative"),
        (Histogram(offset=1), "bucket indices are outside"),
        (Histogram(offset=-17185, counts=[1], count=1), "bucket indices are outside"),
        (Histogram(offset=16385, counts=[1], count=1), "bucket indices are outside"),
        (
            Histogram(counts=[1] * (MAX_BUCKETS + 1), count=MAX_BUCKETS + 1),
            "exceeds 160 buckets",
        ),
    ],
)
def test_validate_rejects_malformed_snapshots(histogram: Histogram, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        histogram.validate()
