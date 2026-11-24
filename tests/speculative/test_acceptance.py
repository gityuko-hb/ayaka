"""Gate 1.4 & 1.5 statistical and unit tests for one-point acceptance."""

from __future__ import annotations

import math

import numpy as np
import pytest

from ayaka.speculative.acceptance import accept_one_point, accept_one_point_batch, sanitize_drafts

pytestmark = pytest.mark.unit


def test_accept_one_point_basic_prefix_and_rejections() -> None:
    res = accept_one_point([10, 20, 30, 40], [10, 20, 99])
    assert res.accepted == 2
    assert res.rejected == 1
    assert res.tokens == (10, 20, 30)

    res_all = accept_one_point([10, 20, 30], [10, 20])
    assert res_all.accepted == 2
    assert res_all.rejected == 0
    assert res_all.tokens == (10, 20, 30)

    res_none = accept_one_point([5, 20], [9])
    assert res_none.accepted == 0
    assert res_none.rejected == 1
    assert res_none.tokens == (5,)


def test_accept_one_point_rejects_invalid_and_oov_drafts() -> None:
    clean, valid = sanitize_drafts([5, 9999, -1, 3], vocab_size=100, fallback=0)
    assert clean == (5, 0, 0, 3)
    assert valid == (True, False, False, True)

    res = accept_one_point([5, 0, 0, 3, 4], clean, valid=valid, vocab_size=100)
    assert res.accepted == 1
    assert res.tokens == (5, 0)

    res_direct = accept_one_point([5, 9999, 3], [5, 9999], vocab_size=100)
    assert res_direct.accepted == 1
    assert res_direct.tokens == (5, 9999)


def test_accept_one_point_batch() -> None:
    out = accept_one_point_batch(
        [1, 2, 3, 7, 8],
        [3, 2],
        [[1, 9], [7]],
    )
    assert [r.tokens for r in out] == [(1, 2), (7, 8)]
    assert [r.accepted for r in out] == [1, 1]


@pytest.mark.parametrize(
    ("probs", "draft"),
    [
        ((0.5, 0.5), 0),
        ((0.875, 0.125), 1),
        ((0.1, 0.2, 0.7), 0),
        ((0.1, 0.2, 0.7), 2),
        ((0.05, 0.15, 0.30, 0.50), 1),
        ((0.25, 0.25, 0.25, 0.25), 3),
    ],
)
def test_one_point_rule_exact_distribution_chi_square(probs: tuple[float, ...], draft: int) -> None:
    """Gate 1.4: six (p, d) cases over 2e6 draws match P(accept)=p(d) and Y~p (chi2 p>0.001)."""
    n_draws = 2_000_000
    rng = np.random.default_rng(12345 + draft + int(probs[0] * 1000))
    p_arr = np.asarray(probs, dtype=np.float64)
    cdf = np.cumsum(p_arr)
    u = rng.random(n_draws)
    drawn = np.searchsorted(cdf, u, side="right").clip(0, len(probs) - 1)

    # One-point acceptance: accept iff drawn == draft; emit drawn in both accept & reject cases.
    accept_rate = float(np.mean(drawn == draft))
    expected_acc = probs[draft]
    se_acc = math.sqrt(expected_acc * (1.0 - expected_acc) / n_draws)
    assert abs(accept_rate - expected_acc) < 4.5 * se_acc

    counts = np.bincount(drawn, minlength=len(probs))
    expected = p_arr * n_draws
    chi2 = float(np.sum((counts - expected) ** 2 / expected))
    # df = len(probs) - 1 (between 1 and 3); critical chi2 at alpha=0.001 is 10.83 (df=1),
    # 13.82 (df=2), 16.27 (df=3).
    critical = {1: 10.828, 2: 13.816, 3: 16.266}[len(probs) - 1]
    assert chi2 < critical, f"chi2={chi2:.3f} >= {critical} for probs={probs}, draft={draft}"


def test_lookup_point_mass_accepted_at_target_probability_not_always_rejected() -> None:
    """Gate 1.5: a lookup draft with no candidate list is accepted with frequency p(d) at T>0."""
    n_draws = 200_000
    p_target = np.asarray([0.15, 0.55, 0.30], dtype=np.float64)
    draft = 1
    rng = np.random.default_rng(99)
    cdf = np.cumsum(p_target)
    drawn = np.searchsorted(cdf, rng.random(n_draws), side="right").clip(0, 2)
    accepted = int(np.sum(drawn == draft))
    rate = accepted / n_draws
    se = math.sqrt(p_target[draft] * (1.0 - p_target[draft]) / n_draws)
    # Two-sided z-test at alpha = 0.001 (z = 3.291)
    assert abs(rate - p_target[draft]) < 3.291 * se
