"""Unit tests for FixedController and CostAwareController."""

from __future__ import annotations

import pytest

from ayaka.speculative.policy import CostAwareController, FixedController, match_bucket
from ayaka.speculative.proposer import Proposal

pytestmark = pytest.mark.unit


def test_fixed_controller_caps_and_confidence_cut() -> None:
    ctrl = FixedController(fixed_k=4, confidence_threshold=0.5)
    prop = Proposal(tokens=(1, 2, 3, 4, 5), probs=(0.9, 0.8, 0.5, 0.9, 0.9))
    # 0.9 * 0.8 = 0.72 >= 0.5; * 0.5 = 0.36 < 0.5 -> cut to 2
    assert ctrl.choose_k(prop, 8, batch_size=1, context_len=100) == 2

    ctrl.max_allowed_rows = 2
    assert ctrl.choose_k(prop, 8, batch_size=1, context_len=100) == 1


def test_cost_aware_controller_disables_when_costly_and_enables_when_cheap() -> None:
    ctrl = CostAwareController(max_k=4, min_gain=0.05, ema=0.2)
    prop = Proposal(tokens=(1, 2, 3, 4), confidence=10, source="lookup")

    # Simulate linear-cost step (verify adds 100% per extra row) and 0% acceptance -> k=0.
    for _ in range(30):
        ctrl.observe(
            source="lookup",
            confidence=10,
            k=0,
            accepted=0,
            batch_size=1,
            context_len=512,
            round_seconds=0.010,
        )
        ctrl.observe(
            source="lookup",
            confidence=10,
            k=2,
            accepted=0,
            batch_size=1,
            context_len=512,
            round_seconds=0.030,
        )
    assert ctrl.choose_k(prop, 4, batch_size=1, context_len=512) == 0

    # Now simulate nearly-free verify rows and high acceptance on bucket 16+ -> chooses k > 0.
    high_prop = Proposal(tokens=(1, 2, 3, 4), confidence=20, source="lookup")
    for _ in range(25):
        ctrl.observe(
            source="lookup",
            confidence=20,
            k=4,
            accepted=4,
            batch_size=1,
            context_len=512,
            round_seconds=0.0105,
        )
    k = ctrl.choose_k(high_prop, 4, batch_size=1, context_len=512)
    assert k > 0
    assert match_bucket("lookup", 20) == "lookup:16+"
