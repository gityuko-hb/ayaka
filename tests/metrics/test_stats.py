from __future__ import annotations

from dataclasses import replace

import pytest

from ayaka.metrics import IterationStats

pytestmark = pytest.mark.unit


def test_defaults_are_zero_except_for_the_step_id() -> None:
    stats = IterationStats(step_id=-1)

    assert stats.num_running == 0
    assert stats.num_finished == 0
    assert stats.step_seconds == 0.0


def test_replace_returns_a_new_snapshot() -> None:
    stats = IterationStats(step_id=1)
    bumped = replace(stats, num_settled=2)

    assert bumped.num_settled == 2
    assert stats.num_settled == 0
