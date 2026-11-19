"""Unit tests for the slow-client benchmark schema, ratio math and cancel-delay calculation."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest

if TYPE_CHECKING:
    from types import ModuleType

pytestmark = pytest.mark.unit


def _load_slow_client() -> ModuleType:
    path = Path(__file__).resolve().parents[2] / "benchmarks" / "e2e" / "slow_client.py"
    spec = importlib.util.spec_from_file_location("slow_client_bench", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_build_result_computes_stuck_and_control_ratios_and_cancel_delay() -> None:
    mod = _load_slow_client()
    rows = [
        {"pair": 0, "kind": "alone", "stuck": False, "median_s": 0.010, "gaps_s": [0.010]},
        {
            "pair": 0,
            "kind": "stuck",
            "stuck": True,
            "median_s": 0.012,
            "gaps_s": [0.012],
            "cancel_delay_seconds": 11.4,
            "blocked_during_readers": True,
        },
        {
            "pair": 0,
            "kind": "stuck",
            "stuck": True,
            "median_s": 0.012,
            "gaps_s": [0.012],
            "cancel_delay_seconds": 12.2,
            "blocked_during_readers": True,
        },
        {"pair": 0, "kind": "alone", "stuck": False, "median_s": 0.010, "gaps_s": [0.010]},
        {"pair": 0, "kind": "control", "stuck": False, "median_s": 0.011, "gaps_s": [0.011]},
    ]

    result = mod.build_result(
        rows=rows,
        pairs=1,
        threshold=1.25,
        settings={"fast": 2, "pairs": 1, "control_reader": True},
        machine_info={"os": "test"},
        power=("ac", "ac"),
        control_reader=True,
    )

    assert result["schema"] == "ayaka.slow-client/1"
    assert result["ratios"] == [pytest.approx(1.2)]
    assert result["median_ratio"] == pytest.approx(1.2)
    assert result["control_ratios"] == [pytest.approx(1.1)]
    assert result["median_control_ratio"] == pytest.approx(1.1)
    assert result["cancel_delays_s"] == [11.4, 12.2]
    assert result["median_cancel_delay_seconds"] == pytest.approx(11.8)
    assert result["valid"] is True
    assert result["passed"] is True


def test_build_result_marks_unblocked_stuck_phase_invalid() -> None:
    mod = _load_slow_client()
    rows = [
        {"pair": 0, "kind": "alone", "stuck": False, "median_s": 0.010, "gaps_s": [0.010]},
        {
            "pair": 0,
            "kind": "stuck",
            "stuck": True,
            "median_s": 0.010,
            "gaps_s": [0.010],
            "cancel_delay_seconds": None,
            "blocked_during_readers": False,
        },
    ]

    result = mod.build_result(
        rows=rows,
        pairs=1,
        threshold=1.25,
        settings={"fast": 2, "pairs": 1, "control_reader": False},
        machine_info={"os": "test"},
        power=("ac", "ac"),
    )

    assert result["valid"] is False
    assert result["passed"] is False
    assert result["median_cancel_delay_seconds"] is None
    assert "control_ratios" not in result


def test_main_runs_with_control_reader_and_writes_schema(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    mod = _load_slow_client()
    calls: list[tuple[int, bool]] = []

    def fake_phase(
        url: str,
        model: str,
        fast: int,
        tokens: int,
        key: str | None,
        *,
        stuck: bool,
        settle: float,
        server_timeout: float,
    ) -> Any:
        del url, model, tokens, key, settle, server_timeout
        calls.append((fast, stuck))
        if stuck:
            return mod.Phase(
                [0.012, 0.012], readers_started=5.0, readers_ended=7.0, cancelled_at=15.0
            )
        return mod.Phase([0.010, 0.010], readers_started=0.0, readers_ended=2.0)

    monkeypatch.setattr(mod, "phase", fake_phase)
    monkeypatch.setattr("ayaka.entrypoints.bench_machine.power_source", lambda: {"source": "test"})
    out = tmp_path / "slow.json"
    code = mod.main(
        [
            "--pairs",
            "1",
            "--fast",
            "2",
            "--settle",
            "0",
            "--server-timeout",
            "10",
            "--threshold",
            "1.25",
            "--control-reader",
            "--out",
            str(out),
        ]
    )

    assert code == 0
    # A B B A alone/stuck phases plus the third control reader phase (fast + 1 = 3).
    assert calls == [(2, False), (2, True), (2, True), (2, False), (3, False)]
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["schema"] == "ayaka.slow-client/1"
    assert payload["median_cancel_delay_seconds"] == pytest.approx(15.0)
    assert payload["median_control_ratio"] == pytest.approx(1.0)
