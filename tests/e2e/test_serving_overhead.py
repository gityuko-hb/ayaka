"""Schema smoke test for the serving-overhead benchmark harness on the fake engine."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest

torch: Any = pytest.importorskip("torch")
pytest.importorskip("aiohttp")

if TYPE_CHECKING:
    from types import ModuleType

pytestmark = pytest.mark.integration


def _load_overhead() -> ModuleType:
    path = Path(__file__).resolve().parents[2] / "benchmarks" / "e2e" / "serving_overhead.py"
    spec = importlib.util.spec_from_file_location("serving_overhead_bench", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_fake_serving_overhead_harness_writes_expected_schema(tmp_path: Path) -> None:
    mod = _load_overhead()
    out = tmp_path / "overhead.json"
    code = mod.main(
        [
            "--mode",
            "fake",
            "--streams",
            "2",
            "--tokens",
            "8",
            "--rounds",
            "1",
            "--step-ms",
            "0.2",
            "--burst",
            "--burst-requests",
            "3",
            "--burst-prompt-chars",
            "256",
            "--out",
            str(out),
        ]
    )

    assert code == 0
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["schema"] == "ayaka.serving-overhead/1"
    assert payload["mode"] == "fake"
    assert len(payload["rounds"]) == 1
    assert payload["rounds"][0]["frames"] >= 2
    assert payload["rounds"][0]["step_gap_ms"]["count"] >= 1
    assert payload["steady_summary"]["rounds"] == 1
    assert payload["steady_summary"]["server_cpu_per_frame_us_median"] is not None
    assert payload["burst_summary"]["rounds"] == 1
    assert len(payload["burst_rounds"]) == 1
