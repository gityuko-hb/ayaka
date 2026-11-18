"""Opt-in diagnostics for a long GPU run: device memory around every test, as JSON lines.

Disabled unless ``AYAKA_GPU_DIAG`` names a file, so a normal run never pays for it. Enabled, it
appends one line per test start, per test end and per outcome, each with the device's free, total,
allocated and reserved bytes. The ledger check uses the allocator's reserved bytes; these lines
still show whether another process moved the driver's free memory, and the timestamps line up with
a sampled ``nvidia-smi``.

Run it as::

    AYAKA_GPU_DIAG=/tmp/gpu-diag.jsonl pytest -m gpu -p gpu_diagnostics

``tests/_support`` is on pytest's ``pythonpath``, which is how ``-p gpu_diagnostics`` finds this
module; a run from another directory also needs it on ``PYTHONPATH``.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import TYPE_CHECKING, Final

if TYPE_CHECKING:
    import pytest

PATH_VARIABLE: Final = "AYAKA_GPU_DIAG"

_PATH: Path | None = None


def _memory() -> dict[str, int] | None:
    """What torch reports for the current device, or ``None`` without one."""
    try:
        import torch
    except ImportError:
        return None
    if not torch.cuda.is_available():
        return None
    free, total = torch.cuda.mem_get_info()
    return {
        "free": int(free),
        "total": int(total),
        "allocated": int(torch.cuda.memory_allocated()),
        "reserved": int(torch.cuda.memory_reserved()),
    }


def _record(nodeid: str, phase: str, **extra: object) -> None:
    """Append one JSON line, or do nothing when the diagnostics are off."""
    if _PATH is None:
        return
    record: dict[str, object] = {"nodeid": nodeid, "phase": phase, "time": time.time()}
    record.update(extra)
    memory = _memory()
    if memory is not None:
        record.update(memory)
    with _PATH.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record) + "\n")


def pytest_configure(config: pytest.Config) -> None:
    del config
    raw = os.environ.get(PATH_VARIABLE)
    if not raw:
        return
    global _PATH
    _PATH = Path(raw).expanduser()
    _PATH.parent.mkdir(parents=True, exist_ok=True)


def pytest_runtest_setup(item: pytest.Item) -> None:
    _record(item.nodeid, "setup")


def pytest_runtest_teardown(item: pytest.Item, nextitem: pytest.Item | None) -> None:
    del nextitem
    _record(item.nodeid, "teardown")


def pytest_runtest_logreport(report: pytest.TestReport) -> None:
    if report.when != "call":
        return
    detail = str(report.longrepr or "")[:400]
    _record(report.nodeid, "outcome", outcome=report.outcome, detail=detail)
