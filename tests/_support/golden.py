"""Golden files: bytes a test must produce exactly, rewritten only on request.

``check_golden(name, data)`` compares ``data`` with ``tests/entrypoints/fixtures/golden/<name>``.
With ``AYAKA_UPDATE_GOLDEN=1`` in the environment it writes the file instead, so a change to the
wire is a diff of these files that someone reads before committing. Without the variable a
missing file fails the test: a new case cannot pass by writing its own golden file.

``.gitattributes`` keeps the files byte for byte (``-text``): a checkout on Windows must not turn
their ``\n`` into ``\r\n``.
"""

from __future__ import annotations

import difflib
import os
from pathlib import Path

import pytest

__all__ = ["GOLDEN", "UPDATE_GOLDEN", "check_golden"]

GOLDEN = Path(__file__).resolve().parents[1] / "entrypoints" / "fixtures" / "golden"
UPDATE_GOLDEN = os.environ.get("AYAKA_UPDATE_GOLDEN") == "1"


def check_golden(name: str, data: bytes) -> None:
    """``data`` must equal the golden file ``name`` (written instead with the update flag)."""
    path = GOLDEN / name
    if UPDATE_GOLDEN:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        return
    if not path.exists():
        pytest.fail(f"no golden file {name}; write it with AYAKA_UPDATE_GOLDEN=1 and review it")
    expected = path.read_bytes()
    if data == expected:
        return
    # A diff of the lines reads better than two whole streams, or than errors.json as one frame.
    diff = difflib.unified_diff(
        expected.decode(errors="replace").split("\n"),
        data.decode(errors="replace").split("\n"),
        fromfile=f"golden/{name}",
        tofile="written",
        lineterm="",
        n=0,
    )
    pytest.fail("\n".join([f"{name} is not the golden bytes:", *diff]))
