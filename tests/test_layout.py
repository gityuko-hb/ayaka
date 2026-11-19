"""The size limit of the package: a module of ``python/ayaka`` has at most 700 lines.

A file that grows past the limit is split by what it owns (a new module per thing, never a
``_part2``), in its own pull request; this test is what keeps the limit from drifting again.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

PACKAGE = Path(__file__).resolve().parents[1] / "python" / "ayaka"

#: The most lines a module may have.
MAX_MODULE_LINES = 700

_MODULES = sorted(PACKAGE.rglob("*.py"))


def _relative(path: Path) -> str:
    return path.relative_to(PACKAGE.parent).as_posix()


def test_the_package_has_modules() -> None:
    assert len(_MODULES) > 50


@pytest.mark.parametrize("path", _MODULES, ids=_relative)
def test_a_module_stays_under_the_line_limit(path: Path) -> None:
    lines = len(path.read_text(encoding="utf-8").splitlines())

    assert lines <= MAX_MODULE_LINES, (
        f"{_relative(path)} has {lines} lines; split it by what it owns "
        f"(the limit is {MAX_MODULE_LINES})"
    )
