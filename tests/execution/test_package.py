"""The execution facade resolves every public name from the module that defines it."""

from __future__ import annotations

import subprocess
import sys

import pytest

import ayaka.execution as execution

pytestmark = pytest.mark.unit

_EXPORTS: dict[str, tuple[str, ...]] = execution._EXPORTS  # pyright: ignore[reportAttributeAccessIssue]


def test_all_matches_the_export_table() -> None:
    exported = [name for names in _EXPORTS.values() for name in names]

    assert set(execution.__all__) == set(exported)
    assert len(execution.__all__) == len(set(execution.__all__))


def test_dir_lists_every_public_name() -> None:
    assert set(execution.__all__) <= set(dir(execution))


def test_importing_the_package_does_not_load_torch() -> None:
    code = (
        "import sys\n"
        "import ayaka.execution\n"
        "assert 'torch' not in sys.modules, 'torch was imported'\n"
    )

    result = subprocess.run(  # noqa: S603
        [sys.executable, "-I", "-c", code], capture_output=True, text=True, check=False
    )

    assert result.returncode == 0, result.stderr
