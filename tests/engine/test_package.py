"""The engine facade resolves every public name from the module that defines it."""

from __future__ import annotations

import subprocess
import sys

import pytest

import ayaka.engine as engine

pytestmark = pytest.mark.unit

_EXPORTS: dict[str, tuple[str, ...]] = engine._EXPORTS  # pyright: ignore[reportAttributeAccessIssue]


def test_all_matches_the_export_table() -> None:
    exported = [name for names in _EXPORTS.values() for name in names]

    assert set(engine.__all__) == set(exported)
    assert len(engine.__all__) == len(set(engine.__all__))


def test_dir_lists_every_public_name() -> None:
    assert set(engine.__all__) <= set(dir(engine))


def test_importing_the_package_does_not_load_torch() -> None:
    code = (
        "import sys\nimport ayaka.engine\nassert 'torch' not in sys.modules, 'torch was imported'\n"
    )

    result = subprocess.run(  # noqa: S603
        [sys.executable, "-I", "-c", code], capture_output=True, text=True, check=False
    )

    assert result.returncode == 0, result.stderr
