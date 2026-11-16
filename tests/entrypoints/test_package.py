"""The entrypoints facade resolves every public name from the module that defines it."""

from __future__ import annotations

import subprocess
import sys

import pytest

import ayaka.entrypoints as entrypoints

pytestmark = pytest.mark.unit

_EXPORTS: dict[str, tuple[str, ...]] = entrypoints._EXPORTS  # pyright: ignore[reportAttributeAccessIssue]


def test_all_matches_the_export_table() -> None:
    exported = [name for names in _EXPORTS.values() for name in names]

    assert set(entrypoints.__all__) == set(exported)


def test_main_is_the_function_of_the_cli_module() -> None:
    from ayaka.entrypoints import cli

    assert entrypoints.main is cli.main


def test_dir_lists_every_public_name() -> None:
    assert set(entrypoints.__all__) <= set(dir(entrypoints))


def test_importing_the_package_does_not_load_torch() -> None:
    code = (
        "import sys\n"
        "import ayaka.entrypoints\n"
        "from ayaka.entrypoints.cli import main\n"
        "assert 'torch' not in sys.modules, 'torch was imported'\n"
    )

    result = subprocess.run(  # noqa: S603
        [sys.executable, "-I", "-c", code], capture_output=True, text=True, check=False
    )

    assert result.returncode == 0, result.stderr
