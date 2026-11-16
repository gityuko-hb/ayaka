"""The scheduler facade resolves every public name from the module that defines it."""

from __future__ import annotations

import importlib
import subprocess
import sys
from typing import TYPE_CHECKING

import pytest

import ayaka.scheduler as scheduler

if TYPE_CHECKING:
    from types import ModuleType

pytestmark = pytest.mark.unit

_EXPORTS: dict[str, tuple[str, ...]] = scheduler._EXPORTS  # pyright: ignore[reportAttributeAccessIssue]


def test_all_matches_the_export_table() -> None:
    exported = [name for names in _EXPORTS.values() for name in names]

    assert set(scheduler.__all__) == set(exported)
    assert len(scheduler.__all__) == len(set(scheduler.__all__))


@pytest.mark.parametrize(
    ("module_name", "name"),
    [(f"ayaka.scheduler.{module}", name) for module, names in _EXPORTS.items() for name in names],
)
def test_public_name_resolves_to_its_owner(module_name: str, name: str) -> None:
    owner: ModuleType = importlib.import_module(module_name)

    assert getattr(scheduler, name) is getattr(owner, name)


def test_dir_lists_every_public_name() -> None:
    assert set(scheduler.__all__) <= set(dir(scheduler))


def test_importing_the_scheduler_does_not_load_torch() -> None:
    code = (
        "import sys\n"
        "import ayaka.scheduler\n"
        "from ayaka.scheduler.scheduler import Scheduler\n"
        "assert 'torch' not in sys.modules, 'torch was imported'\n"
    )

    result = subprocess.run(  # noqa: S603
        [sys.executable, "-I", "-c", code], capture_output=True, text=True, check=False
    )

    assert result.returncode == 0, result.stderr
