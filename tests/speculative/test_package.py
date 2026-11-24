"""The speculative facade resolves every public name lazily without loading torch."""

from __future__ import annotations

import importlib
import subprocess
import sys
from typing import TYPE_CHECKING

import pytest

import ayaka.speculative as speculative

if TYPE_CHECKING:
    from types import ModuleType

pytestmark = pytest.mark.unit

_EXPORTS: dict[str, tuple[str, ...]] = speculative._EXPORTS  # pyright: ignore[reportAttributeAccessIssue]


def test_all_matches_the_export_table() -> None:
    exported = [name for names in _EXPORTS.values() for name in names]

    assert set(speculative.__all__) == set(exported)
    assert len(speculative.__all__) == len(set(speculative.__all__))


@pytest.mark.parametrize(
    ("module_name", "name"),
    [(f"ayaka.speculative.{module}", name) for module, names in _EXPORTS.items() for name in names],
)
def test_public_name_resolves_to_its_owner(module_name: str, name: str) -> None:
    owner: ModuleType = importlib.import_module(module_name)

    assert getattr(speculative, name) is getattr(owner, name)


def test_dir_lists_every_public_name() -> None:
    assert set(speculative.__all__) <= set(dir(speculative))


def test_importing_speculative_does_not_load_torch() -> None:
    code = (
        "import sys\n"
        "import ayaka.speculative\n"
        "from ayaka.speculative.ngram import LookupProposer\n"
        "from ayaka.speculative.acceptance import accept_one_point\n"
        "from ayaka.speculative.policy import CostAwareController\n"
        "assert 'torch' not in sys.modules, 'torch was imported'\n"
    )

    result = subprocess.run(  # noqa: S603
        [sys.executable, "-I", "-c", code], capture_output=True, text=True, check=False
    )

    assert result.returncode == 0, result.stderr
