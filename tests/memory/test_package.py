"""The memory facade resolves every public name from the module that defines it."""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

import pytest

import ayaka.memory as memory

if TYPE_CHECKING:
    from types import ModuleType

pytestmark = pytest.mark.unit


def test_all_matches_the_export_table() -> None:
    exported = [name for names in memory._EXPORTS.values() for name in names]

    assert set(memory.__all__) == set(exported)
    assert len(memory.__all__) == len(set(memory.__all__))


@pytest.mark.parametrize(
    ("module_name", "name"),
    [
        (f"ayaka.memory.{module}", name)
        for module, names in memory._EXPORTS.items()
        for name in names
    ],
)
def test_public_name_resolves_to_its_owner(module_name: str, name: str) -> None:
    owner: ModuleType = importlib.import_module(module_name)

    assert getattr(memory, name) is getattr(owner, name)


def test_dir_lists_every_public_name() -> None:
    assert set(memory.__all__) <= set(dir(memory))
