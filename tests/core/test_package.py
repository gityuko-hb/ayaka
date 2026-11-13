from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

import pytest

import ayaka.core as core

if TYPE_CHECKING:
    from types import ModuleType

pytestmark = pytest.mark.unit


def test_all_matches_the_export_table() -> None:
    exported = [name for names in core._EXPORTS.values() for name in names]

    assert set(core.__all__) == set(exported)
    assert len(core.__all__) == len(set(core.__all__))


@pytest.mark.parametrize(
    ("module_name", "name"),
    [(f"ayaka.core.{module}", name) for module, names in core._EXPORTS.items() for name in names],
)
def test_public_name_resolves_to_its_owner(module_name: str, name: str) -> None:
    owner: ModuleType = importlib.import_module(module_name)

    assert getattr(core, name) is getattr(owner, name)


def test_dir_lists_every_public_name() -> None:
    assert set(core.__all__) <= set(dir(core))


def test_unknown_name_raises_attribute_error() -> None:
    with pytest.raises(AttributeError, match="has no attribute 'Missing'"):
        _ = core.Missing
