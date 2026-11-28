from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

import pytest

import ayaka.multimodal as multimodal

if TYPE_CHECKING:
    from types import ModuleType

pytestmark = pytest.mark.unit


def test_all_matches_the_export_table() -> None:
    exported = [name for names in multimodal._EXPORTS.values() for name in names]

    assert set(multimodal.__all__) == set(exported)
    assert len(multimodal.__all__) == len(set(multimodal.__all__))


@pytest.mark.parametrize(
    ("module_name", "name"),
    [
        (f"ayaka.multimodal.{module}", name)
        for module, names in multimodal._EXPORTS.items()
        for name in names
    ],
)
def test_public_name_resolves_to_its_owner(module_name: str, name: str) -> None:
    owner: ModuleType = importlib.import_module(module_name)

    assert getattr(multimodal, name) is getattr(owner, name)


def test_dir_lists_every_public_name() -> None:
    assert set(multimodal.__all__) <= set(dir(multimodal))


def test_unknown_name_raises_attribute_error() -> None:
    with pytest.raises(AttributeError, match="has no attribute 'Missing'"):
        _ = multimodal.Missing
