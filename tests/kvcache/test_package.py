"""The kvcache facades resolve every public name from the module that defines it."""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

import pytest

import ayaka.kvcache as kvcache
import ayaka.kvcache.storage as storage

if TYPE_CHECKING:
    from types import ModuleType

pytestmark = pytest.mark.unit

PACKAGES = {"ayaka.kvcache": kvcache, "ayaka.kvcache.storage": storage}


@pytest.mark.parametrize("package", PACKAGES)
def test_all_matches_the_export_table(package: str) -> None:
    module = PACKAGES[package]
    exported = [name for names in module._EXPORTS.values() for name in names]  # pyright: ignore[reportAttributeAccessIssue]

    assert set(module.__all__) == set(exported)
    assert len(module.__all__) == len(set(module.__all__))


@pytest.mark.parametrize(
    ("package", "module_name", "name"),
    [
        (package, f"{package}.{module}", name)
        for package, facade in PACKAGES.items()
        for module, names in facade._EXPORTS.items()  # pyright: ignore[reportAttributeAccessIssue]
        for name in names
    ],
)
def test_public_name_resolves_to_its_owner(package: str, module_name: str, name: str) -> None:
    owner: ModuleType = importlib.import_module(module_name)

    assert getattr(PACKAGES[package], name) is getattr(owner, name)


@pytest.mark.parametrize("package", PACKAGES)
def test_dir_lists_every_public_name(package: str) -> None:
    module = PACKAGES[package]

    assert set(module.__all__) <= set(dir(module))
