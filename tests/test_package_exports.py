"""Every name in a package ``__all__`` resolves through its lazy export table."""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from types import ModuleType

pytestmark = pytest.mark.unit

PACKAGES = (
    "ayaka.configs",
    "ayaka.metrics",
    "ayaka.tokenizers",
    "ayaka.tokenizers.adapters",
    "ayaka.tokenizers.decoding",
    "ayaka.tokenizers.encoding",
    "ayaka.utils",
)


@pytest.mark.parametrize("package_name", PACKAGES)
def test_every_public_name_resolves(package_name: str) -> None:
    package: ModuleType = importlib.import_module(package_name)
    names = package.__all__

    missing = [name for name in names if not hasattr(package, name)]

    assert missing == []


@pytest.mark.parametrize("package_name", PACKAGES)
def test_dir_lists_every_public_name(package_name: str) -> None:
    package: ModuleType = importlib.import_module(package_name)
    names = package.__all__

    assert set(names) <= set(dir(package))
