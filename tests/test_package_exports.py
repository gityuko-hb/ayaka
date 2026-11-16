"""Every name in a package ``__all__`` resolves through its lazy export table."""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from types import ModuleType

pytestmark = pytest.mark.unit

#: Some lazy packages (layers, models, model_loader) load torch when a name resolves.
pytest.importorskip("torch")

_PACKAGE_ROOT = Path(__file__).resolve().parents[1] / "python" / "ayaka"


def _lazy_packages() -> tuple[str, ...]:
    """Every package whose ``__init__.py`` calls ``build_lazy_exports``."""
    packages: list[str] = []
    for init in _PACKAGE_ROOT.rglob("__init__.py"):
        if "build_lazy_exports(" not in init.read_text(encoding="utf-8"):
            continue
        parts = init.relative_to(_PACKAGE_ROOT).parent.parts
        packages.append(".".join(("ayaka", *parts)))
    return tuple(sorted(packages))


PACKAGES = _lazy_packages()


def test_the_discovery_finds_the_known_facades() -> None:
    assert {"ayaka.configs", "ayaka.layers", "ayaka.models", "ayaka.ops", "ayaka.scheduler"} <= set(
        PACKAGES
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
