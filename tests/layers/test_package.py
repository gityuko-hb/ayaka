"""The layers package: lazy exports that resolve, and no torch until a layer is used."""

from __future__ import annotations

import importlib
import subprocess
import sys

import pytest

pytestmark = pytest.mark.unit

PACKAGES = ("ayaka.layers", "ayaka.layers.attention", "ayaka.layers.embedding", "ayaka.layers.ffn")


@pytest.mark.parametrize("package_name", PACKAGES)
def test_every_public_name_resolves_and_is_listed_by_dir(package_name: str) -> None:
    pytest.importorskip("torch")
    package = importlib.import_module(package_name)

    missing = [name for name in package.__all__ if not hasattr(package, name)]

    assert missing == []
    assert set(package.__all__) <= set(dir(package))


@pytest.mark.parametrize("package_name", PACKAGES)
def test_importing_a_package_does_not_import_torch(package_name: str) -> None:
    program = (
        f"import sys, {package_name}\n"
        "assert 'torch' not in sys.modules, 'torch was imported'\n"
        f"{package_name}.__all__\n"
        "assert 'torch' not in sys.modules, 'torch was imported by listing the names'\n"
    )

    result = subprocess.run(  # noqa: S603  # fixed argv, no input from outside
        [sys.executable, "-I", "-c", program], capture_output=True, text=True, check=False
    )

    assert result.returncode == 0, result.stderr


def test_importing_ayaka_does_not_import_torch() -> None:
    program = "import sys, ayaka\nassert 'torch' not in sys.modules\n"

    result = subprocess.run(  # noqa: S603  # fixed argv, no input from outside
        [sys.executable, "-I", "-c", program], capture_output=True, text=True, check=False
    )

    assert result.returncode == 0, result.stderr
