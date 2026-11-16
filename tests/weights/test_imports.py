"""Reading checkpoint headers must work on a host with no PyTorch."""

from __future__ import annotations

import subprocess
import sys

import pytest

pytestmark = pytest.mark.unit

MODULES = ("spec", "tensor", "safetensors", "manifest", "mapping")


@pytest.mark.parametrize("module", MODULES)
def test_importing_a_module_does_not_import_torch(module: str) -> None:
    program = (
        "import sys\n"
        f"import ayaka.weights.{module}\n"
        "assert 'torch' not in sys.modules, 'torch was imported'\n"
    )

    result = subprocess.run(  # noqa: S603  # fixed argv, no input from outside
        [sys.executable, "-I", "-c", program], capture_output=True, text=True, check=False
    )

    assert result.returncode == 0, result.stderr
