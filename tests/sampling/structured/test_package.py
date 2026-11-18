"""The structured package exports its names lazily and loads no grammar library until a compile."""

from __future__ import annotations

import subprocess
import sys

import pytest

import ayaka.sampling.structured as structured

pytestmark = pytest.mark.unit

_EXPORTS: dict[str, tuple[str, ...]] = structured._EXPORTS  # pyright: ignore[reportAttributeAccessIssue]


def test_all_matches_the_export_table() -> None:
    exported = [name for names in _EXPORTS.values() for name in names]

    assert set(structured.__all__) == set(exported)


@pytest.mark.parametrize(
    "statement",
    [
        "import ayaka",
        "import ayaka.engine.input",
        "from ayaka.sampling.structured.compiler import GrammarCompiler",
        "import ayaka.entrypoints.http",
    ],
)
def test_nothing_loads_llguidance_before_a_compile(statement: str) -> None:
    code = f"import sys\n{statement}\nassert 'llguidance' not in sys.modules, 'llguidance loaded'\n"

    result = subprocess.run(  # noqa: S603
        [sys.executable, "-I", "-c", code], capture_output=True, text=True, check=False
    )

    assert result.returncode == 0, result.stderr
