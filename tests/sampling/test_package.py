"""The sampling facade resolves every public name from its owner, and import stays free of torch."""

from __future__ import annotations

import importlib
import subprocess
import sys

import pytest

import ayaka.sampling as sampling

pytestmark = pytest.mark.unit


def test_all_matches_the_export_table() -> None:
    exported = [name for names in sampling._EXPORTS.values() for name in names]  # pyright: ignore[reportAttributeAccessIssue]

    assert set(sampling.__all__) == set(exported)
    assert len(sampling.__all__) == len(set(sampling.__all__))


def test_dir_lists_every_public_name() -> None:
    assert set(sampling.__all__) <= set(dir(sampling))


@pytest.mark.parametrize(
    ("module_name", "name"),
    [
        (f"ayaka.sampling.{module}", name)
        for module, names in sampling._EXPORTS.items()  # pyright: ignore[reportAttributeAccessIssue]
        for name in names
    ],
)
def test_public_name_resolves_to_its_owner(module_name: str, name: str) -> None:
    pytest.importorskip("torch")

    assert getattr(sampling, name) is getattr(importlib.import_module(module_name), name)


def test_an_unknown_name_is_an_attribute_error() -> None:
    with pytest.raises(AttributeError):
        sampling.nonexistent  # noqa: B018  # pyright: ignore[reportAttributeAccessIssue]


@pytest.mark.parametrize("module_name", ["ayaka.sampling", "ayaka.sampling.rng"])
def test_importing_does_not_import_torch(module_name: str) -> None:
    program = (
        f"import sys, {module_name}\n"
        "assert 'torch' not in sys.modules, 'torch was imported'\n"
        "import ayaka.sampling as s\n"
        "s.draw_uniform, s.philox4x32_10, s.resolve_seed, s.MAX_SEED\n"
        "assert 'torch' not in sys.modules, 'torch was imported by the rng names'\n"
    )

    result = subprocess.run(  # noqa: S603  # fixed argv, no input from outside
        [sys.executable, "-I", "-c", program], capture_output=True, text=True, check=False
    )

    assert result.returncode == 0, result.stderr
