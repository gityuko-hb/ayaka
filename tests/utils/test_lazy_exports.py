"""Tests for lazy package-level exports (PEP 562)."""

from __future__ import annotations

import importlib
import sys
from typing import TYPE_CHECKING

import pytest

from ayaka.utils.lazy_exports import build_lazy_exports

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path
    from types import ModuleType

pytestmark = pytest.mark.unit

PACKAGE = "lazy_exports_test_pkg"

_INIT = """\
from __future__ import annotations

from ayaka.utils.lazy_exports import build_lazy_exports

__all__ = ["Alpha", "Beta"]

_EXPORTS = {
    "alpha": ("Alpha",),
    "beta": ("Beta",),
}

__getattr__, __dir__ = build_lazy_exports(__name__, _EXPORTS, globals())
"""


@pytest.fixture
def package(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[ModuleType]:
    root = tmp_path / PACKAGE
    root.mkdir()
    (root / "__init__.py").write_text(_INIT, encoding="utf-8")
    (root / "alpha.py").write_text("class Alpha: ...\n", encoding="utf-8")
    (root / "beta.py").write_text("class Beta: ...\n", encoding="utf-8")

    monkeypatch.syspath_prepend(str(tmp_path))
    module = importlib.import_module(PACKAGE)
    try:
        yield module
    finally:
        stale = [name for name in sys.modules if name == PACKAGE or name.startswith(f"{PACKAGE}.")]
        for name in stale:
            del sys.modules[name]


class TestLazyExports:
    def test_importing_the_package_does_not_import_submodules(self, package: ModuleType) -> None:
        assert f"{PACKAGE}.alpha" not in sys.modules
        assert f"{PACKAGE}.beta" not in sys.modules
        assert "Alpha" not in vars(package)

    def test_attribute_access_imports_only_the_owning_submodule(self, package: ModuleType) -> None:
        alpha = package.Alpha

        assert alpha.__name__ == "Alpha"
        assert f"{PACKAGE}.alpha" in sys.modules
        assert f"{PACKAGE}.beta" not in sys.modules

    def test_resolved_name_is_cached_in_the_package_namespace(self, package: ModuleType) -> None:
        alpha = package.Alpha

        assert vars(package)["Alpha"] is alpha
        del sys.modules[f"{PACKAGE}.alpha"]
        assert package.Alpha is alpha

    def test_dir_lists_public_names_without_importing_them(self, package: ModuleType) -> None:
        names = dir(package)

        assert "Alpha" in names
        assert "Beta" in names
        assert f"{PACKAGE}.alpha" not in sys.modules

    def test_unknown_name_raises_attribute_error(self, package: ModuleType) -> None:
        with pytest.raises(AttributeError, match="has no attribute 'Missing'"):
            _ = package.Missing


class TestValidation:
    def test_duplicate_export_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="'Alpha' exported by alpha and beta"):
            build_lazy_exports(
                PACKAGE,
                {"alpha": ("Alpha",), "beta": ("Alpha",)},
                {"__all__": ["Alpha"]},
            )

    def test_missing_name_from_all_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match=r"missing from __all__: \['Beta'\]"):
            build_lazy_exports(
                PACKAGE,
                {"alpha": ("Alpha",), "beta": ("Beta",)},
                {"__all__": ["Alpha"]},
            )

    def test_extra_name_in_all_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match=r"not in the export table: \['Ghost'\]"):
            build_lazy_exports(
                PACKAGE,
                {"alpha": ("Alpha",)},
                {"__all__": ["Alpha", "Ghost"]},
            )

    def test_dir_includes_names_that_are_not_yet_resolved(self) -> None:
        getattr_, dir_ = build_lazy_exports(
            PACKAGE,
            {"alpha": ("Alpha",)},
            {"__all__": ["Alpha"], "existing": 1},
        )

        assert "Alpha" in dir_()
        assert "existing" in dir_()
        with pytest.raises(AttributeError, match="has no attribute 'Missing'"):
            getattr_("Missing")
