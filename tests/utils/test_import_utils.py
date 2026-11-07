from __future__ import annotations

import json

import pytest

from ayaka.utils.import_utils import (
    CapabilityError,
    LazyModule,
    has_module,
    require_module,
    resolve_qualname,
)

pytestmark = pytest.mark.unit


MISSING = "ayaka_missing_package_for_tests"


class TestHasModule:
    def test_known_stdlib_module(self) -> None:
        assert has_module("json")

    def test_missing_module(self) -> None:
        assert not has_module(MISSING)


class TestRequireModule:
    def test_imports_a_real_module(self) -> None:
        assert require_module("json") is json

    def test_reports_capability_detail_and_remedy(self) -> None:
        with pytest.raises(CapabilityError) as exc:
            require_module(MISSING, capability="fp8_gemm", remedy="pip install ayaka-kernels")
        assert exc.value.capability == "fp8_gemm"
        assert MISSING in exc.value.detail
        assert exc.value.remedy == "pip install ayaka-kernels"
        assert isinstance(exc.value.__cause__, ImportError)

    def test_defaults_capability_and_remedy_to_module_name(self) -> None:
        with pytest.raises(CapabilityError) as exc:
            require_module(MISSING)
        assert exc.value.capability == MISSING
        assert exc.value.remedy == f"install {MISSING}"


class TestLazyModule:
    def test_imports_on_first_attribute_access(self) -> None:
        lazy = LazyModule("json")
        assert lazy._loader is None
        assert lazy.dumps({"ok": True}) == '{"ok": true}'
        assert lazy._loader is json

    def test_real_attributes_are_visible_to_dir(self) -> None:
        lazy = LazyModule("json")
        assert "dumps" in dir(lazy)

    def test_missing_module_raises_capability_error_on_use(self) -> None:
        lazy = LazyModule(MISSING, capability="demo", remedy="pip install demo")
        with pytest.raises(CapabilityError) as exc:
            _ = lazy.anything
        assert exc.value.capability == "demo"

    def test_dunder_lookup_stays_an_attribute_error(self) -> None:
        lazy = LazyModule(MISSING)
        with pytest.raises(AttributeError):
            _ = lazy.__wrapped__
        assert not hasattr(lazy, "__wrapped__")

    def test_dir_of_a_missing_module_does_not_import_it(self) -> None:
        lazy = LazyModule(MISSING)
        names = dir(lazy)
        assert "_capability" in names


class TestResolveQualname:
    def test_colon_form(self) -> None:
        assert resolve_qualname("json:dumps") is json.dumps

    def test_dotted_form(self) -> None:
        assert resolve_qualname("ayaka.utils.import_utils.CapabilityError") is CapabilityError

    def test_requires_a_module_and_attribute(self) -> None:
        with pytest.raises(ValueError, match="must contain a module and an attribute"):
            resolve_qualname("json")

    def test_missing_module_raises_import_error(self) -> None:
        with pytest.raises(ImportError):
            resolve_qualname(f"{MISSING}:thing")

    def test_missing_attribute_raises_import_error(self) -> None:
        with pytest.raises(ImportError, match="cannot resolve"):
            resolve_qualname("json.no_such_attribute")
