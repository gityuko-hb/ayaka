"""Backend registry: family-wise composition, capability checks and failure reporting."""

from __future__ import annotations

import logging
import subprocess
import sys
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, cast, get_args

import pytest

from ayaka.core.errors import BackendError, CapabilityError, ConfigError
from ayaka.ops import backend as backend_module
from ayaka.ops.backend import (
    BackendCaps,
    BackendFamilies,
    FamilyName,
    available_backends,
    load_backend,
    register_backend,
)
from ayaka.utils.dtypes import COMPUTE_DTYPES, DType

if TYPE_CHECKING:
    from collections.abc import Callable

pytestmark = pytest.mark.unit

ALL: tuple[FamilyName, ...] = (
    "norm",
    "activation",
    "embedding",
    "linear",
    "rope",
    "kvcache",
    "attention",
    "sampling",
    "moe",
)
LINUX = frozenset({sys.platform})


def _caps(name: str, families: tuple[str, ...] = ALL, **overrides: Any) -> BackendCaps:
    """A record with every capability, unless ``overrides`` takes some away."""
    fields: dict[str, Any] = {
        "name": name,
        "families": frozenset(families),
        "platforms": LINUX,
        "devices": frozenset({"cpu", "cuda"}),
        "dtypes": frozenset(COMPUTE_DTYPES),
        "graph_safe": frozenset(families),
        "external_uniform": True,
        "multi_row_attention": True,
    }
    return BackendCaps(**{**fields, **overrides})


def _impls(label: str, families: tuple[str, ...] = ALL) -> BackendFamilies:
    return cast("BackendFamilies", {family: SimpleNamespace(owner=label) for family in families})


class _Loaders:
    """Records which backends were actually loaded."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    def add(self, caps: BackendCaps, factory: Callable[[], BackendFamilies] | None = None) -> None:
        def loader() -> BackendFamilies:
            self.calls.append(caps.name)
            return factory() if factory else _impls(caps.name, tuple(caps.families))

        register_backend(caps, loader)


@pytest.fixture
def loaders(monkeypatch: pytest.MonkeyPatch) -> _Loaders:
    """An empty registry for one test, so the real backends cannot leak in."""
    monkeypatch.setattr(backend_module, "_registry", {})
    return _Loaders()


def _owner(implementation: object) -> str:
    return cast("Any", implementation).owner


def test_family_name_lists_exactly_the_registered_families() -> None:
    value = cast("Any", FamilyName).__value__

    assert get_args(value) == backend_module._FAMILIES


class TestComposition:
    def test_first_listed_backend_wins_every_family_it_serves(self, loaders: _Loaders) -> None:
        loaders.add(_caps("fast"))
        loaders.add(_caps("slow"))

        ops = load_backend(["fast", "slow"], device="cpu", dtype=DType.FP32)

        assert dict(ops.providers) == dict.fromkeys(ALL, "fast")
        assert loaders.calls == ["fast"]

    def test_families_are_taken_from_different_backends(self, loaders: _Loaders) -> None:
        loaders.add(_caps("partial", families=("norm", "linear")))
        loaders.add(_caps("full"))

        ops = load_backend(["partial", "full"], device="cpu", dtype=DType.FP16)

        assert dict(ops.providers) == {
            "norm": "partial",
            "activation": "full",
            "embedding": "full",
            "linear": "partial",
            "rope": "full",
            "kvcache": "full",
            "attention": "full",
            "sampling": "full",
            "moe": "full",
        }
        assert _owner(ops.norm) == "partial"
        assert _owner(ops.activation) == "full"

    def test_a_backend_that_does_not_serve_the_device_is_skipped(
        self, loaders: _Loaders, caplog: pytest.LogCaptureFixture
    ) -> None:
        loaders.add(_caps("gpu-only", devices=frozenset({"cuda"})))
        loaders.add(_caps("everywhere"))

        with caplog.at_level(logging.INFO, logger=backend_module.logger.name):
            ops = load_backend(["gpu-only", "everywhere"], device="cpu", dtype=DType.FP32)

        assert set(ops.providers.values()) == {"everywhere"}
        assert loaders.calls == ["everywhere"]
        assert "gpu-only has no cpu support; using everywhere" in caplog.text

    def test_a_backend_without_the_dtype_is_skipped(self, loaders: _Loaders) -> None:
        loaders.add(_caps("fp16-only", dtypes=frozenset({DType.FP16})))
        loaders.add(_caps("everywhere"))

        ops = load_backend(["fp16-only", "everywhere"], device="cpu", dtype=DType.BF16)

        assert set(ops.providers.values()) == {"everywhere"}

    def test_a_backend_for_another_platform_is_skipped(self, loaders: _Loaders) -> None:
        loaders.add(_caps("elsewhere", platforms=frozenset({"plan9"})))
        loaders.add(_caps("here"))

        ops = load_backend(["elsewhere", "here"], device="cpu", dtype=DType.FP32)

        assert set(ops.providers.values()) == {"here"}

    def test_a_backend_that_cannot_load_is_skipped_with_a_warning(
        self, loaders: _Loaders, caplog: pytest.LogCaptureFixture
    ) -> None:
        def missing_library() -> BackendFamilies:
            raise CapabilityError(
                "ops.fancy", detail="no module named fancy", remedy="install fancy"
            )

        loaders.add(_caps("fancy"), missing_library)
        loaders.add(_caps("plain"))

        with caplog.at_level(logging.INFO, logger=backend_module.logger.name):
            ops = load_backend(["fancy", "plain"], device="cpu", dtype=DType.FP32)

        assert set(ops.providers.values()) == {"plain"}
        assert loaders.calls == ["fancy", "plain"]
        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warnings) == 1  # one per backend, not one per family
        assert "no module named fancy; install fancy" in warnings[0].getMessage()

    def test_the_selection_is_logged_once(
        self, loaders: _Loaders, caplog: pytest.LogCaptureFixture
    ) -> None:
        loaders.add(_caps("only"))

        with caplog.at_level(logging.INFO, logger=backend_module.logger.name):
            load_backend(["only"], device="cuda", dtype=DType.BF16)

        messages = [record.getMessage() for record in caplog.records]
        assert len(messages) == 1
        assert messages[0].startswith("ops backends for cuda bf16: ")
        for family in ALL:
            assert f"{family}=only" in messages[0]

    def test_providers_cannot_be_modified(self, loaders: _Loaders) -> None:
        loaders.add(_caps("only"))
        ops = load_backend(["only"], device="cpu", dtype=DType.FP32)

        with pytest.raises(TypeError):
            cast("Any", ops.providers)["norm"] = "other"


class TestMissingCapabilities:
    def test_every_missing_family_is_listed_with_its_reasons(self, loaders: _Loaders) -> None:
        loaders.add(_caps("cuda-only", families=("norm", "linear"), devices=frozenset({"cuda"})))
        loaders.add(_caps("tiny", families=("norm", "embedding"), dtypes=frozenset({DType.FP16})))

        with pytest.raises(ConfigError) as excinfo:
            load_backend(["cuda-only", "tiny"], device="cpu", dtype=DType.FP32)

        error = excinfo.value
        assert error.code == "BACKEND_CAPABILITY_MISSING"
        assert error.path == "ops.backends"
        assert (
            "provides norm, activation, embedding, linear, rope, kvcache, attention, sampling, moe "
            "for device 'cpu' "
            "and dtype fp32" in str(error)
        )
        assert "norm: cuda-only has no cpu support; tiny has no fp32 support" in str(error)
        assert "activation: cuda-only does not implement this family; tiny does not implement" in (
            str(error)
        )
        assert "linear: cuda-only has no cpu support; tiny does not implement this family" in str(
            error
        )
        assert set(cast("dict[str, Any]", error.context["missing"])) == set(ALL)

    def test_one_family_missing_does_not_hide_the_others(self, loaders: _Loaders) -> None:
        loaders.add(_caps("partial", families=tuple(f for f in ALL if f != "linear")))

        with pytest.raises(ConfigError, match=r"provides linear for device"):
            load_backend(["partial"], device="cpu", dtype=DType.FP32)

    def test_an_unavailable_backend_is_named_in_the_error(self, loaders: _Loaders) -> None:
        def missing_library() -> BackendFamilies:
            raise CapabilityError(
                "ops.fancy", detail="no module named fancy", remedy="install fancy"
            )

        loaders.add(_caps("fancy"), missing_library)

        with pytest.raises(ConfigError, match=r"fancy is unavailable \(no module named fancy"):
            load_backend(["fancy"], device="cpu", dtype=DType.FP32)

    def test_a_loader_that_omits_a_declared_family_is_a_backend_error(
        self, loaders: _Loaders
    ) -> None:
        loaders.add(_caps("liar"), lambda: _impls("liar", tuple(f for f in ALL if f != "linear")))

        with pytest.raises(BackendError, match=r"declares \['linear'\]"):
            load_backend(["liar"], device="cpu", dtype=DType.FP32)

    def test_a_loader_error_other_than_capability_is_not_swallowed(self, loaders: _Loaders) -> None:
        def broken() -> BackendFamilies:
            raise RuntimeError("boom")

        loaders.add(_caps("broken"), broken)
        loaders.add(_caps("fallback"))

        with pytest.raises(RuntimeError, match="boom"):
            load_backend(["broken", "fallback"], device="cpu", dtype=DType.FP32)


class TestCapabilities:
    def test_with_graphs_a_family_comes_only_from_a_graph_safe_backend(
        self, loaders: _Loaders, caplog: pytest.LogCaptureFixture
    ) -> None:
        loaders.add(_caps("fast", families=("attention",), graph_safe=frozenset()))
        loaders.add(_caps("reference"))

        with caplog.at_level(logging.INFO, logger=backend_module.logger.name):
            eager = load_backend(["fast", "reference"], device="cuda", dtype=DType.FP16)
            graphed = load_backend(
                ["fast", "reference"], device="cuda", dtype=DType.FP16, cuda_graph=True
            )

        assert eager.providers["attention"] == "fast"
        assert not eager.graph_safe
        assert graphed.providers["attention"] == "reference"
        assert graphed.graph_safe
        assert "fast cannot run inside a CUDA graph; using reference" in caplog.text

    def test_a_missing_graph_capability_is_listed(self, loaders: _Loaders) -> None:
        loaders.add(_caps("eager", graph_safe=frozenset({"norm"})))

        with pytest.raises(ConfigError) as excinfo:
            load_backend(["eager"], device="cuda", dtype=DType.FP16, cuda_graph=True)

        message = str(excinfo.value)
        assert "inside a CUDA graph" in message
        assert "sampling: eager cannot run inside a CUDA graph" in message
        assert "norm:" not in message

    def test_attention_needs_several_rows_per_request(self, loaders: _Loaders) -> None:
        loaders.add(_caps("decode-only", families=("attention",), multi_row_attention=False))
        loaders.add(_caps("reference"))

        ops = load_backend(["decode-only", "reference"], device="cpu", dtype=DType.FP32)

        assert ops.providers["attention"] == "reference"

    def test_attention_for_a_head_width_without_kernels_comes_from_the_next_backend(
        self, loaders: _Loaders
    ) -> None:
        loaders.add(
            _caps("fast", families=("attention",), attention_head_dims=frozenset({64, 128}))
        )
        loaders.add(_caps("reference"))

        phi = load_backend(["fast", "reference"], device="cpu", dtype=DType.FP32, head_dim=96)
        llama = load_backend(["fast", "reference"], device="cpu", dtype=DType.FP32, head_dim=128)
        unknown = load_backend(["fast", "reference"], device="cpu", dtype=DType.FP32)

        assert phi.providers["attention"] == "reference"
        assert llama.providers["attention"] == "fast"
        assert unknown.providers["attention"] == "fast"

    def test_a_head_width_no_backend_has_is_named_in_the_error(self, loaders: _Loaders) -> None:
        loaders.add(_caps("fast", attention_head_dims=frozenset({64, 128})))

        with pytest.raises(
            ConfigError, match=r"fast has no kernels for heads of 96 \(only \[64, 128\]\)"
        ):
            load_backend(["fast"], device="cpu", dtype=DType.FP32, head_dim=96)

    def test_flashinfer_declares_the_head_widths_of_its_kernels(self) -> None:
        [caps] = [caps for caps in available_backends() if caps.name == "flashinfer"]

        assert caps.attention_head_dims == frozenset({64, 128, 256})

    def test_a_decode_only_attention_alone_is_refused(self, loaders: _Loaders) -> None:
        loaders.add(_caps("decode-only", multi_row_attention=False))

        with pytest.raises(
            ConfigError, match="attention: decode-only serves only one query row per request"
        ):
            load_backend(["decode-only"], device="cpu", dtype=DType.FP32)

    def test_sampling_must_take_its_random_numbers_from_outside(self) -> None:
        with pytest.raises(ValueError, match="external_uniform=True"):
            _caps("own-rng", external_uniform=False)

        assert not _caps("no-sampling", families=("norm",), external_uniform=False).external_uniform

    def test_graph_safe_must_be_a_subset_of_the_families(self) -> None:
        with pytest.raises(ValueError, match=r"does not implement: \['attention'\]"):
            _caps("x", families=("norm",), graph_safe=frozenset({"attention"}))


class TestBackendList:
    @pytest.mark.usefixtures("loaders")
    def test_empty_list(self) -> None:
        with pytest.raises(ConfigError, match="at least one backend") as excinfo:
            load_backend([], device="cpu", dtype=DType.FP32)

        assert excinfo.value.code == "BACKEND_LIST_EMPTY"

    def test_unknown_name_lists_the_registered_ones(self, loaders: _Loaders) -> None:
        loaders.add(_caps("real"))

        with pytest.raises(ConfigError, match=r"not registered: \['imaginary'\].*\['real'\]"):
            load_backend(["real", "imaginary"], device="cpu", dtype=DType.FP32)

    def test_repeated_name(self, loaders: _Loaders) -> None:
        loaders.add(_caps("real"))

        with pytest.raises(ConfigError, match=r"more than once: \['real'\]"):
            load_backend(["real", "real"], device="cpu", dtype=DType.FP32)

    def test_a_bare_string_is_not_a_list_of_letters(self, loaders: _Loaders) -> None:
        loaders.add(_caps("real"))

        with pytest.raises(TypeError, match="not one string"):
            load_backend(cast("Any", "real"), device="cpu", dtype=DType.FP32)


class TestRegistration:
    def test_duplicate_name_is_refused(self, loaders: _Loaders) -> None:
        loaders.add(_caps("once"))

        with pytest.raises(ValueError, match="'once' is already registered"):
            loaders.add(_caps("once"))

    def test_available_backends_lists_records_without_loading(self, loaders: _Loaders) -> None:
        loaders.add(_caps("a"))
        loaders.add(_caps("b"))

        assert [caps.name for caps in available_backends()] == ["a", "b"]
        assert loaders.calls == []

    def test_caps_reject_an_unknown_family(self) -> None:
        with pytest.raises(ValueError, match=r"unknown op families \['not-a-family'\]"):
            _caps("x", families=("norm", "not-a-family"))

    def test_caps_reject_a_blank_name(self) -> None:
        with pytest.raises(ValueError, match="name must be a non-empty string"):
            _caps(" ")


class TestTorchBackend:
    def test_is_registered_by_default(self) -> None:
        torch_caps = next(caps for caps in available_backends() if caps.name == "torch")

        assert torch_caps.families == frozenset(ALL)
        assert torch_caps.devices == {"cpu", "cuda"}
        assert torch_caps.dtypes == frozenset(COMPUTE_DTYPES)
        assert torch_caps.graph_safe == frozenset(ALL)
        assert torch_caps.external_uniform
        assert torch_caps.multi_row_attention

    def test_serves_every_family_on_cpu(self) -> None:
        pytest.importorskip("torch")

        ops = load_backend(["torch"], device="cpu", dtype=DType.FP32)

        assert dict(ops.providers) == dict.fromkeys(ALL, "torch")
        assert ops.norm.__name__ == "ayaka.ops.backends.torch.norm"  # type: ignore[attr-defined]  # modules are the implementations

    def test_refuses_dtypes_it_does_not_compute_in(self) -> None:
        with pytest.raises(ConfigError, match="has no fp8_e4m3 support"):
            load_backend(["torch"], device="cpu", dtype=DType.FP8_E4M3)

    def test_flashinfer_is_registered_for_attention_on_linux_gpus_only(self) -> None:
        caps = next(caps for caps in available_backends() if caps.name == "flashinfer")

        assert caps.families == {"attention"}
        assert caps.platforms == {"linux"}
        assert caps.devices == {"cuda"}
        assert caps.dtypes == {DType.FP16, DType.BF16}

    def test_without_torch_the_backend_is_reported_unavailable(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setitem(sys.modules, "torch", None)

        with pytest.raises(ConfigError, match=r"torch is unavailable \(cannot import torch"):
            load_backend(["torch"], device="cpu", dtype=DType.FP32)


def test_importing_the_package_does_not_import_torch() -> None:
    program = (
        "import sys, ayaka.ops, ayaka.ops.backend\n"
        "assert 'torch' not in sys.modules, 'torch was imported'\n"
        "ayaka.ops.load_backend\n"
        "assert 'torch' not in sys.modules, 'torch was imported by name resolution'\n"
    )

    result = subprocess.run(  # noqa: S603  # fixed argv, no input from outside
        [sys.executable, "-I", "-c", program], capture_output=True, text=True, check=False
    )

    assert result.returncode == 0, result.stderr
