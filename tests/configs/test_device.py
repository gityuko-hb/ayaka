from __future__ import annotations

import subprocess
import sys
from dataclasses import FrozenInstanceError

import pytest

from ayaka.configs.device import DeviceConfig
from ayaka.core.errors import ConfigError
from ayaka.utils.dtypes import DType


@pytest.mark.unit
class TestDeviceConfig:
    def test_defaults_are_constants_of_the_cpu_reference_run(self) -> None:
        config = DeviceConfig()

        assert config.device == "cpu"
        assert config.dtype is DType.FP32
        assert config.backends == ("torch",)
        assert not config.cuda_graph
        assert config.graph_buckets == (1, 2, 4, 8)
        assert config.vision_load == "eager"

    def test_vision_load_modes(self) -> None:
        assert DeviceConfig(vision_load="lazy").vision_load == "lazy"
        with pytest.raises(ConfigError) as caught:
            DeviceConfig(vision_load="never")  # type: ignore[arg-type]  # deliberately invalid
        assert caught.value.code == "INVALID_CHOICE"

    def test_is_frozen(self) -> None:
        with pytest.raises(FrozenInstanceError):
            DeviceConfig().device = "cuda"  # type: ignore[misc]  # frozen dataclass

    @pytest.mark.parametrize(
        ("device", "kind", "index"),
        [
            ("cpu", "cpu", None),
            ("cuda", "cuda", None),
            ("cuda:0", "cuda", 0),
            ("cuda:12", "cuda", 12),
        ],
    )
    def test_device_parts(self, device: str, kind: str, index: int | None) -> None:
        config = DeviceConfig(device=device)

        assert (config.device_type, config.device_index) == (kind, index)

    @pytest.mark.parametrize(
        "device", ["gpu", "cpu:0", "cuda:", "cuda:-1", "cuda:x", "CUDA", "mps", " cpu", "cuda0"]
    )
    def test_rejects_unknown_devices(self, device: str) -> None:
        with pytest.raises(ConfigError) as caught:
            DeviceConfig(device=device)

        assert caught.value.code == "INVALID_DEVICE"

    @pytest.mark.parametrize("device", ["", "  ", None])
    def test_rejects_blank_device(self, device: object) -> None:
        with pytest.raises(ValueError, match=r"device\.device must be a non-empty string"):
            DeviceConfig(device=device)  # type: ignore[arg-type]  # deliberately wrong type

    @pytest.mark.parametrize("dtype", [DType.FP16, DType.BF16, DType.FP32])
    def test_accepts_compute_dtypes(self, dtype: DType) -> None:
        assert DeviceConfig(dtype=dtype).dtype is dtype

    @pytest.mark.parametrize("dtype", [DType.FP8_E4M3, DType.INT8, DType.FP64])
    def test_rejects_non_compute_dtypes(self, dtype: DType) -> None:
        with pytest.raises(ConfigError) as caught:
            DeviceConfig(dtype=dtype)

        assert caught.value.code == "INVALID_CHOICE"
        assert "fp32" in caught.value.message

    @pytest.mark.parametrize("dtype", ["bf16", None, 16])
    def test_rejects_raw_dtype_values(self, dtype: object) -> None:
        with pytest.raises(TypeError, match="DType member"):
            DeviceConfig(dtype=dtype)  # type: ignore[arg-type]  # deliberately wrong type

    @pytest.mark.parametrize("backends", [(), ["torch"], "torch", ("torch", "")])
    def test_rejects_malformed_backends(self, backends: object) -> None:
        with pytest.raises(ValueError, match="backends"):
            DeviceConfig(backends=backends)  # type: ignore[arg-type]  # deliberately wrong type

    def test_rejects_repeated_backend(self) -> None:
        with pytest.raises(ValueError, match="must not repeat"):
            DeviceConfig(backends=("torch", "torch"))

    def test_backend_order_is_kept(self) -> None:
        assert DeviceConfig(backends=("flashinfer", "torch")).backends == ("flashinfer", "torch")

    @pytest.mark.parametrize("flag", [1, "yes", None])
    def test_rejects_non_bool_graph_flag(self, flag: object) -> None:
        with pytest.raises(TypeError, match="cuda_graph must be bool"):
            DeviceConfig(device="cuda", cuda_graph=flag)  # type: ignore[arg-type]  # deliberately wrong type

    def test_graph_on_cpu_is_rejected(self) -> None:
        with pytest.raises(ConfigError) as caught:
            DeviceConfig(device="cpu", cuda_graph=True)

        assert caught.value.code == "GRAPH_NEEDS_CUDA"

    def test_graph_on_cuda_is_accepted(self) -> None:
        config = DeviceConfig(device="cuda:1", cuda_graph=True, graph_buckets=(1, 4))

        assert config.cuda_graph
        assert config.graph_buckets == (1, 4)

    @pytest.mark.parametrize("buckets", [(), (2, 1), (1, 1), (0, 1), [1, 2]])
    def test_buckets_are_validated_even_when_graphs_are_off(self, buckets: object) -> None:
        with pytest.raises(ValueError, match="bucket"):
            DeviceConfig(graph_buckets=buckets)  # type: ignore[arg-type]  # deliberately wrong type

    def test_dtype_serializes_as_its_label(self) -> None:
        payload = DeviceConfig(dtype=DType.BF16).to_dict()

        assert payload["dtype"] == "bf16"
        assert payload["graph_buckets"] == [1, 2, 4, 8]

    def test_fingerprint_separates_dtypes(self) -> None:
        assert (
            DeviceConfig(dtype=DType.FP16).fingerprint != DeviceConfig(dtype=DType.BF16).fingerprint
        )


@pytest.mark.unit
def test_importing_the_configs_does_not_import_torch() -> None:
    program = (
        "import sys, ayaka.configs\n"
        "from ayaka.configs import CacheConfig, DeviceConfig, EngineConfig, ModelConfig, "
        "SchedulerConfig\n"
        "from ayaka.configs.model_source import ModelSourceConfig\n"
        "DeviceConfig(); CacheConfig(); SchedulerConfig(); EngineConfig(ModelSourceConfig('a/b'))\n"
        "assert 'torch' not in sys.modules, 'torch was imported'\n"
    )

    result = subprocess.run(  # noqa: S603  # fixed argv, no input from outside
        [sys.executable, "-I", "-c", program], capture_output=True, text=True, check=False
    )

    assert result.returncode == 0, result.stderr
