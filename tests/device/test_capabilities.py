"""What a probed device can run, and the configurations it refuses."""

from __future__ import annotations

import pytest

from ayaka.configs.device import DeviceConfig
from ayaka.core.errors import ConfigError
from ayaka.device.capabilities import DeviceCapabilities, device_capabilities
from ayaka.device.info import DeviceInfo
from ayaka.utils.dtypes import DType

pytestmark = pytest.mark.unit

GIB = 2**30


def _gpu(capability: tuple[int, int], index: int = 0) -> DeviceInfo:
    return DeviceInfo(
        kind="cuda",
        index=index,
        name="Test GPU",
        total_bytes=4 * GIB,
        free_bytes=3 * GIB,
        compute_capability=capability,
    )


HOST = DeviceInfo(kind="cpu", index=0, name="cpu", total_bytes=16 * GIB, free_bytes=8 * GIB)


def _caps(info: DeviceInfo) -> DeviceCapabilities:
    return device_capabilities(info, platform="linux", wsl=True)


class TestFromInfo:
    def test_ampere_computes_in_bf16_and_captures_graphs(self) -> None:
        caps = _caps(_gpu((8, 6)))

        assert caps.arch == "sm_86"
        assert caps.dtypes == {DType.FP32, DType.FP16, DType.BF16}
        assert caps.cuda_graph

    def test_turing_has_no_bf16(self) -> None:
        caps = _caps(_gpu((7, 5)))

        assert DType.BF16 not in caps.dtypes
        assert DType.FP16 in caps.dtypes

    def test_the_host_has_no_graphs(self) -> None:
        caps = _caps(HOST)

        assert caps.arch == "cpu"
        assert not caps.cuda_graph
        assert caps.dtypes == {DType.FP32, DType.FP16, DType.BF16}

    def test_the_description_names_the_device_and_wsl(self) -> None:
        assert _caps(_gpu((8, 6))).describe() == "cuda:0 Test GPU sm_86 4.0 GiB (WSL2)"
        assert device_capabilities(HOST, platform="linux", wsl=False).describe() == "cpu"


class TestCheck:
    def test_a_config_that_fits_passes(self) -> None:
        _caps(_gpu((8, 6))).check(DeviceConfig(device="cuda:0", dtype=DType.BF16, cuda_graph=True))

    def test_every_problem_is_listed(self) -> None:
        caps = _caps(HOST)
        config = DeviceConfig(device="cuda", dtype=DType.BF16)

        problems = caps.problems(config)

        assert len(problems) == 1  # the device; the host computes in bf16
        assert "names 'cuda'" in problems[0]

    def test_bf16_on_turing_is_refused_with_a_code(self) -> None:
        with pytest.raises(ConfigError) as caught:
            _caps(_gpu((7, 5))).check(DeviceConfig(device="cuda", dtype=DType.BF16))

        assert caught.value.code == "DEVICE_CAPABILITY_MISSING"
        assert "sm_75 has no bf16 arithmetic" in str(caught.value)

    def test_another_gpu_index_is_a_mismatch(self) -> None:
        problems = _caps(_gpu((8, 6), index=0)).problems(DeviceConfig(device="cuda:1"))

        assert len(problems) == 1
        assert "cuda:1" in problems[0]
