# ruff: noqa: N802 - the fake must mirror pynvml's camelCase API

from __future__ import annotations

from typing import Any

import pytest

import ayaka.utils.nvml_utils as nvml_utils
from ayaka.utils.nvml_utils import (
    NvmlDeviceRaw,
    get_attribute_int,
    get_driver_version,
    nvml_available,
    nvml_session,
    probe_device,
    probe_nvlink_matrix,
)

pytestmark = pytest.mark.unit


class FakeMemory:
    def __init__(self, total: int) -> None:
        self.total = total


class FakePci:
    def __init__(self, bus_id: bytes) -> None:
        self.busId = bus_id


class FakeNvml:
    NVML_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT = 16

    def __init__(self) -> None:
        self.init_calls = 0
        self.shutdown_calls = 0
        self.init_error: Exception | None = None
        self.shutdown_error: Exception | None = None

    def nvmlInit(self) -> None:
        self.init_calls += 1
        if self.init_error is not None:
            raise self.init_error

    def nvmlShutdown(self) -> None:
        self.shutdown_calls += 1
        if self.shutdown_error is not None:
            raise self.shutdown_error

    def nvmlDeviceGetName(self, handle: Any) -> bytes:
        return b"Fake GPU"

    def nvmlDeviceGetCudaComputeCapability(self, handle: Any) -> tuple[int, int]:
        return (9, 0)

    def nvmlDeviceGetMemoryInfo(self, handle: Any) -> FakeMemory:
        return FakeMemory(24 * 2**30)

    def nvmlDeviceGetUUID(self, handle: Any) -> bytes:
        return b"GPU-fake"

    def nvmlDeviceGetPciInfo(self, handle: Any) -> FakePci:
        return FakePci(b"00000000:01:00.0")

    def nvmlDeviceGetAttribute(self, handle: Any, attribute: Any) -> int:
        return 132


class TestAvailabilityAndSession:
    def test_nvml_available_delegates_to_has_module(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(nvml_utils, "has_module", lambda name: name == "pynvml")
        assert nvml_available()

    def test_missing_package_yields_none(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(nvml_utils, "nvml_available", lambda: False)
        with nvml_session() as handle:
            assert handle is None

    def test_successful_session_initializes_and_shuts_down(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fake = FakeNvml()
        monkeypatch.setattr(nvml_utils, "nvml_available", lambda: True)
        monkeypatch.setattr(nvml_utils, "nvml", fake)
        with nvml_session() as handle:
            assert handle is fake
        assert (fake.init_calls, fake.shutdown_calls) == (1, 1)

    def test_init_failure_yields_none_without_shutdown(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fake = FakeNvml()
        fake.init_error = RuntimeError("no driver")
        monkeypatch.setattr(nvml_utils, "nvml_available", lambda: True)
        monkeypatch.setattr(nvml_utils, "nvml", fake)
        with nvml_session() as handle:
            assert handle is None
        assert (fake.init_calls, fake.shutdown_calls) == (1, 0)

    def test_shutdown_runs_when_the_body_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        fake = FakeNvml()
        monkeypatch.setattr(nvml_utils, "nvml_available", lambda: True)
        monkeypatch.setattr(nvml_utils, "nvml", fake)
        with pytest.raises(ValueError, match="boom"), nvml_session():
            raise ValueError("boom")
        assert fake.shutdown_calls == 1

    def test_shutdown_error_is_suppressed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        fake = FakeNvml()
        fake.shutdown_error = RuntimeError("already gone")
        monkeypatch.setattr(nvml_utils, "nvml_available", lambda: True)
        monkeypatch.setattr(nvml_utils, "nvml", fake)
        with nvml_session() as handle:
            assert handle is fake


class TestQueries:
    def test_driver_version_bytes_and_strings(self) -> None:
        class BytesModule:
            def nvmlSystemGetDriverVersion(self) -> bytes:
                return b"580.1"

        class StringModule:
            def nvmlSystemGetDriverVersion(self) -> str:
                return "580.2"

        class BrokenModule:
            def nvmlSystemGetDriverVersion(self) -> str:
                raise RuntimeError("nope")

        assert get_driver_version(BytesModule()) == "580.1"
        assert get_driver_version(StringModule()) == "580.2"
        assert get_driver_version(BrokenModule()) == ""

    def test_attribute_int_success_and_failure(self) -> None:
        fake = FakeNvml()
        assert get_attribute_int(fake, 0, "NVML_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT") == 132
        assert get_attribute_int(fake, 0, "NOPE") == 0

        class Broken(FakeNvml):
            def nvmlDeviceGetAttribute(self, handle: Any, attribute: Any) -> int:
                raise RuntimeError("nope")

        assert get_attribute_int(Broken(), 0, "NVML_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT") == 0

    def test_probe_device(self) -> None:
        probe = probe_device(FakeNvml(), 0, handle=object())
        assert probe == NvmlDeviceRaw(
            index=0,
            name="Fake GPU",
            sm_major=9,
            sm_minor=0,
            num_sms=132,
            hbm_bytes=24 * 2**30,
            uuid="GPU-fake",
            pci_bus_id="00000000:01:00.0",
        )

    def test_probe_device_degrades_when_queries_fail(self) -> None:
        class Broken:
            NVML_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT = 16

            def nvmlDeviceGetName(self, handle: Any) -> bytes:
                raise RuntimeError("name")

            def nvmlDeviceGetCudaComputeCapability(self, handle: Any) -> tuple[int, int]:
                raise RuntimeError("cc")

            def nvmlDeviceGetMemoryInfo(self, handle: Any) -> FakeMemory:
                raise RuntimeError("mem")

            def nvmlDeviceGetUUID(self, handle: Any) -> bytes:
                raise RuntimeError("uuid")

            def nvmlDeviceGetPciInfo(self, handle: Any) -> FakePci:
                raise RuntimeError("pci")

            def nvmlDeviceGetAttribute(self, handle: Any, attribute: Any) -> int:
                raise RuntimeError("sms")

        probe = probe_device(Broken(), 1, handle=object())
        assert probe == NvmlDeviceRaw(
            index=1,
            name="",
            sm_major=0,
            sm_minor=0,
            num_sms=0,
            hbm_bytes=0,
            uuid="",
            pci_bus_id="",
        )


class TestNvlink:
    def test_matrix_marks_only_supported_pairs(self) -> None:
        class P2P:
            NVML_P2P_STATUS_OK = 0
            NVML_P2P_CAPS_INDEX_NVLINK = 1

            def nvmlDeviceGetP2PStatus(self, left: Any, right: Any, capability: Any) -> int:
                return 0 if (left, right) == (0, 1) else 1

        assert probe_nvlink_matrix(P2P(), [0, 1]) == [[False, True], [False, False]]

    def test_missing_p2p_api_returns_all_false(self) -> None:
        class NoP2P:
            pass

        assert probe_nvlink_matrix(NoP2P(), [0, 1]) == [[False, False], [False, False]]
