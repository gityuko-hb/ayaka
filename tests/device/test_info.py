# ruff: noqa: N802 - the fake must mirror pynvml's camelCase API

"""Device probing with a fake NVML and fake host readings: no GPU, no torch."""

from __future__ import annotations

import contextlib
import subprocess
import sys
from typing import TYPE_CHECKING, Any

import pytest

import ayaka.device.info as info
from ayaka.device.info import DeviceInfo, probe_devices

if TYPE_CHECKING:
    from collections.abc import Generator

pytestmark = pytest.mark.unit

GIB = 2**30


class _Memory:
    def __init__(self, total: int, free: int) -> None:
        self.total = total
        self.free = free


class _Pci:
    def __init__(self, bus_id: str) -> None:
        self.busId = bus_id


class FakeNvml:
    NVML_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT = 16

    def __init__(self, gpus: list[tuple[str, int, int]], *, memory_fails: bool = False) -> None:
        self.gpus = gpus
        self.memory_fails = memory_fails

    def nvmlDeviceGetCount(self) -> int:
        return len(self.gpus)

    def nvmlDeviceGetHandleByIndex(self, index: int) -> int:
        return index

    def nvmlDeviceGetName(self, handle: int) -> bytes:
        return self.gpus[handle][0].encode()

    def nvmlDeviceGetCudaComputeCapability(self, handle: int) -> tuple[int, int]:
        return (8, 6)

    def nvmlDeviceGetMemoryInfo(self, handle: int) -> _Memory:
        if self.memory_fails:
            raise RuntimeError("no memory info")
        _, total, free = self.gpus[handle]
        return _Memory(total, free)

    def nvmlDeviceGetUUID(self, handle: int) -> str:
        return f"GPU-{handle}"

    def nvmlDeviceGetPciInfo(self, handle: int) -> _Pci:
        return _Pci(f"00000000:0{handle}:00.0")

    def nvmlDeviceGetAttribute(self, handle: int, attr: int) -> int:
        return 20


def _install(monkeypatch: pytest.MonkeyPatch, session: Any, *, total: int, available: int) -> None:
    monkeypatch.setattr(info, "nvml_session", session)
    monkeypatch.setattr(info, "host_total_ram_bytes", lambda: total)
    monkeypatch.setattr(info, "host_available_bytes", lambda: available)


def _session_of(fake: FakeNvml | None) -> Any:
    @contextlib.contextmanager
    def session() -> Generator[Any]:
        yield fake

    return session


def test_gpus_come_first_in_nvml_order_and_the_host_last(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = FakeNvml([("RTX 3050", 4 * GIB, 3 * GIB), ("RTX 4090", 24 * GIB, 23 * GIB)])
    _install(monkeypatch, _session_of(fake), total=16 * GIB, available=9 * GIB)

    devices = probe_devices()

    assert [(d.kind, d.index, d.name) for d in devices] == [
        ("cuda", 0, "RTX 3050"),
        ("cuda", 1, "RTX 4090"),
        ("cpu", 0, "cpu"),
    ]
    gpu = devices[0]
    assert (gpu.total_bytes, gpu.free_bytes) == (4 * GIB, 3 * GIB)
    assert gpu.compute_capability == (8, 6)
    assert gpu.num_sms == 20
    assert gpu.uuid == "GPU-0"
    assert gpu.pci_bus_id == "00000000:00:00.0"
    assert gpu.is_cuda
    assert (devices[2].total_bytes, devices[2].free_bytes) == (16 * GIB, 9 * GIB)
    assert not devices[2].is_cuda


def test_without_nvml_the_host_is_the_only_device(monkeypatch: pytest.MonkeyPatch) -> None:
    _install(monkeypatch, _session_of(None), total=8 * GIB, available=2 * GIB)

    devices = probe_devices()

    assert devices == (
        DeviceInfo(kind="cpu", index=0, name="cpu", total_bytes=8 * GIB, free_bytes=2 * GIB),
    )


def test_unreadable_free_memory_degrades_to_zero(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = FakeNvml([("RTX 3050", 4 * GIB, 3 * GIB)], memory_fails=True)
    _install(monkeypatch, _session_of(fake), total=0, available=0)

    gpu, host = probe_devices()

    assert gpu.free_bytes == 0
    assert (host.total_bytes, host.free_bytes) == (0, 0)


def test_host_readings_that_are_missing_become_zero(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(info, "nvml_session", _session_of(None))
    monkeypatch.setattr(info, "host_total_ram_bytes", lambda: None)
    monkeypatch.setattr(info, "host_available_bytes", lambda: None)

    (host,) = probe_devices()

    assert (host.total_bytes, host.free_bytes) == (0, 0)


def test_available_memory_is_capped_at_the_total(monkeypatch: pytest.MonkeyPatch) -> None:
    # A container limit below the host's MemAvailable must not yield free > total.
    _install(monkeypatch, _session_of(None), total=2 * GIB, available=10 * GIB)

    (host,) = probe_devices()

    assert host.free_bytes == 2 * GIB


def test_a_failing_device_count_means_no_gpus(monkeypatch: pytest.MonkeyPatch) -> None:
    class Broken(FakeNvml):
        def nvmlDeviceGetCount(self) -> int:
            raise RuntimeError("driver gone")

    _install(monkeypatch, _session_of(Broken([])), total=GIB, available=GIB)

    assert [d.kind for d in probe_devices()] == ["cpu"]


def test_a_gpu_whose_handle_fails_is_skipped(monkeypatch: pytest.MonkeyPatch) -> None:
    class Flaky(FakeNvml):
        def nvmlDeviceGetHandleByIndex(self, index: int) -> int:
            if index == 0:
                raise RuntimeError("lost")
            return index

    fake = Flaky([("a", GIB, GIB), ("b", GIB, GIB)])
    _install(monkeypatch, _session_of(fake), total=GIB, available=GIB)

    assert [(d.kind, d.index) for d in probe_devices()] == [("cuda", 1), ("cpu", 0)]


def test_a_nameless_gpu_gets_a_placeholder_name(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = FakeNvml([("", GIB, GIB)])
    _install(monkeypatch, _session_of(fake), total=GIB, available=GIB)

    assert probe_devices()[0].name == "cuda:0"


@pytest.mark.parametrize(
    ("overrides", "error"),
    [
        ({"kind": "tpu"}, ValueError),
        ({"index": -1}, ValueError),
        ({"name": " "}, ValueError),
        ({"total_bytes": 1.5}, TypeError),
        ({"free_bytes": 2 * GIB, "total_bytes": GIB}, ValueError),
        ({"num_sms": -1}, ValueError),
    ],
)
def test_device_info_validates_its_fields(
    overrides: dict[str, Any], error: type[Exception]
) -> None:
    fields: dict[str, Any] = {
        "kind": "cuda",
        "index": 0,
        "name": "gpu",
        "total_bytes": GIB,
        "free_bytes": GIB,
    }
    fields.update(overrides)

    with pytest.raises(error):
        DeviceInfo(**fields)


def test_the_real_probe_returns_the_host_and_never_loads_torch() -> None:
    program = (
        "import sys\n"
        "from ayaka.device import probe_devices\n"
        "devices = probe_devices()\n"
        "assert devices[-1].kind == 'cpu'\n"
        "assert all(d.free_bytes <= d.total_bytes for d in devices)\n"
        "assert 'torch' not in sys.modules, 'torch was imported'\n"
    )

    result = subprocess.run(  # noqa: S603  # fixed argv, no input from outside
        [sys.executable, "-I", "-c", program], capture_output=True, text=True, check=False
    )

    assert result.returncode == 0, result.stderr
