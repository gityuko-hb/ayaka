from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

import ayaka.utils.host_info as host_info
from ayaka.utils.host_info import CgroupMemory

if TYPE_CHECKING:
    from pathlib import Path

pytestmark = pytest.mark.unit


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="ascii")


@pytest.fixture
def isolated_paths(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(host_info, "_MEMINFO", tmp_path / "meminfo")
    monkeypatch.setattr(host_info, "_PRESSURE_MEMORY", tmp_path / "pressure")
    monkeypatch.setattr(host_info, "_CGROUP_V2", tmp_path / "cgroup")
    monkeypatch.setattr(host_info, "_CGROUP_V1_MEM", tmp_path / "cgroup_v1")
    monkeypatch.setattr(host_info, "_NUMA_BASE", tmp_path / "node")
    monkeypatch.setattr(host_info, "_PCI_DEVICES", tmp_path / "pci")
    return tmp_path


class TestMeminfo:
    def test_reads_total_and_available(self, isolated_paths: Path) -> None:
        _write(isolated_paths / "meminfo", "MemTotal: 1024 kB\nMemAvailable: 512 kB\n")
        assert host_info.host_ram_bytes() == 1024 * 1024
        assert host_info.host_available_bytes() == 512 * 1024

    def test_missing_file_degrades(self, isolated_paths: Path) -> None:
        assert host_info.host_ram_bytes() is None
        assert host_info.host_available_bytes() is None

    def test_malformed_lines_are_ignored(self, isolated_paths: Path) -> None:
        _write(isolated_paths / "meminfo", "MemTotal: not-a-number kB\nMemAvailable: 1 kB\n")
        assert host_info.host_ram_bytes() is None
        assert host_info.host_available_bytes() == 1024


class TestCgroup:
    def test_missing_hierarchies(self, isolated_paths: Path) -> None:
        assert host_info.cgroup_memory() == CgroupMemory()

    def test_v2_limit_and_current(self, isolated_paths: Path) -> None:
        _write(isolated_paths / "cgroup" / "memory.max", "2048\n")
        _write(isolated_paths / "cgroup" / "memory.current", "1024\n")
        assert host_info.cgroup_memory() == CgroupMemory(2048, 1024, "cgroup-v2")

    def test_v2_unlimited_is_none(self, isolated_paths: Path) -> None:
        _write(isolated_paths / "cgroup" / "memory.max", "max\n")
        _write(isolated_paths / "cgroup" / "memory.current", "1024\n")
        assert host_info.cgroup_memory() == CgroupMemory(None, 1024, "cgroup-v2")

    def test_v1_sentinel_is_unlimited(self, isolated_paths: Path) -> None:
        _write(
            isolated_paths / "cgroup_v1" / "memory.limit_in_bytes",
            f"{1 << 62}\n",
        )
        _write(isolated_paths / "cgroup_v1" / "memory.usage_in_bytes", "100\n")
        assert host_info.cgroup_memory() == CgroupMemory(None, 100, "cgroup-v1")

    def test_v1_real_limit(self, isolated_paths: Path) -> None:
        _write(isolated_paths / "cgroup_v1" / "memory.limit_in_bytes", "4096\n")
        _write(isolated_paths / "cgroup_v1" / "memory.usage_in_bytes", "100\n")
        assert host_info.cgroup_memory() == CgroupMemory(4096, 100, "cgroup-v1")

    def test_total_uses_the_smallest_limit(self, isolated_paths: Path) -> None:
        _write(isolated_paths / "meminfo", "MemTotal: 4096 kB\n")
        _write(isolated_paths / "cgroup" / "memory.max", "1048576\n")
        _write(isolated_paths / "cgroup" / "memory.current", "1\n")
        assert host_info.host_total_ram_bytes() == 1_048_576

    def test_total_falls_back_to_meminfo(self, isolated_paths: Path) -> None:
        _write(isolated_paths / "meminfo", "MemTotal: 4096 kB\n")
        _write(isolated_paths / "cgroup" / "memory.max", "max\n")
        _write(isolated_paths / "cgroup" / "memory.current", "1\n")
        assert host_info.host_total_ram_bytes() == 4096 * 1024

    def test_total_without_any_source(self, isolated_paths: Path) -> None:
        assert host_info.host_total_ram_bytes() is None


class TestNuma:
    def test_node_count(self, isolated_paths: Path) -> None:
        base = isolated_paths / "node"
        (base / "node0").mkdir(parents=True)
        (base / "node1").mkdir()
        (base / "node10").mkdir()
        (base / "not-a-node").mkdir()
        assert host_info.numa_node_count() == 3

    def test_node_count_unknown(self, isolated_paths: Path) -> None:
        assert host_info.numa_node_count() == 1

    def test_free_bytes_in_node_order(self, isolated_paths: Path) -> None:
        base = isolated_paths / "node"
        _write(base / "node0" / "meminfo", "Node 0 MemFree: 1024 kB\n")
        _write(base / "node1" / "meminfo", "Node 1 MemFree: 2048 kB\n")
        (base / "node2").mkdir()
        assert host_info.numa_free_bytes() == (1024 * 1024, 2048 * 1024, 0)

    def test_free_bytes_unknown(self, isolated_paths: Path) -> None:
        assert host_info.numa_free_bytes() == ()


class TestPci:
    @pytest.mark.parametrize(
        ("bus_id", "expected"),
        [
            ("0000:01:00.0", "0000:01:00.0"),
            ("00000000:01:00.0", "0000:01:00.0"),
            ("01:00.0", "0000:01:00.0"),
            ("0000:AF:1F.7", "0000:af:1f.7"),
        ],
    )
    def test_normalizes_domains(self, bus_id: str, expected: str) -> None:
        assert host_info._normalize_pci_bus_id(bus_id) == expected

    @pytest.mark.parametrize("bus_id", ["", "nope", "0000:01:00", "0000:zz:00.0", "0000:01:00.8"])
    def test_rejects_invalid_addresses(self, bus_id: str) -> None:
        assert host_info._normalize_pci_bus_id(bus_id) == ""

    def test_numa_node_reads_the_normalized_path(self, monkeypatch: pytest.MonkeyPatch) -> None:
        class FakePciRoot:
            def __init__(self) -> None:
                self.parts: list[str] = []

            def __truediv__(self, part: str) -> FakePciRoot:
                self.parts.append(part)
                return self

            def read_text(self, encoding: str) -> str:
                assert encoding == "ascii"
                return "1\n"

        root = FakePciRoot()
        monkeypatch.setattr(host_info, "_PCI_DEVICES", root)
        assert host_info.pci_numa_node("00000000:01:00.0") == 1
        assert root.parts == ["0000:01:00.0", "numa_node"]

    def test_numa_node_missing_returns_minus_one(self, isolated_paths: Path) -> None:
        assert host_info.pci_numa_node("0000:01:00.0") == -1


class TestMemoryPressure:
    def test_reads_some_and_full(self, isolated_paths: Path) -> None:
        _write(
            isolated_paths / "pressure",
            "some avg10=1.25 avg60=0.50 avg300=0.10 total=10\n"
            "full avg10=0.50 avg60=0.25 avg300=0.05 total=5\n",
        )
        assert host_info.memory_pressure() == (1.25, 0.5)

    def test_missing_file_degrades(self, isolated_paths: Path) -> None:
        assert host_info.memory_pressure() == (0.0, 0.0)

    def test_ignores_unrelated_lines(self, isolated_paths: Path) -> None:
        _write(isolated_paths / "pressure", "some avg10=2.0\n\nnot psi\n")
        assert host_info.memory_pressure() == (2.0, 0.0)
