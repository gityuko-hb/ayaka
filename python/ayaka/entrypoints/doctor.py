"""``ayaka doctor``: environment, memory, toolchain and profile diagnostics.

Runs targeted read-only checks and prints ``pass``, ``warn`` or ``fail`` per
item along with one concrete actionable fix for every non-passing check.
Nothing here loads a model or imports ``torch`` at module import time.
"""

from __future__ import annotations

import ctypes
import importlib.metadata
import json
import os
import shutil
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Final, Literal

from ayaka.device.capabilities import running_on_wsl
from ayaka.device.info import probe_devices
from ayaka.engine.profile import StartupProfile, profile_directory
from ayaka.entrypoints.console import EXIT_BUG, EXIT_OK
from ayaka.utils.host_info import (
    CgroupMemory,
    cgroup_memory,
    host_available_bytes,
    host_total_ram_bytes,
    memory_pressure,
)
from ayaka.utils.import_utils import has_module

if TYPE_CHECKING:
    import argparse
    from collections.abc import Callable

    from ayaka.device.info import DeviceInfo

__all__ = [
    "CheckStatus",
    "DoctorCheck",
    "DoctorProbes",
    "DoctorReport",
    "doctor_command",
    "doctor_lines",
    "run_doctor",
]

type CheckStatus = Literal["pass", "warn", "fail"]

_MIB: Final = 2**20
_GIB: Final = 2**30
_GAP_ABS_THRESHOLD: Final = 512 * _MIB
_GAP_RATIO_THRESHOLD: Final = 0.15
_MIN_HOST_RAM_WARN: Final = 2 * _GIB
_MIN_HOST_RAM_FAIL: Final = 512 * _MIB
_MIN_MEMLOCK_WARN: Final = 64 * _MIB
_PSI_FULL_WARN: Final = 5.0


@dataclass(frozen=True, slots=True)
class DoctorCheck:
    """Result of one diagnostic check.

    Attributes:
        name: Short machine-readable check identifier.
        status: ``"pass"``, ``"warn"`` or ``"fail"``.
        summary: One-line finding.
        detail: Optional extra context (measurements, paths).
        fix: Concrete remediation required when ``status`` is ``"warn"`` or ``"fail"``.
    """

    name: str
    status: CheckStatus
    summary: str
    detail: str = ""
    fix: str | None = None

    def __post_init__(self) -> None:
        if self.status not in ("pass", "warn", "fail"):
            raise ValueError(f"status must be pass, warn or fail, got {self.status!r}")
        if self.status != "pass" and (not self.fix or not self.fix.strip()):
            raise ValueError(f"check {self.name!r} with status {self.status!r} needs a fix")


@dataclass(frozen=True, slots=True)
class DoctorReport:
    """Collected output of every check ``ayaka doctor`` ran."""

    checks: tuple[DoctorCheck, ...]

    @property
    def ok(self) -> bool:
        """Whether zero checks failed (warnings are tolerated)."""
        return all(check.status != "fail" for check in self.checks)

    @property
    def has_warnings(self) -> bool:
        """Whether at least one check emitted a warning."""
        return any(check.status == "warn" for check in self.checks)

    def as_json(self) -> dict[str, object]:
        """Serialize the report for ``--output json``."""
        return {
            "ok": self.ok,
            "checks": [asdict(check) for check in self.checks],
        }


def _default_torch_cuda() -> tuple[str | None, str | None, bool, tuple[tuple[int, int, int], ...]]:
    """Return ``(torch_version, torch_cuda, cuda_available, ((index, free, total), ...))``."""
    if not has_module("torch"):
        return None, None, False, ()
    import torch

    usable = bool(torch.cuda.is_available())
    memories: list[tuple[int, int, int]] = []
    if usable:
        for index in range(torch.cuda.device_count()):
            try:
                free, total = torch.cuda.mem_get_info(index)
                memories.append((index, int(free), int(total)))
            except Exception:  # noqa: S112 - inaccessible device is omitted from CUDA readings
                continue
    return str(torch.__version__), torch.version.cuda, usable, tuple(memories)


def _default_host_ram() -> tuple[int | None, int | None]:
    """Host ``(total_bytes, available_bytes)`` across Linux and Windows."""
    total = host_total_ram_bytes()
    avail = host_available_bytes()
    if (total is None or avail is None) and sys.platform == "win32":

        class _MemoryStatus(ctypes.Structure):
            _fields_ = (
                ("length", ctypes.c_ulong),
                ("memory_load", ctypes.c_ulong),
                ("total_phys", ctypes.c_ulonglong),
                ("avail_phys", ctypes.c_ulonglong),
                ("total_page_file", ctypes.c_ulonglong),
                ("avail_page_file", ctypes.c_ulonglong),
                ("total_virtual", ctypes.c_ulonglong),
                ("avail_virtual", ctypes.c_ulonglong),
                ("avail_extended_virtual", ctypes.c_ulonglong),
            )

        status = _MemoryStatus()
        status.length = ctypes.sizeof(_MemoryStatus)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            total = int(status.total_phys)
            avail = int(status.avail_phys)
    return total, avail


def _default_memlock_limit() -> int | None:
    """Soft ``RLIMIT_MEMLOCK`` limit in bytes on POSIX, or ``None`` when unlimited/unavailable."""
    if sys.platform == "win32":
        return None
    import resource

    try:
        soft, _ = resource.getrlimit(resource.RLIMIT_MEMLOCK)
    except (OSError, ValueError):
        return None
    if soft < 0 or soft == getattr(resource, "RLIM_INFINITY", -1):
        return None
    return int(soft)


def _default_package_installed(name: str) -> bool:
    try:
        importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return False
    return True


def _default_disk_free(path: Path) -> int | None:
    candidate = path
    while not candidate.exists() and candidate.parent != candidate:
        candidate = candidate.parent
    try:
        return int(shutil.disk_usage(candidate).free)
    except OSError:
        return None


def _hub_cache_dir() -> Path:
    configured = os.environ.get("HF_HUB_CACHE")
    if configured:
        return Path(configured).expanduser()
    home = os.environ.get("HF_HOME")
    base = Path(home).expanduser() if home else Path.home() / ".cache" / "huggingface"
    return base / "hub"


@dataclass(frozen=True, slots=True)
class DoctorProbes:
    """Injectable system probes used by :func:`run_doctor`."""

    devices: Callable[[], tuple[DeviceInfo, ...]] = probe_devices
    torch_cuda: Callable[
        [], tuple[str | None, str | None, bool, tuple[tuple[int, int, int], ...]]
    ] = _default_torch_cuda
    wsl2: Callable[[], bool] = running_on_wsl
    platform: str = field(default_factory=lambda: sys.platform)
    host_ram: Callable[[], tuple[int | None, int | None]] = _default_host_ram
    cgroup: Callable[[], CgroupMemory] = cgroup_memory
    psi_memory: Callable[[], tuple[float, float]] = memory_pressure
    memlock_limit: Callable[[], int | None] = _default_memlock_limit
    package_installed: Callable[[str], bool] = _default_package_installed
    which: Callable[[str], str | None] = shutil.which
    profile_dir: Callable[[], Path] = profile_directory
    hub_cache_dir: Callable[[], Path] = _hub_cache_dir
    disk_free: Callable[[Path], int | None] = _default_disk_free


def _check_cuda_driver(
    devices: tuple[DeviceInfo, ...],
    torch_version: str | None,
    torch_cuda: str | None,
    cuda_usable: bool,
) -> DoctorCheck:
    cuda_devices = [d for d in devices if d.kind == "cuda"]
    if torch_version is None:
        fix = (
            "run `uv sync --extra cu130` for CUDA or `uv sync --extra cpu` for CPU"
            if cuda_devices
            else "run `uv sync --extra cpu` (or `--extra cu130` on an NVIDIA GPU)"
        )
        return DoctorCheck("cuda_driver", "fail", "PyTorch is not installed", fix=fix)
    if cuda_devices and not cuda_usable:
        names = ", ".join(d.name for d in cuda_devices)
        return DoctorCheck(
            "cuda_driver",
            "fail",
            f"NVIDIA GPU detected ({names}) but PyTorch CUDA is not usable",
            detail=f"torch {torch_version} (CUDA {torch_cuda or 'none'})",
            fix=(
                "install the CUDA PyTorch build with `uv sync --extra cu130` "
                "and verify `nvidia-smi`"
            ),
        )
    if not cuda_devices and not cuda_usable:
        return DoctorCheck(
            "cuda_driver",
            "warn",
            "no NVIDIA GPU detected; CPU execution only",
            detail=f"torch {torch_version}",
            fix="check `nvidia-smi` and driver visibility if an NVIDIA GPU is expected",
        )
    summary = f"CUDA {torch_cuda or 'unknown'} usable ({len(cuda_devices)} GPU)"
    detail = ", ".join(
        f"cuda:{d.index} {d.name} ({d.total_bytes / _MIB:,.0f} MiB)" for d in cuda_devices
    )
    return DoctorCheck("cuda_driver", "pass", summary, detail=detail)


def _check_nvml_cuda_gap(
    devices: tuple[DeviceInfo, ...],
    cuda_memories: tuple[tuple[int, int, int], ...],
    *,
    is_wsl2: bool,
) -> DoctorCheck:
    if not cuda_memories:
        return DoctorCheck("nvml_cuda_gap", "pass", "no active CUDA device to compare")
    nvml_by_index = {d.index: d for d in devices if d.kind == "cuda"}
    worst_gap = 0
    worst_detail = ""
    for index, cuda_free, cuda_total in cuda_memories:
        nvml = nvml_by_index.get(index)
        nvml_free = nvml.free_bytes if nvml is not None else cuda_free
        total = nvml.total_bytes if nvml is not None and nvml.total_bytes else cuda_total
        if cuda_free <= 0:
            return DoctorCheck(
                "nvml_cuda_gap",
                "fail",
                f"cuda:{index} reports 0 bytes CUDA free memory",
                detail=f"NVML free {nvml_free / _MIB:,.0f} MiB, CUDA free 0 MiB",
                fix="close competing GPU processes or lower `--memory-ratio` (e.g. `0.80`)",
            )
        gap = max(nvml_free - cuda_free, 0)
        threshold = max(_GAP_ABS_THRESHOLD, int(total * _GAP_RATIO_THRESHOLD))
        if gap > threshold and gap >= worst_gap:
            worst_gap = gap
            worst_detail = (
                f"cuda:{index} NVML free {nvml_free / _MIB:,.0f} MiB vs "
                f"CUDA free {cuda_free / _MIB:,.0f} MiB (gap {gap / _MIB:,.0f} MiB)"
            )
    if worst_gap > 0:
        env = "WSL2/WDDM oversubscription risk" if is_wsl2 else "driver-vs-CUDA free gap"
        return DoctorCheck(
            "nvml_cuda_gap",
            "warn",
            f"{env}: {worst_gap / _MIB:,.0f} MiB discrepancy",
            detail=worst_detail,
            fix=(
                "start with a lower `--memory-ratio` (e.g. `0.80`) "
                "so planning stays inside CUDA free memory"
            ),
        )
    return DoctorCheck("nvml_cuda_gap", "pass", "NVML and CUDA free memory are aligned")


def _check_host_ram(probes: DoctorProbes, *, is_wsl2: bool) -> DoctorCheck:
    total, available = probes.host_ram()
    cgroup = probes.cgroup()
    _, psi_full = probes.psi_memory()
    memlock = probes.memlock_limit()
    if available is not None and available < _MIN_HOST_RAM_FAIL:
        fix = (
            "close heavy processes or increase `[wsl2] memory=` in `%UserProfile%\\.wslconfig`"
            if is_wsl2
            else "free host RAM or raise container cgroup memory limits before loading models"
        )
        return DoctorCheck(
            "host_pinned_ram",
            "fail",
            f"critically low available host RAM ({available / _MIB:,.0f} MiB)",
            detail=f"total={total / _GIB:.1f} GiB" if total else "",
            fix=fix,
        )
    if memlock is not None and memlock < _MIN_MEMLOCK_WARN:
        return DoctorCheck(
            "host_pinned_ram",
            "warn",
            f"RLIMIT_MEMLOCK is restricted ({memlock / _MIB:,.0f} MiB)",
            fix=(
                "raise locked-memory limits with `ulimit -l unlimited` "
                "before enabling host KV pinning"
            ),
        )
    if psi_full > _PSI_FULL_WARN:
        return DoctorCheck(
            "host_pinned_ram",
            "warn",
            f"host memory pressure is elevated (PSI full avg10={psi_full:.1f}%)",
            fix="close background memory consumers or reduce host KV cache size",
        )
    if available is not None and available < _MIN_HOST_RAM_WARN:
        return DoctorCheck(
            "host_pinned_ram",
            "warn",
            f"low available host RAM ({available / _GIB:.2f} GiB)",
            fix="keep host KV tiering disabled or cap `--host-kv-max-mb` conservatively",
        )
    parts: list[str] = []
    if available is not None and total is not None:
        parts.append(f"{available / _GIB:.1f} GiB available / {total / _GIB:.1f} GiB total")
    if cgroup.limit_bytes is not None:
        parts.append(f"{cgroup.flavour} limit {cgroup.limit_bytes / _GIB:.1f} GiB")
    return DoctorCheck(
        "host_pinned_ram",
        "pass",
        "host RAM headroom is healthy",
        detail=", ".join(parts),
    )


def _check_flashinfer(probes: DoctorProbes) -> DoctorCheck:
    installed = probes.package_installed("flashinfer-python")
    if not installed:
        return DoctorCheck(
            "flashinfer_nvcc",
            "pass",
            "flashinfer-python is not installed (using torch backend)",
        )
    if probes.platform != "linux":
        return DoctorCheck(
            "flashinfer_nvcc",
            "warn",
            f"flashinfer-python is installed on unsupported platform {probes.platform!r}",
            fix="pass `--backends torch` or run inside Linux / WSL2",
        )
    nvcc = probes.which("nvcc")
    if nvcc is None:
        return DoctorCheck(
            "flashinfer_nvcc",
            "fail",
            "flashinfer-python is installed but `nvcc` was not found on PATH",
            fix=(
                "add CUDA toolkit binaries to PATH "
                "(`export PATH=/usr/local/cuda/bin:$PATH`) or pass `--backends torch`"
            ),
        )
    return DoctorCheck("flashinfer_nvcc", "pass", f"flashinfer ready with nvcc at {nvcc}")


def _check_disk_space(probes: DoctorProbes, *, min_disk_gib: float) -> DoctorCheck:
    targets = (
        ("startup profiles", probes.profile_dir()),
        ("Hugging Face cache", probes.hub_cache_dir()),
    )
    threshold = int(max(min_disk_gib, 0.1) * _GIB)
    low: list[tuple[str, Path, int]] = []
    details: list[str] = []
    for label, path in targets:
        free = probes.disk_free(path)
        if free is None:
            continue
        details.append(f"{label}: {free / _GIB:.1f} GiB free ({path})")
        if free < threshold:
            low.append((label, path, free))
    if not low:
        return DoctorCheck(
            "cache_disk_space",
            "pass",
            f"cache volumes have >= {min_disk_gib:g} GiB free",
            detail="; ".join(details),
        )
    label, path, free = min(low, key=lambda item: item[2])
    status: CheckStatus = "fail" if free < _GIB else "warn"
    return DoctorCheck(
        "cache_disk_space",
        status,
        f"low disk space for {label}: {free / _GIB:.2f} GiB free",
        detail=str(path),
        fix="free disk space or set `HF_HOME` / `AYAKA_CACHE_DIR` to a larger volume",
    )


def _check_startup_profiles(probes: DoctorProbes) -> DoctorCheck:
    directory = probes.profile_dir()
    if not directory.is_dir():
        return DoctorCheck("startup_profiles", "pass", "no startup profiles saved yet")
    files = sorted(directory.glob("*.json"))
    if not files:
        return DoctorCheck("startup_profiles", "pass", "no startup profiles saved yet")
    bad: list[Path] = []
    for path in files:
        try:
            StartupProfile.from_json(path.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError, KeyError):
            bad.append(path)
    if bad:
        names = ", ".join(str(p) for p in bad)
        return DoctorCheck(
            "startup_profiles",
            "fail",
            f"{len(bad)} invalid startup profile file(s)",
            detail=names,
            fix=f"remove the corrupt startup profile file(s): {names}",
        )
    return DoctorCheck(
        "startup_profiles",
        "pass",
        f"{len(files)} startup profile(s) valid",
        detail=str(directory),
    )


def run_doctor(
    *,
    min_disk_gib: float = 5.0,
    probes: DoctorProbes | None = None,
) -> DoctorReport:
    """Run every diagnostic check and return a structured :class:`DoctorReport`."""
    active = probes or DoctorProbes()
    devices = active.devices()
    torch_version, torch_cuda, cuda_usable, cuda_memories = active.torch_cuda()
    is_wsl2 = active.wsl2()
    checks = (
        _check_cuda_driver(devices, torch_version, torch_cuda, cuda_usable),
        _check_nvml_cuda_gap(devices, cuda_memories, is_wsl2=is_wsl2),
        _check_host_ram(active, is_wsl2=is_wsl2),
        _check_flashinfer(active),
        _check_disk_space(active, min_disk_gib=min_disk_gib),
        _check_startup_profiles(active),
    )
    return DoctorReport(checks=checks)


def doctor_lines(report: DoctorReport) -> list[str]:
    """Render ``report`` into human-readable lines for ``ayaka doctor``."""
    lines: list[str] = []
    for check in report.checks:
        tag = check.status.upper()
        lines.append(f"[{tag:<4}] {check.name}: {check.summary}")
        if check.detail:
            lines.append(f"       detail: {check.detail}")
        if check.fix:
            lines.append(f"       fix: {check.fix}")
    return lines


def doctor_command(args: argparse.Namespace) -> int:
    """Execute ``ayaka doctor`` and return ``0`` when all checks pass/warn, ``1`` on failure."""
    report = run_doctor(min_disk_gib=float(args.min_disk_gib))
    if args.output == "json":
        print(json.dumps(report.as_json(), indent=2))
    else:
        for line in doctor_lines(report):
            print(line)
    return EXIT_OK if report.ok else EXIT_BUG
