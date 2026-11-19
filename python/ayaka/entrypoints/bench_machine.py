"""The machine an ``ayaka bench`` run measured: versions, the commit, power, GPU and memory.

Everything here is read, never changed: :func:`machine_info` for ``results.json``,
:func:`power_source` and :func:`gpu_state` at the start and the end of a run, and
:class:`MemorySampler`, a thread that keeps the peak VRAM and RAM while the run goes.
"""

from __future__ import annotations

import contextlib
import logging
import os
import platform
import re
import subprocess
import sys
import threading
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import TYPE_CHECKING

from ayaka.__version__ import __version__

if TYPE_CHECKING:
    from collections.abc import Callable

__all__ = ["MemorySampler", "gpu_state", "machine_info", "machine_slug", "power_source"]

_LOG = logging.getLogger(__name__)


def _version_of(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def _git(*args: str) -> str | None:
    root = Path(__file__).resolve().parents[3]
    if not (root / ".git").exists():
        return None
    try:
        result = subprocess.run(  # noqa: S603  # a fixed argv
            ["git", "-C", str(root), *args],  # noqa: S607  # git from PATH
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip()


def _dirty() -> bool | None:
    """Whether the tree differs from its commit; line endings alone (a Windows checkout's CRLF)
    do not count, nor do untracked files."""
    root = Path(__file__).resolve().parents[3]
    if not (root / ".git").exists():
        return None
    try:
        result = subprocess.run(  # noqa: S603  # a fixed argv
            ["git", "-C", str(root), "diff", "--quiet", "--ignore-cr-at-eol", "HEAD", "--"],  # noqa: S607
            capture_output=True,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return {0: False, 1: True}.get(result.returncode)


def _running_on_wsl() -> bool:
    try:
        return "microsoft" in Path("/proc/version").read_text(encoding="utf-8").lower()
    except OSError:
        return False


def power_source(
    *,
    platform_name: str | None = None,
    sysfs: Path = Path("/sys/class/power_supply"),
    wsl: bool | None = None,
    windows_status: Callable[[], bool | None] | None = None,
) -> str:
    """``"ac"``, ``"battery"`` or ``"unknown"``.

    Linux reads ``/sys/class/power_supply``; WSL2 has none there and asks Windows through
    ``powershell.exe``; Windows asks ``GetSystemPowerStatus``.
    """
    name = platform_name or sys.platform
    if name == "win32" or (wsl if wsl is not None else _running_on_wsl()):
        online = (windows_status or _windows_on_ac)()
        return "unknown" if online is None else ("ac" if online else "battery")
    if not sysfs.is_dir():
        return "unknown"
    mains = [entry for entry in sysfs.iterdir() if _read(entry / "type") == "Mains"]
    batteries = [entry for entry in sysfs.iterdir() if _read(entry / "type") == "Battery"]
    if any(_read(entry / "online") == "1" for entry in mains):
        return "ac"
    if batteries:
        return "battery"
    return "ac" if not mains else "unknown"  # a machine with neither is a desktop on mains


def _read(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8").strip()
    except OSError:
        return None


def _windows_on_ac() -> bool | None:
    if sys.platform == "win32":
        import ctypes

        class _Status(ctypes.Structure):
            _fields_ = (  # the ctypes layout
                ("ac_line", ctypes.c_byte),
                ("battery_flag", ctypes.c_byte),
                ("battery_percent", ctypes.c_byte),
                ("system_status", ctypes.c_byte),
                ("battery_life", ctypes.c_ulong),
                ("battery_full_life", ctypes.c_ulong),
            )

        status = _Status()
        kernel = getattr(ctypes, "windll").kernel32  # noqa: B009  # windll exists on Windows only
        if not kernel.GetSystemPowerStatus(ctypes.byref(status)):
            return None
        return {0: False, 1: True}.get(int(status.ac_line))
    try:  # WSL2: Windows answers through interop
        result = subprocess.run(
            [  # noqa: S607  # Windows' own PowerShell through WSL interop
                "powershell.exe",
                "-NoProfile",
                "-Command",
                "Add-Type -AssemblyName System.Windows.Forms;"
                "[System.Windows.Forms.SystemInformation]::PowerStatus.PowerLineStatus",
            ],
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    answer = result.stdout.strip().lower()
    return {"online": True, "offline": False}.get(answer)


def gpu_state() -> dict[str, object]:
    """Temperature, memory and driver of GPU 0 from NVML; empty without one."""
    from ayaka.utils.nvml_utils import get_driver_version, nvml_session

    with nvml_session() as nvml:
        if nvml is None:
            return {}
        try:
            handle = nvml.nvmlDeviceGetHandleByIndex(0)
            memory = nvml.nvmlDeviceGetMemoryInfo(handle)
            return {
                "temperature_c": int(nvml.nvmlDeviceGetTemperature(handle, 0)),
                "memory_used_bytes": int(memory.used),
                "memory_total_bytes": int(memory.total),
                "driver": get_driver_version(nvml),
            }
        except Exception:  # NVML differs between drivers; a missing value is not an error
            _LOG.debug("bench: NVML did not answer", exc_info=True)
            return {}


def machine_slug(gpu: str | None) -> str:
    """A name for the results folder from the GPU, never the host name: ``rtx-3050-laptop``."""
    if not gpu:
        return "cpu"
    words = re.sub(r"[^a-z0-9]+", "-", gpu.lower()).strip("-")
    return re.sub(r"^(nvidia-)?(geforce-)?", "", words) or "gpu"


def machine_info() -> dict[str, object]:
    import torch

    cuda = torch.cuda.is_available()
    free, total = torch.cuda.mem_get_info() if cuda else (None, None)
    commit = _git("rev-parse", "HEAD")
    return {
        "gpu": torch.cuda.get_device_name(0) if cuda else None,
        "capability": list(torch.cuda.get_device_capability(0)) if cuda else None,
        "vram_free_bytes": free,
        "vram_total_bytes": total,
        "os": platform.platform(),
        "wsl2": _running_on_wsl(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "flashinfer": _version_of("flashinfer-python"),
        "llguidance": _version_of("llguidance"),
        "transformers": _version_of("transformers"),
        "ayaka": {
            "version": __version__,
            "commit": commit,
            "dirty": _dirty(),
        },
    }


def _rss_bytes() -> int | None:
    """The process's resident memory now."""
    if sys.platform == "win32":
        import ctypes
        from ctypes import wintypes

        class _Counters(ctypes.Structure):
            _fields_ = (  # the ctypes layout
                ("cb", wintypes.DWORD),
                ("page_faults", wintypes.DWORD),
                ("peak_working_set", ctypes.c_size_t),
                ("working_set", ctypes.c_size_t),
                ("quota_peak_paged", ctypes.c_size_t),
                ("quota_paged", ctypes.c_size_t),
                ("quota_peak_nonpaged", ctypes.c_size_t),
                ("quota_nonpaged", ctypes.c_size_t),
                ("pagefile", ctypes.c_size_t),
                ("peak_pagefile", ctypes.c_size_t),
            )

        counters = _Counters()
        counters.cb = ctypes.sizeof(counters)
        windll = getattr(ctypes, "windll")  # noqa: B009  # windll exists on Windows only
        process = windll.kernel32.GetCurrentProcess()
        if not windll.psapi.GetProcessMemoryInfo(process, ctypes.byref(counters), counters.cb):
            return None
        return int(counters.working_set)
    try:
        pages = int(Path("/proc/self/statm").read_text(encoding="utf-8").split()[1])
    except (OSError, ValueError, IndexError):
        return None
    return pages * os.sysconf("SC_PAGE_SIZE")


class MemorySampler:
    """Samples VRAM (NVML, the whole device) and RAM (this process) on a thread."""

    def __init__(self, interval: float = 0.1) -> None:
        self.interval = interval
        self.peak_vram: int | None = None
        self.peak_rss: int | None = None
        self.samples = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def __enter__(self) -> MemorySampler:
        self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._stop.set()
        self._thread.join()

    def _run(self) -> None:
        from ayaka.utils.nvml_utils import nvml_session

        with nvml_session() as nvml:
            handle = None
            if nvml is not None:
                with contextlib.suppress(Exception):
                    handle = nvml.nvmlDeviceGetHandleByIndex(0)
            while True:
                if handle is not None and nvml is not None:
                    with contextlib.suppress(Exception):
                        used = int(nvml.nvmlDeviceGetMemoryInfo(handle).used)
                        self.peak_vram = max(self.peak_vram or 0, used)
                rss = _rss_bytes()
                if rss is not None:
                    self.peak_rss = max(self.peak_rss or 0, rss)
                self.samples += 1
                if self._stop.wait(self.interval):
                    return
