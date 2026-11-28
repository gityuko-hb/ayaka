"""What every benchmark result records about where it ran: machine, versions, commit."""

from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


def _version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def _commit() -> str | None:
    try:
        result = subprocess.run(  # noqa: S603  # fixed argv
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"],  # noqa: S607
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    # Line endings alone (a copy of a Windows checkout) are not a change; untracked files neither.
    dirty = subprocess.run(  # noqa: S603
        ["git", "-C", str(ROOT), "diff", "--quiet", "--ignore-cr-at-eol", "HEAD", "--"],  # noqa: S607
        capture_output=True,
        check=False,
    ).returncode
    return result.stdout.strip() + ("-dirty" if dirty else "")


def _cpu() -> str:
    """The CPU's model name (``/proc/cpuinfo`` on Linux), else what ``platform`` reports."""
    try:
        for line in Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor()


def machine() -> dict[str, Any]:
    """The GPU and its memory, the driver, the libraries, the OS (and whether it is WSL2), the CPU
    and the commit."""
    import torch

    from ayaka.device.capabilities import running_on_wsl
    from ayaka.utils.nvml_utils import get_driver_version, nvml_session

    with nvml_session() as nvml:
        driver = get_driver_version(nvml) if nvml is not None else None
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    return {
        "date": datetime.now(UTC).isoformat(timespec="seconds"),
        "gpu": gpu,
        "capability": list(torch.cuda.get_device_capability(0)) if gpu else None,
        "vram_bytes": torch.cuda.get_device_properties(0).total_memory if gpu else None,
        "driver": driver,
        "os": platform.platform(),
        "cpu": _cpu(),
        "cpu_count": os.cpu_count(),
        "wsl2": running_on_wsl(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "flashinfer": _version("flashinfer-python"),
        "ayaka": _version("ayaka-llm"),
        "commit": _commit(),
    }


def write(result: dict[str, Any], out: str | None) -> None:
    """Print ``result`` as JSON, and write it to ``out`` too when given."""
    text = json.dumps(result, indent=2)
    print(text)
    if out:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        Path(out).write_text(text + "\n", encoding="utf-8")
        print(f"wrote {out}", file=sys.stderr)
