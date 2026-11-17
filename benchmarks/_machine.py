"""What every benchmark result records about where it ran (QA-5): machine, versions, commit."""

from __future__ import annotations

import json
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
    dirty = subprocess.run(  # noqa: S603
        ["git", "-C", str(ROOT), "status", "--porcelain"],  # noqa: S607
        capture_output=True,
        text=True,
        check=False,
    ).stdout.strip()
    return result.stdout.strip() + ("-dirty" if dirty else "")


def machine() -> dict[str, Any]:
    """The GPU, the driver, the libraries, the OS (and whether it is WSL2) and the commit."""
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
        "driver": driver,
        "os": platform.platform(),
        "wsl2": running_on_wsl(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "flashinfer": _version("flashinfer-python"),
        "ayaka": _version("ayaka"),
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
