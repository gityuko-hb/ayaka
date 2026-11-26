"""``ayaka info`` and ``ayaka config show``: what this installation has, and what a run would use.

``info`` prints the version, the devices, the ops backends that can run here, the extras that are
installed, and the directories Ayaka reads and writes with their sizes. It loads torch to ask about
CUDA, but no model.

``config show`` prints the server settings ``ayaka serve`` would use with the same flags, each
with where it came from: a flag, the configuration file, or the default. The API key is never
printed, only whether one is set.
"""

from __future__ import annotations

import os
import platform
import sys
from dataclasses import asdict, dataclass, fields
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import TYPE_CHECKING, Final, cast

from ayaka.__version__ import __version__
from ayaka.configs.server import ServerConfig, user_config_path
from ayaka.utils.import_utils import has_module

if TYPE_CHECKING:
    from collections.abc import Mapping

__all__ = [
    "EXTRAS",
    "BackendLine",
    "DeviceLine",
    "InfoReport",
    "PathLine",
    "config_rows",
    "info_lines",
    "info_report",
]

#: What each extra brings, as the distribution names to look for.
EXTRAS: Final[Mapping[str, tuple[str, ...]]] = {
    "cpu / cu130": ("torch",),
    "flashinfer": ("flashinfer-python",),
    "serve": ("aiohttp",),
    "structured": ("llguidance",),
    "xgrammar": ("xgrammar",),
    "grpc": ("grpcio", "protobuf"),
    "hf": ("transformers", "tokenizers", "huggingface-hub"),
    "tiktoken": ("tiktoken", "jinja2"),
}

_MIB: Final = 2**20


@dataclass(frozen=True, slots=True)
class DeviceLine:
    """One device the probe found."""

    kind: str
    index: int
    name: str
    total_bytes: int
    free_bytes: int


@dataclass(frozen=True, slots=True)
class BackendLine:
    """One ops backend registered for this platform."""

    name: str
    devices: tuple[str, ...]
    families: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class PathLine:
    """A directory or file Ayaka uses."""

    name: str
    path: str
    exists: bool
    bytes: int | None


@dataclass(frozen=True, slots=True)
class InfoReport:
    """Everything ``ayaka info`` prints."""

    version: str
    python: str
    platform: str
    wsl2: bool
    torch: str | None
    torch_cuda: str | None
    cuda_usable: bool
    devices: tuple[DeviceLine, ...]
    backends: tuple[BackendLine, ...]
    extras: dict[str, dict[str, str | None]]
    paths: tuple[PathLine, ...]

    def as_json(self) -> dict[str, object]:
        """The report as JSON-ready data."""
        return asdict(self)


def _version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def _size(path: Path) -> int | None:
    """Bytes under a directory, or ``None`` when it does not exist or is a file."""
    if not path.is_dir():
        return None
    total = 0
    for root, _, files in os.walk(path):
        for name in files:
            item = Path(root) / name
            try:
                if not item.is_symlink():
                    total += item.stat().st_size
            except OSError:
                continue
    return total


def _hub_cache() -> Path:
    """The Hugging Face Hub cache, where the Hub client keeps downloaded models."""
    configured = os.environ.get("HF_HUB_CACHE")
    if configured:
        return Path(configured)
    home = os.environ.get("HF_HOME")
    base = Path(home) if home else Path.home() / ".cache" / "huggingface"
    return base / "hub"


def info_report() -> InfoReport:
    """Probe this installation."""
    from ayaka.device.capabilities import running_on_wsl
    from ayaka.device.info import probe_devices
    from ayaka.engine.profile import profile_directory
    from ayaka.ops.backend import available_backends

    torch_version: str | None = None
    torch_cuda: str | None = None
    usable = False
    if has_module("torch"):
        import torch

        torch_version = str(torch.__version__)
        torch_cuda = torch.version.cuda
        usable = bool(torch.cuda.is_available())
    config = user_config_path()
    return InfoReport(
        version=__version__,
        python=platform.python_version(),
        platform=platform.platform(),
        wsl2=running_on_wsl(),
        torch=torch_version,
        torch_cuda=torch_cuda,
        cuda_usable=usable,
        devices=tuple(
            DeviceLine(d.kind, d.index, d.name, d.total_bytes, d.free_bytes)
            for d in probe_devices()
        ),
        backends=tuple(
            BackendLine(
                caps.name,
                tuple(sorted(caps.devices)),
                tuple(sorted(str(family) for family in caps.families)),
            )
            for caps in available_backends()
            if sys.platform in caps.platforms
        ),
        extras={
            extra: {name: _version(name) for name in packages} for extra, packages in EXTRAS.items()
        },
        paths=tuple(
            PathLine(name, str(path), path.exists(), _size(path))
            for name, path in (
                ("startup profiles", profile_directory()),
                ("Hugging Face cache", _hub_cache()),
                ("configuration file", config),
            )
        ),
    )


def info_lines(report: InfoReport) -> list[str]:
    """The report as the lines ``ayaka info`` prints."""
    lines = [
        f"ayaka      {report.version} on Python {report.python}",
        f"platform   {report.platform}" + (" (WSL2)" if report.wsl2 else ""),
    ]
    if report.torch is None:
        lines.append("torch      not installed (uv sync --extra cpu, or --extra cu130 on a GPU)")
    else:
        usable = "usable" if report.cuda_usable else "not usable"
        lines.append(f"torch      {report.torch} (CUDA {report.torch_cuda or 'none'}, {usable})")
    for device in report.devices:
        memory = (
            f"{device.total_bytes / _MIB:,.0f} MiB, {device.free_bytes / _MIB:,.0f} MiB free"
            if device.total_bytes
            else "memory not readable on this OS"
        )
        lines.append(f"device     {device.kind}:{device.index} {device.name}: {memory}")
    for backend in report.backends:
        lines.append(
            f"backend    {backend.name} on {', '.join(backend.devices)}: "
            f"{', '.join(backend.families)}"
        )
    for extra, packages in report.extras.items():
        if not any(packages.values()):
            lines.append(f"extra      {extra}: not installed")
            continue
        found = [f"{name} {got or 'missing'}" for name, got in packages.items()]
        lines.append(f"extra      {extra}: {', '.join(found)}")
    for entry in report.paths:
        if not entry.exists:
            detail = "missing"
        elif entry.bytes is None:
            detail = "present"
        else:
            detail = f"{entry.bytes / _MIB:,.0f} MiB"
        lines.append(f"path       {entry.name}: {entry.path} ({detail})")
    return lines


def config_rows(
    flags: Mapping[str, object], file_values: Mapping[str, object], file: Path | None
) -> list[tuple[str, str, str]]:
    """``(key, value, source)`` for every server setting, as ``ayaka config show`` prints them.

    Args:
        flags: The values given as flags, by setting name.
        file_values: The ``[server]`` table of the configuration file.
        file: The file's path, or ``None`` when none was read.
    """
    defaults = ServerConfig()
    rows: list[tuple[str, str, str]] = []
    for item in fields(ServerConfig):
        name = item.name
        value: object
        if name in flags:
            value, source = flags[name], "flag"
        elif name in file_values:
            value, source = file_values[name], f"file {file}"
        else:
            value, source = getattr(defaults, name), "default"
        if name == "api_key":
            text = "set" if value else "not set"
        elif isinstance(value, (list, tuple)):
            items = cast("list[object] | tuple[object, ...]", value)
            text = ", ".join(str(element) for element in items) or "(none)"
        else:
            text = str(value)
        rows.append((name, text, source))
    return rows
