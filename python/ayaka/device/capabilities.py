"""What one device can run: dtypes, CUDA graphs, platform; checked once, at start-up.

:func:`device_capabilities` turns a :class:`~ayaka.device.info.DeviceInfo` into a
:class:`DeviceCapabilities` record, and :meth:`DeviceCapabilities.check` refuses a
:class:`~ayaka.configs.device.DeviceConfig` that asks for something the device
does not have, listing every problem at once. Layers never test a compute
capability themselves; this record is the one place that does. Which ops
backend serves which family is ``ayaka.ops.backend.load_backend``'s question;
the platform recorded here (``linux``, ``win32``; WSL2 or not) is what it, and
``ayaka info``, report next to it.

The architecture is passed on as it is (``"sm_86"``), never rounded to a family:
a backend decides what it supports for an exact architecture.

Nothing here imports torch.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import TYPE_CHECKING, Final

from ayaka.core.errors import ConfigError
from ayaka.utils.dtypes import DType

if TYPE_CHECKING:
    from ayaka.configs.device import DeviceConfig
    from ayaka.device.info import DeviceInfo, DeviceKind

__all__ = ["DeviceCapabilities", "device_capabilities", "running_on_wsl"]

#: The first architecture with bf16 arithmetic (Ampere).
_BF16_FROM: Final = (8, 0)
#: CUDA graphs need a GPU the driver can capture on; every architecture CUDA 12+ supports can.
_GRAPH_FROM: Final = (5, 0)
_OSRELEASE: Final = Path("/proc/sys/kernel/osrelease")


@cache
def running_on_wsl() -> bool:
    """Whether this Linux is WSL2 (its kernel release names Microsoft); ``False`` elsewhere."""
    if not sys.platform.startswith("linux"):
        return False
    try:
        release = _OSRELEASE.read_text(encoding="ascii", errors="replace")
    except OSError:
        return False
    return "microsoft" in release.lower()


@dataclass(frozen=True, slots=True)
class DeviceCapabilities:
    """The features of one device that decide what a configuration may ask of it.

    Attributes:
        kind: ``"cuda"`` or ``"cpu"``.
        index: Position among devices of the same kind.
        name: Marketing name, or ``"cpu"``.
        arch: ``"sm_<major><minor>"`` for a GPU, ``"cpu"`` for the host.
        dtypes: Compute dtypes the device computes in natively.
        cuda_graph: Whether decode steps can be captured in CUDA graphs here.
        platform: ``sys.platform`` of the process.
        wsl: Whether the process runs under WSL2.
        total_bytes: Device memory, or host RAM.
    """

    kind: DeviceKind
    index: int
    name: str
    arch: str
    dtypes: frozenset[DType]
    cuda_graph: bool
    platform: str
    wsl: bool
    total_bytes: int

    def problems(self, config: DeviceConfig) -> list[str]:
        """Everything ``config`` asks for that this device lacks; empty when it fits."""
        found: list[str] = []
        if config.device_type != self.kind or (
            self.kind == "cuda" and (config.device_index or 0) != self.index
        ):
            found.append(f"the config names {config.device!r} but this is {self.describe()}")
        if config.dtype not in self.dtypes:
            supported = sorted(dtype.label for dtype in self.dtypes)
            found.append(f"{self.arch} has no {config.dtype.label} arithmetic (it has {supported})")
        # Only an explicit request is checked: the default (unset) asks nothing of the device,
        # and every GPU CUDA 13 runs on (sm_75 and newer) captures graphs.
        if config.cuda_graph and not self.cuda_graph:
            found.append(f"{self.describe()} cannot capture CUDA graphs")
        return found

    def check(self, config: DeviceConfig) -> None:
        """Refuse a configuration this device cannot run.

        Raises:
            ConfigError: ``DEVICE_CAPABILITY_MISSING``, with every problem in the
                message and in ``context["problems"]``.
        """
        found = self.problems(config)
        if found:
            raise ConfigError(
                "device",
                "DEVICE_CAPABILITY_MISSING",
                "; ".join(found),
                context={"problems": tuple(found)},
            )

    def describe(self) -> str:
        """One line for start-up logs: ``cuda:0 NVIDIA GeForce RTX 3050 sm_86 4.0 GiB (WSL2)``."""
        where = " (WSL2)" if self.wsl else ""
        if self.kind == "cpu":
            return f"cpu{where}"
        return (
            f"cuda:{self.index} {self.name} {self.arch} {self.total_bytes / 2**30:.1f} GiB{where}"
        )


def device_capabilities(
    info: DeviceInfo, *, platform: str | None = None, wsl: bool | None = None
) -> DeviceCapabilities:
    """The capabilities of a probed device.

    Args:
        info: A device from :func:`~ayaka.device.info.probe_devices`.
        platform: ``sys.platform`` to record; the running one by default.
        wsl: Whether this is WSL2; detected by default.
    """
    if info.kind == "cpu":
        dtypes = frozenset({DType.FP32, DType.FP16, DType.BF16})
        arch, graph = "cpu", False
    else:
        major, minor = info.compute_capability
        arch = f"sm_{major}{minor}"
        dtypes = frozenset({DType.FP32, DType.FP16})
        if info.compute_capability >= _BF16_FROM:
            dtypes |= {DType.BF16}
        graph = info.compute_capability >= _GRAPH_FROM
    return DeviceCapabilities(
        kind=info.kind,
        index=info.index,
        name=info.name,
        arch=arch,
        dtypes=dtypes,
        cuda_graph=graph,
        platform=sys.platform if platform is None else platform,
        wsl=running_on_wsl() if wsl is None else wsl,
        total_bytes=info.total_bytes,
    )
