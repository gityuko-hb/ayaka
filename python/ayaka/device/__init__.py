"""The machine Ayaka runs on: what each device is and how much memory it has.

Probing uses NVML and ``/proc`` only, so importing this package or calling
``probe_devices`` never loads torch or creates a CUDA context. Public names are
resolved from their submodule on first access.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ayaka.utils.lazy_exports import build_lazy_exports

if TYPE_CHECKING:
    from ayaka.device.info import DeviceInfo, probe_devices

__all__ = [
    "DeviceInfo",
    "probe_devices",
]

_EXPORTS = {
    "info": ("DeviceInfo", "probe_devices"),
}

__getattr__, __dir__ = build_lazy_exports(__name__, _EXPORTS, globals())
