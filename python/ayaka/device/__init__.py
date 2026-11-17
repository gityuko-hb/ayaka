"""The machine Ayaka runs on: what each device is, what it can run and how much memory it has.

Probing uses NVML and ``/proc`` only, so importing this package, calling
``probe_devices`` or reading ``device_capabilities`` never loads torch or creates
a CUDA context. :class:`Device` (``context``) is the torch side and loads torch
when it is first used. Public names are resolved from their submodule on first
access.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ayaka.utils.lazy_exports import build_lazy_exports

if TYPE_CHECKING:
    from ayaka.device.capabilities import DeviceCapabilities, device_capabilities, running_on_wsl
    from ayaka.device.context import Device, MemoryReading
    from ayaka.device.info import DeviceInfo, DeviceKind, probe_devices

__all__ = [
    "Device",
    "DeviceCapabilities",
    "DeviceInfo",
    "DeviceKind",
    "MemoryReading",
    "device_capabilities",
    "probe_devices",
    "running_on_wsl",
]

_EXPORTS = {
    "capabilities": ("DeviceCapabilities", "device_capabilities", "running_on_wsl"),
    "context": ("Device", "MemoryReading"),
    "info": ("DeviceInfo", "DeviceKind", "probe_devices"),
}

__getattr__, __dir__ = build_lazy_exports(__name__, _EXPORTS, globals())
