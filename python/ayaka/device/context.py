"""The device a model runs on, as the engine uses it: torch device, stream, events, memory.

:class:`Device` gathers the few CUDA calls the execution layer makes (the
current stream, events to time a step or wait for a copy, synchronisation, the
memory the allocator holds and the driver reports free) behind one object, so
the rest of the engine does not branch on ``torch.cuda`` itself and the same
code runs on the CPU, where every one of them is a no-op or a zero.

This module imports torch; ``ayaka.device`` loads it on first use.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from ayaka.utils.torch_utils import resolve_device

__all__ = ["Device", "MemoryReading"]


@dataclass(frozen=True, slots=True)
class MemoryReading:
    """Device memory at one moment, in bytes.

    Attributes:
        free: What the driver reports free on the whole device (other processes included).
        total: The device's memory.
        allocated: What tensors of this process occupy, through the caching allocator.
        reserved: What the caching allocator holds from the driver (``allocated`` plus its cache).
        peak_allocated: The most ``allocated`` reached since the last
            :meth:`Device.reset_peak`.
    """

    free: int
    total: int
    allocated: int
    reserved: int
    peak_allocated: int

    @property
    def used(self) -> int:
        """``total - free``: everything on the device, this process or not."""
        return self.total - self.free


class Device:
    """One device: ``"cpu"`` or a CUDA GPU.

    Args:
        spec: ``"cpu"``, ``"cuda"``, ``"cuda:N"`` or a ``torch.device``.

    Raises:
        CapabilityError: If torch is missing or the GPU is not available.
    """

    def __init__(self, spec: str | torch.device) -> None:
        self._device: torch.device = resolve_device(spec, capability="device")

    @property
    def torch_device(self) -> torch.device:
        """The resolved ``torch.device`` (a CUDA device always has its index)."""
        return self._device

    @property
    def is_cuda(self) -> bool:
        """Whether this is a GPU."""
        return self._device.type == "cuda"

    def __repr__(self) -> str:
        return f"Device({str(self._device)!r})"

    def synchronize(self) -> None:
        """Wait for every kernel queued on this device; a no-op on the CPU."""
        if self.is_cuda:
            torch.cuda.synchronize(self._device)

    def event(self, *, timing: bool = False) -> Any:
        """A CUDA event recorded nowhere yet, or ``None`` on the CPU.

        Args:
            timing: Whether the event can measure elapsed time (it costs a little more).
        """
        if not self.is_cuda:
            return None
        return torch.cuda.Event(enable_timing=timing)

    def memory(self) -> MemoryReading:
        """What the device and this process's allocator hold now; zeros on the CPU.

        On a GPU this asks the driver for the free memory, which does not
        synchronise the device.
        """
        if not self.is_cuda:
            return MemoryReading(free=0, total=0, allocated=0, reserved=0, peak_allocated=0)
        free, total = torch.cuda.mem_get_info(self._device)
        return MemoryReading(
            free=int(free),
            total=int(total),
            allocated=int(torch.cuda.memory_allocated(self._device)),
            reserved=int(torch.cuda.memory_reserved(self._device)),
            peak_allocated=int(torch.cuda.max_memory_allocated(self._device)),
        )

    def release_cache(self) -> None:
        """Give the memory torch's allocator caches but no tensor uses back to the driver.

        Waits for the device first, so blocks freed by queued kernels are released too. A no-op
        on the CPU.
        """
        if self.is_cuda:
            torch.cuda.synchronize(self._device)
            torch.cuda.empty_cache()

    def reset_peak(self) -> None:
        """Start a new peak for :attr:`MemoryReading.peak_allocated`; a no-op on the CPU."""
        if self.is_cuda:
            torch.cuda.reset_peak_memory_stats(self._device)

    def current_stream(self) -> Any:
        """The stream work is queued on now, or ``None`` on the CPU."""
        if not self.is_cuda:
            return None
        return torch.cuda.current_stream(self._device)
