"""What the machine offers: a description of each device and a probe that finds them.

The probe reads NVML and the host's ``/proc`` and cgroup files. It never imports
torch and never creates a CUDA context, so the baseline it reports is the memory
that was free before Ayaka (or any CUDA library in this process) took any
of it, and it is safe to call before a fork.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from ayaka.utils.host_info import host_available_bytes, host_total_ram_bytes
from ayaka.utils.nvml_utils import nvml_session, probe_device
from ayaka.utils.validation import require_int, require_text

__all__ = ["DeviceInfo", "DeviceKind", "probe_devices"]

type DeviceKind = Literal["cuda", "cpu"]
"""The kinds of device Ayaka runs on."""


@dataclass(frozen=True, slots=True)
class DeviceInfo:
    """One device a model can run on.

    A byte count of ``0`` means the probe could not read it. Planning treats that
    as "nothing known to be free" and refuses to start, never as "unlimited".

    Attributes:
        kind: ``"cuda"`` for an NVIDIA GPU, ``"cpu"`` for the host itself.
        index: Position among devices of the same kind.
        name: Marketing name, or ``"cpu"`` for the host.
        total_bytes: Device memory, or host RAM (honouring a cgroup limit).
        free_bytes: Memory free at probe time. This is the baseline for
            :func:`~ayaka.memory.budget.plan_budget`: read it once, before the
            first allocation, and keep it.
        compute_capability: ``(major, minor)`` CUDA capability; ``(0, 0)`` for the host.
        num_sms: Streaming multiprocessors; ``0`` for the host or when unknown.
        uuid: NVML UUID; empty when unknown or for the host.
        pci_bus_id: PCI address; empty when unknown or for the host.

    Raises:
        TypeError: If a byte count or index is not an ``int``.
        ValueError: If a count is negative, ``name`` is blank, ``free_bytes``
            exceeds ``total_bytes`` or ``kind`` is not ``"cuda"`` or ``"cpu"``.
    """

    kind: DeviceKind
    index: int
    name: str
    total_bytes: int
    free_bytes: int
    compute_capability: tuple[int, int] = (0, 0)
    num_sms: int = 0
    uuid: str = ""
    pci_bus_id: str = ""

    def __post_init__(self) -> None:
        if self.kind not in ("cuda", "cpu"):
            raise ValueError("kind must be 'cuda' or 'cpu'")
        require_int(self.index, "index")
        require_text(self.name, "name")
        require_int(self.total_bytes, "total_bytes")
        require_int(self.free_bytes, "free_bytes")
        require_int(self.num_sms, "num_sms")
        if self.free_bytes > self.total_bytes:
            raise ValueError("free_bytes must not exceed total_bytes")

    @property
    def is_cuda(self) -> bool:
        """Whether this is an NVIDIA GPU."""
        return self.kind == "cuda"


def _free_bytes(pynvml_mod: Any, handle: Any) -> int:
    try:
        return int(pynvml_mod.nvmlDeviceGetMemoryInfo(handle).free)
    except Exception:
        return 0


def _cuda_devices() -> tuple[DeviceInfo, ...]:
    with nvml_session() as nvml:
        if nvml is None:
            return ()
        try:
            count = int(nvml.nvmlDeviceGetCount())
        except Exception:
            return ()
        found: list[DeviceInfo] = []
        for index in range(count):
            try:
                handle = nvml.nvmlDeviceGetHandleByIndex(index)
            except Exception:  # noqa: S112 - a device whose handle is lost is skipped
                continue
            raw = probe_device(nvml, index, handle)
            found.append(
                DeviceInfo(
                    kind="cuda",
                    index=raw.index,
                    name=raw.name or f"cuda:{index}",
                    total_bytes=raw.hbm_bytes,
                    free_bytes=min(_free_bytes(nvml, handle), raw.hbm_bytes),
                    compute_capability=(raw.sm_major, raw.sm_minor),
                    num_sms=raw.num_sms,
                    uuid=raw.uuid,
                    pci_bus_id=raw.pci_bus_id,
                )
            )
        return tuple(found)


def _host_device() -> DeviceInfo:
    total = host_total_ram_bytes() or 0
    available = min(host_available_bytes() or 0, total)
    return DeviceInfo(kind="cpu", index=0, name="cpu", total_bytes=total, free_bytes=available)


def probe_devices() -> tuple[DeviceInfo, ...]:
    """Find every device: the CUDA GPUs in NVML order, then the host.

    The host entry is always last, so a machine without a GPU (or without
    ``pynvml``) still answers with one device and CPU runs plan against RAM.
    Every probe degrades to empty or zero values instead of raising.

    Returns:
        The GPUs followed by exactly one ``"cpu"`` entry.
    """
    return (*_cuda_devices(), _host_device())
