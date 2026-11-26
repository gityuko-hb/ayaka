"""Host-RAM KV cache tiering configuration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

from ayaka.configs.base import ConfigMixin
from ayaka.core.errors import ConfigError
from ayaka.utils.validation import require_int, require_number

__all__ = ["HostMemoryConfig"]


@dataclass(frozen=True, slots=True)
class HostMemoryConfig(ConfigMixin):
    """Settings for offloading evicted KV blocks to pinned or pageable host RAM.

    Host KV tiering is opt-in (``enabled=False`` by default) so machines with tight
    host RAM or missing GPU bandwidth measurements never allocate host pools unasked.

    Attributes:
        enabled: Whether host-RAM KV offloading and promotion are active.
        ram_keep_fraction: Share of available/cgroup host RAM left untouched for the
            operating system and other processes, in ``[0.05, 0.90]``.
        max_host_kv_mb: Optional hard cap on host KV storage in MiB; ``None`` derives
            the ceiling solely from available host/cgroup RAM and ``ram_keep_fraction``.
        pinned: Whether to request CUDA pinned (page-locked) host buffers.
        allow_pageable: Whether to fall back to pageable host memory when pinned
            allocation fails on CPU-only hosts or under strict ``RLIMIT_MEMLOCK``.
        max_inflight_offloads: Bounded queue depth for asynchronous write-behind
            offloads before applying backpressure.
    """

    enabled: bool = False
    ram_keep_fraction: float = 0.20
    max_host_kv_mb: int | None = None
    pinned: bool = True
    allow_pageable: bool = False
    max_inflight_offloads: int = 8

    def __post_init__(self) -> None:
        if not isinstance(cast("object", self.enabled), bool):
            raise TypeError("host_memory.enabled must be a bool")
        if not isinstance(cast("object", self.pinned), bool):
            raise TypeError("host_memory.pinned must be a bool")
        if not isinstance(cast("object", self.allow_pageable), bool):
            raise TypeError("host_memory.allow_pageable must be a bool")
        keep = require_number(self.ram_keep_fraction, "host_memory.ram_keep_fraction")
        if not 0.05 <= keep <= 0.90:
            raise ConfigError(
                "host_memory.ram_keep_fraction",
                "INVALID_RANGE",
                f"ram_keep_fraction must be in [0.05, 0.90], got {keep}",
            )
        object.__setattr__(self, "ram_keep_fraction", keep)
        if self.max_host_kv_mb is not None:
            require_int(self.max_host_kv_mb, "host_memory.max_host_kv_mb", minimum=1)
        require_int(self.max_inflight_offloads, "host_memory.max_inflight_offloads", minimum=1)
