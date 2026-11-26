"""Pure host-RAM KV tier budget planning from available memory and cgroup limits."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from ayaka.utils.host_info import cgroup_memory, host_available_bytes
from ayaka.utils.validation import require_int

if TYPE_CHECKING:
    from ayaka.configs.host_memory import HostMemoryConfig

__all__ = ["HostMemoryPlan", "plan_host_memory"]

_MIB = 1024 * 1024


@dataclass(frozen=True, slots=True)
class HostMemoryPlan:
    """Computed capacity of the optional host-RAM KV cache tier."""

    enabled: bool
    budget_bytes: int
    max_blocks: int
    keep_bytes: int
    available_ram_bytes: int | None
    cgroup_limit_bytes: int | None
    pinned: bool
    allow_pageable: bool


def plan_host_memory(
    config: HostMemoryConfig,
    *,
    bytes_per_block: int,
    available_ram_bytes: int | None = None,
    cgroup_limit_bytes: int | None = None,
) -> HostMemoryPlan:
    """Plan host-RAM KV tier capacity while preserving OS headroom.

    Args:
        config: Host memory tier configuration.
        bytes_per_block: Total host bytes required per KV block across all groups.
        available_ram_bytes: Optional injected available host RAM (bytes) for testing.
        cgroup_limit_bytes: Optional injected container memory limit (bytes) for testing.
    """
    block_bytes = require_int(bytes_per_block, "bytes_per_block", minimum=1)
    avail = available_ram_bytes if available_ram_bytes is not None else host_available_bytes()
    cg_limit = cgroup_limit_bytes
    cg_headroom: int | None = cg_limit
    if cg_limit is None and available_ram_bytes is None:
        cg = cgroup_memory()
        cg_limit = cg.limit_bytes
        if cg.limit_bytes is not None and cg.current_bytes is not None:
            cg_headroom = max(0, cg.limit_bytes - cg.current_bytes)
        else:
            cg_headroom = cg.limit_bytes

    if not config.enabled:
        return HostMemoryPlan(
            enabled=False,
            budget_bytes=0,
            max_blocks=0,
            keep_bytes=0,
            available_ram_bytes=avail,
            cgroup_limit_bytes=cg_limit,
            pinned=config.pinned,
            allow_pageable=config.allow_pageable,
        )

    candidates = [x for x in (avail, cg_headroom) if x is not None]
    effective_ram = min(candidates) if candidates else 0
    keep_bytes = int(effective_ram * config.ram_keep_fraction)
    budget_bytes = max(0, effective_ram - keep_bytes)
    if config.max_host_kv_mb is not None:
        cap = config.max_host_kv_mb * _MIB
        budget_bytes = min(budget_bytes, cap) if effective_ram > 0 else cap
    max_blocks = budget_bytes // block_bytes
    return HostMemoryPlan(
        enabled=True,
        budget_bytes=budget_bytes,
        max_blocks=max_blocks,
        keep_bytes=keep_bytes,
        available_ram_bytes=avail,
        cgroup_limit_bytes=cg_limit,
        pinned=config.pinned,
        allow_pageable=config.allow_pageable,
    )
