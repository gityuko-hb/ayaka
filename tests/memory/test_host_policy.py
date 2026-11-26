"""Unit tests for pure host memory tier policy planning."""

from __future__ import annotations

import pytest

from ayaka.configs.host_memory import HostMemoryConfig
from ayaka.memory.host_policy import plan_host_memory

pytestmark = pytest.mark.unit

_MIB = 1024 * 1024
_GIB = 1024 * _MIB


def test_disabled_host_memory_returns_zero_blocks() -> None:
    plan = plan_host_memory(
        HostMemoryConfig(enabled=False),
        bytes_per_block=4 * _MIB,
        available_ram_bytes=8 * _GIB,
    )
    assert plan.enabled is False
    assert plan.budget_bytes == 0
    assert plan.max_blocks == 0


def test_plan_host_memory_respects_keep_fraction_cgroup_and_cap() -> None:
    # 10 GiB available, 4 GiB cgroup -> min = 4 GiB, keep 25% = 1 GiB -> 3 GiB budget.
    plan = plan_host_memory(
        HostMemoryConfig(enabled=True, ram_keep_fraction=0.25),
        bytes_per_block=64 * _MIB,
        available_ram_bytes=10 * _GIB,
        cgroup_limit_bytes=4 * _GIB,
    )
    assert plan.enabled is True
    assert plan.keep_bytes == 1 * _GIB
    assert plan.budget_bytes == 3 * _GIB
    assert plan.max_blocks == (3 * _GIB) // (64 * _MIB)

    # Explicit max_host_kv_mb caps below derived RAM ceiling.
    capped = plan_host_memory(
        HostMemoryConfig(enabled=True, ram_keep_fraction=0.20, max_host_kv_mb=512),
        bytes_per_block=16 * _MIB,
        available_ram_bytes=16 * _GIB,
        cgroup_limit_bytes=16 * _GIB,
    )
    assert capped.budget_bytes == 512 * _MIB
    assert capped.max_blocks == 32
