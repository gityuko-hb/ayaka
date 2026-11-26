"""Engine-level host-RAM KV tier initialization and storage attachment."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from ayaka.kvcache.host_store import HostKVStore
from ayaka.memory.host_policy import plan_host_memory

if TYPE_CHECKING:
    from collections.abc import Sequence

    from ayaka.configs.engine import EngineConfig
    from ayaka.kvcache.manager import KVCacheManager
    from ayaka.kvcache.storage.pool import KVPool

__all__ = ["configure_host_kv"]

_LOG = logging.getLogger("ayaka.engine")


def configure_host_kv(
    kv: KVCacheManager,
    pools: Sequence[KVPool],
    config: EngineConfig,
) -> HostKVStore | None:
    """Attach storage pools to ``kv`` and build an opt-in :class:`HostKVStore` when enabled."""
    host_cfg = config.host_memory
    if not host_cfg.enabled:
        kv.attach_storage(pools)
        return None
    bytes_per_block = sum(pool.group.spec.bytes_per_block() for pool in pools)
    plan = plan_host_memory(host_cfg, bytes_per_block=bytes_per_block)
    if plan.max_blocks < 1:
        _LOG.warning(
            "host KV tier requested but computed budget (%d bytes) fits 0 blocks; "
            "keeping host tier off",
            plan.budget_bytes,
        )
        kv.attach_storage(pools)
        return None
    store = HostKVStore(
        [pool.group for pool in pools],
        plan.max_blocks,
        pinned=plan.pinned,
        allow_pageable=plan.allow_pageable,
        max_inflight_offloads=host_cfg.max_inflight_offloads,
    )
    kv.attach_storage(pools, host_store=store)
    _LOG.info(
        "host KV tier enabled: %d blocks (%d MiB, pinned=%s)",
        plan.max_blocks,
        plan.budget_bytes // (1024 * 1024),
        store.pinned_memory,
    )
    return store
