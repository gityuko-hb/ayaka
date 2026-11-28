"""The KV and recurrent-state cache: block/slot geometry, storage, and ownership.

Public names are resolved from the module that defines them on first access, so
importing this package never loads torch.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ayaka.utils.lazy_exports import build_lazy_exports

if TYPE_CHECKING:
    from ayaka.kvcache.blocks import Block, BlockPool
    from ayaka.kvcache.manager import KVCacheManager, KVCacheStats
    from ayaka.kvcache.prefix.base import PrefixCache, PrefixMatch
    from ayaka.kvcache.prefix.hash_cache import HashPrefixCache
    from ayaka.kvcache.state_slots import (
        PADDING_STATE_SLOT,
        StateCheckpointStore,
        StateSlotPool,
    )
    from ayaka.kvcache.storage.pool import KVPool, LatentKVPool, RecurrentStatePool
    from ayaka.kvcache.storage.spec import (
        PADDING_BLOCK_ID,
        KVGroup,
        KVSpec,
        LatentKVGroup,
        LatentKVSpec,
        RecurrentStateGroup,
        RecurrentStateSpec,
    )
    from ayaka.kvcache.table import BlockTable

__all__ = [
    "PADDING_BLOCK_ID",
    "PADDING_STATE_SLOT",
    "Block",
    "BlockPool",
    "BlockTable",
    "HashPrefixCache",
    "KVCacheManager",
    "KVCacheStats",
    "KVGroup",
    "KVPool",
    "KVSpec",
    "LatentKVGroup",
    "LatentKVPool",
    "LatentKVSpec",
    "PrefixCache",
    "PrefixMatch",
    "RecurrentStateGroup",
    "RecurrentStatePool",
    "RecurrentStateSpec",
    "StateCheckpointStore",
    "StateSlotPool",
]

_EXPORTS = {
    "blocks": ("Block", "BlockPool"),
    "manager": ("KVCacheManager", "KVCacheStats"),
    "prefix.base": ("PrefixCache", "PrefixMatch"),
    "prefix.hash_cache": ("HashPrefixCache",),
    "state_slots": ("PADDING_STATE_SLOT", "StateCheckpointStore", "StateSlotPool"),
    "storage.pool": ("KVPool", "LatentKVPool", "RecurrentStatePool"),
    "storage.spec": (
        "PADDING_BLOCK_ID",
        "KVGroup",
        "KVSpec",
        "LatentKVGroup",
        "LatentKVSpec",
        "RecurrentStateGroup",
        "RecurrentStateSpec",
    ),
    "table": ("BlockTable",),
}

__getattr__, __dir__ = build_lazy_exports(__name__, _EXPORTS, globals())
