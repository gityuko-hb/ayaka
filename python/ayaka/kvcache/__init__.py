"""The KV cache: block geometry and storage, and who owns which block.

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
    from ayaka.kvcache.storage.pool import KVPool
    from ayaka.kvcache.storage.spec import PADDING_BLOCK_ID, KVGroup, KVSpec
    from ayaka.kvcache.table import BlockTable

__all__ = [
    "PADDING_BLOCK_ID",
    "Block",
    "BlockPool",
    "BlockTable",
    "HashPrefixCache",
    "KVCacheManager",
    "KVCacheStats",
    "KVGroup",
    "KVPool",
    "KVSpec",
    "PrefixCache",
    "PrefixMatch",
]

_EXPORTS = {
    "blocks": ("Block", "BlockPool"),
    "manager": ("KVCacheManager", "KVCacheStats"),
    "prefix.base": ("PrefixCache", "PrefixMatch"),
    "prefix.hash_cache": ("HashPrefixCache",),
    "storage.pool": ("KVPool",),
    "storage.spec": ("PADDING_BLOCK_ID", "KVGroup", "KVSpec"),
    "table": ("BlockTable",),
}

__getattr__, __dir__ = build_lazy_exports(__name__, _EXPORTS, globals())
