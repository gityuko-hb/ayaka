"""Reuse of the KV of full prompt blocks across requests, named by a salted hash chain.

Public names are resolved from the module that defines them on first access.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ayaka.utils.lazy_exports import build_lazy_exports

if TYPE_CHECKING:
    from ayaka.kvcache.prefix.base import PrefixCache, PrefixMatch
    from ayaka.kvcache.prefix.hash_cache import HashPrefixCache
    from ayaka.kvcache.prefix.hashing import (
        CACHE_FORMAT_VERSION,
        DIGEST_SIZE,
        ChainRoot,
        chain_digests,
        root_digest,
    )

__all__ = [
    "CACHE_FORMAT_VERSION",
    "DIGEST_SIZE",
    "ChainRoot",
    "HashPrefixCache",
    "PrefixCache",
    "PrefixMatch",
    "chain_digests",
    "root_digest",
]

_EXPORTS = {
    "base": ("PrefixCache", "PrefixMatch"),
    "hash_cache": ("HashPrefixCache",),
    "hashing": ("CACHE_FORMAT_VERSION", "DIGEST_SIZE", "ChainRoot", "chain_digests", "root_digest"),
}

__getattr__, __dir__ = build_lazy_exports(__name__, _EXPORTS, globals())
