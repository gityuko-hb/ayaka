"""The physical side of the KV and state cache: geometries and tensor pools.

Public names are resolved from their submodule on first access, so importing
this package never loads torch; the pool classes import it when instantiated.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ayaka.utils.lazy_exports import build_lazy_exports

if TYPE_CHECKING:
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

__all__ = [
    "PADDING_BLOCK_ID",
    "KVGroup",
    "KVPool",
    "KVSpec",
    "LatentKVGroup",
    "LatentKVPool",
    "LatentKVSpec",
    "RecurrentStateGroup",
    "RecurrentStatePool",
    "RecurrentStateSpec",
]

_EXPORTS = {
    "pool": ("KVPool", "LatentKVPool", "RecurrentStatePool"),
    "spec": (
        "PADDING_BLOCK_ID",
        "KVGroup",
        "KVSpec",
        "LatentKVGroup",
        "LatentKVSpec",
        "RecurrentStateGroup",
        "RecurrentStateSpec",
    ),
}

__getattr__, __dir__ = build_lazy_exports(__name__, _EXPORTS, globals())
