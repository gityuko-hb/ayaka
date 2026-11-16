"""The physical side of the KV cache: the geometry of a group and the tensors that hold it.

Public names are resolved from their submodule on first access, so importing
this package never loads torch; :class:`KVPool` imports it when a pool is built.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ayaka.utils.lazy_exports import build_lazy_exports

if TYPE_CHECKING:
    from ayaka.kvcache.storage.pool import KVPool
    from ayaka.kvcache.storage.spec import PADDING_BLOCK_ID, KVGroup, KVSpec

__all__ = [
    "PADDING_BLOCK_ID",
    "KVGroup",
    "KVPool",
    "KVSpec",
]

_EXPORTS = {
    "pool": ("KVPool",),
    "spec": ("PADDING_BLOCK_ID", "KVGroup", "KVSpec"),
}

__getattr__, __dir__ = build_lazy_exports(__name__, _EXPORTS, globals())
