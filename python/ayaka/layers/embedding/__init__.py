"""Token embeddings and the output head.

Public names are resolved from their submodule on first access.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ayaka.utils.lazy_exports import build_lazy_exports

if TYPE_CHECKING:
    from ayaka.layers.embedding.token import LMHead, TokenEmbedding

__all__ = ["LMHead", "TokenEmbedding"]

_EXPORTS = {
    "token": ("LMHead", "TokenEmbedding"),
}

__getattr__, __dir__ = build_lazy_exports(__name__, _EXPORTS, globals())
