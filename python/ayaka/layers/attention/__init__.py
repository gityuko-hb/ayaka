"""Attention layers.

Public names are resolved from their submodule on first access.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ayaka.utils.lazy_exports import build_lazy_exports

if TYPE_CHECKING:
    from ayaka.layers.attention.dense import Attention, QKNorm
    from ayaka.layers.attention.linear import GatedDeltaNetAttention, KimiDeltaAttention
    from ayaka.layers.attention.mla import MultiHeadLatentAttention

__all__ = [
    "Attention",
    "GatedDeltaNetAttention",
    "KimiDeltaAttention",
    "MultiHeadLatentAttention",
    "QKNorm",
]

_EXPORTS = {
    "dense": ("Attention", "QKNorm"),
    "linear": ("GatedDeltaNetAttention", "KimiDeltaAttention"),
    "mla": ("MultiHeadLatentAttention",),
}

__getattr__, __dir__ = build_lazy_exports(__name__, _EXPORTS, globals())
