"""Feed-forward blocks.

Public names are resolved from their submodule on first access.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ayaka.utils.lazy_exports import build_lazy_exports

if TYPE_CHECKING:
    from ayaka.layers.ffn.dense import MLP, GatedMLP
    from ayaka.layers.ffn.moe import MoEBlock, StackedExperts

__all__ = ["MLP", "GatedMLP", "MoEBlock", "StackedExperts"]

_EXPORTS = {
    "dense": ("MLP", "GatedMLP"),
    "moe": ("MoEBlock", "StackedExperts"),
}

__getattr__, __dir__ = build_lazy_exports(__name__, _EXPORTS, globals())
