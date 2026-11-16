"""Feed-forward blocks.

Public names are resolved from their submodule on first access.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ayaka.utils.lazy_exports import build_lazy_exports

if TYPE_CHECKING:
    from ayaka.layers.ffn.dense import MLP, GatedMLP

__all__ = ["MLP", "GatedMLP"]

_EXPORTS = {
    "dense": ("MLP", "GatedMLP"),
}

__getattr__, __dir__ = build_lazy_exports(__name__, _EXPORTS, globals())
