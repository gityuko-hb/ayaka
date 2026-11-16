"""One model step: building its inputs and running it.

Public names are resolved from the module that defines them on first access. The modules import
torch, so importing this package does not.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ayaka.utils.lazy_exports import build_lazy_exports

if TYPE_CHECKING:
    from ayaka.execution.inputs import InputBuilder
    from ayaka.execution.runner import ModelRunner

__all__ = ["InputBuilder", "ModelRunner"]

_EXPORTS = {
    "inputs": ("InputBuilder",),
    "runner": ("ModelRunner",),
}

__getattr__, __dir__ = build_lazy_exports(__name__, _EXPORTS, globals())
