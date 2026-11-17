"""One model step: building its inputs, running it, and replaying decode steps as CUDA graphs.

Public names are resolved from the module that defines them on first access. The modules import
torch, so importing this package does not.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ayaka.utils.lazy_exports import build_lazy_exports

if TYPE_CHECKING:
    from ayaka.execution.graph import CapturedBucket, GraphRunner
    from ayaka.execution.inputs import InputBuilder, ModelInputs, StagedDecode
    from ayaka.execution.runner import ModelRunner

__all__ = [
    "CapturedBucket",
    "GraphRunner",
    "InputBuilder",
    "ModelInputs",
    "ModelRunner",
    "StagedDecode",
]

_EXPORTS = {
    "graph": ("CapturedBucket", "GraphRunner"),
    "inputs": ("InputBuilder", "ModelInputs", "StagedDecode"),
    "runner": ("ModelRunner",),
}

__getattr__, __dir__ = build_lazy_exports(__name__, _EXPORTS, globals())
