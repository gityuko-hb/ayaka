"""What tensors look like on disk: descriptions, file formats and name mapping.

``weights`` knows checkpoints, not models. A model declares the parameters it
needs as :class:`TensorSpec` and a :class:`WeightMapping`; a checkpoint is read
into a :class:`WeightManifest` from file headers alone; matching the two before
any tensor is read is :meth:`WeightMapping.bind`. Reading the bytes and
building a model from them is ``ayaka.model_loader``'s job.

Public names are resolved from their submodule on first access.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ayaka.utils.lazy_exports import build_lazy_exports

if TYPE_CHECKING:
    from ayaka.weights.manifest import WeightManifest
    from ayaka.weights.mapping import WeightMapping
    from ayaka.weights.safetensors import SafetensorsFile
    from ayaka.weights.spec import TensorSpec
    from ayaka.weights.tensor import TensorRef

__all__ = [
    "SafetensorsFile",
    "TensorRef",
    "TensorSpec",
    "WeightManifest",
    "WeightMapping",
]

_EXPORTS = {
    "manifest": ("WeightManifest",),
    "mapping": ("WeightMapping",),
    "safetensors": ("SafetensorsFile",),
    "spec": ("TensorSpec",),
    "tensor": ("TensorRef",),
}

__getattr__, __dir__ = build_lazy_exports(__name__, _EXPORTS, globals())
