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
    from ayaka.weights.gguf import GGMLType, GGUFFile, GGUFTensor, gguf_hf_config, read_gguf
    from ayaka.weights.manifest import WeightManifest
    from ayaka.weights.mapping import WeightMapping
    from ayaka.weights.mxfp4 import FP4_E2M1_LUT, dequantize_mxfp4, mxfp4_lut
    from ayaka.weights.quant import QuantFormat, ScaleLayout, read_quant_format
    from ayaka.weights.safetensors import SafetensorsFile
    from ayaka.weights.spec import TensorSpec
    from ayaka.weights.tensor import TensorRef

__all__ = [
    "FP4_E2M1_LUT",
    "GGMLType",
    "GGUFFile",
    "GGUFTensor",
    "QuantFormat",
    "SafetensorsFile",
    "ScaleLayout",
    "TensorRef",
    "TensorSpec",
    "WeightManifest",
    "WeightMapping",
    "dequantize_mxfp4",
    "gguf_hf_config",
    "mxfp4_lut",
    "read_gguf",
    "read_quant_format",
]

_EXPORTS = {
    "gguf": ("GGMLType", "GGUFFile", "GGUFTensor", "gguf_hf_config", "read_gguf"),
    "manifest": ("WeightManifest",),
    "mapping": ("WeightMapping",),
    "mxfp4": ("FP4_E2M1_LUT", "dequantize_mxfp4", "mxfp4_lut"),
    "quant": ("QuantFormat", "ScaleLayout", "read_quant_format"),
    "safetensors": ("SafetensorsFile",),
    "spec": ("TensorSpec",),
    "tensor": ("TensorRef",),
}

__getattr__, __dir__ = build_lazy_exports(__name__, _EXPORTS, globals())
