"""Quantized linear layers: the method contract and its formats.

Public names are resolved from their submodule on first access.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ayaka.utils.lazy_exports import build_lazy_exports

if TYPE_CHECKING:
    from ayaka.layers.quantization.fp8 import Fp8Method
    from ayaka.layers.quantization.method import (
        UNQUANTIZED,
        QuantMethod,
        Unquantized,
        quant_method,
    )

__all__ = ["UNQUANTIZED", "Fp8Method", "QuantMethod", "Unquantized", "quant_method"]

_EXPORTS = {
    "fp8": ("Fp8Method",),
    "method": ("UNQUANTIZED", "QuantMethod", "Unquantized", "quant_method"),
}

__getattr__, __dir__ = build_lazy_exports(__name__, _EXPORTS, globals())
