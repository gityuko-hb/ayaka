"""Linear ops: the contract a backend implements for ``ayaka.layers.linear``.

Two products: :meth:`LinearOps.linear` on weights in the compute dtype, and
:meth:`LinearOps.scaled_linear` on 8-bit floating weights with their dequantization scales
(``ayaka.layers.quantization.fp8``).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from torch import Tensor

__all__ = ["LinearOps"]


class LinearOps(Protocol):
    """Affine projection of the last dimension."""

    def linear(self, x: Tensor, weight: Tensor, bias: Tensor | None, out: Tensor) -> None:
        """Write ``x @ weight.T + bias`` into ``out``.

        ``x`` is ``[..., in_features]``, ``weight`` is
        ``[out_features, in_features]``, ``bias`` is ``[out_features]`` or
        ``None`` and ``out`` is ``[..., out_features]``. All of them share one
        compute dtype. ``out`` must not overlap an input.
        """
        ...

    def scaled_linear(
        self,
        x: Tensor,
        weight: Tensor,
        scale: Tensor,
        bias: Tensor | None,
        out: Tensor,
        *,
        block: tuple[int, int] | None,
    ) -> None:
        """Write ``x @ (scale * weight).T + bias`` into ``out``: a product on 8-bit weights.

        ``weight`` is ``[out_features, in_features]`` in an 8-bit float dtype
        (``float8_e4m3fn`` or ``float8_e5m2``) and ``scale`` is float32:

        * ``block is None``: ``[out_features, 1]``, one scale per output row;
        * ``block == (b0, b1)``: ``[ceil(out_features / b0), ceil(in_features / b1)]``, one
          scale per ``b0 x b1`` block of ``weight``, the last block of each side possibly short.

        ``x``, ``bias`` and ``out`` are as in :meth:`linear`, in the compute dtype. The weight
        is dequantized exactly (``float32(weight) * scale``) and rounded once to the compute
        dtype before the product; a backend may do it in slices of rows, but the dequantized
        weight must not depend on the slicing (the product may differ by its own rounding).
        """
        ...
