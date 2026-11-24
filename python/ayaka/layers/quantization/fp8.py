"""8-bit floating weights with static scales (Hugging Face ``fp8``, ``compressed-tensors``).

The weight stays 8 bit on the device. Its scale is held in float32, in one of two layouts:

* ``"block"`` recipes keep their ``[ceil(out / b0), ceil(in / b1)]`` grid of scales;
* ``"channel"`` and ``"tensor"`` recipes become one scale per output row, ``[out, 1]``. A
  per-tensor scale is repeated over its rows, so q, k and v (or gate and up), each stored with a
  scale of its own, can share one fused ``weight`` and one ``weight_scale``.

The product is :meth:`~ayaka.ops.linear.LinearOps.scaled_linear`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import nn

from ayaka.core.errors import ModelLoadError
from ayaka.utils.dtypes import DType
from ayaka.utils.math_utils import div_ceil

if TYPE_CHECKING:
    from collections.abc import Sequence

    from ayaka.ops.linear import LinearOps
    from ayaka.weights.quant import QuantFormat

__all__ = ["Fp8Method"]

_SCALE = "weight_scale"


class Fp8Method:
    """The :class:`~ayaka.layers.quantization.method.QuantMethod` of 8-bit floating weights.

    Args:
        format: The checkpoint's recipe.
    """

    def __init__(self, format: QuantFormat) -> None:  # noqa: A002  # the record's name
        self._format = format
        self._block = format.block

    @property
    def format(self) -> QuantFormat | None:
        """The recipe given at construction."""
        return self._format

    def create(
        self,
        in_features: int,
        out_features: int,
        *,
        dtype: torch.dtype,
        device: torch.device | str | None,
    ) -> tuple[nn.Parameter, nn.Parameter | None]:
        """An 8-bit ``weight`` and its float32 ``weight_scale``; ``dtype`` is the activations'."""
        del dtype
        stored = self._format.weight_dtype.torch_dtype
        weight = nn.Parameter(
            torch.empty(out_features, in_features, dtype=stored, device=device),
            requires_grad=False,
        )
        scale = nn.Parameter(
            torch.empty(self._scale_shape(out_features, in_features), device=device),
            requires_grad=False,
        )
        return weight, scale

    def stored_shape(self, parameter: str, rows: int, in_features: int) -> tuple[int, ...]:
        """The weight and bias as stored; the scale in the recipe's own layout."""
        if parameter != _SCALE:
            return (rows, in_features) if parameter == "weight" else (rows,)
        match self._format.scale_layout:
            case "tensor":
                return (1,)
            case "channel":
                return (rows, 1)
            case "block":
                return self._scale_shape(rows, in_features)

    def part_rows(self, parameter: str, rows: Sequence[int]) -> tuple[int, ...]:
        """Scale rows of block parts: every part but the last must be whole blocks.

        Raises:
            ModelLoadError: If a part other than the last ends inside a block of scales.
        """
        if parameter != _SCALE or self._block is None:
            return tuple(rows)
        side = self._block[0]
        uneven = [count for count in rows[:-1] if count % side]
        if uneven:
            raise ModelLoadError(
                f"cannot fuse parts of {list(rows)} rows into one weight with {side}-row scale "
                f"blocks: a part of {uneven[0]} rows ends inside a block",
                context={"rows": list(rows), "block": list(self._block)},
            )
        return tuple(div_ceil(count, side) for count in rows)

    def stored_dtype(self, parameter: str, compute: DType) -> DType:
        """The recipe's 8-bit dtype for the weight, float32 for the scale."""
        if parameter == "weight":
            return self._format.weight_dtype
        return DType.FP32 if parameter == _SCALE else compute

    def fill(self, parameter: str, target: torch.Tensor, tensor: torch.Tensor) -> None:
        """Copy; a per-tensor scale is repeated over every row it covers."""
        if parameter == _SCALE and self._format.scale_layout == "tensor":
            target.copy_(tensor.reshape(1, 1).expand_as(target))
            return
        target.copy_(tensor)

    def apply(
        self,
        ops: LinearOps,
        x: torch.Tensor,
        weight: torch.Tensor,
        scale: torch.Tensor | None,
        bias: torch.Tensor | None,
        out: torch.Tensor,
    ) -> None:
        """``ops.scaled_linear`` with this recipe's block."""
        if scale is None:  # create() always makes one; this narrows the type
            raise ModelLoadError("an 8-bit linear layer has no weight_scale")
        ops.scaled_linear(x, weight, scale, bias, out, block=self._block)

    def _scale_shape(self, rows: int, in_features: int) -> tuple[int, ...]:
        if self._block is None:
            return (rows, 1)
        return (div_ceil(rows, self._block[0]), div_ceil(in_features, self._block[1]))
