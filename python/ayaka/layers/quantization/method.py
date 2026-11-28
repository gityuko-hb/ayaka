"""The quantization contract of a linear layer, and the method of an unquantized one.

A :class:`~ayaka.layers.linear.Linear` is built with a :class:`QuantMethod` and asks it three
things, so the layer itself has no branch on the format:

1. **Parameters**: the shape and dtype of ``weight`` and of its scale (:meth:`QuantMethod.create`).
2. **Checkpoint**: what the checkpoint stores for each parameter, also for one part of a
   parameter that Ayaka fuses from several stored tensors (q, k, v; gate, up), and how a stored
   tensor fills its rows (:meth:`QuantMethod.stored_shape`, :meth:`QuantMethod.part_rows`,
   :meth:`QuantMethod.stored_dtype`, :meth:`QuantMethod.fill`).
3. **Op**: which op of :class:`~ayaka.ops.linear.LinearOps` runs it (:meth:`QuantMethod.apply`).

Parameter names are the layer's own: ``weight``, ``bias`` and ``weight_scale``.
:func:`quant_method` returns the method of a recipe (:class:`~ayaka.weights.quant.QuantFormat`).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

import torch
from torch import nn

if TYPE_CHECKING:
    from collections.abc import Sequence

    from ayaka.ops.linear import LinearOps
    from ayaka.utils.dtypes import DType
    from ayaka.weights.quant import QuantFormat

__all__ = ["UNQUANTIZED", "QuantMethod", "Unquantized", "quant_method"]


class QuantMethod(Protocol):
    """How a linear layer stores, loads and multiplies its weight."""

    @property
    def format(self) -> QuantFormat | None:
        """The checkpoint recipe this method reads; ``None`` for weights in the compute dtype."""
        ...

    def create(
        self,
        in_features: int,
        out_features: int,
        *,
        dtype: torch.dtype,
        device: torch.device | str | None,
    ) -> tuple[nn.Parameter, nn.Parameter | None]:
        """The uninitialised ``weight`` and its ``weight_scale`` (``None`` when there is none)."""
        ...

    def stored_shape(self, parameter: str, rows: int, in_features: int) -> tuple[int, ...]:
        """The shape the checkpoint stores for ``rows`` output rows of ``parameter``."""
        ...

    def part_rows(self, parameter: str, rows: Sequence[int]) -> tuple[int, ...]:
        """The rows of ``parameter`` that stored parts of ``rows`` output rows each fill.

        Raises:
            ModelLoadError: If the parts cannot be fused into one parameter of this layout.
        """
        ...

    def stored_dtype(self, parameter: str, compute: DType) -> DType:
        """The dtype ``parameter`` holds: the stored dtype must be this one, or convert to it."""
        ...

    def fill(self, parameter: str, target: torch.Tensor, tensor: torch.Tensor) -> None:
        """Copy a stored tensor into ``target``, the rows of ``parameter`` it fills."""
        ...

    def apply(
        self,
        ops: LinearOps,
        x: torch.Tensor,
        weight: torch.Tensor,
        scale: torch.Tensor | None,
        bias: torch.Tensor | None,
        out: torch.Tensor,
    ) -> None:
        """Write ``x``'s projection into ``out`` with the op of this format."""
        ...


class Unquantized:
    """Weights in the compute dtype, multiplied by :meth:`~ayaka.ops.linear.LinearOps.linear`."""

    @property
    def format(self) -> QuantFormat | None:
        """Always ``None``."""
        return None

    def create(
        self,
        in_features: int,
        out_features: int,
        *,
        dtype: torch.dtype,
        device: torch.device | str | None,
    ) -> tuple[nn.Parameter, nn.Parameter | None]:
        """``weight`` ``[out_features, in_features]`` in ``dtype``; no scale."""
        weight = nn.Parameter(
            torch.empty(out_features, in_features, dtype=dtype, device=device), requires_grad=False
        )
        return weight, None

    def stored_shape(self, parameter: str, rows: int, in_features: int) -> tuple[int, ...]:
        """``[rows, in_features]`` for the weight, ``[rows]`` for the bias."""
        return (rows, in_features) if parameter == "weight" else (rows,)

    def part_rows(self, parameter: str, rows: Sequence[int]) -> tuple[int, ...]:
        """Every stored row is one row of the parameter."""
        del parameter
        return tuple(rows)

    def stored_dtype(self, parameter: str, compute: DType) -> DType:
        """The compute dtype: any 16- or 32-bit float is converted to it."""
        del parameter
        return compute

    def fill(self, parameter: str, target: torch.Tensor, tensor: torch.Tensor) -> None:
        """A plain copy, converting the dtype."""
        del parameter
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
        """``ops.linear``."""
        del scale
        ops.linear(x, weight, bias, out)


#: The method every linear layer uses unless it is built with another one.
UNQUANTIZED = Unquantized()


def quant_method(format: QuantFormat | None) -> QuantMethod:  # noqa: A002  # the record's name
    """The method that reads and runs checkpoints of ``format``; ``None`` is :data:`UNQUANTIZED`."""
    if format is None or format.producer == "mxfp4":
        return UNQUANTIZED
    from ayaka.layers.quantization.fp8 import Fp8Method

    return Fp8Method(format)
