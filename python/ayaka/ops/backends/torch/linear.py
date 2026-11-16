"""Reference linear ops; implements :class:`ayaka.ops.linear.LinearOps`."""

from __future__ import annotations

from typing import TYPE_CHECKING

from torch.nn import functional

from ayaka.core.errors import BackendError
from ayaka.ops.backends.torch._checks import require_compute_dtype, require_like

if TYPE_CHECKING:
    import torch

__all__ = ["linear"]


def linear(
    x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None, out: torch.Tensor
) -> None:
    op = "linear.linear"
    require_compute_dtype(op, "x", x)
    if weight.ndim != 2:
        raise BackendError(
            f"{op}: weight must be [out_features, in_features], got {list(weight.shape)}",
            context={"op": op, "argument": "weight"},
        )
    out_features = weight.shape[0]
    require_like(op, "weight", weight, (out_features, x.shape[-1]), x.dtype)
    if bias is not None:
        require_like(op, "bias", bias, (out_features,), x.dtype)
    require_like(op, "out", out, (*x.shape[:-1], out_features), x.dtype)
    # Computed in the dtype of the inputs: widening would copy the whole weight.
    out.copy_(functional.linear(x, weight, bias))
