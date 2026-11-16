"""The affine projection layer."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import nn

if TYPE_CHECKING:
    from ayaka.ops.backend import OpsBackend

__all__ = ["Linear"]


class Linear(nn.Module):
    """``y = x @ weight.T + bias`` over the last dimension, computed by the backend's linear op.

    The parameters are named and shaped like those of :class:`torch.nn.Linear`
    (``weight`` is ``[out_features, in_features]``). They are allocated
    uninitialised and hold no gradient: a loader writes the real values.

    Args:
        in_features: Width of the input.
        out_features: Width of the output.
        bias: Whether the layer has a bias.
        ops: The backend whose ``linear`` op does the work.
        dtype: Compute dtype of the parameters; the input must have it too.
        device: Where the parameters live.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        bias: bool = True,
        ops: OpsBackend,
        dtype: torch.dtype,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(
            torch.empty(out_features, in_features, dtype=dtype, device=device), requires_grad=False
        )
        self.bias: nn.Parameter | None = (
            nn.Parameter(torch.empty(out_features, dtype=dtype, device=device), requires_grad=False)
            if bias
            else None
        )
        self._ops = ops.linear

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Project ``x`` of shape ``[..., in_features]`` to ``[..., out_features]``."""
        out = x.new_empty((*x.shape[:-1], self.out_features))
        self._ops.linear(x, self.weight, self.bias, out)
        return out
