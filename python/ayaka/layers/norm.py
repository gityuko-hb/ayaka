"""Normalisation layers."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import nn

if TYPE_CHECKING:
    from ayaka.ops.backend import OpsBackend

__all__ = ["LayerNorm", "RMSNorm"]


class RMSNorm(nn.Module):
    """``x / sqrt(mean(x**2) + eps) * weight`` over the last dimension.

    Args:
        hidden_size: Width of the normalised dimension.
        eps: Added to the mean square; the checkpoint's value, not a default.
        ops: The backend whose ``norm`` ops do the work.
        dtype: Compute dtype of the weight; the input must have it too.
        device: Where the weight lives.
    """

    def __init__(
        self,
        hidden_size: int,
        *,
        eps: float,
        ops: OpsBackend,
        dtype: torch.dtype,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.eps = eps
        self.weight = nn.Parameter(
            torch.empty(hidden_size, dtype=dtype, device=device), requires_grad=False
        )
        self._ops = ops.norm

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return the normalised ``x`` as a new tensor."""
        out = torch.empty_like(x)
        self._ops.rms_norm(x, self.weight, out, eps=self.eps)
        return out

    def fused_add(self, x: torch.Tensor, residual: torch.Tensor) -> None:
        """Set ``residual`` to ``x + residual`` and ``x`` to its normalisation, in place.

        This is the residual-stream step of a pre-norm block: the sum is kept
        for the next addition and its normalised form feeds the next sublayer.
        """
        self._ops.fused_add_rms_norm(x, residual, self.weight, eps=self.eps)


class LayerNorm(nn.Module):
    """``(x - mean) / sqrt(var + eps) * weight + bias`` over the last dimension.

    The variance is the biased one, as in :class:`torch.nn.LayerNorm`.

    Args:
        hidden_size: Width of the normalised dimension.
        eps: Added to the variance; the checkpoint's value, not a default.
        bias: Whether the layer has a bias.
        ops: The backend whose ``norm`` ops do the work.
        dtype: Compute dtype of the parameters; the input must have it too.
        device: Where the parameters live.
    """

    def __init__(
        self,
        hidden_size: int,
        *,
        eps: float,
        bias: bool = True,
        ops: OpsBackend,
        dtype: torch.dtype,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.eps = eps
        self.weight = nn.Parameter(
            torch.empty(hidden_size, dtype=dtype, device=device), requires_grad=False
        )
        self.bias: nn.Parameter | None = (
            nn.Parameter(torch.empty(hidden_size, dtype=dtype, device=device), requires_grad=False)
            if bias
            else None
        )
        self._ops = ops.norm

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return the normalised ``x`` as a new tensor."""
        out = torch.empty_like(x)
        self._ops.layer_norm(x, self.weight, self.bias, out, eps=self.eps)
        return out
