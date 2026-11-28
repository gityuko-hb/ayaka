"""Normalisation layers."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import nn
from torch.nn import functional

if TYPE_CHECKING:
    from ayaka.ops.backend import OpsBackend

__all__ = ["LayerNorm", "RMSNorm", "RMSNormGated", "ScalelessRMSNorm"]


class RMSNorm(nn.Module):
    """``x / sqrt(mean(x**2) + eps) * weight`` (or ``* (1 + weight)`` when ``unit_offset``)."""

    def __init__(
        self,
        hidden_size: int,
        *,
        eps: float,
        unit_offset: bool = False,
        ops: OpsBackend,
        dtype: torch.dtype,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.eps = eps
        self.unit_offset = unit_offset
        self.weight = nn.Parameter(
            torch.empty(hidden_size, dtype=dtype, device=device), requires_grad=False
        )
        self._ops = ops.norm

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return the normalised ``x`` as a new tensor."""
        if self.unit_offset:
            xf = x.float()
            normed = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)
            return (normed * (1.0 + self.weight.float())).to(x.dtype)
        out = torch.empty_like(x)
        self._ops.rms_norm(x, self.weight, out, eps=self.eps)
        return out

    def fused_add(self, x: torch.Tensor, residual: torch.Tensor) -> None:
        """Set ``residual`` to ``x + residual`` and ``x`` to its normalisation, in place."""
        if self.unit_offset:
            residual.add_(x)
            x.copy_(self.forward(residual))
            return
        self._ops.fused_add_rms_norm(x, residual, self.weight, eps=self.eps)


class ScalelessRMSNorm(nn.Module):
    """RMS normalization without a learned affine weight (used by Gemma 4 ``v_norm``)."""

    def __init__(self, hidden_size: int, *, eps: float) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return RMS-normalized ``x`` without weight scaling."""
        xf = x.float()
        mean_sq = xf.pow(2).mean(-1, keepdim=True) + self.eps
        return (xf * torch.pow(mean_sq, -0.5)).to(x.dtype)


class RMSNormGated(nn.Module):
    """SiLU-gated RMS normalization matching Qwen3.5 linear-attention output norm."""

    def __init__(
        self,
        hidden_size: int,
        *,
        eps: float,
        dtype: torch.dtype,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.eps = eps
        self.weight = nn.Parameter(
            torch.empty(hidden_size, dtype=dtype, device=device), requires_grad=False
        )

    def forward(self, x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        """Normalise ``x`` in float32, scale in ``x.dtype``, then multiply by ``silu(gate)``."""
        xf = x.float()
        normed = (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)).to(x.dtype)
        weighted = self.weight * normed
        return (weighted * functional.silu(gate.float())).to(x.dtype)


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
