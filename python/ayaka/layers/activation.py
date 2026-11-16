"""Activation layers: plain GELU and the gated forms fused with the gate of a gated MLP."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import nn

if TYPE_CHECKING:
    from ayaka.ops.backend import OpsBackend

__all__ = ["GELU", "GELUAndMul", "SiLUAndMul"]


class GELU(nn.Module):
    """Element-wise GELU, exact or the tanh approximation.

    Args:
        tanh: Use the tanh approximation (GPT-2's ``gelu_new``) instead of the erf form.
        ops: The backend whose ``activation`` ops do the work.
    """

    def __init__(self, *, tanh: bool, ops: OpsBackend) -> None:
        super().__init__()
        self.tanh = tanh
        self._ops = ops.activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return ``gelu(x)`` as a new tensor of the shape of ``x``."""
        out = torch.empty_like(x)
        self._ops.gelu(x, out, tanh=self.tanh)
        return out


class SiLUAndMul(nn.Module):
    """``silu(a) * b`` for ``x = [a, b]`` packed along the last dimension.

    The input is ``[..., 2 * d]`` and the output ``[..., d]``.

    Args:
        ops: The backend whose ``activation`` ops do the work.
    """

    def __init__(self, *, ops: OpsBackend) -> None:
        super().__init__()
        self._ops = ops.activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return ``silu(a) * b`` as a new tensor."""
        out = x.new_empty((*x.shape[:-1], x.shape[-1] // 2))
        self._ops.silu_and_mul(x, out)
        return out


class GELUAndMul(nn.Module):
    """``gelu(a) * b`` for ``x = [a, b]`` packed along the last dimension.

    The input is ``[..., 2 * d]`` and the output ``[..., d]``.

    Args:
        tanh: Use the tanh approximation of GELU instead of the erf form.
        ops: The backend whose ``activation`` ops do the work.
    """

    def __init__(self, *, tanh: bool, ops: OpsBackend) -> None:
        super().__init__()
        self.tanh = tanh
        self._ops = ops.activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return ``gelu(a) * b`` as a new tensor."""
        out = x.new_empty((*x.shape[:-1], x.shape[-1] // 2))
        self._ops.gelu_and_mul(x, out, tanh=self.tanh)
        return out
