"""Dense feed-forward blocks: the plain MLP and the gated MLP."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import torch
from torch import nn

from ayaka.layers.activation import GELU, GELUAndMul, SiLUAndMul
from ayaka.layers.linear import Linear

if TYPE_CHECKING:
    from ayaka.layers.quantization.method import QuantMethod
    from ayaka.ops.backend import OpsBackend

__all__ = ["MLP", "GatedActivation", "GatedMLP"]

type GatedActivation = Literal["silu", "gelu", "gelu_tanh"]
"""The activations a :class:`GatedMLP` can gate with."""


class MLP(nn.Module):
    """``down_proj(gelu(up_proj(x)))``: the feed-forward block of GPT-2.

    Args:
        hidden_size: Width of the input and the output.
        intermediate_size: Width between the two projections.
        bias: Whether both projections have a bias.
        tanh: Use the tanh approximation of GELU (GPT-2's ``gelu_new``).
        ops: The backend whose ops do the work.
        dtype: Compute dtype of the parameters.
        device: Where the parameters live.
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        *,
        bias: bool = True,
        tanh: bool,
        ops: OpsBackend,
        dtype: torch.dtype,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__()
        self.up_proj = Linear(
            hidden_size, intermediate_size, bias=bias, ops=ops, dtype=dtype, device=device
        )
        self.act = GELU(tanh=tanh, ops=ops)
        self.down_proj = Linear(
            intermediate_size, hidden_size, bias=bias, ops=ops, dtype=dtype, device=device
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Map ``[..., hidden_size]`` to ``[..., hidden_size]``."""
        return self.down_proj(self.act(self.up_proj(x)))


class GatedMLP(nn.Module):
    """``down_proj(act(gate) * up)``: the feed-forward block of Llama-style models.

    The gate and up projections are one :class:`Linear`, ``gate_up_proj``, whose
    output rows are the gate rows followed by the up rows. That is the packing
    the ``*_and_mul`` ops take, and it makes the first projection one matrix
    product. A checkpoint that stores the two separately is concatenated, gate
    first, when it is loaded.

    Args:
        hidden_size: Width of the input and the output.
        intermediate_size: Width of the gate and of the up projection.
        activation: ``"silu"``, or ``"gelu"`` (exact) or ``"gelu_tanh"`` for the gate.
        bias: Whether the projections have a bias.
        ops: The backend whose ops do the work.
        dtype: Compute dtype of the parameters.
        device: Where the parameters live.
        quant: How both projections store and multiply their weights; ``None`` for the
            compute dtype.

    Raises:
        ValueError: If ``activation`` is not one of the names above.
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        *,
        activation: GatedActivation = "silu",
        bias: bool = False,
        ops: OpsBackend,
        dtype: torch.dtype,
        device: torch.device | str | None = None,
        quant: QuantMethod | None = None,
    ) -> None:
        super().__init__()
        self.gate_up_proj = Linear(
            hidden_size,
            2 * intermediate_size,
            bias=bias,
            ops=ops,
            dtype=dtype,
            device=device,
            quant=quant,
        )
        self.act: nn.Module
        match activation:
            case "silu":
                self.act = SiLUAndMul(ops=ops)
            case "gelu" | "gelu_tanh":
                self.act = GELUAndMul(tanh=activation == "gelu_tanh", ops=ops)
            case _:
                raise ValueError(
                    f"activation must be 'silu', 'gelu' or 'gelu_tanh', got {activation!r}"
                )
        self.down_proj = Linear(
            intermediate_size,
            hidden_size,
            bias=bias,
            ops=ops,
            dtype=dtype,
            device=device,
            quant=quant,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Map ``[..., hidden_size]`` to ``[..., hidden_size]``."""
        return self.down_proj(self.act(self.gate_up_proj(x)))
