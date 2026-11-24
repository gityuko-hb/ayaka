"""The mixture-of-experts feed-forward block, with every expert in device memory (``fused``).

A router (``gate``, a :class:`~ayaka.layers.linear.Linear` with one output per expert) scores
each row; :meth:`~ayaka.ops.moe.MoeOps.route` keeps the ``top_k`` experts and their weights; and
:meth:`~ayaka.ops.moe.MoeOps.experts` sums the chosen SiLU-gated expert MLPs, weighted::

    weights, ids = route(gate(x), k)
    y = sum_j weights[j] * down[ids[j]] @ (silu(gate_e) * up_e)

The experts of a layer are two stacked tensors held as matrices, so that a checkpoint that
stores the experts one by one fills them by rows (``CausalLM.stored_parts``), like a fused
q/k/v: ``experts.gate_up_proj`` is ``[E * 2I, H]`` (expert ``e``'s gate rows, then its up rows)
and ``experts.down_proj`` is ``[E * H, I]``. The op sees them as ``[E, 2I, H]`` and
``[E, H, I]``, views of the same memory.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import nn

from ayaka.layers.linear import Linear

if TYPE_CHECKING:
    from ayaka.ops.backend import OpsBackend

__all__ = ["MoEBlock", "StackedExperts"]


class StackedExperts(nn.Module):
    """The expert weights of one layer, stacked: ``gate_up_proj`` and ``down_proj``.

    Args:
        hidden_size: ``H``, the width of the stream.
        intermediate_size: ``I``, the width of one expert's gate and of its up projection.
        num_experts: ``E``.
        dtype: Compute dtype of the parameters.
        device: Where the parameters live.
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_experts: int,
        *,
        dtype: torch.dtype,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_experts = num_experts
        self.gate_up_proj = nn.Parameter(
            torch.empty(
                num_experts * 2 * intermediate_size, hidden_size, dtype=dtype, device=device
            ),
            requires_grad=False,
        )
        self.down_proj = nn.Parameter(
            torch.empty(num_experts * hidden_size, intermediate_size, dtype=dtype, device=device),
            requires_grad=False,
        )

    def stacked(self) -> tuple[torch.Tensor, torch.Tensor]:
        """``gate_up`` ``[E, 2I, H]`` and ``down`` ``[E, H, I]``: views, nothing is copied."""
        count, inner, hidden = self.num_experts, self.intermediate_size, self.hidden_size
        return (
            self.gate_up_proj.view(count, 2 * inner, hidden),
            self.down_proj.view(count, hidden, inner),
        )


class MoEBlock(nn.Module):
    """Top-k routed SiLU-gated experts, all in device memory.

    Args:
        hidden_size: Width of the input and the output.
        intermediate_size: Width of one expert's gate and up projections.
        num_experts: Experts in the layer.
        top_k: Experts each row is sent to.
        renormalize: Divide the ``top_k`` weights of a row by their sum
            (``norm_topk_prob``); otherwise they stay softmax probabilities over all experts.
        ops: The backend whose ``moe`` and ``linear`` ops do the work.
        dtype: Compute dtype of the parameters.
        device: Where the parameters live.

    Raises:
        ValueError: If ``top_k`` is not between 1 and ``num_experts``.
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_experts: int,
        top_k: int,
        *,
        renormalize: bool,
        ops: OpsBackend,
        dtype: torch.dtype,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__()
        if not 1 <= top_k <= num_experts:
            raise ValueError(f"top_k must be between 1 and {num_experts}, got {top_k}")
        self.top_k = top_k
        self.renormalize = renormalize
        self.gate = Linear(
            hidden_size, num_experts, bias=False, ops=ops, dtype=dtype, device=device
        )
        self.experts = StackedExperts(
            hidden_size, intermediate_size, num_experts, dtype=dtype, device=device
        )
        self._moe = ops.moe

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Map ``[..., hidden_size]`` to ``[..., hidden_size]``."""
        rows = x.reshape(-1, x.shape[-1])
        logits = self.gate(rows)
        weights = torch.empty(rows.shape[0], self.top_k, dtype=torch.float32, device=x.device)
        ids = torch.empty(rows.shape[0], self.top_k, dtype=torch.int32, device=x.device)
        self._moe.route(logits, weights, ids, renormalize=self.renormalize)
        gate_up, down = self.experts.stacked()
        out = torch.empty_like(rows)
        self._moe.experts(rows, gate_up, down, weights, ids, out)
        return out.view(x.shape)
