"""What every mixture-of-experts model here shares: its block, its parts and its names.

A layer that routes to experts has a :class:`~ayaka.layers.ffn.moe.MoEBlock` where a dense
model has its gated MLP. Checkpoints store the router as ``mlp.gate.weight`` and the experts one
by one (``mlp.experts.{e}.gate_proj.weight``, ``up_proj``, ``down_proj``); the block stacks them
into two matrices, which :func:`expert_parts` lists as parts, so the loader fills them by rows
like a fused q/k/v.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

from ayaka.layers.ffn.dense import GatedMLP
from ayaka.layers.ffn.moe import MoEBlock

if TYPE_CHECKING:
    import torch
    from torch import nn

    from ayaka.configs.model import ModelConfig
    from ayaka.ops.backend import OpsBackend

__all__ = ["MOE_RENAMES", "expert_parts", "moe_mlp"]

#: Hugging Face names of the router and of each expert's projections.
MOE_RENAMES: Final = (
    (r"model\.layers\.(\d+)\.mlp\.gate\.weight", r"layers.\1.mlp.gate.weight"),
    (
        r"model\.layers\.(\d+)\.mlp\.experts\.(\d+)\.(gate|up|down)_proj\.weight",
        r"layers.\1.mlp.experts.\2.\3_proj.weight",
    ),
)


def moe_mlp(
    config: ModelConfig,
    layer_id: int,
    *,
    ops: OpsBackend,
    dtype: torch.dtype,
    device: torch.device | str | None,
) -> nn.Module:
    """The feed-forward block of ``layer_id``: experts, or a gated MLP for a dense layer."""
    if layer_id in config.dense_layers:
        return GatedMLP(
            config.hidden_size,
            config.intermediate_size,
            activation="silu",
            bias=config.mlp_bias,
            ops=ops,
            dtype=dtype,
            device=device,
        )
    return MoEBlock(
        config.hidden_size,
        config.moe_intermediate_size,
        config.num_experts,
        config.experts_per_token,
        renormalize=config.norm_topk_prob,
        ops=ops,
        dtype=dtype,
        device=device,
    )


def expert_parts(config: ModelConfig, name: str) -> tuple[tuple[str, int], ...]:
    """The stored tensors of a stacked expert matrix, in row order; ``()`` for another name.

    ``layers.N.mlp.experts.gate_up_proj`` is each expert's ``gate_proj`` then ``up_proj``
    (``moe_intermediate_size`` rows each), expert 0 first; ``down_proj`` is each expert's
    ``down_proj`` (``hidden_size`` rows).
    """
    count = config.num_experts
    if name.endswith(".mlp.experts.gate_up_proj"):
        base = name.removesuffix("gate_up_proj")
        rows = config.moe_intermediate_size
        return tuple(
            (f"{base}{expert}.{part}_proj.weight", rows)
            for expert in range(count)
            for part in ("gate", "up")
        )
    if name.endswith(".mlp.experts.down_proj"):
        base = name.removesuffix("down_proj")
        return tuple(
            (f"{base}{expert}.down_proj.weight", config.hidden_size) for expert in range(count)
        )
    return ()
