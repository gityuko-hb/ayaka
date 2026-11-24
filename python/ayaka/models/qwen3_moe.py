"""Qwen3-MoE: Qwen3 attention with mixture-of-experts feed-forward blocks.

Qwen3-MoE checkpoints (``Qwen3MoeForCausalLM``: Qwen3-30B-A3B, 128 experts, 8 per token) are
:mod:`ayaka.models.qwen3` with :class:`~ayaka.layers.ffn.moe.MoEBlock` in place of the dense
MLP: softmax over the router, the 8 largest, weights renormalised when ``norm_topk_prob`` is
set, each expert a SiLU-gated MLP of ``moe_intermediate_size``. A layer named in
``mlp_only_layers``, or skipped by ``decoder_sparse_step``, keeps the dense MLP of
``intermediate_size``.

Every expert lives in device memory. Quantized checkpoints are refused: the experts have no
8-bit path.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from ayaka.core.errors import ModelLoadError
from ayaka.models._moe import MOE_RENAMES, expert_parts, moe_mlp
from ayaka.models.qwen3 import Qwen3ForCausalLM

if TYPE_CHECKING:
    import torch
    from torch import nn

    from ayaka.configs.model import ModelConfig
    from ayaka.ops.backend import OpsBackend
    from ayaka.utils.dtypes import DType
    from ayaka.weights.quant import QuantFormat

__all__ = ["Qwen3MoeForCausalLM"]


class Qwen3MoeForCausalLM(Qwen3ForCausalLM):
    """Qwen3-MoE with a language-model head.

    The parameters are those of :class:`~ayaka.models.qwen3.Qwen3ForCausalLM`, with
    ``layers.N.mlp.gate.weight`` and ``layers.N.mlp.experts.{gate_up,down}_proj`` in every
    layer that routes to experts.

    Raises:
        ModelLoadError: If the config is not an MoE config.
    """

    checkpoint_renames: ClassVar[tuple[tuple[str, str], ...]] = (
        *MOE_RENAMES,
        *Qwen3ForCausalLM.checkpoint_renames,
    )
    quantizable: ClassVar[bool] = False

    def __init__(
        self,
        config: ModelConfig,
        *,
        ops: OpsBackend,
        dtype: DType,
        device: torch.device | str | None = None,
        quant: QuantFormat | None = None,
    ) -> None:
        if not config.is_moe:
            raise ModelLoadError(
                f"{config.architecture} needs num_experts and num_experts_per_tok",
                context={"architecture": config.architecture},
            )
        super().__init__(config, ops=ops, dtype=dtype, device=device, quant=quant)

    def build_mlp(
        self,
        layer_id: int,
        *,
        ops: OpsBackend,
        dtype: torch.dtype,
        device: torch.device | str | None,
    ) -> nn.Module:
        """The experts of block ``layer_id``, or the dense MLP of a dense layer."""
        return moe_mlp(self.config, layer_id, ops=ops, dtype=dtype, device=device)

    def stored_parts(self, name: str) -> tuple[tuple[str, int], ...]:
        """Qwen3's fused q/k/v and gate/up, and each layer's experts stored one by one."""
        return expert_parts(self.config, name) or super().stored_parts(name)
