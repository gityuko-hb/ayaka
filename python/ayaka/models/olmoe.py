"""OLMoE: Llama attention with an RMSNorm over the whole query and key projections, and experts.

OLMoE checkpoints (``OlmoeForCausalLM``: OLMoE-1B-7B, 64 experts, 8 per token) differ from the
Llama block in two places:

* attention normalises the queries and the keys with one RMSNorm over each whole projection
  (``q_norm`` ``[heads * head_dim]``, ``k_norm`` ``[kv_heads * head_dim]``), after the
  projection and before the rotary embedding;
* every feed-forward block is a mixture of experts (:class:`~ayaka.layers.ffn.moe.MoEBlock`):
  softmax over the router, the 8 largest, weights not renormalised (``norm_topk_prob``
  false), each expert a SiLU-gated MLP of ``intermediate_size``.

``clip_qkv`` is refused when set. Every expert lives in device memory. Quantized checkpoints are
refused: the experts have no 8-bit path.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from ayaka.core.errors import ModelLoadError
from ayaka.layers.attention.dense import Attention
from ayaka.models._moe import MOE_RENAMES, expert_parts, moe_mlp
from ayaka.models.llama import LlamaForCausalLM

if TYPE_CHECKING:
    import torch
    from torch import nn

    from ayaka.configs.model import ModelConfig
    from ayaka.ops.backend import OpsBackend
    from ayaka.utils.dtypes import DType
    from ayaka.weights.quant import QuantFormat

__all__ = ["OlmoeForCausalLM"]


class OlmoeForCausalLM(LlamaForCausalLM):
    """OLMoE with a language-model head.

    The parameters are those of :class:`~ayaka.models.llama.LlamaForCausalLM`, with
    ``layers.N.attn.{q,k}_norm.weight`` and, in place of ``layers.N.mlp``'s dense projections,
    ``layers.N.mlp.gate.weight`` ``[E, H]`` and ``layers.N.mlp.experts.{gate_up,down}_proj``.

    Raises:
        ModelLoadError: If the config is not an MoE config, or sets ``clip_qkv``.
    """

    checkpoint_renames: ClassVar[tuple[tuple[str, str], ...]] = (
        *MOE_RENAMES,
        *LlamaForCausalLM.checkpoint_renames,
        (r"model\.layers\.(\d+)\.self_attn\.(q_norm|k_norm)\.weight", r"layers.\1.attn.\2.weight"),
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

    def build_attention(
        self,
        layer_id: int,
        *,
        ops: OpsBackend,
        dtype: torch.dtype,
        device: torch.device | str | None,
    ) -> Attention:
        """Grouped-query attention with an RMSNorm over the whole q and k projections."""
        config = self.config
        return Attention(
            config.hidden_size,
            config.num_attention_heads,
            config.num_kv_heads,
            config.head_dim,
            layer_id=layer_id,
            bias=config.attention_bias,
            rotary=self.rotary,
            qk_norm="full",
            norm_eps=config.norm_eps,
            ops=ops,
            dtype=dtype,
            device=device,
        )

    def build_mlp(
        self,
        layer_id: int,
        *,
        ops: OpsBackend,
        dtype: torch.dtype,
        device: torch.device | str | None,
    ) -> nn.Module:
        """The experts of block ``layer_id``."""
        return moe_mlp(self.config, layer_id, ops=ops, dtype=dtype, device=device)

    def stored_parts(self, name: str) -> tuple[tuple[str, int], ...]:
        """Llama's fused q/k/v, and each layer's experts stored one by one."""
        return expert_parts(self.config, name) or super().stored_parts(name)
