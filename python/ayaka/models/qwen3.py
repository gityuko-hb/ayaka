"""Qwen3: the Llama block with an RMSNorm over each query head and each key head.

Qwen3 checkpoints (``Qwen3ForCausalLM``) have no projection bias by default
(``attention_bias``) and normalise each head of the queries and of the keys with an RMSNorm of
width ``head_dim`` (``q_norm``, ``k_norm``, one weight shared by every head) after the
projection and before the rotary embedding::

    q = rope(q_norm(q_proj(h).view(heads, head_dim)))
    k = rope(k_norm(k_proj(h).view(kv_heads, head_dim)))

``head_dim`` is set in ``config.json`` apart from ``hidden_size`` (Qwen3-1.7B: 16 heads of 128
on a 2,048-wide stream). Everything else is :mod:`ayaka.models.llama`: names, fused
projections, rotary embedding, norms and the SiLU-gated MLP.

Qwen3 writes its reasoning between ``<think>`` and ``</think>`` and its tool calls as
``<tool_call>`` JSON (the ``think`` and ``hermes`` parsers).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from ayaka.layers.attention.dense import Attention
from ayaka.models.llama import LlamaForCausalLM

if TYPE_CHECKING:
    import torch

    from ayaka.ops.backend import OpsBackend

__all__ = ["Qwen3ForCausalLM"]


class Qwen3ForCausalLM(LlamaForCausalLM):
    """Qwen3 with a language-model head.

    The parameters are those of :class:`~ayaka.models.llama.LlamaForCausalLM` plus
    ``layers.N.attn.q_norm.weight`` and ``layers.N.attn.k_norm.weight``, each ``[head_dim]``.
    """

    checkpoint_renames: ClassVar[tuple[tuple[str, str], ...]] = (
        *LlamaForCausalLM.checkpoint_renames,
        (r"model\.layers\.(\d+)\.self_attn\.(q_norm|k_norm)\.weight", r"layers.\1.attn.\2.weight"),
    )

    def build_attention(
        self,
        layer_id: int,
        *,
        ops: OpsBackend,
        dtype: torch.dtype,
        device: torch.device | str | None,
    ) -> Attention:
        """Grouped-query attention with a per-head RMSNorm on q and k before the rotary."""
        config = self.config
        return Attention(
            config.hidden_size,
            config.num_attention_heads,
            config.num_kv_heads,
            config.head_dim,
            layer_id=layer_id,
            bias=config.attention_bias,
            rotary=self.rotary,
            qk_norm="head",
            norm_eps=config.norm_eps,
            ops=ops,
            dtype=dtype,
            device=device,
        )
