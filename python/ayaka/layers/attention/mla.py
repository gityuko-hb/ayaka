"""Multi-Head Latent Attention (MLA) layer storing compressed latent KV vectors."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import nn

from ayaka.core.errors import EngineError
from ayaka.layers.linear import Linear
from ayaka.layers.norm import RMSNorm

if TYPE_CHECKING:
    from ayaka.layers.context import ForwardContext
    from ayaka.layers.quantization.method import QuantMethod
    from ayaka.layers.rope import RotaryEmbedding
    from ayaka.ops.backend import OpsBackend

__all__ = ["MultiHeadLatentAttention"]


class MultiHeadLatentAttention(nn.Module):
    """Absorbed Multi-Head Latent Attention over a paged :class:`LatentKVBuffers` cache.

    During forward, only compressed latent ``ckv`` and decoupled rotary ``kpe`` are
    written into the cache; ``kv_b_proj`` up-projection weights are absorbed into the
    queries and attended outputs via batched matrix absorption.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        *,
        q_lora_rank: int | None,
        kv_lora_rank: int,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
        v_head_dim: int,
        layer_id: int,
        rotary: RotaryEmbedding | None = None,
        norm_eps: float = 1e-6,
        scale: float | None = None,
        ops: OpsBackend,
        dtype: torch.dtype,
        device: torch.device | str | None = None,
        quant: QuantMethod | None = None,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.q_lora_rank = q_lora_rank
        self.kv_lora_rank = kv_lora_rank
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.v_head_dim = v_head_dim
        self.layer_id = layer_id
        self.rotary = rotary
        self.scale = (qk_nope_head_dim + qk_rope_head_dim) ** -0.5 if scale is None else scale

        q_out_dim = num_heads * (qk_nope_head_dim + qk_rope_head_dim)
        if q_lora_rank is None:
            self.q_proj: Linear | None = Linear(
                hidden_size, q_out_dim, bias=False, ops=ops, dtype=dtype, device=device, quant=quant
            )
            self.q_a_proj: Linear | None = None
            self.q_a_layernorm: RMSNorm | None = None
            self.q_b_proj: Linear | None = None
        else:
            self.q_proj = None
            self.q_a_proj = Linear(
                hidden_size,
                q_lora_rank,
                bias=False,
                ops=ops,
                dtype=dtype,
                device=device,
                quant=quant,
            )
            self.q_a_layernorm = RMSNorm(
                q_lora_rank, eps=norm_eps, ops=ops, dtype=dtype, device=device
            )
            self.q_b_proj = Linear(
                q_lora_rank, q_out_dim, bias=False, ops=ops, dtype=dtype, device=device, quant=quant
            )

        self.kv_a_proj_with_mqa = Linear(
            hidden_size,
            kv_lora_rank + qk_rope_head_dim,
            bias=False,
            ops=ops,
            dtype=dtype,
            device=device,
            quant=quant,
        )
        self.kv_a_layernorm = RMSNorm(
            kv_lora_rank, eps=norm_eps, ops=ops, dtype=dtype, device=device
        )
        # Note: kv_b_proj is kept unquantized (absorbs directly via weight matrix).
        self.kv_b_proj = Linear(
            kv_lora_rank,
            num_heads * (qk_nope_head_dim + v_head_dim),
            bias=False,
            ops=ops,
            dtype=dtype,
            device=device,
            quant=None,
        )
        self.o_proj = Linear(
            num_heads * v_head_dim,
            hidden_size,
            bias=False,
            ops=ops,
            dtype=dtype,
            device=device,
            quant=quant,
        )
        self._mla = ops.mla

    def forward(self, hidden: torch.Tensor, context: ForwardContext) -> torch.Tensor:
        """Compute absorbed MLA over ``hidden`` using compressed latent KV buffers."""
        rows = hidden.shape[0]
        if rows == 0:
            return hidden

        if self.q_proj is not None:
            q_full = self.q_proj(hidden)
        elif (
            self.q_a_proj is not None
            and self.q_a_layernorm is not None
            and self.q_b_proj is not None
        ):
            q_full = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(hidden)))
        else:
            raise EngineError(f"MLA layer {self.layer_id} missing query projection modules")
        q_view = q_full.reshape(rows, self.num_heads, self.qk_nope_head_dim + self.qk_rope_head_dim)
        q_nope, q_pe = q_view.split([self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
        q_pe = q_pe.contiguous()

        kv_comp = self.kv_a_proj_with_mqa(hidden)
        ckv_raw, kpe = kv_comp.split([self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        ckv = self.kv_a_layernorm(ckv_raw.contiguous())
        kpe_heads = kpe.reshape(rows, 1, self.qk_rope_head_dim).contiguous()

        meta = context.attention_for(self.layer_id)
        if self.rotary is not None:
            self.rotary(meta.positions, q_pe, kpe_heads)
        kpe_flat = kpe_heads.squeeze(1).contiguous()

        w_kv_b = self.kv_b_proj.weight.reshape(
            self.num_heads, self.qk_nope_head_dim + self.v_head_dim, self.kv_lora_rank
        )
        w_k = w_kv_b[:, : self.qk_nope_head_dim, :]  # [H, D_nope, kv_lora_rank]
        w_v = w_kv_b[:, self.qk_nope_head_dim :, :]  # [H, D_v, kv_lora_rank]

        q_nope_absorbed = torch.einsum("nhd,hdc->nhc", q_nope, w_k).contiguous()
        buffers = context.latent_buffers(self.layer_id)
        self._mla.store_latent(ckv, kpe_flat, buffers, meta)

        latent_out = torch.empty(
            (rows, self.num_heads, self.kv_lora_rank),
            dtype=q_nope.dtype,
            device=hidden.device,
        )
        self._mla.attend_mla(q_nope_absorbed, q_pe, buffers, meta, latent_out, scale=self.scale)
        v_out = torch.einsum("nhc,hvc->nhv", latent_out, w_v)
        return self.o_proj(v_out.reshape(rows, self.num_heads * self.v_head_dim))
