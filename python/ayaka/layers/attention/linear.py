"""Reusable linear-attention layers (Gated DeltaNet and Kimi Delta Attention)."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch
from torch import nn
from torch.nn import functional

from ayaka.core.errors import EngineError
from ayaka.layers.linear import Linear

if TYPE_CHECKING:
    from ayaka.layers.context import ForwardContext, StateCheckpointCapture
    from ayaka.layers.quantization.method import QuantMethod
    from ayaka.ops.backend import OpsBackend
    from ayaka.ops.linear_attention import LinearAttentionMetadata, RecurrentStateBuffers

__all__ = ["GatedDeltaNetAttention", "KimiDeltaAttention"]


def _conv_tail_at(
    prev_conv: torch.Tensor,
    req_x: torch.Tensor,
    offset: int,
    kernel_size: int,
) -> torch.Tensor:
    """Construct the exact ``[conv_dim, K]`` conv ring buffer at token ``offset``."""
    if offset >= kernel_size:
        return req_x[offset - kernel_size : offset].transpose(0, 1).to(prev_conv.dtype)
    take_prev = kernel_size - offset
    tail_x = req_x[:offset].transpose(0, 1).to(prev_conv.dtype)
    return torch.cat([prev_conv[:, -take_prev:], tail_x], dim=-1)


class _BaseDeltaAttention(nn.Module):
    """Shared implementation of gated delta-rule and KDA recurrent attention layers."""

    def __init__(
        self,
        hidden_size: int,
        num_k_heads: int,
        num_v_heads: int,
        key_head_dim: int,
        value_head_dim: int,
        conv_kernel_size: int,
        *,
        layer_id: int,
        per_channel_decay: bool,
        use_qk_l2norm: bool = True,
        norm_eps: float = 1e-6,
        ops: OpsBackend,
        dtype: torch.dtype,
        device: torch.device | str | None = None,
        quant: QuantMethod | None = None,
    ) -> None:
        super().__init__()
        if num_k_heads <= 0 or num_v_heads % num_k_heads:
            raise ValueError(
                f"num_k_heads ({num_k_heads}) must positively divide num_v_heads ({num_v_heads})"
            )
        self.hidden_size = hidden_size
        self.num_k_heads = num_k_heads
        self.num_v_heads = num_v_heads
        self.key_head_dim = key_head_dim
        self.value_head_dim = value_head_dim
        self.key_dim = num_k_heads * key_head_dim
        self.value_dim = num_v_heads * value_head_dim
        self.conv_dim = 2 * self.key_dim + self.value_dim
        self.conv_kernel_size = conv_kernel_size
        self.layer_id = layer_id
        self.per_channel_decay = per_channel_decay
        self.use_qk_l2norm = use_qk_l2norm
        self.norm_eps = norm_eps
        self.scale = key_head_dim**-0.5

        self.in_proj_qkv = Linear(
            hidden_size, self.conv_dim, bias=False, ops=ops, dtype=dtype, device=device, quant=quant
        )
        self.in_proj_z = Linear(
            hidden_size,
            self.value_dim,
            bias=False,
            ops=ops,
            dtype=dtype,
            device=device,
            quant=quant,
        )
        a_width = num_v_heads * key_head_dim if per_channel_decay else num_v_heads
        self.in_proj_a = Linear(
            hidden_size, a_width, bias=False, ops=ops, dtype=dtype, device=device, quant=quant
        )
        self.in_proj_b = Linear(
            hidden_size, num_v_heads, bias=False, ops=ops, dtype=dtype, device=device, quant=quant
        )
        self.conv1d_weight = nn.Parameter(
            torch.empty((self.conv_dim, 1, conv_kernel_size), dtype=dtype, device=device),
            requires_grad=False,
        )
        self.a_log = nn.Parameter(
            torch.empty(
                (num_v_heads, key_head_dim) if per_channel_decay else (num_v_heads,),
                dtype=dtype,
                device=device,
            ),
            requires_grad=False,
        )
        self.dt_bias = nn.Parameter(
            torch.empty(
                (num_v_heads, key_head_dim) if per_channel_decay else (num_v_heads,),
                dtype=dtype,
                device=device,
            ),
            requires_grad=False,
        )
        self.norm_weight = nn.Parameter(
            torch.empty(value_head_dim, dtype=dtype, device=device), requires_grad=False
        )
        self.out_proj = Linear(
            self.value_dim,
            hidden_size,
            bias=False,
            ops=ops,
            dtype=dtype,
            device=device,
            quant=quant,
        )
        self._linear_attn = ops.linear_attention

    def forward(self, hidden: torch.Tensor, context: ForwardContext) -> torch.Tensor:
        """Apply linear-attention step over ``hidden`` and return ``[N, hidden_size]``."""
        lin_meta = context.linear_attention
        if lin_meta is None:
            raise EngineError(
                f"linear-attention layer {self.layer_id} requires ForwardContext.linear_attention",
                context={"layer_id": self.layer_id},
            )
        state = context.state_for(self.layer_id)
        rows = hidden.shape[0]
        if rows == 0:
            return hidden

        mixed_raw = self.in_proj_qkv(hidden)
        z = self.in_proj_z(hidden).reshape(rows, self.num_v_heads, self.value_head_dim)
        a = self.in_proj_a(hidden)
        b = self.in_proj_b(hidden)
        if self.per_channel_decay:
            a = a.reshape(rows, self.num_v_heads, self.key_head_dim)
        g = -self.a_log.float().exp() * functional.softplus(a.float() + self.dt_bias.float())
        beta = torch.sigmoid(b.float())

        capture = context.checkpoint_capture
        has_ckpts = (
            lin_meta.is_prefill
            and capture is not None
            and any(bool(entry) for entry in capture.boundaries)
        )
        conv_out = torch.empty_like(mixed_raw)
        core_out = torch.empty(
            (rows, self.num_v_heads, self.value_head_dim), dtype=hidden.dtype, device=hidden.device
        )
        if has_ckpts and capture is not None:
            self._run_prefill_with_checkpoints(
                mixed_raw, g, beta, conv_out, core_out, state, lin_meta, capture
            )
        else:
            self._linear_attn.causal_conv1d(
                mixed_raw,
                self.conv1d_weight,
                state.conv_state,
                lin_meta.state_slots,
                lin_meta.cu_seqlens,
                is_prefill=lin_meta.is_prefill,
                out=conv_out,
            )
            q, k, v = self._split_qkv(conv_out, rows)
            recurrence = (
                self._linear_attn.kda_recurrence
                if self.per_channel_decay
                else self._linear_attn.gated_delta_rule
            )
            recurrence(
                q,
                k,
                v,
                g,
                beta,
                state.ssm_state,
                lin_meta.state_slots,
                lin_meta.cu_seqlens,
                is_prefill=lin_meta.is_prefill,
                scale=self.scale,
                use_qk_l2norm=self.use_qk_l2norm,
                out=core_out,
            )

        normed = self._gated_rms_norm(core_out, z)
        return self.out_proj(normed.reshape(rows, self.value_dim))

    def _split_qkv(
        self, conv_out: torch.Tensor, rows: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        q_end = self.key_dim
        k_end = q_end + self.key_dim
        return (
            conv_out[:, :q_end].reshape(rows, self.num_k_heads, self.key_head_dim).contiguous(),
            conv_out[:, q_end:k_end]
            .reshape(rows, self.num_k_heads, self.key_head_dim)
            .contiguous(),
            conv_out[:, k_end:].reshape(rows, self.num_v_heads, self.value_head_dim).contiguous(),
        )

    def _gated_rms_norm(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        xf = x.float()
        variance = xf.pow(2).mean(dim=-1, keepdim=True)
        normed = xf * torch.rsqrt(variance + self.norm_eps) * self.norm_weight.float()
        gated = normed * functional.silu(z.float())
        return gated.to(x.dtype)

    def _run_prefill_with_checkpoints(
        self,
        mixed_raw: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        conv_out: torch.Tensor,
        core_out: torch.Tensor,
        state: RecurrentStateBuffers,
        lin_meta: LinearAttentionMetadata,
        capture: StateCheckpointCapture,
    ) -> None:
        """Compute conv1d and split delta-rule spans at requested checkpoint boundaries."""
        offsets = [
            int(x)
            for x in cast("list[int]", lin_meta.cu_seqlens.tolist())  # pyright: ignore[reportUnknownMemberType]
        ]
        slots = [
            int(x)
            for x in cast("list[int]", lin_meta.state_slots.tolist())  # pyright: ignore[reportUnknownMemberType]
        ]
        prev_conv = state.conv_state[lin_meta.state_slots.long()].clone()

        self._linear_attn.causal_conv1d(
            mixed_raw,
            self.conv1d_weight,
            state.conv_state,
            lin_meta.state_slots,
            lin_meta.cu_seqlens,
            is_prefill=True,
            out=conv_out,
        )
        rows = mixed_raw.shape[0]
        q_all, k_all, v_all = self._split_qkv(conv_out, rows)
        recurrence = (
            self._linear_attn.kda_recurrence
            if self.per_channel_decay
            else self._linear_attn.gated_delta_rule
        )

        for req_idx, (slot, boundaries) in enumerate(zip(slots, capture.boundaries, strict=True)):
            start = offsets[req_idx]
            stop = offsets[req_idx + 1]
            seq_len = stop - start
            if seq_len <= 0 or slot == 0:
                if seq_len > 0:
                    core_out[start:stop].zero_()
                continue
            req_slot = lin_meta.state_slots[req_idx : req_idx + 1]
            req_x = mixed_raw[start:stop]
            spans = [off for off, _ in boundaries if 0 < off <= seq_len]
            points = [*spans, seq_len] if not spans or spans[-1] != seq_len else spans
            digest_by_off = dict(boundaries)
            cursor = 0
            for end in points:
                sub_len = end - cursor
                sub_cu = torch.tensor([0, sub_len], dtype=torch.int32, device=mixed_raw.device)
                s0 = start + cursor
                s1 = start + end
                recurrence(
                    q_all[s0:s1],
                    k_all[s0:s1],
                    v_all[s0:s1],
                    g[s0:s1],
                    beta[s0:s1],
                    state.ssm_state,
                    req_slot,
                    sub_cu,
                    is_prefill=True,
                    scale=self.scale,
                    use_qk_l2norm=self.use_qk_l2norm,
                    out=core_out[s0:s1],
                )
                digest = digest_by_off.get(end)
                if digest is not None:
                    conv_snap = _conv_tail_at(prev_conv[req_idx], req_x, end, self.conv_kernel_size)
                    capture.snapshots[(self.layer_id, digest)] = (
                        state.ssm_state[slot].clone().cpu(),
                        conv_snap.cpu(),
                    )
                cursor = end


class GatedDeltaNetAttention(_BaseDeltaAttention):
    """Gated DeltaNet linear-attention layer with scalar per-head decay."""

    def __init__(
        self,
        hidden_size: int,
        num_k_heads: int,
        num_v_heads: int,
        key_head_dim: int,
        value_head_dim: int,
        conv_kernel_size: int,
        *,
        layer_id: int,
        use_qk_l2norm: bool = True,
        norm_eps: float = 1e-6,
        ops: OpsBackend,
        dtype: torch.dtype,
        device: torch.device | str | None = None,
        quant: QuantMethod | None = None,
    ) -> None:
        super().__init__(
            hidden_size,
            num_k_heads,
            num_v_heads,
            key_head_dim,
            value_head_dim,
            conv_kernel_size,
            layer_id=layer_id,
            per_channel_decay=False,
            use_qk_l2norm=use_qk_l2norm,
            norm_eps=norm_eps,
            ops=ops,
            dtype=dtype,
            device=device,
            quant=quant,
        )


class KimiDeltaAttention(_BaseDeltaAttention):
    """Kimi Delta Attention linear-attention layer with per-key-channel decay."""

    def __init__(
        self,
        hidden_size: int,
        num_k_heads: int,
        num_v_heads: int,
        key_head_dim: int,
        value_head_dim: int,
        conv_kernel_size: int,
        *,
        layer_id: int,
        use_qk_l2norm: bool = True,
        norm_eps: float = 1e-6,
        ops: OpsBackend,
        dtype: torch.dtype,
        device: torch.device | str | None = None,
        quant: QuantMethod | None = None,
    ) -> None:
        super().__init__(
            hidden_size,
            num_k_heads,
            num_v_heads,
            key_head_dim,
            value_head_dim,
            conv_kernel_size,
            layer_id=layer_id,
            per_channel_decay=True,
            use_qk_l2norm=use_qk_l2norm,
            norm_eps=norm_eps,
            ops=ops,
            dtype=dtype,
            device=device,
            quant=quant,
        )
