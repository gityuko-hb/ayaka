"""Reference linear-attention ops (causal conv1d, Gated DeltaNet, Kimi Delta Attention)."""

from __future__ import annotations

from typing import cast

import torch
from torch.nn import functional

from ayaka.core.errors import BackendError
from ayaka.ops.backends.torch._checks import (
    require_compute_dtype,
    require_index_tensor,
    require_like,
)

__all__ = [
    "causal_conv1d",
    "gated_delta_rule",
    "kda_recurrence",
    "validate_conv1d_inputs",
    "validate_delta_inputs",
]


def _l2_normalize(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    inv_norm = torch.rsqrt((x * x).sum(dim=-1, keepdim=True) + eps)
    return x * inv_norm


def validate_conv1d_inputs(
    op: str,
    x: torch.Tensor,
    weight: torch.Tensor,
    conv_state: torch.Tensor,
    state_slots: torch.Tensor,
    out: torch.Tensor,
    bias: torch.Tensor | None,
) -> tuple[torch.Tensor, int, int, int]:
    """Validate shapes/dtypes for causal conv1d and return ``(weight_2d, rows, conv_dim, K)``."""
    require_compute_dtype(op, "x", x)
    if x.ndim != 2:
        raise BackendError(f"{op}: x must be 2D [N, conv_dim], got {list(x.shape)}")
    rows, conv_dim = x.shape
    require_like(op, "out", out, x.shape, x.dtype)
    if weight.ndim == 3 and weight.shape[1] == 1:
        w2 = weight[:, 0, :]
    elif weight.ndim == 2:
        w2 = weight
    else:
        raise BackendError(
            f"{op}: weight must be [conv_dim, K] or [conv_dim, 1, K], got {list(weight.shape)}"
        )
    if w2.shape[0] != conv_dim:
        raise BackendError(f"{op}: weight conv_dim {w2.shape[0]} != x conv_dim {conv_dim}")
    kernel_size = int(w2.shape[1])
    if (
        conv_state.ndim != 3
        or conv_state.shape[1] != conv_dim
        or conv_state.shape[2] != kernel_size
    ):
        raise BackendError(
            f"{op}: conv_state must be [max_slots, {conv_dim}, {kernel_size}], "
            f"got {list(conv_state.shape)}"
        )
    require_index_tensor(op, "state_slots", state_slots, ndim=1)
    if bias is not None:
        require_like(op, "bias", bias, (conv_dim,), bias.dtype)
    return w2, rows, conv_dim, kernel_size


def _host_ints(tensor: torch.Tensor) -> list[int]:
    return [int(x) for x in cast("list[int]", tensor.tolist())]  # pyright: ignore[reportUnknownMemberType]


def causal_conv1d(
    x: torch.Tensor,
    weight: torch.Tensor,
    conv_state: torch.Tensor,
    state_slots: torch.Tensor,
    cu_seqlens: torch.Tensor,
    *,
    is_prefill: bool,
    out: torch.Tensor,
    bias: torch.Tensor | None = None,
) -> None:
    op = "linear_attention.causal_conv1d"
    w2, rows, conv_dim, kernel_size = validate_conv1d_inputs(
        op, x, weight, conv_state, state_slots, out, bias
    )
    if rows == 0:
        return

    w_fp32 = w2.float()
    b_fp32 = bias.float() if bias is not None else None
    if not is_prefill:
        # Single-row decode: CUDA-graph safe (fixed-shape tensor ops, zero host sync).
        slots = state_slots.long()
        if rows != slots.shape[0]:
            raise BackendError(
                f"{op}: decode requires 1 row per slot ({slots.shape[0]}), got {rows}"
            )
        prev = conv_state[slots]  # [B, C, K]
        updated = torch.cat([prev[:, :, 1:], x.to(conv_state.dtype).unsqueeze(-1)], dim=-1)
        conv_state.index_copy_(0, slots, updated)
        acc = (updated.float() * w_fp32[None, :, :]).sum(dim=-1)
        if b_fp32 is not None:
            acc = acc + b_fp32[None, :]
        res = functional.silu(acc).to(out.dtype)
        res = torch.where((slots > 0)[:, None], res, 0.0)
        out.copy_(res)
        return

    # Variable-length ragged prefill: continues seamlessly from prior conv_state (if chunk > 1).
    require_index_tensor(op, "cu_seqlens", cu_seqlens, ndim=1)
    offsets = _host_ints(cu_seqlens)
    slots_list = _host_ints(state_slots)
    if len(offsets) != len(slots_list) + 1:
        raise BackendError(
            f"{op}: cu_seqlens length ({len(offsets)}) must equal len(state_slots) + 1"
        )
    for req_idx, slot in enumerate(slots_list):
        start = int(offsets[req_idx])
        stop = int(offsets[req_idx + 1])
        seq_len = stop - start
        if seq_len <= 0:
            continue
        if slot == 0:
            out[start:stop].zero_()
            continue
        seq_x = x[start:stop].float().transpose(0, 1)  # [C, seq_len]
        if kernel_size > 1:
            prefix = conv_state[slot, :, -(kernel_size - 1) :].float()  # [C, K-1]
            padded = torch.cat([prefix, seq_x], dim=-1)  # [C, K-1 + seq_len]
        else:
            padded = seq_x
        conv_out = functional.conv1d(
            padded.unsqueeze(0),
            w_fp32.unsqueeze(1),
            bias=b_fp32,
            groups=conv_dim,
        ).squeeze(0)  # [C, seq_len]
        out[start:stop] = functional.silu(conv_out).transpose(0, 1).to(out.dtype)
        # Update conv_state tail to last `kernel_size` inputs (padded on the left if shorter).
        if seq_len >= kernel_size:
            conv_state[slot].copy_(seq_x[:, -kernel_size:].to(conv_state.dtype))
        else:
            merged = torch.cat([conv_state[slot, :, seq_len:].float(), seq_x], dim=-1)
            conv_state[slot].copy_(merged.to(conv_state.dtype))


def _delta_step(
    q_t: torch.Tensor,
    k_t: torch.Tensor,
    v_t: torch.Tensor,
    decay: torch.Tensor,
    beta_t: torch.Tensor,
    state: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Single recurrence step in fp32 for both scalar-per-head GDN and per-channel KDA.

    ``decay`` is ``exp(g)`` with shape ``[B, HV, 1, 1]`` (GDN) or ``[B, HV, K, 1]`` (KDA).
    ``state`` has shape ``[B, HV, K, V]``.
    """
    decayed = state * decay
    kv_mem = (decayed * k_t.unsqueeze(-1)).sum(dim=-2)  # [B, HV, V]
    delta = (v_t - kv_mem) * beta_t.unsqueeze(-1)  # [B, HV, V]
    next_state = decayed + k_t.unsqueeze(-1) * delta.unsqueeze(-2)
    y_t = (next_state * q_t.unsqueeze(-1)).sum(dim=-2)  # [B, HV, V]
    return y_t, next_state


def validate_delta_inputs(
    op: str,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    ssm_state: torch.Tensor,
    state_slots: torch.Tensor,
    out: torch.Tensor,
    *,
    per_channel_g: bool,
) -> tuple[int, int, int, int, int]:
    """Validate GDN/KDA inputs; return ``(rows, k_heads, v_heads, k_dim, v_dim)``."""
    require_compute_dtype(op, "q", q)
    if q.ndim != 3 or k.ndim != 3 or v.ndim != 3:
        raise BackendError(f"{op}: q, k, v must be 3D [N, heads, dim]")
    rows, num_k_heads, k_dim = q.shape
    num_v_heads, v_dim = v.shape[1], v.shape[2]
    require_like(op, "k", k, (rows, num_k_heads, k_dim), q.dtype)
    require_like(op, "v", v, (rows, num_v_heads, v_dim), q.dtype)
    require_like(op, "out", out, (rows, num_v_heads, v_dim), q.dtype)
    if num_k_heads == 0 or num_v_heads % num_k_heads:
        raise BackendError(
            f"{op}: num_v_heads ({num_v_heads}) must be a positive multiple of "
            f"num_k_heads ({num_k_heads})"
        )
    expected_g = (rows, num_v_heads, k_dim) if per_channel_g else (rows, num_v_heads)
    if tuple(g.shape) != expected_g:
        raise BackendError(f"{op}: g must have shape {list(expected_g)}, got {list(g.shape)}")
    if tuple(beta.shape) != (rows, num_v_heads):
        raise BackendError(f"{op}: beta must be [{rows}, {num_v_heads}], got {list(beta.shape)}")
    if ssm_state.ndim != 4 or ssm_state.shape[1:] != (num_v_heads, k_dim, v_dim):
        raise BackendError(
            f"{op}: ssm_state must be [max_slots, {num_v_heads}, {k_dim}, {v_dim}], "
            f"got {list(ssm_state.shape)}"
        )
    require_index_tensor(op, "state_slots", state_slots, ndim=1)
    return rows, num_k_heads, num_v_heads, k_dim, v_dim


def _delta_recurrence_impl(
    op: str,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    ssm_state: torch.Tensor,
    state_slots: torch.Tensor,
    cu_seqlens: torch.Tensor,
    *,
    is_prefill: bool,
    scale: float | None,
    use_qk_l2norm: bool,
    out: torch.Tensor,
    per_channel_g: bool,
) -> None:
    rows, num_k_heads, num_v_heads, k_dim, _ = validate_delta_inputs(
        op, q, k, v, g, beta, ssm_state, state_slots, out, per_channel_g=per_channel_g
    )
    if rows == 0:
        return
    q_scale = (k_dim**-0.5) if scale is None else scale
    qf = q.float()
    kf = k.float()
    if use_qk_l2norm:
        qf = _l2_normalize(qf)
        kf = _l2_normalize(kf)
    qf = qf * q_scale
    repeat = num_v_heads // num_k_heads
    if repeat > 1:
        qf = qf.repeat_interleave(repeat, dim=1)
        kf = kf.repeat_interleave(repeat, dim=1)
    vf = v.float()
    bf = beta.float()
    decay_all = torch.exp(g.float())
    decay_all = decay_all.unsqueeze(-1) if per_channel_g else decay_all.unsqueeze(-1).unsqueeze(-1)

    if not is_prefill:
        slots = state_slots.long()
        if rows != slots.shape[0]:
            raise BackendError(
                f"{op}: decode requires 1 row per slot ({slots.shape[0]}), got {rows}"
            )
        state = ssm_state[slots].float()
        y_t, next_state = _delta_step(qf, kf, vf, decay_all, bf, state)
        ssm_state.index_copy_(0, slots, next_state.to(ssm_state.dtype))
        y_t = torch.where((slots > 0)[:, None, None], y_t, 0.0)
        out.copy_(y_t.to(out.dtype))
        return

    require_index_tensor(op, "cu_seqlens", cu_seqlens, ndim=1)
    offsets = _host_ints(cu_seqlens)
    slots_list = _host_ints(state_slots)
    if len(offsets) != len(slots_list) + 1:
        raise BackendError(
            f"{op}: cu_seqlens length ({len(offsets)}) must equal len(state_slots) + 1"
        )
    for req_idx, slot in enumerate(slots_list):
        start = int(offsets[req_idx])
        stop = int(offsets[req_idx + 1])
        if stop <= start:
            continue
        if slot == 0:
            out[start:stop].zero_()
            continue
        state = ssm_state[slot].float()  # [HV, K, V]
        for t in range(start, stop):
            y_t, state = _delta_step(
                qf[t].unsqueeze(0),
                kf[t].unsqueeze(0),
                vf[t].unsqueeze(0),
                decay_all[t].unsqueeze(0),
                bf[t].unsqueeze(0),
                state.unsqueeze(0),
            )
            state = state.squeeze(0)
            out[t].copy_(y_t.squeeze(0).to(out.dtype))
        ssm_state[slot].copy_(state.to(ssm_state.dtype))


def gated_delta_rule(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    ssm_state: torch.Tensor,
    state_slots: torch.Tensor,
    cu_seqlens: torch.Tensor,
    *,
    is_prefill: bool,
    scale: float | None,
    use_qk_l2norm: bool,
    out: torch.Tensor,
) -> None:
    _delta_recurrence_impl(
        "linear_attention.gated_delta_rule",
        q,
        k,
        v,
        g,
        beta,
        ssm_state,
        state_slots,
        cu_seqlens,
        is_prefill=is_prefill,
        scale=scale,
        use_qk_l2norm=use_qk_l2norm,
        out=out,
        per_channel_g=False,
    )


def kda_recurrence(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    ssm_state: torch.Tensor,
    state_slots: torch.Tensor,
    cu_seqlens: torch.Tensor,
    *,
    is_prefill: bool,
    scale: float | None,
    use_qk_l2norm: bool,
    out: torch.Tensor,
) -> None:
    _delta_recurrence_impl(
        "linear_attention.kda_recurrence",
        q,
        k,
        v,
        g,
        beta,
        ssm_state,
        state_slots,
        cu_seqlens,
        is_prefill=is_prefill,
        scale=scale,
        use_qk_l2norm=use_qk_l2norm,
        out=out,
        per_channel_g=True,
    )
