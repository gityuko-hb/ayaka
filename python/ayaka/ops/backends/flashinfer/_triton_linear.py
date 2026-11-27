# pyright: reportUnknownVariableType=false, reportUnknownMemberType=false, reportUnknownArgumentType=false, reportUnknownParameterType=false, reportMissingParameterType=false, reportInvalidTypeForm=false
# Triton 3.8 kernel DSL has no bundled stubs and requires ``tl.constexpr`` type forms.
"""Triton JIT kernels for CUDA-graph-safe linear-attention decode."""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any, cast

import triton  # pyright: ignore[reportMissingTypeStubs]
import triton.language as tl  # pyright: ignore[reportMissingTypeStubs]

if TYPE_CHECKING:
    from collections.abc import Callable

    import torch

__all__ = ["launch_causal_conv1d_decode", "launch_delta_decode"]


def _triton_jit(fn: Callable[..., Any]) -> Any:
    return cast("Any", triton.jit(fn))


@_triton_jit
def _causal_conv1d_decode_kernel(
    x_ptr,
    weight_ptr,
    bias_ptr,
    state_ptr,
    slots_ptr,
    out_ptr,
    conv_dim,
    stride_xb,
    stride_xc,
    stride_wc,
    stride_wk,
    stride_ss,
    stride_sc,
    stride_sk,
    stride_ob,
    stride_oc,
    has_bias: tl.constexpr,
    kernel_size: tl.constexpr,
    block_c: tl.constexpr,
) -> None:
    row = tl.program_id(0)
    pid_c = tl.program_id(1)
    offs_c = pid_c * block_c + tl.arange(0, block_c)
    mask_c = offs_c < conv_dim

    slot = tl.load(slots_ptr + row).to(tl.int64)
    out_offsets = row * stride_ob + offs_c * stride_oc
    if slot == 0:
        tl.store(out_ptr + out_offsets, tl.zeros([block_c], dtype=tl.float32), mask=mask_c)
        return

    state_base = state_ptr + slot * stride_ss + offs_c * stride_sc
    acc = tl.zeros([block_c], dtype=tl.float32)
    for tap in tl.static_range(kernel_size - 1):  # pyright: ignore[reportGeneralTypeIssues] -- Triton intrinsic
        val = tl.load(state_base + (tap + 1) * stride_sk, mask=mask_c, other=0.0)
        tl.store(state_base + tap * stride_sk, val, mask=mask_c)
        w = tl.load(weight_ptr + offs_c * stride_wc + tap * stride_wk, mask=mask_c, other=0.0).to(
            tl.float32
        )
        acc += val.to(tl.float32) * w

    x_val = tl.load(x_ptr + row * stride_xb + offs_c * stride_xc, mask=mask_c, other=0.0)
    tl.store(
        state_base + (kernel_size - 1) * stride_sk,
        x_val.to(state_ptr.dtype.element_ty),
        mask=mask_c,
    )
    w_last = tl.load(
        weight_ptr + offs_c * stride_wc + (kernel_size - 1) * stride_wk, mask=mask_c, other=0.0
    ).to(tl.float32)
    acc += x_val.to(tl.float32) * w_last
    if has_bias:
        b = tl.load(bias_ptr + offs_c, mask=mask_c, other=0.0).to(tl.float32)
        acc += b
    y = acc * tl.sigmoid(acc)
    tl.store(out_ptr + out_offsets, y.to(out_ptr.dtype.element_ty), mask=mask_c)


@_triton_jit
def _delta_decode_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    g_ptr,
    beta_ptr,
    state_ptr,
    slots_ptr,
    out_ptr,
    k_dim,
    v_dim,
    head_group,
    scale,
    stride_qb,
    stride_qh,
    stride_qd,
    stride_kb,
    stride_kh,
    stride_kd,
    stride_vb,
    stride_vh,
    stride_vd,
    stride_gb,
    stride_gh,
    stride_gd,
    stride_bb,
    stride_bh,
    stride_ss,
    stride_sh,
    stride_sk,
    stride_sv,
    stride_ob,
    stride_oh,
    stride_od,
    use_qk_l2norm: tl.constexpr,
    per_channel_g: tl.constexpr,
    block_k: tl.constexpr,
    block_v: tl.constexpr,
) -> None:
    row = tl.program_id(0)
    hv = tl.program_id(1)
    pid_v = tl.program_id(2)

    offs_v = pid_v * block_v + tl.arange(0, block_v)
    mask_v = offs_v < v_dim
    slot = tl.load(slots_ptr + row).to(tl.int64)
    out_base = out_ptr + row * stride_ob + hv * stride_oh + offs_v * stride_od
    if slot == 0:
        tl.store(out_base, tl.zeros([block_v], dtype=tl.float32), mask=mask_v)
        return

    kh = hv // head_group
    offs_k = tl.arange(0, block_k)
    mask_k = offs_k < k_dim

    q_vec = tl.load(
        q_ptr + row * stride_qb + kh * stride_qh + offs_k * stride_qd, mask=mask_k, other=0.0
    ).to(tl.float32)
    k_vec = tl.load(
        k_ptr + row * stride_kb + kh * stride_kh + offs_k * stride_kd, mask=mask_k, other=0.0
    ).to(tl.float32)
    if use_qk_l2norm:
        q_vec = q_vec * tl.rsqrt(tl.sum(q_vec * q_vec) + 1e-6)
        k_vec = k_vec * tl.rsqrt(tl.sum(k_vec * k_vec) + 1e-6)
    q_vec = q_vec * scale

    v_vec = tl.load(
        v_ptr + row * stride_vb + hv * stride_vh + offs_v * stride_vd, mask=mask_v, other=0.0
    ).to(tl.float32)
    beta_val = tl.load(beta_ptr + row * stride_bb + hv * stride_bh).to(tl.float32)

    state_tile_ptr = (
        state_ptr
        + slot * stride_ss
        + hv * stride_sh
        + offs_k[:, None] * stride_sk
        + offs_v[None, :] * stride_sv
    )
    state_mask = mask_k[:, None] & mask_v[None, :]
    state_mat = tl.load(state_tile_ptr, mask=state_mask, other=0.0).to(tl.float32)

    if per_channel_g:
        g_vec = tl.load(
            g_ptr + row * stride_gb + hv * stride_gh + offs_k * stride_gd,
            mask=mask_k,
            other=0.0,
        ).to(tl.float32)
        state_mat = state_mat * tl.exp(g_vec)[:, None]
    else:
        g_val = tl.load(g_ptr + row * stride_gb + hv * stride_gh).to(tl.float32)
        state_mat = state_mat * tl.exp(g_val)

    kv_mem = tl.sum(state_mat * k_vec[:, None], axis=0)
    delta = (v_vec - kv_mem) * beta_val
    state_mat = state_mat + k_vec[:, None] * delta[None, :]
    y_vec = tl.sum(state_mat * q_vec[:, None], axis=0)

    tl.store(state_tile_ptr, state_mat, mask=state_mask)
    tl.store(out_base, y_vec.to(out_ptr.dtype.element_ty), mask=mask_v)


def launch_causal_conv1d_decode(
    x: torch.Tensor,
    w2: torch.Tensor,
    bias: torch.Tensor | None,
    conv_state: torch.Tensor,
    state_slots: torch.Tensor,
    out: torch.Tensor,
    conv_dim: int,
    kernel_size: int,
) -> None:
    """Launch the Triton single-step causal conv1d decode kernel."""
    rows = int(x.shape[0])
    block_c = 128
    grid = (rows, (conv_dim + block_c - 1) // block_c)
    b_ptr = bias if bias is not None else x
    with warnings.catch_warnings():
        # Triton 3.8 AST visitor emits an AnnAssign DeprecationWarning on Python 3.13 during JIT.
        warnings.filterwarnings("ignore", message=".*AnnAssign.*", category=DeprecationWarning)
        _causal_conv1d_decode_kernel[grid](
            x,
            w2,
            b_ptr,
            conv_state,
            state_slots,
            out,
            conv_dim,
            x.stride(0),
            x.stride(1),
            w2.stride(0),
            w2.stride(1),
            conv_state.stride(0),
            conv_state.stride(1),
            conv_state.stride(2),
            out.stride(0),
            out.stride(1),
            has_bias=bias is not None,
            kernel_size=kernel_size,
            block_c=block_c,
        )


def launch_delta_decode(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    ssm_state: torch.Tensor,
    state_slots: torch.Tensor,
    out: torch.Tensor,
    *,
    num_k_heads: int,
    num_v_heads: int,
    k_dim: int,
    v_dim: int,
    scale: float,
    use_qk_l2norm: bool,
    per_channel_g: bool,
) -> None:
    """Launch the Triton single-step GDN/KDA delta-rule decode kernel."""
    rows = int(q.shape[0])
    block_k = int(triton.next_power_of_2(k_dim))
    block_v = min(64, int(triton.next_power_of_2(v_dim)))
    grid = (rows, num_v_heads, (v_dim + block_v - 1) // block_v)
    g_stride_d = g.stride(2) if per_channel_g else 0
    with warnings.catch_warnings():
        # Triton 3.8 AST visitor emits an AnnAssign DeprecationWarning on Python 3.13 during JIT.
        warnings.filterwarnings("ignore", message=".*AnnAssign.*", category=DeprecationWarning)
        _delta_decode_kernel[grid](
            q,
            k,
            v,
            g,
            beta,
            ssm_state,
            state_slots,
            out,
            k_dim,
            v_dim,
            num_v_heads // num_k_heads,
            scale,
            q.stride(0),
            q.stride(1),
            q.stride(2),
            k.stride(0),
            k.stride(1),
            k.stride(2),
            v.stride(0),
            v.stride(1),
            v.stride(2),
            g.stride(0),
            g.stride(1),
            g_stride_d,
            beta.stride(0),
            beta.stride(1),
            ssm_state.stride(0),
            ssm_state.stride(1),
            ssm_state.stride(2),
            ssm_state.stride(3),
            out.stride(0),
            out.stride(1),
            out.stride(2),
            use_qk_l2norm=use_qk_l2norm,
            per_channel_g=per_channel_g,
            block_k=block_k,
            block_v=block_v,
        )
