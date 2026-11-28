"""FlashInfer and Triton GPU kernels for recurrent linear-attention ops.

Implements :class:`ayaka.ops.linear_attention.LinearAttentionOps`:
- ``causal_conv1d``: Triton fused shift-and-convolve decode kernel (CUDA-graph safe)
  and reference depthwise conv1d prefill over ragged ``cu_seqlens``.
- ``gated_delta_rule``: FlashInfer ``chunk_gated_delta_rule`` prefill (for supported
  head dimensions) and Triton fused decode kernel (CUDA-graph safe) with exact
  fallback to the reference recurrence on other dimensions/dtypes.
- ``kda_recurrence``: Triton fused per-channel decay decode kernel (CUDA-graph safe)
  and reference prefill.
"""

from __future__ import annotations

import warnings
from typing import Any

import torch

from ayaka.core.errors import BackendError
from ayaka.ops.backends.flashinfer._triton_linear import (
    launch_causal_conv1d_decode,
    launch_delta_decode,
)
from ayaka.ops.backends.torch import linear_attention as reference
from ayaka.ops.backends.torch._checks import require_index_tensor
from ayaka.utils.import_utils import require_module

__all__ = ["causal_conv1d", "gated_delta_rule", "kda_recurrence"]


def _flashinfer() -> Any:
    return require_module(
        "flashinfer",
        capability="ops.flashinfer",
        remedy="install it on Linux or WSL2 with `uv sync --extra cu130 --extra flashinfer`",
    )


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
    w2, rows, conv_dim, kernel_size = reference.validate_conv1d_inputs(
        op, x, weight, conv_state, state_slots, out, bias
    )
    if rows == 0:
        return
    if not is_prefill and x.is_cuda and kernel_size >= 1:
        if rows != state_slots.shape[0]:
            raise BackendError(
                f"{op}: decode requires 1 row per slot ({state_slots.shape[0]}), got {rows}"
            )
        launch_causal_conv1d_decode(
            x, w2, bias, conv_state, state_slots, out, conv_dim, kernel_size
        )
        return
    reference.causal_conv1d(
        x,
        weight,
        conv_state,
        state_slots,
        cu_seqlens,
        is_prefill=is_prefill,
        out=out,
        bias=bias,
    )


def _delta_decode_triton(
    op: str,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    ssm_state: torch.Tensor,
    state_slots: torch.Tensor,
    scale: float | None,
    use_qk_l2norm: bool,
    out: torch.Tensor,
    *,
    per_channel_g: bool,
) -> None:
    rows, num_k_heads, num_v_heads, k_dim, v_dim = reference.validate_delta_inputs(
        op, q, k, v, g, beta, ssm_state, state_slots, out, per_channel_g=per_channel_g
    )
    if rows == 0:
        return
    if rows != state_slots.shape[0]:
        raise BackendError(
            f"{op}: decode requires 1 row per slot ({state_slots.shape[0]}), got {rows}"
        )
    if not q.is_cuda or ssm_state.dtype != torch.float32:
        cu_dummy = torch.empty(0, dtype=torch.int32, device=q.device)
        target = reference.kda_recurrence if per_channel_g else reference.gated_delta_rule
        target(
            q,
            k,
            v,
            g,
            beta,
            ssm_state,
            state_slots,
            cu_dummy,
            is_prefill=False,
            scale=scale,
            use_qk_l2norm=use_qk_l2norm,
            out=out,
        )
        return

    q_scale = (k_dim**-0.5) if scale is None else scale
    launch_delta_decode(
        q,
        k,
        v,
        g,
        beta,
        ssm_state,
        state_slots,
        out,
        num_k_heads=num_k_heads,
        num_v_heads=num_v_heads,
        k_dim=k_dim,
        v_dim=v_dim,
        scale=q_scale,
        use_qk_l2norm=use_qk_l2norm,
        per_channel_g=per_channel_g,
    )


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
    op = "linear_attention.gated_delta_rule"
    if not is_prefill:
        _delta_decode_triton(
            op,
            q,
            k,
            v,
            g,
            beta,
            ssm_state,
            state_slots,
            scale,
            use_qk_l2norm,
            out,
            per_channel_g=False,
        )
        return

    rows, _, _, k_dim, v_dim = reference.validate_delta_inputs(
        op, q, k, v, g, beta, ssm_state, state_slots, out, per_channel_g=False
    )
    if rows == 0:
        return
    require_index_tensor(op, "cu_seqlens", cu_seqlens, ndim=1)
    if cu_seqlens.shape[0] != state_slots.shape[0] + 1:
        raise BackendError(
            f"{op}: cu_seqlens length ({cu_seqlens.shape[0]}) must equal len(state_slots) + 1"
        )
    if (
        q.is_cuda
        and q.dtype in (torch.float16, torch.bfloat16)
        and k_dim == 128
        and v_dim == 128
        and ssm_state.dtype == torch.float32
    ):
        slots_long = state_slots.long()
        num_reqs = int(state_slots.shape[0])
        iota = torch.arange(num_reqs, dtype=torch.int32, device=q.device)
        fi_state = ssm_state[slots_long].transpose(-1, -2).contiguous()
        cu32 = cu_seqlens.to(torch.int32)
        g_fp32 = g.to(torch.float32) if g.dtype != torch.float32 else g
        beta_fp32 = beta.to(torch.float32) if beta.dtype != torch.float32 else beta
        q_scale = (k_dim**-0.5) if scale is None else scale
        with warnings.catch_warnings():
            # FlashInfer's CuTe DSL sm80 GDN kernel triggers a cutlass SmemAllocator deprecation.
            warnings.filterwarnings(
                "ignore", message=".*SmemAllocator.*", category=DeprecationWarning
            )
            fi_out, _ = _flashinfer().chunk_gated_delta_rule(
                q,
                k,
                v,
                g=g_fp32,
                beta=beta_fp32,
                scale=q_scale,
                initial_state=fi_state,
                output_final_state=True,
                output_state=fi_state,
                cu_seqlens=cu32,
                use_qk_l2norm_in_kernel=use_qk_l2norm,
                gate_domain="log",
                state_indices=iota,
                output_state_indices=iota,
            )
        real_mask = slots_long > 0
        if bool(real_mask.all()):
            ssm_state.index_copy_(0, slots_long, fi_state.transpose(-1, -2))
            out.copy_(fi_out.to(out.dtype))
        else:
            real_idx = real_mask.nonzero(as_tuple=True)[0]
            if real_idx.numel():
                ssm_state.index_copy_(0, slots_long[real_idx], fi_state[real_idx].transpose(-1, -2))
            lengths = (cu32[1:] - cu32[:-1]).clamp(min=0)
            real_rows = torch.repeat_interleave(real_mask, lengths.long())
            out.copy_(torch.where(real_rows[:, None, None], fi_out.to(out.dtype), 0))
        return

    reference.gated_delta_rule(
        q,
        k,
        v,
        g,
        beta,
        ssm_state,
        state_slots,
        cu_seqlens,
        is_prefill=True,
        scale=scale,
        use_qk_l2norm=use_qk_l2norm,
        out=out,
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
    op = "linear_attention.kda_recurrence"
    if not is_prefill:
        _delta_decode_triton(
            op,
            q,
            k,
            v,
            g,
            beta,
            ssm_state,
            state_slots,
            scale,
            use_qk_l2norm,
            out,
            per_channel_g=True,
        )
        return
    reference.kda_recurrence(
        q,
        k,
        v,
        g,
        beta,
        ssm_state,
        state_slots,
        cu_seqlens,
        is_prefill=True,
        scale=scale,
        use_qk_l2norm=use_qk_l2norm,
        out=out,
    )
