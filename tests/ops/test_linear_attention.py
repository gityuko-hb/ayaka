"""Contract and GPU parity tests for causal conv1d, GDN, and KDA linear-attention ops."""

from __future__ import annotations

import math
from typing import Any

import pytest

from ayaka.core.errors import BackendError
from ayaka.ops.backend import load_backend
from ayaka.utils.dtypes import DType
from ayaka.utils.torch_utils import no_device_sync

torch: Any = pytest.importorskip("torch")


def _fp64_conv1d_oracle(
    sequences: list[Any],
    weight: Any,
    bias: Any | None,
    init_states: list[Any] | None = None,
) -> tuple[list[Any], list[Any]]:
    """Explicit float64 reference for causal depthwise conv1d + SiLU."""
    conv_dim, ks = weight.shape
    w64 = weight.to(torch.float64)
    b64 = bias.to(torch.float64) if bias is not None else None
    outs: list[Any] = []
    states: list[Any] = []
    for i, x_seq in enumerate(sequences):
        x64 = x_seq.to(torch.float64)
        st = (
            init_states[i].to(torch.float64).clone()
            if init_states is not None
            else torch.zeros((conv_dim, ks), dtype=torch.float64)
        )
        y_rows: list[Any] = []
        for t in range(x64.shape[0]):
            st = torch.cat([st[:, 1:], x64[t : t + 1].t()], dim=-1)
            acc = (st * w64).sum(dim=-1)
            if b64 is not None:
                acc = acc + b64
            y_rows.append(acc * torch.sigmoid(acc))
        outs.append(
            torch.stack(y_rows, dim=0)
            if y_rows
            else torch.empty((0, conv_dim), dtype=torch.float64)
        )
        states.append(st)
    return outs, states


def _fp64_delta_oracle(
    q_seq: Any,
    k_seq: Any,
    v_seq: Any,
    g_seq: Any,
    beta_seq: Any,
    init_state: Any,
    *,
    scale: float,
    use_qk_l2norm: bool,
) -> tuple[Any, Any]:
    """Explicit float64 scalar-step recurrence for both GDN and KDA."""
    q64 = q_seq.to(torch.float64)
    k64 = k_seq.to(torch.float64)
    if use_qk_l2norm:
        q64 = q64 * torch.rsqrt((q64 * q64).sum(dim=-1, keepdim=True) + 1e-6)
        k64 = k64 * torch.rsqrt((k64 * k64).sum(dim=-1, keepdim=True) + 1e-6)
    q64 = q64 * scale
    repeat = v_seq.shape[1] // q64.shape[1]
    if repeat > 1:
        q64 = q64.repeat_interleave(repeat, dim=1)
        k64 = k64.repeat_interleave(repeat, dim=1)
    v64 = v_seq.to(torch.float64)
    g64 = g_seq.to(torch.float64)
    b64 = beta_seq.to(torch.float64)
    state = init_state.to(torch.float64).clone()
    ys: list[Any] = []
    for t in range(q64.shape[0]):
        decay = torch.exp(g64[t])
        decay = decay[:, None, None] if decay.ndim == 1 else decay[:, :, None]
        decayed = state * decay
        kv_mem = (decayed * k64[t, :, :, None]).sum(dim=1)  # [HV, V]
        delta = (v64[t] - kv_mem) * b64[t, :, None]
        state = decayed + k64[t, :, :, None] * delta[:, None, :]
        ys.append((state * q64[t, :, :, None]).sum(dim=1))
    return torch.stack(ys, dim=0), state


_REGISTRY_DTYPE = {
    torch.float16: DType.FP16,
    torch.bfloat16: DType.BF16,
    torch.float32: DType.FP32,
}


@pytest.mark.unit
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
@pytest.mark.parametrize("ks", [1, 4])
def test_causal_conv1d_prefill_decode_and_padding(dtype: Any, ks: int) -> None:
    torch.manual_seed(7)
    ops = load_backend(["torch"], device="cpu", dtype=_REGISTRY_DTYPE[dtype]).linear_attention
    dim = 16
    w = torch.randn((dim, 1, ks), dtype=dtype) * 0.3
    bias = torch.randn(dim, dtype=dtype) * 0.2
    seqs = [torch.randn((5, dim), dtype=dtype), torch.randn((3, dim), dtype=dtype)]
    exp_outs, exp_states = _fp64_conv1d_oracle(seqs, w.squeeze(1), bias)

    state = torch.randn((4, dim, ks), dtype=dtype)
    state[1].zero_()
    state[2].zero_()
    slots = torch.tensor([1, 0, 2], dtype=torch.int32)
    cu = torch.tensor([0, 5, 6, 9], dtype=torch.int32)
    x_cat = torch.cat([seqs[0], torch.randn((1, dim), dtype=dtype), seqs[1]], dim=0)
    out = torch.empty_like(x_cat)
    ops.causal_conv1d(x_cat, w, state, slots, cu, is_prefill=True, out=out, bias=bias)

    tol = 2e-2 if dtype is torch.float16 else (4e-2 if dtype is torch.bfloat16 else 1e-5)
    assert torch.allclose(out[:5].float(), exp_outs[0].float(), atol=tol, rtol=tol)
    assert out[5].abs().sum() == 0  # slot 0 padding zeroed
    assert torch.allclose(out[6:].float(), exp_outs[1].float(), atol=tol, rtol=tol)
    assert torch.allclose(state[1].float(), exp_states[0].float(), atol=tol, rtol=tol)

    # Incremental 2-step decode matches continuing the oracle
    x_next = torch.randn((2, dim), dtype=dtype)
    exp_step, exp_next_st = _fp64_conv1d_oracle(
        [x_next[0:1], x_next[1:2]], w.squeeze(1), bias, init_states=exp_states
    )
    dec_out = torch.empty((3, dim), dtype=dtype)
    x_dec = torch.stack([x_next[0], torch.ones(dim, dtype=dtype), x_next[1]], dim=0)
    ops.causal_conv1d(
        x_dec,
        w,
        state,
        slots,
        torch.empty(0, dtype=torch.int32),
        is_prefill=False,
        out=dec_out,
        bias=bias,
    )
    assert dec_out[1].abs().sum() == 0
    assert torch.allclose(dec_out[0].float(), exp_step[0].squeeze(0).float(), atol=tol, rtol=tol)
    assert torch.allclose(dec_out[2].float(), exp_step[1].squeeze(0).float(), atol=tol, rtol=tol)
    assert torch.allclose(state[2].float(), exp_next_st[1].float(), atol=tol, rtol=tol)


@pytest.mark.unit
@pytest.mark.parametrize("per_channel", [False, True])
def test_delta_and_kda_chunk_invariance_and_oracle(per_channel: bool) -> None:
    torch.manual_seed(13)
    ops = load_backend(["torch"], device="cpu", dtype=DType.FP32).linear_attention
    hk, hv, kd, vd = 2, 4, 8, 8
    seq_len = 11
    q = torch.randn((seq_len, hk, kd), dtype=torch.float32)
    k = torch.randn((seq_len, hk, kd), dtype=torch.float32)
    v = torch.randn((seq_len, hv, vd), dtype=torch.float32)
    g = -torch.rand((seq_len, hv, kd) if per_channel else (seq_len, hv), dtype=torch.float32) * 0.4
    beta = torch.sigmoid(torch.randn((seq_len, hv), dtype=torch.float32))
    init = torch.randn((hv, kd, vd), dtype=torch.float32) * 0.2
    scale = 1.0 / math.sqrt(kd)

    exp_y, exp_s = _fp64_delta_oracle(q, k, v, g, beta, init, scale=scale, use_qk_l2norm=True)

    # Full prefill
    st_full = torch.zeros((3, hv, kd, vd), dtype=torch.float32)
    st_full[1].copy_(init)
    out_full = torch.empty_like(v)
    fn = ops.kda_recurrence if per_channel else ops.gated_delta_rule
    fn(
        q,
        k,
        v,
        g,
        beta,
        st_full,
        torch.tensor([1], dtype=torch.int32),
        torch.tensor([0, seq_len], dtype=torch.int32),
        is_prefill=True,
        scale=None,
        use_qk_l2norm=True,
        out=out_full,
    )
    assert torch.allclose(out_full.double(), exp_y, atol=1e-5, rtol=1e-5)
    assert torch.allclose(st_full[1].double(), exp_s, atol=1e-5, rtol=1e-5)

    # Chunked prefill (6 + 4) + 1 decode step -> identical state & outputs
    st_chunk = torch.zeros((3, hv, kd, vd), dtype=torch.float32)
    st_chunk[2].copy_(init)
    slot2 = torch.tensor([2], dtype=torch.int32)
    out_parts = [torch.empty((6, hv, vd)), torch.empty((4, hv, vd)), torch.empty((1, hv, vd))]
    fn(
        q[:6],
        k[:6],
        v[:6],
        g[:6],
        beta[:6],
        st_chunk,
        slot2,
        torch.tensor([0, 6]),
        is_prefill=True,
        scale=None,
        use_qk_l2norm=True,
        out=out_parts[0],
    )
    fn(
        q[6:10],
        k[6:10],
        v[6:10],
        g[6:10],
        beta[6:10],
        st_chunk,
        slot2,
        torch.tensor([0, 4]),
        is_prefill=True,
        scale=None,
        use_qk_l2norm=True,
        out=out_parts[1],
    )
    fn(
        q[10:11],
        k[10:11],
        v[10:11],
        g[10:11],
        beta[10:11],
        st_chunk,
        slot2,
        torch.empty(0, dtype=torch.int32),
        is_prefill=False,
        scale=None,
        use_qk_l2norm=True,
        out=out_parts[2],
    )
    assert torch.allclose(torch.cat(out_parts, dim=0), out_full, atol=1e-5, rtol=1e-5)
    assert torch.allclose(st_chunk[2], st_full[1], atol=1e-5, rtol=1e-5)


@pytest.mark.unit
def test_linear_attention_rejects_bad_shapes() -> None:
    ops = load_backend(["torch"], device="cpu", dtype=DType.FP32).linear_attention
    with pytest.raises(BackendError, match="num_v_heads"):
        ops.gated_delta_rule(
            torch.zeros((2, 3, 4)),
            torch.zeros((2, 3, 4)),
            torch.zeros((2, 4, 4)),
            torch.zeros((2, 4)),
            torch.zeros((2, 4)),
            torch.zeros((2, 4, 4, 4)),
            torch.tensor([1], dtype=torch.int32),
            torch.tensor([0, 2], dtype=torch.int32),
            is_prefill=True,
            scale=None,
            use_qk_l2norm=True,
            out=torch.zeros((2, 4, 4)),
        )


@pytest.mark.gpu
@pytest.mark.parametrize("backends", [("torch",), ("flashinfer", "torch")])
@pytest.mark.parametrize("head_dim", [64, 128])
def test_gpu_gdn_kda_conv1d_parity_and_cuda_graph(backends: tuple[str, ...], head_dim: int) -> None:
    if "flashinfer" in backends:
        pytest.importorskip("flashinfer")
    torch.manual_seed(42)
    dev = torch.device("cuda")
    ops = load_backend(backends, device="cuda", dtype=DType.BF16, cuda_graph=True, head_dim=128)
    la = ops.linear_attention

    hk, hv, vd = 4, 8, head_dim
    conv_dim = 2 * hk * head_dim + hv * vd
    ks = 4

    # 1. Conv1d prefill + decode in CUDA graph
    conv_w = torch.randn((conv_dim, 1, ks), dtype=torch.bfloat16, device=dev) * 0.25
    conv_b = torch.randn(conv_dim, dtype=torch.bfloat16, device=dev) * 0.1
    conv_st = torch.zeros((4, conv_dim, ks), dtype=torch.bfloat16, device=dev)
    x_pre = torch.randn((10, conv_dim), dtype=torch.bfloat16, device=dev)
    slots = torch.tensor([1, 2], dtype=torch.int32, device=dev)
    cu = torch.tensor([0, 6, 10], dtype=torch.int32, device=dev)
    out_pre = torch.empty_like(x_pre)
    la.causal_conv1d(x_pre, conv_w, conv_st, slots, cu, is_prefill=True, out=out_pre, bias=conv_b)
    exp_conv, exp_conv_st = _fp64_conv1d_oracle(
        [x_pre[:6].cpu(), x_pre[6:].cpu()], conv_w.squeeze(1).cpu(), conv_b.cpu()
    )
    assert torch.allclose(out_pre[:6].float().cpu(), exp_conv[0].float(), atol=5e-2, rtol=5e-2)
    assert torch.allclose(conv_st[1].float().cpu(), exp_conv_st[0].float(), atol=5e-2, rtol=5e-2)

    # 2. GDN prefill parity vs fp64
    seq = [torch.randn((9, hk, head_dim), dtype=torch.bfloat16, device=dev) for _ in range(2)]
    q_cat = torch.cat(seq, dim=0)
    k_cat = torch.randn_like(q_cat)
    v_cat = torch.randn((18, hv, vd), dtype=torch.bfloat16, device=dev)
    g_cat = -torch.rand((18, hv), dtype=torch.float32, device=dev) * 0.3
    beta_cat = torch.sigmoid(torch.randn((18, hv), dtype=torch.float32, device=dev))
    ssm_st = torch.zeros((4, hv, head_dim, vd), dtype=torch.float32, device=dev)
    cu_gdn = torch.tensor([0, 9, 18], dtype=torch.int32, device=dev)
    out_gdn = torch.empty_like(v_cat)
    la.gated_delta_rule(
        q_cat,
        k_cat,
        v_cat,
        g_cat,
        beta_cat,
        ssm_st,
        slots,
        cu_gdn,
        is_prefill=True,
        scale=None,
        use_qk_l2norm=True,
        out=out_gdn,
    )
    exp_y0, exp_s0 = _fp64_delta_oracle(
        q_cat[:9].cpu(),
        k_cat[:9].cpu(),
        v_cat[:9].cpu(),
        g_cat[:9].cpu(),
        beta_cat[:9].cpu(),
        torch.zeros((hv, head_dim, vd)),
        scale=head_dim**-0.5,
        use_qk_l2norm=True,
    )
    assert torch.allclose(out_gdn[:9].float().cpu(), exp_y0.float(), atol=5e-2, rtol=5e-2)
    assert torch.allclose(ssm_st[1].float().cpu(), exp_s0.float(), atol=5e-2, rtol=5e-2)

    # 3. CUDA graph capture + replay (GDN + KDA + conv1d decode, slot-0 pad, no_device_sync)
    slots_g = torch.tensor([1, 0], dtype=torch.int32, device=dev)
    qd = torch.randn((2, hk, head_dim), dtype=torch.bfloat16, device=dev)
    kd = torch.randn((2, hk, head_dim), dtype=torch.bfloat16, device=dev)
    vd_t = torch.randn((2, hv, vd), dtype=torch.bfloat16, device=dev)
    gd = -torch.rand((2, hv), dtype=torch.float32, device=dev) * 0.2
    gk_d = -torch.rand((2, hv, head_dim), dtype=torch.float32, device=dev) * 0.2
    bd = torch.sigmoid(torch.randn((2, hv), dtype=torch.float32, device=dev))
    xd = torch.randn((2, conv_dim), dtype=torch.bfloat16, device=dev)
    out_cd = torch.empty_like(xd)
    out_gd = torch.empty_like(vd_t)
    out_kd = torch.empty_like(vd_t)
    cu_d = torch.tensor([0, 1, 2], dtype=torch.int32, device=dev)
    kda_st = torch.zeros_like(ssm_st)

    # Warm-up
    la.causal_conv1d(xd, conv_w, conv_st, slots_g, cu_d, is_prefill=False, out=out_cd, bias=conv_b)
    la.gated_delta_rule(
        qd,
        kd,
        vd_t,
        gd,
        bd,
        ssm_st,
        slots_g,
        cu_d,
        is_prefill=False,
        scale=None,
        use_qk_l2norm=True,
        out=out_gd,
    )
    la.kda_recurrence(
        qd,
        kd,
        vd_t,
        gk_d,
        bd,
        kda_st,
        slots_g,
        cu_d,
        is_prefill=False,
        scale=None,
        use_qk_l2norm=True,
        out=out_kd,
    )

    graph = torch.cuda.CUDAGraph()
    torch.cuda.synchronize()
    with torch.inference_mode(), torch.cuda.graph(graph):
        la.causal_conv1d(
            xd, conv_w, conv_st, slots_g, cu_d, is_prefill=False, out=out_cd, bias=conv_b
        )
        la.gated_delta_rule(
            qd,
            kd,
            vd_t,
            gd,
            bd,
            ssm_st,
            slots_g,
            cu_d,
            is_prefill=False,
            scale=None,
            use_qk_l2norm=True,
            out=out_gd,
        )
        la.kda_recurrence(
            qd,
            kd,
            vd_t,
            gk_d,
            bd,
            kda_st,
            slots_g,
            cu_d,
            is_prefill=False,
            scale=None,
            use_qk_l2norm=True,
            out=out_kd,
        )
    next_slots = torch.tensor([2, 0], dtype=torch.int32).pin_memory()
    with torch.inference_mode(), no_device_sync(strict=True):
        slots_g.copy_(next_slots, non_blocking=True)
        graph.replay()
    torch.cuda.synchronize()
    assert out_cd[1].abs().sum() == 0
    assert out_gd[1].abs().sum() == 0
    assert out_kd[1].abs().sum() == 0
