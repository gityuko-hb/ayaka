"""Contract and GPU parity tests for compressed Multi-Head Latent Attention (MLA) ops."""

from __future__ import annotations

import math
from typing import Any

import pytest

from ayaka.core.types import ForwardMode
from ayaka.ops.backend import load_backend
from ayaka.ops.kvcache import FP8_E4M3_CODEC, RAW_CODEC, LatentKVBuffers
from ayaka.utils.dtypes import FP8_E4M3_MAX, FP8_E4M3_MIN, DType
from ayaka.utils.torch_utils import no_device_sync

torch: Any = pytest.importorskip("torch")


def _uncompressed_mha_oracle(
    q_nope: Any,
    q_pe: Any,
    w_uk: Any,
    w_uv: Any,
    ckv_seqs: list[Any],
    kpe_seqs: list[Any],
    *,
    rows_per_req: list[int],
    scale: float,
) -> list[Any]:
    """Explicit uncompressed float64 MHA oracle from latent sequences and absorption matrices."""
    qn64 = q_nope.to(torch.float64)
    qp64 = q_pe.to(torch.float64)
    wk64 = w_uk.to(torch.float64)
    wv64 = w_uv.to(torch.float64)
    outs: list[Any] = []
    first = 0
    for ckv_seq, kpe_seq, count in zip(ckv_seqs, kpe_seqs, rows_per_req, strict=True):
        length = ckv_seq.shape[0]
        if length == 0 or count == 0:
            outs.append(torch.zeros((count, qn64.shape[1], wv64.shape[1]), dtype=torch.float64))
            first += count
            continue
        k_nope = torch.einsum("lc,hdc->lhd", ckv_seq.to(torch.float64), wk64)
        k_pe = kpe_seq.to(torch.float64)[:, None, :].expand(-1, qn64.shape[1], -1)
        v_full = torch.einsum("lc,hvc->lhv", ckv_seq.to(torch.float64), wv64)

        q_n = qn64[first : first + count].transpose(0, 1)  # [H, R, D_nope]
        q_p = qp64[first : first + count].transpose(0, 1)  # [H, R, D_pe]
        k_n = k_nope.transpose(0, 1)  # [H, L, D_nope]
        k_p = k_pe.transpose(0, 1)  # [H, L, D_pe]
        v_h = v_full.transpose(0, 1)  # [H, L, D_v]

        scores = (
            torch.matmul(q_n, k_n.transpose(-2, -1)) + torch.matmul(q_p, k_p.transpose(-2, -1))
        ) * scale
        k_pos = torch.arange(length)[None, :]
        q_pos = torch.arange(length - count, length)[:, None]
        scores = scores.masked_fill(~(k_pos <= q_pos)[None], float("-inf"))
        weights = torch.softmax(scores, dim=-1)
        outs.append(torch.matmul(weights, v_h).transpose(0, 1))
        first += count
    return outs


_REGISTRY_DTYPE = {
    torch.float16: DType.FP16,
    torch.bfloat16: DType.BF16,
    torch.float32: DType.FP32,
}


@pytest.mark.unit
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
@pytest.mark.parametrize("use_fp8", [False, True])
def test_absorbed_mla_matches_uncompressed_mha_oracle(dtype: Any, use_fp8: bool) -> None:
    if use_fp8 and dtype is torch.float32:
        return
    torch.manual_seed(17)
    ops = load_backend(["torch"], device="cpu", dtype=_REGISTRY_DTYPE[dtype])
    block_size = 4
    heads, d_nope, d_pe, d_v, ckv_dim = 4, 16, 8, 12, 32
    scale = 1.0 / math.sqrt(d_nope + d_pe)

    w_uk = torch.randn((heads, d_nope, ckv_dim), dtype=dtype) * 0.2
    w_uv = torch.randn((heads, d_v, ckv_dim), dtype=dtype) * 0.2
    ckv_seqs = [torch.randn((6, ckv_dim), dtype=dtype), torch.randn((5, ckv_dim), dtype=dtype)]
    kpe_seqs = [torch.randn((6, d_pe), dtype=dtype), torch.randn((5, d_pe), dtype=dtype)]

    if use_fp8:
        ckv_scale, kpe_scale = 0.05, 0.05
        ckv_buf = torch.zeros((8, block_size, ckv_dim), dtype=torch.float8_e4m3fn)
        kpe_buf = torch.zeros((8, block_size, d_pe), dtype=torch.float8_e4m3fn)
        buffers = LatentKVBuffers(ckv_buf, kpe_buf, FP8_E4M3_CODEC, ckv_scale, kpe_scale)
        # Oracle uses the exact quantized-then-dequantized values so quantization rounding is clean
        dequant_ckv = [
            (s.float() / ckv_scale)
            .clamp(FP8_E4M3_MIN, FP8_E4M3_MAX)
            .to(torch.float8_e4m3fn)
            .float()
            * ckv_scale
            for s in ckv_seqs
        ]
        dequant_kpe = [
            (s.float() / kpe_scale)
            .clamp(FP8_E4M3_MIN, FP8_E4M3_MAX)
            .to(torch.float8_e4m3fn)
            .float()
            * kpe_scale
            for s in kpe_seqs
        ]
    else:
        ckv_buf = torch.zeros((8, block_size, ckv_dim), dtype=dtype)
        kpe_buf = torch.zeros((8, block_size, d_pe), dtype=dtype)
        buffers = LatentKVBuffers(ckv_buf, kpe_buf, RAW_CODEC)
        dequant_ckv, dequant_kpe = ckv_seqs, kpe_seqs

    table = torch.tensor([[1, 2], [3, 4]], dtype=torch.int32)
    # 1. Prefill first chunk (5 and 4 tokens)
    m_pre = ops.attention.plan(
        ForwardMode.PREFILL,
        torch.tensor([5, 4], dtype=torch.int32),
        torch.tensor([5, 4], dtype=torch.int32),
        table,
        block_size,
    )
    ops.mla.store_latent(
        torch.cat([ckv_seqs[0][:5], ckv_seqs[1][:4]], 0),
        torch.cat([kpe_seqs[0][:5], kpe_seqs[1][:4]], 0),
        buffers,
        m_pre,
    )

    q_nope_pre = torch.randn((9, heads, d_nope), dtype=dtype)
    q_pe_pre = torch.randn((9, heads, d_pe), dtype=dtype)
    q_abs = torch.einsum("nhd,hdc->nhc", q_nope_pre.float(), w_uk.float()).to(dtype)
    out_pre_lat = torch.empty((9, heads, ckv_dim), dtype=dtype)
    ops.mla.attend_mla(q_abs, q_pe_pre, buffers, m_pre, out_pre_lat, scale=scale)
    out_pre = torch.einsum("nhc,hvc->nhv", out_pre_lat.float(), w_uv.float())

    oracle_pre = _uncompressed_mha_oracle(
        q_nope_pre,
        q_pe_pre,
        w_uk,
        w_uv,
        [dequant_ckv[0][:5], dequant_ckv[1][:4]],
        [dequant_kpe[0][:5], dequant_kpe[1][:4]],
        rows_per_req=[5, 4],
        scale=scale,
    )
    tol = 3e-2 if dtype in (torch.float16, torch.bfloat16) else 1e-4
    assert torch.allclose(out_pre[:5], oracle_pre[0].float(), atol=tol, rtol=tol)
    assert torch.allclose(out_pre[5:], oracle_pre[1].float(), atol=tol, rtol=tol)

    # 2. Decode step (token 6 and token 5)
    m_dec = ops.attention.plan(
        ForwardMode.DECODE,
        torch.tensor([1, 1], dtype=torch.int32),
        torch.tensor([6, 5], dtype=torch.int32),
        table,
        block_size,
    )
    ops.mla.store_latent(
        torch.stack([ckv_seqs[0][5], ckv_seqs[1][4]], 0),
        torch.stack([kpe_seqs[0][5], kpe_seqs[1][4]], 0),
        buffers,
        m_dec,
    )
    q_nope_d = torch.randn((2, heads, d_nope), dtype=dtype)
    q_pe_d = torch.randn((2, heads, d_pe), dtype=dtype)
    q_abs_d = torch.einsum("nhd,hdc->nhc", q_nope_d.float(), w_uk.float()).to(dtype)
    out_d_lat = torch.empty((2, heads, ckv_dim), dtype=dtype)
    ops.mla.attend_mla(q_abs_d, q_pe_d, buffers, m_dec, out_d_lat, scale=scale)
    out_d = torch.einsum("nhc,hvc->nhv", out_d_lat.float(), w_uv.float())

    oracle_dec = _uncompressed_mha_oracle(
        q_nope_d, q_pe_d, w_uk, w_uv, dequant_ckv, dequant_kpe, rows_per_req=[1, 1], scale=scale
    )
    assert torch.allclose(out_d[0:1], oracle_dec[0].float(), atol=tol, rtol=tol)
    assert torch.allclose(out_d[1:2], oracle_dec[1].float(), atol=tol, rtol=tol)


@pytest.mark.gpu
@pytest.mark.parametrize("backends", [("torch",), ("flashinfer", "torch")])
def test_gpu_mla_canonical_prefill_decode_and_cuda_graph(backends: tuple[str, ...]) -> None:
    if "flashinfer" in backends:
        pytest.importorskip("flashinfer")
    torch.manual_seed(99)
    dev = torch.device("cuda")
    ops = load_backend(backends, device="cuda", dtype=DType.BF16, cuda_graph=True, head_dim=128)
    ref = load_backend(["torch"], device="cuda", dtype=DType.BF16, cuda_graph=False, head_dim=128)

    ckv = torch.randn((8, 16, 512), dtype=torch.bfloat16, device=dev)
    kpe = torch.randn((8, 16, 64), dtype=torch.bfloat16, device=dev)
    buf = LatentKVBuffers(ckv, kpe, RAW_CODEC)

    rows = torch.tensor([5, 3], dtype=torch.int32, device=dev)
    lens = torch.tensor([16, 18], dtype=torch.int32, device=dev)
    tbl = torch.tensor([[1, 0], [2, 3]], dtype=torch.int32, device=dev)
    qn = torch.randn((8, 8, 512), dtype=torch.bfloat16, device=dev)
    qp = torch.randn((8, 8, 64), dtype=torch.bfloat16, device=dev)
    m_test = ops.attention.plan(ForwardMode.PREFILL, rows, lens, tbl, 16)
    m_ref = ref.attention.plan(ForwardMode.PREFILL, rows, lens, tbl, 16)
    o_test, o_ref = torch.empty_like(qn), torch.empty_like(qn)
    ops.mla.attend_mla(qn, qp, buf, m_test, o_test, scale=0.1)
    ref.mla.attend_mla(qn, qp, buf, m_ref, o_ref, scale=0.1)
    assert torch.allclose(o_test.float(), o_ref.float(), atol=5e-2, rtol=5e-2)

    # CUDA graph decode with padding request
    gmeta = ops.attention.graph_metadata(2, 2, 16, dev)
    qd_n = torch.randn((2, 8, 512), dtype=torch.bfloat16, device=dev)
    qd_p = torch.randn((2, 8, 64), dtype=torch.bfloat16, device=dev)
    g_out = torch.empty_like(qd_n)
    ops.attention.update_graph_metadata(
        gmeta, torch.zeros(2, dtype=torch.int32), torch.zeros((2, 2), dtype=torch.int32)
    )
    ops.mla.attend_mla(qd_n, qd_p, buf, gmeta, g_out, scale=0.1)  # plan before capture
    graph = torch.cuda.CUDAGraph()
    torch.cuda.synchronize()
    with torch.inference_mode(), torch.cuda.graph(graph):
        ops.mla.attend_mla(qd_n, qd_p, buf, gmeta, g_out, scale=0.1)
    with torch.inference_mode(), no_device_sync(strict=False):
        ops.attention.update_graph_metadata(
            gmeta,
            torch.tensor([16, 0], dtype=torch.int32),
            torch.tensor([[1, 0], [0, 0]], dtype=torch.int32),
        )
        graph.replay()
    torch.cuda.synchronize()
    assert g_out[1].abs().sum() == 0
    # Compare active row 0 against eager reference
    m_dec_ref = ref.attention.plan(
        ForwardMode.DECODE,
        torch.tensor([1, 1], dtype=torch.int32, device=dev),
        torch.tensor([16, 0], dtype=torch.int32, device=dev),
        tbl,
        16,
    )
    r_dec = torch.empty_like(qd_n)
    ref.mla.attend_mla(qd_n, qd_p, buf, m_dec_ref, r_dec, scale=0.1)
    assert torch.allclose(g_out[0].float(), r_dec[0].float(), atol=5e-2, rtol=5e-2)
