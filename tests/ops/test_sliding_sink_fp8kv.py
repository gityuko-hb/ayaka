"""Contract and GPU parity tests for sliding-window, attention sinks, and opt-in FP8 KV."""

from __future__ import annotations

import math
from typing import Any

import pytest

from ayaka.core.errors import BackendError
from ayaka.core.types import ForwardMode, KVLayoutKind
from ayaka.ops.backend import load_backend
from ayaka.ops.kvcache import FP8_E4M3_CODEC, RAW_CODEC, KVBuffers
from ayaka.utils.dtypes import DType
from ayaka.utils.torch_utils import no_device_sync

torch: Any = pytest.importorskip("torch")


def _fp64_window_sink_oracle(
    q: Any,
    k_seqs: list[Any],
    v_seqs: list[Any],
    *,
    rows_per_req: list[int],
    scale: float,
    window: int | None,
    sinks: Any | None,
) -> list[Any]:
    """Float64 oracle for causal sliding-window attention with virtual per-head sinks."""
    q64 = q.to(torch.float64)
    heads = q64.shape[1]
    kv_heads = k_seqs[0].shape[1]
    group = heads // kv_heads
    outs: list[Any] = []
    first = 0
    for k_seq, v_seq, count in zip(k_seqs, v_seqs, rows_per_req, strict=True):
        length = k_seq.shape[0]
        if length == 0 or count == 0:
            outs.append(torch.zeros((count, heads, q64.shape[2]), dtype=torch.float64))
            first += count
            continue
        k_exp = k_seq.to(torch.float64).repeat_interleave(group, dim=1).transpose(0, 1)
        v_exp = v_seq.to(torch.float64).repeat_interleave(group, dim=1).transpose(0, 1)
        q_req = q64[first : first + count].transpose(0, 1)  # [H, R, D]
        scores = torch.matmul(q_req, k_exp.transpose(-2, -1)) * scale
        k_pos = torch.arange(length)[None, :]
        q_pos = torch.arange(length - count, length)[:, None]
        visible = k_pos <= q_pos
        if window is not None:
            visible = visible & (k_pos >= (q_pos - window + 1))
        scores = scores.masked_fill(~visible[None], float("-inf"))
        if sinks is not None:
            sink_col = sinks.to(torch.float64).view(heads, 1, 1).expand(heads, count, 1)
            weights = torch.softmax(torch.cat([scores, sink_col], dim=-1), dim=-1)[..., :-1]
        else:
            weights = torch.softmax(scores, dim=-1)
        outs.append(torch.matmul(weights, v_exp).transpose(0, 1))
        first += count
    return outs


def _make_kv_buffers(
    k_seqs: list[Any],
    v_seqs: list[Any],
    block_size: int,
    layout: KVLayoutKind,
    codec: str,
    *,
    k_scale: float = 1.0,
    v_scale: float = 1.0,
    device: str = "cpu",
    ops_kvcache: Any = None,
    plan_fn: Any = None,
) -> tuple[KVBuffers, Any, list[Any], list[Any]]:
    kv_heads, head_dim = k_seqs[0].shape[1], k_seqs[0].shape[2]
    compute_dtype = k_seqs[0].dtype
    storage_dtype = torch.float8_e4m3fn if codec == FP8_E4M3_CODEC else compute_dtype
    pages_per_req = [math.ceil(k.shape[0] / block_size) if k.shape[0] else 0 for k in k_seqs]
    total_blocks = 1 + sum(pages_per_req)
    shape = (
        (total_blocks, block_size, kv_heads, head_dim)
        if layout is KVLayoutKind.NHD
        else (total_blocks, kv_heads, block_size, head_dim)
    )
    k_buf = torch.zeros(shape, dtype=storage_dtype, device=device)
    v_buf = torch.zeros(shape, dtype=storage_dtype, device=device)
    buffers = KVBuffers(k_buf, v_buf, layout, codec, k_scale=k_scale, v_scale=v_scale)

    width = max(max(pages_per_req, default=1), 1)
    table_rows: list[list[int]] = []
    cursor = 1
    lengths = [k.shape[0] for k in k_seqs]
    for pages in pages_per_req:
        row = list(range(cursor, cursor + pages))
        cursor += pages
        row.extend([0] * (width - len(row)))
        table_rows.append(row)
    table = torch.tensor(table_rows, dtype=torch.int32, device=device)

    rows_nonempty = [k.shape[0] for k in k_seqs]
    meta_store = plan_fn(
        ForwardMode.PREFILL,
        torch.tensor(rows_nonempty, dtype=torch.int32, device=device),
        torch.tensor(lengths, dtype=torch.int32, device=device),
        table,
        block_size,
    )
    cat_k = torch.cat(k_seqs, dim=0)
    cat_v = torch.cat(v_seqs, dim=0)
    ops_kvcache.store(cat_k, cat_v, buffers, meta_store)

    if codec == FP8_E4M3_CODEC:
        # Read back exact stored dequantized tensors for fair quantization-isolated oracle check
        nhd_k = k_buf if layout is KVLayoutKind.NHD else k_buf.permute(0, 2, 1, 3)
        nhd_v = v_buf if layout is KVLayoutKind.NHD else v_buf.permute(0, 2, 1, 3)
        deq_k, deq_v = [], []
        for req_idx, length in enumerate(lengths):
            if length == 0:
                deq_k.append(torch.empty((0, kv_heads, head_dim), dtype=torch.float32))
                deq_v.append(torch.empty((0, kv_heads, head_dim), dtype=torch.float32))
                continue
            blk_ids = table[req_idx, : pages_per_req[req_idx]].long()
            deq_k.append(
                (nhd_k[blk_ids].reshape(-1, kv_heads, head_dim)[:length].float() * k_scale).cpu()
            )
            deq_v.append(
                (nhd_v[blk_ids].reshape(-1, kv_heads, head_dim)[:length].float() * v_scale).cpu()
            )
    else:
        deq_k = [s.cpu() for s in k_seqs]
        deq_v = [s.cpu() for s in v_seqs]
    return buffers, table, deq_k, deq_v


@pytest.mark.unit
@pytest.mark.parametrize("layout", [KVLayoutKind.NHD, KVLayoutKind.HND])
@pytest.mark.parametrize("window", [None, 1, 5])
@pytest.mark.parametrize("with_sinks", [False, True])
def test_torch_window_sinks_and_fp8_kv_match_oracle(
    layout: KVLayoutKind, window: int | None, with_sinks: bool
) -> None:
    torch.manual_seed(21)
    ops = load_backend(["torch"], device="cpu", dtype=DType.BF16)
    block_size = 4
    heads, kv_heads, head_dim = 6, 2, 16
    scale = 1.0 / math.sqrt(head_dim)
    k_seqs = [
        torch.randn((11, kv_heads, head_dim), dtype=torch.bfloat16) * 0.5,
        torch.randn((7, kv_heads, head_dim), dtype=torch.bfloat16) * 0.5,
    ]
    v_seqs = [
        torch.randn((11, kv_heads, head_dim), dtype=torch.bfloat16) * 0.5,
        torch.randn((7, kv_heads, head_dim), dtype=torch.bfloat16) * 0.5,
    ]
    sinks = torch.randn(heads, dtype=torch.float32) * 1.5 if with_sinks else None

    for codec, k_s, v_s in [(RAW_CODEC, 1.0, 1.0), (FP8_E4M3_CODEC, 0.25, 0.25)]:
        buffers, table, deq_k, deq_v = _make_kv_buffers(
            k_seqs,
            v_seqs,
            block_size,
            layout,
            codec,
            k_scale=k_s,
            v_scale=v_s,
            ops_kvcache=ops.kvcache,
            plan_fn=ops.attention.plan,
        )
        # Prefill suffix queries
        q_pre = torch.randn((7, heads, head_dim), dtype=torch.bfloat16)
        m_pre = ops.attention.plan(
            ForwardMode.PREFILL,
            torch.tensor([4, 3], dtype=torch.int32),
            torch.tensor([11, 7], dtype=torch.int32),
            table,
            block_size,
        )
        o_pre = torch.empty_like(q_pre)
        ops.attention.attend(q_pre, buffers, m_pre, o_pre, scale=scale, window=window, sinks=sinks)
        exp_pre = _fp64_window_sink_oracle(
            q_pre, deq_k, deq_v, rows_per_req=[4, 3], scale=scale, window=window, sinks=sinks
        )
        assert torch.allclose(o_pre[:4].float(), exp_pre[0].float(), atol=5e-2, rtol=5e-2)
        assert torch.allclose(o_pre[4:].float(), exp_pre[1].float(), atol=5e-2, rtol=5e-2)

        # Decode
        q_dec = torch.randn((2, heads, head_dim), dtype=torch.bfloat16)
        m_dec = ops.attention.plan(
            ForwardMode.DECODE,
            torch.tensor([1, 1], dtype=torch.int32),
            torch.tensor([11, 7], dtype=torch.int32),
            table,
            block_size,
        )
        o_dec = torch.empty_like(q_dec)
        ops.attention.attend(q_dec, buffers, m_dec, o_dec, scale=scale, window=window, sinks=sinks)
        exp_dec = _fp64_window_sink_oracle(
            q_dec, deq_k, deq_v, rows_per_req=[1, 1], scale=scale, window=window, sinks=sinks
        )
        assert torch.allclose(o_dec[0:1].float(), exp_dec[0].float(), atol=5e-2, rtol=5e-2)
        assert torch.allclose(o_dec[1:2].float(), exp_dec[1].float(), atol=5e-2, rtol=5e-2)


@pytest.mark.unit
def test_invalid_window_and_sinks_raise_backend_error() -> None:
    ops = load_backend(["torch"], device="cpu", dtype=DType.FP32)
    meta = ops.attention.plan(
        ForwardMode.DECODE,
        torch.tensor([1], dtype=torch.int32),
        torch.tensor([1], dtype=torch.int32),
        torch.tensor([[1]], dtype=torch.int32),
        4,
    )
    buf = KVBuffers(
        torch.zeros((2, 4, 2, 4)), torch.zeros((2, 4, 2, 4)), KVLayoutKind.NHD, RAW_CODEC
    )
    q = torch.zeros((1, 2, 4))
    out = torch.empty_like(q)
    with pytest.raises(BackendError, match="window must be positive"):
        ops.attention.attend(q, buf, meta, out, scale=0.5, window=0)
    with pytest.raises(BackendError, match="sinks"):
        ops.attention.attend(q, buf, meta, out, scale=0.5, sinks=torch.zeros(5))


@pytest.mark.gpu
@pytest.mark.parametrize("backends", [("torch",), ("flashinfer", "torch")])
def test_gpu_heterogeneous_window_sinks_and_fp8_in_cuda_graph(backends: tuple[str, ...]) -> None:
    if "flashinfer" in backends:
        pytest.importorskip("flashinfer")
    torch.manual_seed(77)
    dev = torch.device("cuda")
    ops = load_backend(backends, device="cuda", dtype=DType.BF16, cuda_graph=True, head_dim=64)
    block_size = 16
    heads, kv_heads, head_dim = 8, 2, 64
    scale = 1.0 / math.sqrt(head_dim)
    k_seqs = [
        torch.randn((25, kv_heads, head_dim), dtype=torch.bfloat16, device=dev) * 0.5,
        torch.randn((19, kv_heads, head_dim), dtype=torch.bfloat16, device=dev) * 0.5,
    ]
    v_seqs = [
        torch.randn((25, kv_heads, head_dim), dtype=torch.bfloat16, device=dev) * 0.5,
        torch.randn((19, kv_heads, head_dim), dtype=torch.bfloat16, device=dev) * 0.5,
    ]
    sinks = torch.randn(heads, dtype=torch.float32, device=dev)

    # Layer 0: sliding window=8 + sinks in FP8 KV; Layer 1: full attention in FP8 KV
    buf_win, tbl, deq_k, deq_v = _make_kv_buffers(
        k_seqs,
        v_seqs,
        block_size,
        ops.kvcache.layout(),
        FP8_E4M3_CODEC,
        k_scale=0.25,
        v_scale=0.25,
        device="cuda",
        ops_kvcache=ops.kvcache,
        plan_fn=ops.attention.plan,
    )
    buf_full, _, deq_k2, deq_v2 = _make_kv_buffers(
        k_seqs,
        v_seqs,
        block_size,
        ops.kvcache.layout(),
        FP8_E4M3_CODEC,
        k_scale=0.25,
        v_scale=0.25,
        device="cuda",
        ops_kvcache=ops.kvcache,
        plan_fn=ops.attention.plan,
    )

    # Eager prefill check (prefill both layers in the SAME planned step!)
    q_pre = torch.randn((9, heads, head_dim), dtype=torch.bfloat16, device=dev)
    m_pre = ops.attention.plan(
        ForwardMode.PREFILL,
        torch.tensor([5, 4], dtype=torch.int32, device=dev),
        torch.tensor([25, 19], dtype=torch.int32, device=dev),
        tbl,
        block_size,
    )
    o_win_pre, o_full_pre = torch.empty_like(q_pre), torch.empty_like(q_pre)
    ops.attention.attend(q_pre, buf_win, m_pre, o_win_pre, scale=scale, window=8, sinks=sinks)
    ops.attention.attend(q_pre, buf_full, m_pre, o_full_pre, scale=scale, window=None, sinks=None)
    exp_win_pre = _fp64_window_sink_oracle(
        q_pre.cpu(), deq_k, deq_v, rows_per_req=[5, 4], scale=scale, window=8, sinks=sinks.cpu()
    )
    exp_full_pre = _fp64_window_sink_oracle(
        q_pre.cpu(), deq_k2, deq_v2, rows_per_req=[5, 4], scale=scale, window=None, sinks=None
    )
    assert torch.allclose(o_win_pre[:5].float().cpu(), exp_win_pre[0].float(), atol=5e-2, rtol=5e-2)
    assert torch.allclose(
        o_full_pre[5:].float().cpu(), exp_full_pre[1].float(), atol=5e-2, rtol=5e-2
    )

    # Captured CUDA graph running BOTH alternating layers together in FP8
    gmeta = ops.attention.graph_metadata(2, 2, block_size, dev)
    qd = torch.randn((2, heads, head_dim), dtype=torch.bfloat16, device=dev)
    gw, gf = torch.empty_like(qd), torch.empty_like(qd)
    ops.attention.update_graph_metadata(
        gmeta, torch.zeros(2, dtype=torch.int32), torch.zeros((2, 2), dtype=torch.int32)
    )
    ops.attention.attend(qd, buf_win, gmeta, gw, scale=scale, window=8, sinks=sinks)
    ops.attention.attend(qd, buf_full, gmeta, gf, scale=scale, window=None, sinks=None)

    graph = torch.cuda.CUDAGraph()
    torch.cuda.synchronize()
    with torch.inference_mode(), torch.cuda.graph(graph):
        ops.attention.attend(qd, buf_win, gmeta, gw, scale=scale, window=8, sinks=sinks)
        ops.attention.attend(qd, buf_full, gmeta, gf, scale=scale, window=None, sinks=None)
    lens_cpu = torch.tensor([25, 19], dtype=torch.int32)
    tbl_cpu = tbl.cpu()
    with torch.inference_mode(), no_device_sync(strict=True):
        ops.attention.update_graph_metadata(gmeta, lens_cpu, tbl_cpu)
        graph.replay()
    torch.cuda.synchronize()

    exp_gw = _fp64_window_sink_oracle(
        qd.cpu(), deq_k, deq_v, rows_per_req=[1, 1], scale=scale, window=8, sinks=sinks.cpu()
    )
    exp_gf = _fp64_window_sink_oracle(
        qd.cpu(), deq_k2, deq_v2, rows_per_req=[1, 1], scale=scale, window=None, sinks=None
    )
    assert torch.allclose(gw[0:1].float().cpu(), exp_gw[0].float(), atol=5e-2, rtol=5e-2)
    assert torch.allclose(gw[1:2].float().cpu(), exp_gw[1].float(), atol=5e-2, rtol=5e-2)
    assert torch.allclose(gf[0:1].float().cpu(), exp_gf[0].float(), atol=5e-2, rtol=5e-2)
    assert torch.allclose(gf[1:2].float().cpu(), exp_gf[1].float(), atol=5e-2, rtol=5e-2)
