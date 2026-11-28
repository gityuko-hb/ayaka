"""Reference Multi-Head Latent Attention (MLA) ops; implements :class:`ayaka.ops.mla.MLAOps`."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch.nn import functional

from ayaka.core.errors import BackendError
from ayaka.core.types import ForwardMode
from ayaka.ops.backends.torch._checks import require_compute_dtype, require_like
from ayaka.ops.backends.torch.attention import (
    DECODE_CHUNK_BYTES,
    TorchAttentionPlan,
    TorchGraphPlan,
)
from ayaka.ops.kvcache import FP8_E4M3_CODEC, RAW_CODEC, SUPPORTED_KV_CODECS
from ayaka.utils.dtypes import FP8_E4M3_MAX, FP8_E4M3_MIN

if TYPE_CHECKING:
    from ayaka.ops.attention import AttentionMetadata
    from ayaka.ops.kvcache import LatentKVBuffers

__all__ = [
    "attend_mla",
    "attend_mla_paged",
    "dequant_latent",
    "require_latent_buffers",
    "store_latent",
    "validate_mla_inputs",
]


def require_latent_buffers(op: str, buffers: LatentKVBuffers) -> None:
    if buffers.codec not in SUPPORTED_KV_CODECS:
        raise BackendError(
            f"{op}: unsupported latent KV codec {buffers.codec!r}",
            context={"op": op, "argument": "buffers"},
        )
    if buffers.codec == RAW_CODEC:
        require_compute_dtype(op, "buffers.ckv", buffers.ckv)
    elif buffers.ckv.dtype != torch.float8_e4m3fn:
        raise BackendError(
            f"{op}: fp8_e4m3 latent KV requires float8_e4m3fn, got {buffers.ckv.dtype}",
            context={"op": op, "argument": "buffers"},
        )
    if buffers.ckv.ndim != 3 or buffers.kpe.ndim != 3:
        raise BackendError(
            f"{op}: latent buffers must be 3D [blocks, block_size, dim]",
            context={"op": op, "argument": "buffers"},
        )
    if buffers.ckv.shape[:2] != buffers.kpe.shape[:2] or buffers.ckv.dtype != buffers.kpe.dtype:
        raise BackendError(
            f"{op}: ckv {list(buffers.ckv.shape)} and kpe {list(buffers.kpe.shape)} disagree",
            context={"op": op, "argument": "buffers"},
        )


def store_latent(
    ckv: torch.Tensor,
    kpe: torch.Tensor,
    buffers: LatentKVBuffers,
    meta: AttentionMetadata,
) -> None:
    op = "mla.store_latent"
    require_latent_buffers(op, buffers)
    if meta.block_size != buffers.block_size:
        raise BackendError(
            f"{op}: step block_size {meta.block_size} != buffer block_size {buffers.block_size}",
            context={"op": op, "argument": "meta"},
        )
    rows = meta.slot_mapping.shape[0]
    if buffers.codec == RAW_CODEC:
        require_like(op, "ckv", ckv, (rows, buffers.kv_lora_rank), buffers.ckv.dtype)
        require_like(op, "kpe", kpe, (rows, buffers.qk_rope_head_dim), buffers.kpe.dtype)
        ckv_store, kpe_store = ckv, kpe
    else:
        require_compute_dtype(op, "ckv", ckv)
        require_like(op, "ckv", ckv, (rows, buffers.kv_lora_rank), ckv.dtype)
        require_like(op, "kpe", kpe, (rows, buffers.qk_rope_head_dim), ckv.dtype)
        ckv_store = (
            (ckv.float() / buffers.ckv_scale)
            .clamp(FP8_E4M3_MIN, FP8_E4M3_MAX)
            .to(torch.float8_e4m3fn)
        )
        kpe_store = (
            (kpe.float() / buffers.kpe_scale)
            .clamp(FP8_E4M3_MIN, FP8_E4M3_MAX)
            .to(torch.float8_e4m3fn)
        )
    blocks = meta.slot_mapping // buffers.block_size
    offsets = meta.slot_mapping % buffers.block_size
    buffers.ckv[blocks, offsets] = ckv_store
    buffers.kpe[blocks, offsets] = kpe_store


def dequant_latent(
    ckv: torch.Tensor, kpe: torch.Tensor, buffers: LatentKVBuffers
) -> tuple[torch.Tensor, torch.Tensor]:
    if buffers.codec == FP8_E4M3_CODEC:
        return ckv.float() * buffers.ckv_scale, kpe.float() * buffers.kpe_scale
    return ckv.float(), kpe.float()


def validate_mla_inputs(
    op: str,
    q_nope: torch.Tensor,
    q_pe: torch.Tensor,
    buffers: LatentKVBuffers,
    meta: AttentionMetadata,
    out: torch.Tensor,
) -> int:
    """Validate MLA query and latent KV shapes/dtypes, returning row count."""
    require_latent_buffers(op, buffers)
    if meta.block_size != buffers.block_size:
        raise BackendError(
            f"{op}: step block_size {meta.block_size} != buffer block_size {buffers.block_size}",
            context={"op": op, "argument": "meta"},
        )
    rows = meta.request_of_row.shape[0]
    heads = q_nope.shape[1] if q_nope.ndim == 3 else 0
    if buffers.codec == RAW_CODEC:
        require_like(op, "q_nope", q_nope, (rows, heads, buffers.kv_lora_rank), buffers.ckv.dtype)
    else:
        require_compute_dtype(op, "q_nope", q_nope)
        require_like(op, "q_nope", q_nope, (rows, heads, buffers.kv_lora_rank), q_nope.dtype)
    require_like(op, "q_pe", q_pe, (rows, heads, buffers.qk_rope_head_dim), q_nope.dtype)
    require_like(op, "out", out, q_nope.shape, q_nope.dtype)
    return rows


def attend_mla_paged(
    q_nope: torch.Tensor,
    q_pe: torch.Tensor,
    buffers: LatentKVBuffers,
    block_table: torch.Tensor,
    kv_lengths: torch.Tensor,
    mode: ForwardMode,
    rows_per_req: tuple[int, ...] | None,
    lengths_per_req: tuple[int, ...] | None,
    out: torch.Tensor,
    *,
    scale: float,
) -> None:
    """Reference paged MLA forward shared by backends when no fused kernel applies."""
    if mode is ForwardMode.DECODE:
        _attend_mla_decode(q_nope, q_pe, buffers, block_table, kv_lengths, out, scale)
        return
    if rows_per_req is None or lengths_per_req is None:
        raise BackendError("mla.attend_mla: prefill requires per-request row and length counts")
    out.zero_()
    first = 0
    for req, (count, length) in enumerate(zip(rows_per_req, lengths_per_req, strict=True)):
        if length and count:
            blocks = block_table[req, : -(-length // buffers.block_size)].long()
            ckv, kpe = dequant_latent(
                buffers.ckv[blocks].flatten(0, 1)[:length],
                buffers.kpe[blocks].flatten(0, 1)[:length],
                buffers,
            )
            qn = q_nope[first : first + count].float().transpose(0, 1)  # [H, R, ckv_dim]
            qp = q_pe[first : first + count].float().transpose(0, 1)  # [H, R, kpe_dim]
            scores = (torch.matmul(qn, ckv.t()) + torch.matmul(qp, kpe.t())) * scale
            visible = (
                torch.arange(length, device=q_nope.device)[None, :]
                <= torch.arange(length - count, length, device=q_nope.device)[:, None]
            )
            scores = scores.masked_fill(~visible[None, :, :], float("-inf"))
            weights = functional.softmax(scores, dim=-1)
            res = torch.matmul(weights, ckv)  # [H, R, ckv_dim]
            out[first : first + count] = res.transpose(0, 1).to(out.dtype)
        first += count


def attend_mla(
    q_nope: torch.Tensor,
    q_pe: torch.Tensor,
    buffers: LatentKVBuffers,
    meta: AttentionMetadata,
    out: torch.Tensor,
    *,
    scale: float,
) -> None:
    op = "mla.attend_mla"
    if not isinstance(meta.plan, (TorchAttentionPlan, TorchGraphPlan)) or not isinstance(
        meta.block_table, torch.Tensor
    ):
        raise BackendError(
            f"{op}: step metadata was planned by another backend",
            context={"op": op, "argument": "meta"},
        )
    rows = validate_mla_inputs(op, q_nope, q_pe, buffers, meta, out)
    if rows == 0:
        return
    if meta.mode is not ForwardMode.DECODE and not isinstance(meta.plan, TorchAttentionPlan):
        raise BackendError(f"{op}: graph metadata must be DECODE", context={"op": op})
    rows_per_req = meta.plan.rows if isinstance(meta.plan, TorchAttentionPlan) else None
    lengths_per_req = meta.plan.kv_lengths if isinstance(meta.plan, TorchAttentionPlan) else None
    attend_mla_paged(
        q_nope,
        q_pe,
        buffers,
        meta.block_table,
        meta.kv_lengths,
        meta.mode,
        rows_per_req,
        lengths_per_req,
        out,
        scale=scale,
    )


def _attend_mla_decode(
    q_nope: torch.Tensor,
    q_pe: torch.Tensor,
    buffers: LatentKVBuffers,
    table: torch.Tensor,
    lengths: torch.Tensor,
    out: torch.Tensor,
    scale: float,
) -> None:
    requests, heads, ckv_dim = q_nope.shape
    block_size = buffers.block_size
    width = table.shape[1]
    per_block = max(requests * block_size * (ckv_dim + buffers.qk_rope_head_dim) * 4, 1)
    step = max(1, min(width, DECODE_CHUNK_BYTES // per_block))
    qn = q_nope.float()
    qp = q_pe.float()
    limit = lengths.long()[:, None]
    running_max = torch.full(
        (requests, heads), float("-inf"), dtype=torch.float32, device=qn.device
    )
    total = torch.zeros((requests, heads), dtype=torch.float32, device=qn.device)
    acc = torch.zeros((requests, heads, ckv_dim), dtype=torch.float32, device=qn.device)
    for first in range(0, width, step):
        blocks = table[:, first : first + step].long()
        ckv, kpe = dequant_latent(
            buffers.ckv[blocks].flatten(1, 2), buffers.kpe[blocks].flatten(1, 2), buffers
        )
        start = first * block_size
        positions = torch.arange(start, start + ckv.shape[1], device=qn.device)
        visible = positions[None, :] < limit
        scores = (
            torch.einsum("bhd,bld->bhl", qn, ckv) + torch.einsum("bhd,bld->bhl", qp, kpe)
        ) * scale
        scores = scores.masked_fill(~visible[:, None, :], float("-inf"))
        new_max = torch.maximum(running_max, scores.amax(dim=-1))
        shift = torch.where(torch.isfinite(new_max), new_max, 0.0)
        correction = torch.exp(running_max - shift)
        weights = torch.exp(scores - shift[..., None])
        ckv = torch.where(visible[:, :, None], ckv, 0.0)
        acc = acc * correction[..., None] + torch.einsum("bhl,bld->bhd", weights, ckv)
        total = total * correction + weights.sum(dim=-1)
        running_max = new_max
    seen = (total > 0) & (lengths > 0)[:, None]
    result = torch.where(seen[..., None], acc / torch.where(seen, total, 1.0)[..., None], 0.0)
    out.copy_(result.to(out.dtype))
