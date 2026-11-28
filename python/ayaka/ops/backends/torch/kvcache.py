"""Reference KV-cache ops; implements :class:`ayaka.ops.kvcache.KVCacheOps`.

Both NHD and HND layouts and both ``raw`` and ``fp8_e4m3`` codecs are served.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from ayaka.core.errors import BackendError
from ayaka.core.types import KVLayoutKind
from ayaka.ops.backends.torch._checks import (
    require_compute_dtype,
    require_index_tensor,
    require_like,
)
from ayaka.ops.kvcache import FP8_E4M3_CODEC, RAW_CODEC, SUPPORTED_KV_CODECS
from ayaka.utils.dtypes import FP8_E4M3_MAX, FP8_E4M3_MIN

if TYPE_CHECKING:
    from ayaka.ops.attention import AttentionMetadata
    from ayaka.ops.kvcache import KVBuffers

__all__ = [
    "copy_blocks",
    "dequantize_kv",
    "gather",
    "layout",
    "nhd_view",
    "require_buffers",
    "store",
    "zero_blocks",
]


def layout() -> KVLayoutKind:
    return KVLayoutKind.NHD


def nhd_view(tensor: torch.Tensor, buffers: KVBuffers) -> torch.Tensor:
    """View ``tensor``, laid out like ``buffers.k``, as ``[block, offset, head, dim]``."""
    return tensor if buffers.layout is KVLayoutKind.NHD else tensor.permute(0, 2, 1, 3)


def require_buffers(op: str, buffers: KVBuffers) -> None:
    """Refuse a cache this backend cannot read: layout, codec, shape, dtype and strides."""
    if buffers.layout not in (KVLayoutKind.NHD, KVLayoutKind.HND):
        raise BackendError(
            f"{op}: the torch backend serves the NHD and HND layouts, not {buffers.layout}",
            context={"op": op, "argument": "buffers"},
        )
    if buffers.codec not in SUPPORTED_KV_CODECS:
        raise BackendError(
            f"{op}: the torch backend stores {list(SUPPORTED_KV_CODECS)} codecs, "
            f"not {buffers.codec!r}",
            context={"op": op, "argument": "buffers"},
        )
    if buffers.codec == RAW_CODEC:
        require_compute_dtype(op, "buffers.k", buffers.k)
    else:
        if buffers.k.dtype != torch.float8_e4m3fn:
            raise BackendError(
                f"{op}: fp8_e4m3 KV buffers must have dtype float8_e4m3fn, got {buffers.k.dtype}",
                context={"op": op, "argument": "buffers"},
            )
        if buffers.k_scale <= 0 or buffers.v_scale <= 0:
            raise BackendError(
                f"{op}: fp8_e4m3 scales must be positive",
                context={"op": op, "argument": "buffers"},
            )
    if buffers.k.ndim != 4:
        raise BackendError(
            f"{op}: buffers.k must have four dimensions, got {list(buffers.k.shape)}",
            context={"op": op, "argument": "buffers"},
        )
    require_like(op, "buffers.v", buffers.v, buffers.k.shape, buffers.k.dtype)
    if not (buffers.k.is_contiguous() and buffers.v.is_contiguous()):
        raise BackendError(
            f"{op}: buffers.k and buffers.v must be contiguous",
            context={"op": op, "argument": "buffers"},
        )


def dequantize_kv(
    k: torch.Tensor, v: torch.Tensor, buffers: KVBuffers
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert raw or FP8 KV slices to float32 with scaling."""
    if buffers.codec == FP8_E4M3_CODEC:
        return k.float() * buffers.k_scale, v.float() * buffers.v_scale
    return k.float(), v.float()


def store(k: torch.Tensor, v: torch.Tensor, buffers: KVBuffers, meta: AttentionMetadata) -> None:
    op = "kvcache.store"
    require_buffers(op, buffers)
    if meta.block_size != buffers.block_size:
        raise BackendError(
            f"{op}: the step was planned for blocks of {meta.block_size} positions "
            f"but the cache has blocks of {buffers.block_size}",
            context={"op": op, "argument": "meta"},
        )
    shape = (meta.slot_mapping.shape[0], buffers.num_kv_heads, buffers.head_dim)
    if buffers.codec == RAW_CODEC:
        require_like(op, "k", k, shape, buffers.k.dtype)
        require_like(op, "v", v, shape, buffers.k.dtype)
        k_store, v_store = k, v
    else:
        require_compute_dtype(op, "k", k)
        require_like(op, "k", k, shape, k.dtype)
        require_like(op, "v", v, shape, k.dtype)
        k_store = (
            (k.float() / buffers.k_scale).clamp(FP8_E4M3_MIN, FP8_E4M3_MAX).to(torch.float8_e4m3fn)
        )
        v_store = (
            (v.float() / buffers.v_scale).clamp(FP8_E4M3_MIN, FP8_E4M3_MAX).to(torch.float8_e4m3fn)
        )
    blocks = meta.slot_mapping // buffers.block_size
    offsets = meta.slot_mapping % buffers.block_size
    nhd_view(buffers.k, buffers)[blocks, offsets] = k_store
    nhd_view(buffers.v, buffers)[blocks, offsets] = v_store


def gather(
    buffers: KVBuffers, block_ids: torch.Tensor, out_k: torch.Tensor, out_v: torch.Tensor
) -> None:
    op = "kvcache.gather"
    require_buffers(op, buffers)
    require_index_tensor(op, "block_ids", block_ids, ndim=1)
    shape = (block_ids.shape[0], *buffers.k.shape[1:])
    require_like(op, "out_k", out_k, shape, buffers.k.dtype)
    require_like(op, "out_v", out_v, shape, buffers.k.dtype)
    out_k.copy_(buffers.k[block_ids.long()])
    out_v.copy_(buffers.v[block_ids.long()])


def copy_blocks(buffers: KVBuffers, src: torch.Tensor, dst: torch.Tensor) -> None:
    op = "kvcache.copy_blocks"
    require_buffers(op, buffers)
    require_index_tensor(op, "src", src, ndim=1)
    require_index_tensor(op, "dst", dst, ndim=1)
    if src.shape != dst.shape:
        raise BackendError(
            f"{op}: src has {src.shape[0]} blocks but dst has {dst.shape[0]}",
            context={"op": op, "argument": "dst"},
        )
    buffers.k[dst.long()] = buffers.k[src.long()]
    buffers.v[dst.long()] = buffers.v[src.long()]


def zero_blocks(buffers: KVBuffers, block_ids: torch.Tensor) -> None:
    op = "kvcache.zero_blocks"
    require_buffers(op, buffers)
    require_index_tensor(op, "block_ids", block_ids, ndim=1)
    buffers.k[block_ids.long()] = 0
    buffers.v[block_ids.long()] = 0
