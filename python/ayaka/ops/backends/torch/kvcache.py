"""Reference KV-cache ops; implements :class:`ayaka.ops.kvcache.KVCacheOps`.

Both layouts are served. Every op reads the cache through :func:`nhd_view`, a
view with the axes in ``[block, offset, head, dim]`` order, so the code below
is written once.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ayaka.core.errors import BackendError
from ayaka.core.types import KVLayoutKind
from ayaka.ops.backends.torch._checks import (
    require_compute_dtype,
    require_index_tensor,
    require_like,
)
from ayaka.ops.kvcache import RAW_CODEC

if TYPE_CHECKING:
    import torch

    from ayaka.ops.attention import AttentionMetadata
    from ayaka.ops.kvcache import KVBuffers

__all__ = ["copy_blocks", "gather", "layout", "nhd_view", "require_buffers", "store", "zero_blocks"]


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
    if buffers.codec != RAW_CODEC:
        raise BackendError(
            f"{op}: the torch backend stores the {RAW_CODEC!r} codec, not {buffers.codec!r}",
            context={"op": op, "argument": "buffers"},
        )
    require_compute_dtype(op, "buffers.k", buffers.k)
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
    require_like(op, "k", k, shape, buffers.k.dtype)
    require_like(op, "v", v, shape, buffers.k.dtype)
    blocks = meta.slot_mapping // buffers.block_size
    offsets = meta.slot_mapping % buffers.block_size
    nhd_view(buffers.k, buffers)[blocks, offsets] = k
    nhd_view(buffers.v, buffers)[blocks, offsets] = v


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
