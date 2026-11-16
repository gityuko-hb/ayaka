"""Reference attention ops; implements :class:`ayaka.ops.attention.AttentionOps`.

Written to be read, not to be fast. ``attend`` loops over the requests of the
step, gathers each one's keys and values out of the paged cache through its
block table and runs ``scaled_dot_product_attention`` with a causal mask. The
gather copies the whole context every call, so its cost grows with the context;
do not measure speed with it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import torch
from torch.nn import functional

from ayaka.core.errors import BackendError
from ayaka.core.types import ForwardMode
from ayaka.ops.attention import AttentionMetadata
from ayaka.ops.backends.torch._checks import require_index_tensor, require_like
from ayaka.ops.backends.torch.kvcache import nhd_view, require_buffers

if TYPE_CHECKING:
    from ayaka.ops.kvcache import KVBuffers

__all__ = ["TorchAttentionPlan", "attend", "plan"]


@dataclass(frozen=True, slots=True)
class TorchAttentionPlan:
    """The host-side lists ``attend`` loops over, read once by :func:`plan`."""

    rows: tuple[int, ...]
    kv_lengths: tuple[int, ...]


def plan(
    mode: ForwardMode,
    rows_per_request: torch.Tensor,
    kv_lengths: torch.Tensor,
    block_rows: torch.Tensor,
    block_size: int,
) -> AttentionMetadata:
    op = "attention.plan"
    require_index_tensor(op, "rows_per_request", rows_per_request, ndim=1)
    require_index_tensor(op, "kv_lengths", kv_lengths, ndim=1)
    require_index_tensor(op, "block_rows", block_rows, ndim=2)
    batch = rows_per_request.shape[0]
    if kv_lengths.shape[0] != batch or block_rows.shape[0] != batch:
        raise BackendError(
            f"{op}: rows_per_request, kv_lengths and block_rows must all describe the same "
            f"requests, got {batch}, {kv_lengths.shape[0]} and {block_rows.shape[0]}",
            context={"op": op},
        )
    if block_size < 1:
        raise BackendError(f"{op}: block_size must be positive, got {block_size}")
    rows, lengths = _host_ints(rows_per_request), _host_ints(kv_lengths)
    _check_step(op, mode, rows, lengths)

    device = rows_per_request.device
    table = block_rows.to(torch.int64)
    _check_table(op, table, lengths, block_size, has_rows=sum(rows) > 0)

    rows64, lengths64 = rows_per_request.to(torch.int64), kv_lengths.to(torch.int64)
    request_of_row = torch.repeat_interleave(torch.arange(batch, device=device), rows64)
    first_row = torch.cumsum(rows64, dim=0) - rows64
    within = torch.arange(sum(rows), device=device) - first_row[request_of_row]
    row_length = lengths64[request_of_row]
    is_padding = row_length == 0
    positions = torch.where(is_padding, 0, row_length - rows64[request_of_row] + within)
    block = table[request_of_row, positions // block_size]
    slots = torch.where(is_padding, 0, block * block_size + positions % block_size)
    return AttentionMetadata(
        mode=mode,
        block_size=block_size,
        rows_per_request=rows64.to(torch.int32),
        request_of_row=request_of_row.to(torch.int32),
        positions=positions.to(torch.int32),
        slot_mapping=slots,
        kv_lengths=lengths64.to(torch.int32),
        block_table=table.to(torch.int32),
        plan=TorchAttentionPlan(rows, lengths),
    )


def _host_ints(tensor: torch.Tensor) -> tuple[int, ...]:
    """Read a 1-D integer tensor on the host; ``plan`` runs outside any graph."""
    # torch types ``tolist`` as ``list[Unknown]``.
    return tuple(cast("list[int]", tensor.tolist()))  # pyright: ignore[reportUnknownMemberType]


def _check_step(
    op: str, mode: ForwardMode, rows: tuple[int, ...], lengths: tuple[int, ...]
) -> None:
    for index, (count, length) in enumerate(zip(rows, lengths, strict=True)):
        if count < 0 or length < 0:
            raise BackendError(
                f"{op}: request {index} has {count} rows and {length} tokens; neither can be "
                "negative",
                context={"op": op, "request": index},
            )
        if length and count > length:
            raise BackendError(
                f"{op}: request {index} has {count} query rows but only {length} tokens with "
                "keys after this step; kv_lengths counts this step's rows",
                context={"op": op, "request": index},
            )
        if mode is ForwardMode.DECODE and count != 1:
            raise BackendError(
                f"{op}: a DECODE step has one row per request, request {index} has {count}",
                context={"op": op, "request": index},
            )
    if mode is ForwardMode.IDLE and sum(rows):
        raise BackendError(f"{op}: an IDLE step has no rows, got {sum(rows)}", context={"op": op})


def _check_table(
    op: str, table: torch.Tensor, lengths: tuple[int, ...], block_size: int, *, has_rows: bool
) -> None:
    """Refuse a block table that is too short or has the padding block inside a request."""
    needed = tuple(-(-length // block_size) for length in lengths)
    max_blocks = table.shape[1]
    longest = max(needed, default=0)
    if longest > max_blocks or (has_rows and max_blocks == 0):
        raise BackendError(
            f"{op}: block_rows has {max_blocks} column(s) but the longest request needs "
            f"{longest} blocks of {block_size}",
            context={"op": op, "argument": "block_rows"},
        )
    used = torch.arange(max_blocks, device=table.device) < torch.tensor(
        needed, dtype=torch.int64, device=table.device
    ).reshape(-1, 1)
    if bool((used & (table == 0)).any()):
        raise BackendError(
            f"{op}: block_rows has the padding block 0 inside a request's blocks",
            context={"op": op, "argument": "block_rows"},
        )


def attend(
    q: torch.Tensor,
    buffers: KVBuffers,
    meta: AttentionMetadata,
    out: torch.Tensor,
    *,
    scale: float,
) -> None:
    op = "attention.attend"
    require_buffers(op, buffers)
    if not isinstance(meta.plan, TorchAttentionPlan) or not isinstance(
        meta.block_table, torch.Tensor
    ):
        raise BackendError(
            f"{op}: this step was planned by another backend; plan it with the same ops",
            context={"op": op, "argument": "meta"},
        )
    if meta.block_size != buffers.block_size:
        raise BackendError(
            f"{op}: the step was planned for blocks of {meta.block_size} positions "
            f"but the cache has blocks of {buffers.block_size}",
            context={"op": op, "argument": "meta"},
        )
    heads = q.shape[1] if q.ndim == 3 else 0
    require_like(
        op, "q", q, (meta.request_of_row.shape[0], heads, buffers.head_dim), buffers.k.dtype
    )
    require_like(op, "out", out, q.shape, q.dtype)
    if heads == 0 or heads % buffers.num_kv_heads:
        raise BackendError(
            f"{op}: q has {heads} heads, which is not a positive multiple of the cache's "
            f"{buffers.num_kv_heads} key/value heads",
            context={"op": op, "argument": "q"},
        )

    keys, values = nhd_view(buffers.k, buffers), nhd_view(buffers.v, buffers)
    group = heads // buffers.num_kv_heads
    out.zero_()  # the rows of padding requests stay zero
    first = 0
    for request, (count, length) in enumerate(
        zip(meta.plan.rows, meta.plan.kv_lengths, strict=True)
    ):
        if length and count:
            blocks = meta.block_table[request, : -(-length // buffers.block_size)].long()
            # [length, kv_heads, head_dim] -> [heads, length, head_dim], repeating each kv head
            # for the `group` query heads that share it.
            context_k = keys[blocks].flatten(0, 1)[:length].float()
            context_v = values[blocks].flatten(0, 1)[:length].float()
            context_k = context_k.repeat_interleave(group, dim=1).transpose(0, 1)
            context_v = context_v.repeat_interleave(group, dim=1).transpose(0, 1)
            query = q[first : first + count].float().transpose(0, 1)
            # The rows are the last `count` positions: row j sees keys 0 .. length - count + j.
            visible = (
                torch.arange(length, device=q.device)[None, :]
                <= (torch.arange(length - count, length, device=q.device)[:, None])
            )
            result = functional.scaled_dot_product_attention(
                query, context_k, context_v, attn_mask=visible, scale=scale
            )
            out[first : first + count] = result.transpose(0, 1)
        first += count
