"""Reference attention ops; implements :class:`ayaka.ops.attention.AttentionOps`.

Written to be read, not to be fast. For a step with several rows per request,
``attend`` loops over the requests, gathers each one's keys and values out of
the paged cache through its block table and runs
``scaled_dot_product_attention`` with a causal mask. For a decode step, planned
or on graph metadata, it gathers every request's table at once, padded to the
table's width, and masks the positions past each length: no Python value
depends on the step, so it can be captured, and an eager decode step computes
exactly what its replay does. Both gathers copy the
context every call, so the cost grows with the context; do not measure speed
with it.
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

__all__ = [
    "TorchAttentionPlan",
    "TorchGraphPlan",
    "attend",
    "graph_metadata",
    "plan",
    "stage_graph_step",
    "update_graph_metadata",
]


@dataclass(frozen=True, slots=True)
class TorchAttentionPlan:
    """The host-side lists ``attend`` loops over, read once by :func:`plan`."""

    rows: tuple[int, ...]
    kv_lengths: tuple[int, ...]


@dataclass(frozen=True, slots=True, eq=False)
class TorchGraphPlan:
    """Host staging of the graph metadata: what :func:`update_graph_metadata` fills.

    Pinned when the metadata lives on a GPU, so the copies to the device do not wait.
    """

    positions: torch.Tensor
    slot_mapping: torch.Tensor
    kv_lengths: torch.Tensor
    block_table: torch.Tensor


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
    if not isinstance(meta.plan, (TorchAttentionPlan, TorchGraphPlan)) or not isinstance(
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
    # Decode steps, planned or captured, take the one path a graph can replay, so a decode step
    # computes the same thing eagerly and in a graph.
    if meta.mode is ForwardMode.DECODE:
        _attend_decode(q, keys, values, meta.block_table, meta.kv_lengths, out, scale=scale)
        return
    if not isinstance(meta.plan, TorchAttentionPlan):  # graph metadata is always DECODE
        raise BackendError(f"{op}: graph metadata must be a DECODE step", context={"op": op})
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


def _attend_decode(
    q: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
    table: torch.Tensor,
    lengths: torch.Tensor,
    out: torch.Tensor,
    *,
    scale: float,
) -> None:
    """One row per request, every request's whole table at once; nothing depends on the host.

    ``keys`` and ``values`` are ``[blocks, block_size, kv_heads, head_dim]`` views. Positions at
    or past a request's length are masked, so the unused tail of its table (block 0) is never
    read into the result; a padding request (length 0) gets zeros.
    """
    requests, heads, head_dim = q.shape
    kv_heads = keys.shape[2]
    group = heads // kv_heads
    blocks = table.long()
    # [B, max_blocks, block_size, kv_heads, dim] -> [B, context, kv_heads, dim]
    context_k = keys[blocks].flatten(1, 2).float()
    context_v = values[blocks].flatten(1, 2).float()
    query = q.float().reshape(requests, kv_heads, group, head_dim)
    scores = torch.einsum("bkgd,blkd->bkgl", query, context_k) * scale
    width = context_k.shape[1]
    visible = torch.arange(width, device=q.device)[None, :] < lengths.long()[:, None]  # [B, L]
    scores = scores.masked_fill(~visible[:, None, None, :], float("-inf"))
    weights = torch.softmax(scores, dim=-1)
    # A padding request masks every position, so its softmax is NaN; it must give zeros.
    weights = torch.where(visible.any(dim=-1)[:, None, None, None], weights, 0.0)
    # A slot past the length may hold anything, NaN included, and 0 * NaN is NaN.
    context_v = torch.where(visible[:, :, None, None], context_v, 0.0)
    result = torch.einsum("bkgl,blkd->bkgd", weights, context_v)
    out.copy_(result.reshape(requests, heads, head_dim))


def graph_metadata(
    num_requests: int, max_blocks: int, block_size: int, device: torch.device
) -> AttentionMetadata:
    if num_requests < 1 or max_blocks < 1 or block_size < 1:
        raise BackendError(
            "attention.graph_metadata: num_requests, max_blocks and block_size must be "
            f"positive, got {num_requests}, {max_blocks} and {block_size}",
            context={"op": "attention.graph_metadata"},
        )
    target = torch.device(device)
    pinned = target.type == "cuda"

    def host(shape: tuple[int, ...], dtype: torch.dtype) -> torch.Tensor:
        return torch.zeros(shape, dtype=dtype, pin_memory=pinned)

    staging = TorchGraphPlan(
        positions=host((num_requests,), torch.int32),
        slot_mapping=host((num_requests,), torch.int64),
        kv_lengths=host((num_requests,), torch.int32),
        block_table=host((num_requests, max_blocks), torch.int32),
    )
    return AttentionMetadata(
        mode=ForwardMode.DECODE,
        block_size=block_size,
        rows_per_request=torch.ones(num_requests, dtype=torch.int32, device=target),
        request_of_row=torch.arange(num_requests, dtype=torch.int32, device=target),
        positions=torch.zeros(num_requests, dtype=torch.int32, device=target),
        slot_mapping=torch.zeros(num_requests, dtype=torch.int64, device=target),
        kv_lengths=torch.zeros(num_requests, dtype=torch.int32, device=target),
        block_table=torch.zeros((num_requests, max_blocks), dtype=torch.int32, device=target),
        plan=staging,
    )


def update_graph_metadata(
    meta: AttentionMetadata, kv_lengths: torch.Tensor, block_rows: torch.Tensor
) -> None:
    op = "attention.update_graph_metadata"
    staging = meta.plan
    if not isinstance(staging, TorchGraphPlan) or not isinstance(meta.block_table, torch.Tensor):
        raise BackendError(
            f"{op}: meta was not made by this backend's graph_metadata",
            context={"op": op, "argument": "meta"},
        )
    stage_graph_step(op, meta, staging, kv_lengths, block_rows)


def stage_graph_step(
    op: str,
    meta: AttentionMetadata,
    staging: TorchGraphPlan,
    kv_lengths: torch.Tensor,
    block_rows: torch.Tensor,
) -> None:
    """Check a decode step given on the host and copy its positions, slots, lengths and table.

    The device copies are asynchronous, from the pinned ``staging``. Shared with the backends whose
    graph metadata keeps the same fields (flashinfer).
    """
    table = meta.block_table
    if not isinstance(table, torch.Tensor):
        raise BackendError(
            f"{op}: meta has no block table tensor", context={"op": op, "argument": "meta"}
        )
    requests, max_blocks = staging.block_table.shape
    for name, tensor, shape in (
        ("kv_lengths", kv_lengths, (requests,)),
        ("block_rows", block_rows, (requests, max_blocks)),
    ):
        if (
            tuple(tensor.shape) != shape
            or tensor.dtype != torch.int32
            or tensor.device.type != "cpu"
        ):
            raise BackendError(
                f"{op}: {name} must be an int32 CPU tensor of shape {list(shape)}, got "
                f"{tensor.dtype} {list(tensor.shape)} on {tensor.device}",
                context={"op": op, "argument": name},
            )
    block_size = meta.block_size
    lengths = kv_lengths.long()
    if bool((lengths < 0).any()) or bool((lengths > max_blocks * block_size).any()):
        raise BackendError(
            f"{op}: every length must be in [0, {max_blocks * block_size}] for a table of "
            f"{max_blocks} blocks of {block_size}",
            context={"op": op, "argument": "kv_lengths"},
        )
    real = lengths > 0
    positions = torch.where(real, lengths - 1, 0)
    block = block_rows.long().gather(1, (positions // block_size)[:, None])[:, 0]
    staging.positions.copy_(positions)
    staging.slot_mapping.copy_(torch.where(real, block * block_size + positions % block_size, 0))
    staging.kv_lengths.copy_(kv_lengths)
    staging.block_table.copy_(block_rows)
    for device_tensor, host_tensor in (
        (meta.positions, staging.positions),
        (meta.slot_mapping, staging.slot_mapping),
        (meta.kv_lengths, staging.kv_lengths),
        (table, staging.block_table),
    ):
        device_tensor.copy_(host_tensor, non_blocking=True)
