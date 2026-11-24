"""Reference attention ops; implements :class:`ayaka.ops.attention.AttentionOps`."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Final, cast

import torch
from torch.nn import functional

from ayaka.core.errors import BackendError
from ayaka.core.types import ForwardMode
from ayaka.ops.attention import AttentionMetadata
from ayaka.ops.backends.torch._checks import (
    require_compute_dtype,
    require_index_tensor,
    require_like,
)
from ayaka.ops.backends.torch.kvcache import dequantize_kv, nhd_view, require_buffers
from ayaka.ops.kvcache import RAW_CODEC

if TYPE_CHECKING:
    from ayaka.ops.kvcache import KVBuffers

__all__ = [
    "DECODE_CHUNK_BYTES",
    "TorchAttentionPlan",
    "TorchGraphPlan",
    "TorchVerifyGraphPlan",
    "attend",
    "graph_metadata",
    "plan",
    "stage_graph_step",
    "stage_verify_graph_step",
    "update_graph_metadata",
    "update_verify_graph_metadata",
    "verify_graph_metadata",
]

#: Bytes of float32 keys a decode step gathers at a time.
DECODE_CHUNK_BYTES: Final = 64 * 2**20


@dataclass(frozen=True, slots=True)
class TorchAttentionPlan:
    """The host-side lists ``attend`` loops over, read once by :func:`plan`."""

    rows: tuple[int, ...]
    kv_lengths: tuple[int, ...]


@dataclass(frozen=True, slots=True, eq=False)
class TorchGraphPlan:
    """Host staging of the graph metadata: what :func:`update_graph_metadata` fills."""

    positions: torch.Tensor
    slot_mapping: torch.Tensor
    kv_lengths: torch.Tensor
    block_table: torch.Tensor


@dataclass(frozen=True, slots=True, eq=False)
class TorchVerifyGraphPlan:
    """Host staging of a captured ``VERIFY`` bucket of ``num_rows`` rows across ``max_requests``."""

    rows_per_request: torch.Tensor
    request_of_row: torch.Tensor
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
    window: int | None = None,
    sinks: torch.Tensor | None = None,
) -> None:
    op = "attention.attend"
    require_buffers(op, buffers)
    if not isinstance(
        meta.plan, (TorchAttentionPlan, TorchGraphPlan, TorchVerifyGraphPlan)
    ) or not isinstance(meta.block_table, torch.Tensor):
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
    if buffers.codec == RAW_CODEC:
        require_like(
            op, "q", q, (meta.request_of_row.shape[0], heads, buffers.head_dim), buffers.k.dtype
        )
    else:
        require_compute_dtype(op, "q", q)
        require_like(op, "q", q, (meta.request_of_row.shape[0], heads, buffers.head_dim), q.dtype)
    require_like(op, "out", out, q.shape, q.dtype)
    if heads == 0 or heads % buffers.num_kv_heads:
        raise BackendError(
            f"{op}: q has {heads} heads, which is not a positive multiple of the cache's "
            f"{buffers.num_kv_heads} key/value heads",
            context={"op": op, "argument": "q"},
        )
    if window is not None and window < 1:
        raise BackendError(
            f"{op}: window must be positive when set, got {window}",
            context={"op": op, "argument": "window"},
        )
    if sinks is not None:
        require_like(op, "sinks", sinks, (heads,), sinks.dtype)

    keys, values = nhd_view(buffers.k, buffers), nhd_view(buffers.v, buffers)
    if meta.mode is ForwardMode.DECODE:
        _attend_decode(
            q,
            keys,
            values,
            buffers,
            meta.block_table,
            meta.kv_lengths,
            out,
            scale=scale,
            window=window,
            sinks=sinks,
        )
        return
    if isinstance(meta.plan, TorchVerifyGraphPlan):
        _attend_verify(
            q,
            keys,
            values,
            buffers,
            meta.block_table,
            meta.kv_lengths,
            meta.request_of_row,
            meta.positions,
            out,
            scale=scale,
            window=window,
            sinks=sinks,
        )
        return
    if not isinstance(meta.plan, TorchAttentionPlan):  # graph metadata is always DECODE
        raise BackendError(f"{op}: graph metadata must be a DECODE step", context={"op": op})
    group = heads // buffers.num_kv_heads
    out.zero_()
    first = 0
    for request, (count, length) in enumerate(
        zip(meta.plan.rows, meta.plan.kv_lengths, strict=True)
    ):
        if length and count:
            blocks = meta.block_table[request, : -(-length // buffers.block_size)].long()
            raw_k = keys[blocks].flatten(0, 1)[:length]
            raw_v = values[blocks].flatten(0, 1)[:length]
            context_k, context_v = dequantize_kv(raw_k, raw_v, buffers)
            context_k = context_k.repeat_interleave(group, dim=1).transpose(0, 1)
            context_v = context_v.repeat_interleave(group, dim=1).transpose(0, 1)
            query = q[first : first + count].float().transpose(0, 1)
            k_pos = torch.arange(length, device=q.device)[None, :]
            q_pos = torch.arange(length - count, length, device=q.device)[:, None]
            visible = k_pos <= q_pos
            if window is not None:
                visible = visible & (k_pos >= (q_pos - window + 1))
            if sinks is None:
                result = functional.scaled_dot_product_attention(
                    query, context_k, context_v, attn_mask=visible, scale=scale
                )
            else:
                scores = torch.matmul(query, context_k.transpose(-2, -1)) * scale
                scores = scores.masked_fill(~visible[None, :, :], float("-inf"))
                sink_logits = sinks.float().view(heads, 1, 1).expand(heads, count, 1)
                weights = torch.softmax(torch.cat([scores, sink_logits], dim=-1), dim=-1)[..., :-1]
                result = torch.matmul(weights, context_v)
            out[first : first + count] = result.transpose(0, 1)
        first += count


def _attend_decode(
    q: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
    buffers: KVBuffers,
    table: torch.Tensor,
    lengths: torch.Tensor,
    out: torch.Tensor,
    *,
    scale: float,
    window: int | None = None,
    sinks: torch.Tensor | None = None,
) -> None:
    """One row per request, every request's whole table; CUDA-graph safe."""
    requests, heads, head_dim = q.shape
    block_size, kv_heads = keys.shape[1], keys.shape[2]
    group = heads // kv_heads
    width = table.shape[1]
    per_block = requests * block_size * kv_heads * head_dim * 4
    step = max(1, min(width, DECODE_CHUNK_BYTES // max(per_block, 1)))
    query = q.float().reshape(requests, kv_heads, group, head_dim)
    limit = lengths.long()[:, None]
    shape = (requests, kv_heads, group)
    running_max = torch.full(shape, float("-inf"), dtype=torch.float32, device=q.device)
    total = torch.zeros(shape, dtype=torch.float32, device=q.device)
    acc = torch.zeros((*shape, head_dim), dtype=torch.float32, device=q.device)
    for first in range(0, width, step):
        blocks = table[:, first : first + step].long()
        context_k, context_v = dequantize_kv(
            keys[blocks].flatten(1, 2), values[blocks].flatten(1, 2), buffers
        )
        start = first * block_size
        positions = torch.arange(start, start + context_k.shape[1], device=q.device)
        visible = positions[None, :] < limit
        if window is not None:
            visible = visible & (positions[None, :] >= (limit - window))
        scores = torch.einsum("bkgd,blkd->bkgl", query, context_k) * scale
        scores = scores.masked_fill(~visible[:, None, None, :], float("-inf"))
        new_max = torch.maximum(running_max, scores.amax(dim=-1))
        shift = torch.where(torch.isfinite(new_max), new_max, 0.0)
        correction = torch.exp(running_max - shift)
        weights = torch.exp(scores - shift[..., None])
        context_v = torch.where(visible[:, :, None, None], context_v, 0.0)
        acc = acc * correction[..., None] + torch.einsum("bkgl,blkd->bkgd", weights, context_v)
        total = total * correction + weights.sum(dim=-1)
        running_max = new_max
    if sinks is not None:
        sink_logits = sinks.float().reshape(1, kv_heads, group)
        final_max = torch.maximum(running_max, sink_logits)
        shift = torch.where(torch.isfinite(final_max), final_max, 0.0)
        correction = torch.exp(running_max - shift)
        acc = acc * correction[..., None]
        total = total * correction + torch.exp(sink_logits - shift)
    real = (lengths > 0)[:, None, None]
    seen = (total > 0) & real
    result = torch.where(seen[..., None], acc / torch.where(seen, total, 1.0)[..., None], 0.0)
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
    """Check a decode step given on the host and copy its positions, slots, lengths and table."""
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


def _attend_verify(
    q: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
    buffers: KVBuffers,
    table: torch.Tensor,
    lengths: torch.Tensor,
    request_of_row: torch.Tensor,
    row_positions: torch.Tensor,
    out: torch.Tensor,
    *,
    scale: float,
    window: int | None = None,
    sinks: torch.Tensor | None = None,
) -> None:
    """Graph-safe causal attention for a ``VERIFY`` bucket of fixed total rows.

    Masks, the sliding window, attention sinks and the cache's dtype are applied as
    :func:`_attend_decode` applies them, per row instead of per request.
    """
    num_rows, heads, head_dim = q.shape
    block_size, kv_heads = keys.shape[1], keys.shape[2]
    group = heads // kv_heads
    _, width = table.shape
    req_idx = request_of_row.long()
    row_blocks = table[req_idx]  # [N, width]
    row_kv_len = lengths.long()[req_idx]  # [N]
    # A row sees positions <= min(row_positions[r], row_kv_len[r] - 1) when row_kv_len > 0.
    max_pos = torch.where(row_kv_len > 0, torch.minimum(row_positions.long(), row_kv_len - 1), -1)

    per_block = num_rows * block_size * kv_heads * head_dim * 4
    step = max(1, min(width, DECODE_CHUNK_BYTES // max(per_block, 1)))
    query = q.float().reshape(num_rows, kv_heads, group, head_dim)
    shape = (num_rows, kv_heads, group)
    running_max = torch.full(shape, float("-inf"), dtype=torch.float32, device=q.device)
    total = torch.zeros(shape, dtype=torch.float32, device=q.device)
    acc = torch.zeros((*shape, head_dim), dtype=torch.float32, device=q.device)
    for first in range(0, width, step):
        blocks = row_blocks[:, first : first + step].long()
        context_k, context_v = dequantize_kv(
            keys[blocks].flatten(1, 2), values[blocks].flatten(1, 2), buffers
        )
        start = first * block_size
        positions = torch.arange(start, start + context_k.shape[1], device=q.device)
        visible = positions[None, :] <= max_pos[:, None]
        if window is not None:
            visible = visible & (positions[None, :] > (max_pos[:, None] - window))
        scores = torch.einsum("bkgd,blkd->bkgl", query, context_k) * scale
        scores = scores.masked_fill(~visible[:, None, None, :], float("-inf"))
        new_max = torch.maximum(running_max, scores.amax(dim=-1))
        shift = torch.where(torch.isfinite(new_max), new_max, 0.0)
        correction = torch.exp(running_max - shift)
        weights = torch.exp(scores - shift[..., None])
        context_v = torch.where(visible[:, :, None, None], context_v, 0.0)
        acc = acc * correction[..., None] + torch.einsum("bkgl,blkd->bkgd", weights, context_v)
        total = total * correction + weights.sum(dim=-1)
        running_max = new_max
    if sinks is not None:
        sink_logits = sinks.float().reshape(1, kv_heads, group)
        final_max = torch.maximum(running_max, sink_logits)
        shift = torch.where(torch.isfinite(final_max), final_max, 0.0)
        correction = torch.exp(running_max - shift)
        acc = acc * correction[..., None]
        total = total * correction + torch.exp(sink_logits - shift)
    seen = (total > 0) & (max_pos >= 0)[:, None, None]
    result = torch.where(seen[..., None], acc / torch.where(seen, total, 1.0)[..., None], 0.0)
    out.copy_(result.reshape(num_rows, heads, head_dim))


def verify_graph_metadata(
    num_rows: int, max_requests: int, max_blocks: int, block_size: int, device: torch.device
) -> AttentionMetadata:
    if num_rows < 1 or max_requests < 1 or max_blocks < 1 or block_size < 1:
        raise BackendError(
            "attention.verify_graph_metadata: dimensions must be positive",
            context={"op": "attention.verify_graph_metadata"},
        )
    target = torch.device(device)
    pinned = target.type == "cuda"

    def host(shape: tuple[int, ...], dtype: torch.dtype) -> torch.Tensor:
        return torch.zeros(shape, dtype=dtype, pin_memory=pinned)

    staging = TorchVerifyGraphPlan(
        rows_per_request=host((max_requests,), torch.int32),
        request_of_row=host((num_rows,), torch.int32),
        positions=host((num_rows,), torch.int32),
        slot_mapping=host((num_rows,), torch.int64),
        kv_lengths=host((max_requests,), torch.int32),
        block_table=host((max_requests, max_blocks), torch.int32),
    )
    return AttentionMetadata(
        mode=ForwardMode.VERIFY,
        block_size=block_size,
        rows_per_request=torch.zeros(max_requests, dtype=torch.int32, device=target),
        request_of_row=torch.zeros(num_rows, dtype=torch.int32, device=target),
        positions=torch.zeros(num_rows, dtype=torch.int32, device=target),
        slot_mapping=torch.zeros(num_rows, dtype=torch.int64, device=target),
        kv_lengths=torch.zeros(max_requests, dtype=torch.int32, device=target),
        block_table=torch.zeros((max_requests, max_blocks), dtype=torch.int32, device=target),
        plan=staging,
    )


def update_verify_graph_metadata(
    meta: AttentionMetadata,
    rows_per_request: torch.Tensor,
    kv_lengths: torch.Tensor,
    block_rows: torch.Tensor,
) -> None:
    op = "attention.update_verify_graph_metadata"
    staging = meta.plan
    if not isinstance(staging, TorchVerifyGraphPlan) or not isinstance(
        meta.block_table, torch.Tensor
    ):
        raise BackendError(
            f"{op}: meta was not made by this backend's verify_graph_metadata",
            context={"op": op, "argument": "meta"},
        )
    stage_verify_graph_step(op, meta, staging, rows_per_request, kv_lengths, block_rows)


def stage_verify_graph_step(
    op: str,
    meta: AttentionMetadata,
    staging: TorchVerifyGraphPlan,
    rows_per_request: torch.Tensor,
    kv_lengths: torch.Tensor,
    block_rows: torch.Tensor,
) -> None:
    """Check a ``VERIFY`` step given on the host and asynchronously copy it to device."""
    table = meta.block_table
    if not isinstance(table, torch.Tensor):
        raise BackendError(
            f"{op}: meta has no block table tensor", context={"op": op, "argument": "meta"}
        )
    max_requests, max_blocks = staging.block_table.shape
    num_rows = staging.positions.shape[0]
    for name, tensor, shape in (
        ("rows_per_request", rows_per_request, (max_requests,)),
        ("kv_lengths", kv_lengths, (max_requests,)),
        ("block_rows", block_rows, (max_requests, max_blocks)),
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
    total_rows = int(rows_per_request.sum().item())
    if total_rows != num_rows:
        raise BackendError(
            f"{op}: rows_per_request sums to {total_rows}, expected {num_rows}",
            context={"op": op, "argument": "rows_per_request"},
        )
    block_size = meta.block_size
    rows64 = rows_per_request.long()
    lengths64 = kv_lengths.long()
    if (
        bool((rows64 < 0).any())
        or bool((lengths64 < 0).any())
        or bool((lengths64 > max_blocks * block_size).any())
    ):
        raise BackendError(f"{op}: invalid row counts or kv_lengths", context={"op": op})
    req_of_row = torch.repeat_interleave(torch.arange(max_requests, dtype=torch.int64), rows64)
    first_row = torch.cumsum(rows64, dim=0) - rows64
    within = torch.arange(num_rows, dtype=torch.int64) - first_row[req_of_row]
    row_len = lengths64[req_of_row]
    is_padding = row_len == 0
    positions = torch.where(is_padding, 0, row_len - rows64[req_of_row] + within)
    block = block_rows.long()[req_of_row, positions // block_size]
    slots = torch.where(is_padding, 0, block * block_size + positions % block_size)

    staging.rows_per_request.copy_(rows_per_request)
    staging.request_of_row.copy_(req_of_row.to(torch.int32))
    staging.positions.copy_(positions.to(torch.int32))
    staging.slot_mapping.copy_(slots)
    staging.kv_lengths.copy_(kv_lengths)
    staging.block_table.copy_(block_rows)
    for device_tensor, host_tensor in (
        (meta.rows_per_request, staging.rows_per_request),
        (meta.request_of_row, staging.request_of_row),
        (meta.positions, staging.positions),
        (meta.slot_mapping, staging.slot_mapping),
        (meta.kv_lengths, staging.kv_lengths),
        (table, staging.block_table),
    ):
        device_tensor.copy_(host_tensor, non_blocking=True)
