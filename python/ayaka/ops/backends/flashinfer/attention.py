"""flashinfer attention over the paged cache; implements :class:`ayaka.ops.attention.AttentionOps`.

A step is planned in two halves. :func:`plan` (on the host, once per step) checks
the step and builds what every backend shares, by the rules of the reference
backend (positions, slots, lengths), plus the CSR form of the block table that
flashinfer reads (``indptr``, ``indices``, ``last_page_len``). flashinfer's own
plan also needs the head counts, the head width, the dtype and the scale, which
only the first :func:`attend` of the step knows; that call plans the wrapper and
every later layer reuses it. Neither half reads the device.

Two wrappers do the work. A decode step (one row per request) runs
``BatchDecodeWithPagedKVCacheWrapper`` with tensor cores; any other step runs
``BatchPrefillWithPagedKVCacheWrapper`` with the causal mask aligned to the end
of each request's context, which is :mod:`ayaka.ops.attention`'s row convention.
Captured decode steps use a decode wrapper of their own per bucket, built over
fixed index buffers (``use_cuda_graph=True``), and :func:`update_graph_metadata`
plans it before each replay.

A padding request (``kv_length`` 0) is given a context of its rows' length in
block 0, the padding block, so the kernels read valid memory; its rows are then
set to zero, as the contract says. flashinfer computes in fp16 or bf16 and
compiles for head widths of 64, 128 and 256; anything else is a
:class:`~ayaka.core.errors.BackendError`.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, Final, cast

import torch

from ayaka.core.errors import BackendError
from ayaka.core.types import ForwardMode, KVLayoutKind
from ayaka.ops.backends.flashinfer import HEAD_DIMS
from ayaka.ops.backends.torch import attention as reference
from ayaka.ops.backends.torch._checks import capturing, require_like
from ayaka.ops.backends.torch.kvcache import require_buffers
from ayaka.utils.import_utils import require_module

if TYPE_CHECKING:
    from ayaka.ops.attention import AttentionMetadata
    from ayaka.ops.kvcache import KVBuffers

__all__ = [
    "WORKSPACE_BYTES",
    "FlashInferPlan",
    "attend",
    "graph_metadata",
    "plan",
    "update_graph_metadata",
]

#: Head widths flashinfer compiles attention for.
#: The float scratch buffer every wrapper on one device shares; flashinfer's recommended size.
WORKSPACE_BYTES: Final = 128 * 2**20
_DTYPES: Final = (torch.float16, torch.bfloat16)
_LAYOUT_NAMES: Final = {KVLayoutKind.NHD: "NHD", KVLayoutKind.HND: "HND"}

_workspaces: dict[torch.device, torch.Tensor] = {}
_eager_wrappers: dict[tuple[torch.device, str, bool], Any] = {}

type _PlanKey = tuple[int, int, int, torch.dtype, KVLayoutKind, float]


@dataclass(eq=False, slots=True)
class FlashInferPlan:
    """A step's block table in CSR form, and the wrapper planned for it.

    The host tensors are int32 CPU tensors. ``planned_for`` is the
    ``(heads, kv_heads, head_dim, dtype, layout, scale)`` the wrapper was last
    planned with; ``None`` until the first :func:`attend`.

    Attributes:
        decode: One row per request: the decode wrapper runs the step.
        block_size: Positions per block of the cache the step was planned for.
        qo_indptr: ``[B + 1]`` first row of each request (prefill only).
        kv_indptr: ``[B + 1]`` first entry of each request in ``kv_indices``.
        kv_indices: The blocks of every request, concatenated.
        last_page_len: ``[B]`` positions used in each request's last block.
        real_rows: ``[N]`` bool, on the device: the rows of real requests.
        graph: The fixed buffers of a captured bucket, or ``None`` for an eager step.
    """

    decode: bool
    block_size: int
    qo_indptr: torch.Tensor
    kv_indptr: torch.Tensor
    kv_indices: torch.Tensor
    last_page_len: torch.Tensor
    real_rows: torch.Tensor
    graph: _GraphBuffers | None = None
    wrapper: Any = None
    planned_for: _PlanKey | None = None


@dataclass(eq=False, slots=True)
class _GraphBuffers:
    """What a captured bucket keeps: the reference staging and flashinfer's fixed buffers.

    flashinfer's ``plan`` copies a host ``indices`` tensor into its buffer with a blocking copy,
    so the indices go to the device first, asynchronously, from pinned memory
    (``indices_host`` to ``indices_device``); ``indptr`` and ``last_page_len`` stay on the host,
    pinned, because ``plan`` reads them there.
    """

    staging: reference.TorchGraphPlan
    indptr: torch.Tensor
    indices: torch.Tensor
    last_page_len: torch.Tensor
    indptr_host: torch.Tensor
    indices_host: torch.Tensor
    indices_device: torch.Tensor
    last_host: torch.Tensor
    real_host: torch.Tensor
    wrappers: dict[KVLayoutKind, Any]


def _flashinfer() -> Any:
    """The library, imported when a wrapper is first made (``load`` has checked it is there)."""
    return require_module(
        "flashinfer",
        capability="ops.flashinfer",
        remedy="install it on Linux or WSL2 with `uv sync --extra cu130 --extra flashinfer`",
    )


def _workspace(device: torch.device) -> torch.Tensor:
    found = _workspaces.get(device)
    if found is None:
        # Never an inference tensor: the buffer outlives the step that made it, and flashinfer
        # writes it in place whether or not the next caller is in inference mode.
        with torch.inference_mode(False):
            found = torch.empty(WORKSPACE_BYTES, dtype=torch.uint8, device=device)
        _workspaces[device] = found
    return found


def _eager_wrapper(device: torch.device, layout: KVLayoutKind, decode: bool) -> Any:
    key = (device, _LAYOUT_NAMES[layout], decode)
    wrapper = _eager_wrappers.get(key)
    if wrapper is None:
        with torch.inference_mode(False):
            if decode:
                wrapper = _flashinfer().BatchDecodeWithPagedKVCacheWrapper(
                    _workspace(device), kv_layout=key[1], use_tensor_cores=True
                )
            else:
                wrapper = _flashinfer().BatchPrefillWithPagedKVCacheWrapper(
                    _workspace(device), kv_layout=key[1]
                )
        _eager_wrappers[key] = wrapper
    return wrapper


def _csr(
    rows: list[int], lengths: list[int], table: torch.Tensor, block_size: int
) -> tuple[list[int], list[int], list[int], list[int]]:
    """``qo_indptr``, ``kv_indptr``, ``kv_indices``, ``last_page_len`` of a step, on the host.

    A padding request reads ``max(rows, 1)`` positions of block 0.
    """
    qo_indptr: list[int] = [0]
    kv_indptr: list[int] = [0]
    indices: list[int] = []
    last: list[int] = []
    for index, (count, length) in enumerate(zip(rows, lengths, strict=True)):
        qo_indptr.append(qo_indptr[-1] + count)
        if length:
            pages = -(-length // block_size)
            indices.extend(_host_ints(table[index, :pages]))
        else:
            length = max(count, 1)
            pages = -(-length // block_size)
            indices.extend([0] * pages)
        kv_indptr.append(kv_indptr[-1] + pages)
        last.append(length - (pages - 1) * block_size)
    return qo_indptr, kv_indptr, indices, last


def _ints(values: list[int]) -> torch.Tensor:
    return torch.tensor(values, dtype=torch.int32)


def plan(
    mode: ForwardMode,
    rows_per_request: torch.Tensor,
    kv_lengths: torch.Tensor,
    block_rows: torch.Tensor,
    block_size: int,
) -> AttentionMetadata:
    base = reference.plan(mode, rows_per_request, kv_lengths, block_rows, block_size)
    host = base.plan
    if not isinstance(host, reference.TorchAttentionPlan):  # what the reference plans
        raise BackendError("attention.plan: the reference plan changed shape")
    table = block_rows.to("cpu", torch.int32)
    qo_indptr, kv_indptr, indices, last = _csr(
        list(host.rows), list(host.kv_lengths), table, block_size
    )
    real_rows = base.kv_lengths.long().index_select(0, base.request_of_row.long()) > 0
    holder = FlashInferPlan(
        decode=mode is ForwardMode.DECODE,
        block_size=block_size,
        qo_indptr=_ints(qo_indptr),
        kv_indptr=_ints(kv_indptr),
        kv_indices=_ints(indices),
        last_page_len=_ints(last),
        real_rows=real_rows,
    )
    return replace(base, plan=holder)


def _plan_wrapper(holder: FlashInferPlan, key: _PlanKey, device: torch.device) -> None:
    # What plan() allocates is kept for later run() calls, in or out of inference mode, so it must
    # not be an inference tensor (see _workspace).
    with torch.inference_mode(False):
        _plan(holder, key, device)


def _plan(holder: FlashInferPlan, key: _PlanKey, device: torch.device) -> None:
    heads, kv_heads, head_dim, dtype, layout, scale = key
    if holder.graph is not None:
        wrapper = holder.graph.wrappers.get(layout)
        if wrapper is None:
            wrapper = _flashinfer().BatchDecodeWithPagedKVCacheWrapper(
                _workspace(device),
                kv_layout=_LAYOUT_NAMES[layout],
                use_cuda_graph=True,
                use_tensor_cores=True,
                paged_kv_indptr_buffer=holder.graph.indptr,
                paged_kv_indices_buffer=holder.graph.indices,
                paged_kv_last_page_len_buffer=holder.graph.last_page_len,
            )
            holder.graph.wrappers[layout] = wrapper
    else:
        wrapper = _eager_wrapper(device, layout, holder.decode)
    block_size = holder.block_size
    if holder.decode:
        wrapper.plan(
            holder.kv_indptr,
            holder.kv_indices,
            holder.last_page_len,
            heads,
            kv_heads,
            head_dim,
            block_size,
            q_data_type=dtype,
            kv_data_type=dtype,
            sm_scale=scale,
        )
    else:
        wrapper.plan(
            holder.qo_indptr,
            holder.kv_indptr,
            holder.kv_indices,
            holder.last_page_len,
            heads,
            kv_heads,
            head_dim,
            block_size,
            causal=True,
            sm_scale=scale,
            q_data_type=dtype,
            kv_data_type=dtype,
        )
    holder.wrapper = wrapper
    holder.planned_for = key


def attend(
    q: torch.Tensor,
    buffers: KVBuffers,
    meta: AttentionMetadata,
    out: torch.Tensor,
    *,
    scale: float,
) -> None:
    op = "attention.attend"
    holder = meta.plan
    if not isinstance(holder, FlashInferPlan):
        raise BackendError(
            f"{op}: this step was planned by another backend; plan it with the same ops",
            context={"op": op, "argument": "meta"},
        )
    require_buffers(op, buffers)
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
    if q.dtype not in _DTYPES:
        raise BackendError(
            f"{op}: flashinfer computes in fp16 or bf16, not {q.dtype}",
            context={"op": op, "argument": "q"},
        )
    if buffers.head_dim not in HEAD_DIMS:
        raise BackendError(
            f"{op}: flashinfer has kernels for heads of {sorted(HEAD_DIMS)}, "
            f"not {buffers.head_dim}",
            context={"op": op, "argument": "q"},
        )
    if heads == 0 or heads % buffers.num_kv_heads:
        raise BackendError(
            f"{op}: q has {heads} heads, which is not a positive multiple of the cache's "
            f"{buffers.num_kv_heads} key/value heads",
            context={"op": op, "argument": "q"},
        )
    if q.shape[0] == 0:
        return
    key: _PlanKey = (heads, buffers.num_kv_heads, buffers.head_dim, q.dtype, buffers.layout, scale)
    if holder.planned_for != key:
        if capturing(q):
            raise BackendError(
                f"{op}: the step must be planned for these heads before a CUDA graph capture; "
                "run it once eagerly first",
                context={"op": op},
            )
        _plan_wrapper(holder, key, q.device)
    holder.wrapper.run(q, (buffers.k, buffers.v), out=out)
    out.masked_fill_(~holder.real_rows[:, None, None], 0.0)


def graph_metadata(
    num_requests: int, max_blocks: int, block_size: int, device: torch.device
) -> AttentionMetadata:
    base = reference.graph_metadata(num_requests, max_blocks, block_size, device)
    staging = base.plan
    if not isinstance(staging, reference.TorchGraphPlan):
        raise BackendError("attention.graph_metadata: the reference staging changed shape")
    target = torch.device(device)
    pinned = target.type == "cuda"
    capacity = num_requests * max_blocks

    def host(size: int, dtype: torch.dtype = torch.int32) -> torch.Tensor:
        return torch.zeros(size, dtype=dtype, pin_memory=pinned)

    buffers = _GraphBuffers(
        staging=staging,
        indptr=torch.zeros(num_requests + 1, dtype=torch.int32, device=target),
        indices=torch.zeros(capacity, dtype=torch.int32, device=target),
        last_page_len=torch.zeros(num_requests, dtype=torch.int32, device=target),
        indptr_host=host(num_requests + 1),
        indices_host=host(capacity),
        indices_device=torch.zeros(capacity, dtype=torch.int32, device=target),
        last_host=host(num_requests),
        real_host=host(num_requests, torch.bool),
        wrappers={},
    )
    lengths = [0] * num_requests
    _, kv_indptr, indices, last = _csr(
        [1] * num_requests, lengths, torch.zeros(num_requests, max_blocks), block_size
    )
    holder = FlashInferPlan(
        decode=True,
        block_size=block_size,
        qo_indptr=_ints(list(range(num_requests + 1))),
        kv_indptr=_ints(kv_indptr),
        kv_indices=_ints(indices),
        last_page_len=_ints(last),
        real_rows=torch.zeros(num_requests, dtype=torch.bool, device=target),
        graph=buffers,
    )
    return replace(base, plan=holder)


def update_graph_metadata(
    meta: AttentionMetadata, kv_lengths: torch.Tensor, block_rows: torch.Tensor
) -> None:
    op = "attention.update_graph_metadata"
    holder = meta.plan
    if not isinstance(holder, FlashInferPlan) or holder.graph is None:
        raise BackendError(
            f"{op}: meta was not made by this backend's graph_metadata",
            context={"op": op, "argument": "meta"},
        )
    graph = holder.graph
    reference.stage_graph_step(op, meta, graph.staging, kv_lengths, block_rows)
    lengths = _host_ints(kv_lengths)
    _, kv_indptr, indices, last = _csr([1] * len(lengths), lengths, block_rows, meta.block_size)
    used = len(indices)
    graph.indptr_host.copy_(_ints(kv_indptr))
    graph.last_host.copy_(_ints(last))
    graph.indices_host[:used].copy_(_ints(indices))
    graph.indices_device[:used].copy_(graph.indices_host[:used], non_blocking=True)
    holder.kv_indptr = graph.indptr_host
    holder.kv_indices = graph.indices_device[:used]
    holder.last_page_len = graph.last_host
    graph.real_host.copy_(kv_lengths > 0)
    holder.real_rows.copy_(graph.real_host, non_blocking=True)
    if holder.planned_for is not None:
        _plan_wrapper(holder, holder.planned_for, holder.real_rows.device)


def _host_ints(tensor: torch.Tensor) -> list[int]:
    # torch types ``tolist`` as ``list[Unknown]``.
    return cast("list[int]", tensor.tolist())  # pyright: ignore[reportUnknownMemberType]
