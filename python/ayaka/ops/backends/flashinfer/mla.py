"""FlashInfer Multi-Head Latent Attention (MLA) ops over the compressed latent cache.

Uses FlashInfer's ``BatchMLAPagedAttentionWrapper`` for canonical MLA geometry
(``kv_lora_rank == 512``, ``qk_rope_head_dim == 64``, page size in
``{1, 16, 32, 64}``, fp16/bf16 raw latent buffers) in both eager prefill/decode
and captured decode CUDA graphs, falling back cleanly to paged reference MLA for
arbitrary test shapes or ``fp8_e4m3`` latent storage.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Final

import torch

from ayaka.core.errors import BackendError
from ayaka.ops.backends.flashinfer.attention import FlashInferPlan, workspace
from ayaka.ops.backends.torch import mla as reference
from ayaka.ops.backends.torch._checks import capturing
from ayaka.ops.kvcache import RAW_CODEC
from ayaka.utils.import_utils import require_module

if TYPE_CHECKING:
    from ayaka.ops.attention import AttentionMetadata
    from ayaka.ops.kvcache import LatentKVBuffers

__all__ = ["attend_mla", "store_latent"]

_SUPPORTED_PAGE_SIZES: Final = frozenset({1, 16, 32, 64})
_eager_mla_wrappers: dict[torch.device, Any] = {}
type _MLAPlanKey = tuple[int, int, int, int, torch.dtype, float]


def _flashinfer() -> Any:
    return require_module(
        "flashinfer",
        capability="ops.flashinfer",
        remedy="install it on Linux or WSL2 with `uv sync --extra cu130 --extra flashinfer`",
    )


def _eager_mla_wrapper(device: torch.device) -> Any:
    wrapper = _eager_mla_wrappers.get(device)
    if wrapper is None:
        with torch.inference_mode(False):
            wrapper = _flashinfer().BatchMLAPagedAttentionWrapper(workspace(device), backend="auto")
        _eager_mla_wrappers[device] = wrapper
    return wrapper


def store_latent(
    ckv: torch.Tensor,
    kpe: torch.Tensor,
    buffers: LatentKVBuffers,
    meta: AttentionMetadata,
) -> None:
    reference.store_latent(ckv, kpe, buffers, meta)


def _supports_flashinfer_mla(
    q_nope: torch.Tensor, buffers: LatentKVBuffers, block_size: int
) -> bool:
    return (
        q_nope.is_cuda
        and buffers.codec == RAW_CODEC
        and q_nope.dtype in (torch.float16, torch.bfloat16)
        and buffers.kv_lora_rank == 512
        and buffers.qk_rope_head_dim == 64
        and block_size in _SUPPORTED_PAGE_SIZES
    )


def _plan_mla(
    holder: FlashInferPlan,
    key: _MLAPlanKey,
    device: torch.device,
) -> Any:
    heads, ckv_dim, kpe_dim, page_size, dtype, scale = key
    mla_mod = require_module(
        "flashinfer.mla",
        capability="ops.flashinfer",
        remedy="install flashinfer on Linux/WSL2",
    )
    if holder.graph is not None:
        gkey = (ckv_dim, kpe_dim, page_size)
        wrapper = holder.graph.mla_wrappers.get(gkey)
        if wrapper is None:
            with torch.inference_mode(False):
                wrapper = _flashinfer().BatchMLAPagedAttentionWrapper(
                    workspace(device),
                    use_cuda_graph=True,
                    qo_indptr=holder.graph.mla_qo_indptr,
                    kv_indptr=holder.graph.indptr,
                    kv_indices=holder.graph.indices,
                    kv_len_arr=holder.graph.mla_kv_len,
                    backend="auto",
                )
            holder.graph.mla_wrappers[gkey] = wrapper

        def _replan() -> None:
            _plan_mla(holder, key, device)

        if key not in holder.mla_planned:
            holder.mla_replanners.append(_replan)
    else:
        wrapper = _eager_mla_wrapper(device)

    indices = (
        holder.graph.indices_host[: holder.kv_indices.shape[0]]
        if holder.graph is not None
        else holder.kv_indices
    )
    with torch.inference_mode(False):
        wrapper.plan(
            metadata=mla_mod.MLAPlanMetadata.csr(
                holder.qo_indptr,
                holder.kv_indptr,
                indices,
                holder.mla_kv_len,
            ),
            num_heads=heads,
            head_dim_ckv=ckv_dim,
            head_dim_kpe=kpe_dim,
            page_size=page_size,
            causal=True,
            sm_scale=scale,
            q_data_type=dtype,
            kv_data_type=dtype,
            query_layout="split",
            kv_cache_layout="split",
        )
    holder.mla_planned[key] = wrapper
    return wrapper


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
    holder = meta.plan
    if not isinstance(holder, FlashInferPlan) or not isinstance(meta.block_table, torch.Tensor):
        raise BackendError(
            f"{op}: step metadata was planned by another backend",
            context={"op": op, "argument": "meta"},
        )
    rows = reference.validate_mla_inputs(op, q_nope, q_pe, buffers, meta, out)
    if rows == 0:
        return
    heads = int(q_nope.shape[1])
    if _supports_flashinfer_mla(q_nope, buffers, buffers.block_size):
        key: _MLAPlanKey = (
            heads,
            buffers.kv_lora_rank,
            buffers.qk_rope_head_dim,
            buffers.block_size,
            q_nope.dtype,
            scale,
        )
        wrapper = holder.mla_planned.get(key)
        if wrapper is None:
            if capturing(q_nope):
                raise BackendError(
                    f"{op}: step must be planned before CUDA graph capture; run once eagerly",
                    context={"op": op},
                )
            wrapper = _plan_mla(holder, key, q_nope.device)
        wrapper.run(query=(q_nope, q_pe), kv_cache=(buffers.ckv, buffers.kpe), out=out)
        out.masked_fill_(~holder.real_rows[:, None, None], 0.0)
        return

    reference.attend_mla_paged(
        q_nope,
        q_pe,
        buffers,
        meta.block_table,
        meta.kv_lengths,
        meta.mode,
        holder.host_rows,
        holder.host_kv_lengths,
        out,
        scale=scale,
    )
