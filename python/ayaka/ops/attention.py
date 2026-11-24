"""Attention ops: the contract a backend implements to attend over the paged cache.

One call, :meth:`AttentionOps.attend`, serves every kind of step (prefill,
decode, verify) and supports optional causal sliding windows and per-head
attention sinks. Importing this module does not import torch.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from torch import Tensor, device

    from ayaka.core.types import ForwardMode
    from ayaka.ops.kvcache import KVBuffers

__all__ = ["AttentionMetadata", "AttentionOps"]


@dataclass(frozen=True, slots=True, eq=False)
class AttentionMetadata:
    """Everything about one step that attention and the cache writes need."""

    mode: ForwardMode
    block_size: int
    rows_per_request: Tensor
    request_of_row: Tensor
    positions: Tensor
    slot_mapping: Tensor
    kv_lengths: Tensor
    block_table: object
    plan: object | None


class AttentionOps(Protocol):
    """Planning a step and attending over the paged cache."""

    def plan(
        self,
        mode: ForwardMode,
        rows_per_request: Tensor,
        kv_lengths: Tensor,
        block_rows: Tensor,
        block_size: int,
    ) -> AttentionMetadata:
        """Build the metadata of one step. Runs on the host, outside any graph."""
        ...

    def attend(
        self,
        q: Tensor,
        buffers: KVBuffers,
        meta: AttentionMetadata,
        out: Tensor,
        *,
        scale: float,
        window: int | None = None,
        sinks: Tensor | None = None,
    ) -> None:
        """Write the attention output of every query row into ``out``.

        Args:
            q: Contiguous ``[N, num_heads, head_dim]`` query tensor in compute dtype.
            buffers: Layer KV cache buffers (raw or ``fp8_e4m3``).
            meta: Planned step metadata.
            out: Output tensor with same shape and dtype as ``q``.
            scale: Attention logit scale factor.
            window: Optional causal sliding-window width in tokens (a query at
                position ``p`` attends only to ``[p - window + 1, p]``), or
                ``None`` for full causal attention.
            sinks: Optional 1D tensor of per-head attention sink logits ``[num_heads]``,
                appended as a virtual sink key before softmax normalization.
        """
        ...

    def graph_metadata(
        self, num_requests: int, max_blocks: int, block_size: int, device: device
    ) -> AttentionMetadata:
        """Allocate fixed metadata buffers for CUDA graph decode buckets."""
        ...

    def update_graph_metadata(
        self, meta: AttentionMetadata, kv_lengths: Tensor, block_rows: Tensor
    ) -> None:
        """Update fixed graph metadata buffers from host tensors."""
        ...

    def verify_graph_metadata(
        self, num_rows: int, max_requests: int, max_blocks: int, block_size: int, device: device
    ) -> AttentionMetadata:
        """Allocate fixed metadata for a captured ``VERIFY`` step of ``num_rows`` total rows."""
        ...

    def update_verify_graph_metadata(
        self,
        meta: AttentionMetadata,
        rows_per_request: Tensor,
        kv_lengths: Tensor,
        block_rows: Tensor,
    ) -> None:
        """Write the next ``VERIFY`` step into ``meta`` made by :meth:`verify_graph_metadata`."""
        ...
