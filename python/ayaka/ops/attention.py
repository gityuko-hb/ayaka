"""Attention ops: the contract a backend implements to attend over the paged KV cache.

One call, :meth:`AttentionOps.attend`, serves every kind of step. A step is a
batch of requests and each request contributes one or more consecutive *query
rows*: a decode step has one row per request, a chunk of prefill has many, and
the window a speculative decoder verifies has ``1 + k``. They are the same
computation, so there is no separate decode op. The paged layout, the block
table and the slot mapping are defined in :mod:`ayaka.ops.kvcache`.

Row conventions
---------------

For request ``b`` of a step:

* ``kv_lengths[b]`` counts the tokens that have keys and values **after this
  step's rows have been stored**, so it includes them.
* The request has ``rows_per_request[b]`` query rows, and they are the *last*
  rows of its sequence: row ``j`` is at position
  ``kv_lengths[b] - rows_per_request[b] + j``. Row ``j`` attends to positions
  ``0`` through its own, inclusive (causal), and to nothing later, including
  the rows after it in the same step.
* Rows of all requests are concatenated in request order: the first
  ``rows_per_request[0]`` rows belong to request 0, and so on. ``N`` is the
  total.

A *padding request* has ``kv_lengths[b] == 0``. It keeps its ``rows_per_request``
rows, so a batch can be padded up to a fixed shape; those rows are stored into
the padding slot and :meth:`AttentionOps.attend` writes zeros for them. A real
request always has ``kv_lengths[b] >= rows_per_request[b] >= 1`` or no rows at
all.

Importing this module does not import torch.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from torch import Tensor

    from ayaka.core.types import ForwardMode
    from ayaka.ops.kvcache import KVBuffers

__all__ = ["AttentionMetadata", "AttentionOps"]


@dataclass(frozen=True, slots=True, eq=False)
class AttentionMetadata:
    """Everything about one step that attention and the cache writes need.

    :meth:`AttentionOps.plan` builds it once per step and every layer of the
    group reuses it. It is a snapshot: nothing in it changes after it is built.

    Attributes:
        mode: How the step uses its rows.
        block_size: Positions per block of the cache this was planned for.
        rows_per_request: ``[B]`` int32, query rows of each request.
        request_of_row: ``[N]`` int32, the request each row belongs to.
        positions: ``[N]`` int32, the position of each row in its sequence
            (0 for padding rows). A model reads this for position embeddings.
        slot_mapping: ``[N]`` int64, the cache slot each row is written to
            (0 for padding rows).
        kv_lengths: ``[B]`` int32, tokens with keys and values after this
            step's rows are stored; 0 marks a padding request.
        block_table: The block table in the form the backend reads. Opaque to
            layers. The reference backend keeps the ``[B, max_blocks]`` table
            it was given, as int32.
        plan: The backend's own planning result, or ``None``. Opaque to layers;
            :meth:`AttentionOps.attend` refuses metadata another backend
            planned.
    """

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
        """Build the metadata of one step. Runs on the host, outside any graph.

        ``rows_per_request`` and ``kv_lengths`` are ``[B]`` int32 or int64
        tensors with the meaning given in this module's docstring.
        ``block_rows`` is the ``[B, max_blocks]`` block table, one row per
        request, with 0 for entries a request does not use. Every tensor is on
        the device the ops will run on. ``ops`` does not know the block
        manager, so the table arrives as plain integers.

        The call may read the tensors on the host. It raises
        :class:`~ayaka.core.errors.BackendError` for an inconsistent step: a
        real request with fewer tokens than rows, a mode that does not fit the
        rows (``DECODE`` is one row per request, ``IDLE`` has none), or a block
        table too short for ``kv_lengths`` or with a 0 entry inside it.
        """
        ...

    def attend(
        self, q: Tensor, buffers: KVBuffers, meta: AttentionMetadata, out: Tensor, *, scale: float
    ) -> None:
        """Write the attention output of every query row into ``out``.

        ``q`` and ``out`` are contiguous ``[N, num_heads, head_dim]`` tensors in
        the compute dtype of ``buffers``; ``out`` must not overlap ``q`` or the cache. With
        ``num_kv_heads`` heads in the cache, ``num_heads`` is a multiple of it
        (grouped-query attention): query head ``h`` reads key/value head
        ``h // (num_heads // num_kv_heads)``. Scores are ``scale * q . k``,
        accumulated in at least fp32, and the result is rounded to the dtype of
        ``out`` once. Rows of padding requests are zero.

        Run it after :meth:`~ayaka.ops.kvcache.KVCacheOps.store` has written
        this step's rows. The cache is only read.
        """
        ...
