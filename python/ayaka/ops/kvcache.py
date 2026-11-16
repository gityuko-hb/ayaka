"""KV-cache ops: the paged layout and the contract a backend implements to write it.

This module is the one place that defines how keys and values are laid out in
memory. ``ayaka.kvcache`` allocates tensors to this layout and hands them out as
:class:`KVBuffers`; the attention layer and every backend read and write them
through the ops below. Nothing else describes the layout.

The paged layout
----------------

The cache of one group of layers is a pool of ``num_blocks`` blocks. A block
holds the keys and values of ``block_size`` consecutive positions of one
request, for one layer. A position is stored in a *slot*::

    slot = block_id * block_size + offset          # offset in [0, block_size)

* ``k`` and ``v`` are separate, contiguous tensors with the same shape and
  dtype. For ``KVLayoutKind.NHD`` the shape is
  ``[num_blocks, block_size, kv_heads, head_dim]``; for ``KVLayoutKind.HND`` it
  is ``[num_blocks, kv_heads, block_size, head_dim]``. The layout is chosen
  once per engine, by asking :meth:`KVCacheOps.layout`.
* A request owns a *block table*: ``block_table[i]`` is the block that holds
  its positions ``[i * block_size, (i + 1) * block_size)``. A request of
  ``length`` tokens uses the first ``ceil(length / block_size)`` entries. The
  entries after them are padding and are never read.
* **Block 0 is the padding block.** The allocator never assigns it to a
  request, so a block-table entry of ``0`` means "no block". Rows that do not
  belong to a real request (a batch padded to a fixed size) are stored into
  slot ``0``, which makes the write harmless without a branch. The contents of
  block 0 are unspecified and are never read as context.
* The *slot mapping* is an int64 vector with one slot per query row of a step.
  :meth:`~ayaka.ops.attention.AttentionOps.plan` builds it from the block
  table and the row positions; :meth:`KVCacheOps.store` writes row ``i`` to
  ``slot_mapping[i]``.
* ``codec`` names how values are stored. Only ``"raw"`` exists: the tensors
  hold the compute dtype of the model (fp16, bf16 or fp32) unchanged.

Importing this module does not import torch.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple, Protocol

from ayaka.core.types import KVLayoutKind

if TYPE_CHECKING:
    from torch import Tensor

    from ayaka.ops.attention import AttentionMetadata

__all__ = ["RAW_CODEC", "KVBuffers", "KVCacheOps"]

#: The only codec: keys and values stored in the compute dtype, unchanged.
RAW_CODEC = "raw"


class KVBuffers(NamedTuple):
    """The key and value tensors of one layer: all that a layer or a backend sees of the cache.

    Attributes:
        k: Keys, laid out as described in this module's docstring.
        v: Values, with the same shape, dtype and layout as ``k``.
        layout: ``KVLayoutKind.NHD`` or ``KVLayoutKind.HND``.
        codec: How the values are encoded; ``RAW_CODEC`` is the only one.
    """

    k: Tensor
    v: Tensor
    layout: KVLayoutKind
    codec: str

    @property
    def num_blocks(self) -> int:
        """Blocks in the pool, block 0 included."""
        return int(self.k.shape[0])

    @property
    def block_size(self) -> int:
        """Positions per block."""
        return int(self.k.shape[1 if self.layout is KVLayoutKind.NHD else 2])

    @property
    def num_kv_heads(self) -> int:
        """Key/value heads per position."""
        return int(self.k.shape[2 if self.layout is KVLayoutKind.NHD else 1])

    @property
    def head_dim(self) -> int:
        """Elements per head."""
        return int(self.k.shape[3])


class KVCacheOps(Protocol):
    """Writing, reading and clearing the paged cache.

    Block ids are int32 or int64 tensors on the device of the buffers. The ops
    do not check that ids are in range: on a GPU the check would force a
    device-to-host synchronisation, and the block manager hands out only ids it
    owns.
    """

    def layout(self) -> KVLayoutKind:
        """Return the layout this backend reads fastest, for the pool to allocate in.

        Backends that can serve both layouts return their preference; a
        backend that serves only one returns that one and refuses the other.
        """
        ...

    def store(self, k: Tensor, v: Tensor, buffers: KVBuffers, meta: AttentionMetadata) -> None:
        """Write the keys and values of this step's query rows into the cache.

        ``k`` and ``v`` are contiguous ``[rows, kv_heads, head_dim]`` tensors in
        the dtype of ``buffers``; row ``i`` goes to slot ``meta.slot_mapping[i]``. Slots of
        real rows are distinct. Padding rows all land in slot 0. Only those
        slots change.

        A layer calls this **before**
        :meth:`~ayaka.ops.attention.AttentionOps.attend`, so that each row sees
        its own keys and values and those of the rows before it.
        """
        ...

    def gather(self, buffers: KVBuffers, block_ids: Tensor, out_k: Tensor, out_v: Tensor) -> None:
        """Copy whole blocks out of the cache, in the order of ``block_ids``.

        ``out_k`` and ``out_v`` have the shape of ``buffers.k`` with the first
        dimension replaced by ``len(block_ids)``, and its dtype. They must not
        overlap the cache.
        """
        ...

    def copy_blocks(self, buffers: KVBuffers, src: Tensor, dst: Tensor) -> None:
        """Copy block ``src[i]`` into block ``dst[i]``, for keys and values.

        ``src`` and ``dst`` have the same length. The ids in ``dst`` are
        distinct and none of them appears in ``src``.
        """
        ...

    def zero_blocks(self, buffers: KVBuffers, block_ids: Tensor) -> None:
        """Set every element of the listed blocks to zero, for keys and values."""
        ...
