"""The shape of a KV cache group and what it costs, before any tensor exists.

A :class:`KVSpec` is geometry only: which layers share it, how many heads, how
wide, how many tokens per block, in what dtype. Everything the engine must know
before it allocates, such as bytes per block, follows from it by arithmetic, so
planning memory never needs a tensor or torch. :class:`~ayaka.kvcache.storage.pool.KVPool`
then allocates tensors to exactly this geometry.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, cast

from ayaka.core.types import KVLayoutKind
from ayaka.ops.kvcache import RAW_CODEC
from ayaka.utils.dtypes import COMPUTE_DTYPES, DType
from ayaka.utils.validation import require_int

__all__ = ["PADDING_BLOCK_ID", "KVGroup", "KVSpec"]

#: Block 0 of every pool is padding: it is never given to a request, so a block
#: table entry of 0 means "no block" and padded batch rows have somewhere harmless
#: to write.
PADDING_BLOCK_ID: Final[int] = 0


@dataclass(frozen=True, slots=True)
class KVSpec:
    """Geometry of the KV cache of a group of layers.

    Attributes:
        num_layers: Layers that share this spec.
        num_kv_heads: Key/value heads per layer.
        head_dim: Elements per head.
        block_size: Tokens per block. A value, never a constant: every formula
            takes it from here.
        dtype: Compute dtype of the stored values (fp16, bf16 or fp32).
        codec: How values are encoded; only ``"raw"`` (the compute dtype, unchanged).
        layout: ``NHD`` or ``HND``; asked from the ops backend by whoever builds
            the spec, never chosen here.
        window: Sliding-window width in tokens, or ``None`` for full attention.
            Carried so a model can declare it; no retention policy reads it yet.

    Raises:
        TypeError: If a count is not an ``int``, ``dtype`` is not a ``DType`` or
            ``layout`` is not a ``KVLayoutKind``.
        ValueError: If a count is below 1, ``dtype`` is not a compute dtype,
            ``codec`` is not ``"raw"`` or ``layout`` is not ``NHD`` or ``HND``.
    """

    num_layers: int
    num_kv_heads: int
    head_dim: int
    block_size: int
    dtype: DType
    layout: KVLayoutKind
    codec: str = RAW_CODEC
    window: int | None = None

    def __post_init__(self) -> None:
        require_int(self.num_layers, "num_layers", minimum=1)
        require_int(self.num_kv_heads, "num_kv_heads", minimum=1)
        require_int(self.head_dim, "head_dim", minimum=1)
        require_int(self.block_size, "block_size", minimum=1)
        if not isinstance(cast("object", self.dtype), DType):
            raise TypeError("dtype must be a DType member")
        if self.dtype not in COMPUTE_DTYPES:
            raise ValueError(f"dtype must be a compute dtype, got {self.dtype.label}")
        if not isinstance(cast("object", self.layout), KVLayoutKind):
            raise TypeError("layout must be a KVLayoutKind member")
        if self.layout not in (KVLayoutKind.NHD, KVLayoutKind.HND):
            raise ValueError(f"layout must be NHD or HND, got {self.layout.name}")
        if self.codec != RAW_CODEC:
            raise ValueError(f"codec must be {RAW_CODEC!r}, got {self.codec!r}")
        if self.window is not None:
            require_int(self.window, "window", minimum=1)

    def bytes_per_token(self) -> int:
        """Bytes one token occupies across every layer of the group, keys and values."""
        return self.dtype.nbytes(2 * self.num_layers * self.num_kv_heads * self.head_dim)

    def bytes_per_block(self) -> int:
        """Bytes one block occupies across every layer of the group, keys and values."""
        return self.bytes_per_token() * self.block_size

    def fixed_bytes(self) -> int:
        """Bytes that do not scale with blocks, such as a state slot; zero for attention KV."""
        return 0


@dataclass(frozen=True, slots=True)
class KVGroup:
    """The layers that share one :class:`KVSpec`, and so one pool and one block pool.

    Attributes:
        layer_ids: Model layer ids in this group, in the order the pool stores them.
        spec: Geometry shared by those layers; ``spec.num_layers`` equals ``len(layer_ids)``.

    Raises:
        TypeError: If ``layer_ids`` is not a tuple of ints.
        ValueError: If ``layer_ids`` is empty, repeats an id or holds a negative one,
            or its length differs from ``spec.num_layers``.
    """

    layer_ids: tuple[int, ...]
    spec: KVSpec

    def __post_init__(self) -> None:
        if not isinstance(cast("object", self.layer_ids), tuple):
            raise TypeError("layer_ids must be a tuple of ints")
        for layer in self.layer_ids:
            require_int(layer, "layer id")
        if len(set(self.layer_ids)) != len(self.layer_ids):
            raise ValueError("layer_ids must be distinct")
        if len(self.layer_ids) != self.spec.num_layers:
            raise ValueError(
                f"layer_ids has {len(self.layer_ids)} layers, "
                f"but the spec says {self.spec.num_layers}"
            )

    def layer_index(self, layer_id: int) -> int:
        """Position of a model layer inside the group's tensors.

        Raises:
            KeyError: If the layer is not in this group.
        """
        try:
            return self.layer_ids.index(layer_id)
        except ValueError:
            raise KeyError(f"layer {layer_id} is not in this group {self.layer_ids}") from None
