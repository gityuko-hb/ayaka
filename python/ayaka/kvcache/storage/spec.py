"""The shape of KV, latent KV, and recurrent state cache groups before any tensor exists.

A :class:`KVSpec`, :class:`LatentKVSpec`, or :class:`RecurrentStateSpec` is
geometry only: which layers share it, head counts, widths, block/slot sizes, and
dtypes. Everything the memory planner needs follows by arithmetic without importing
torch or allocating tensors. :mod:`ayaka.kvcache.storage.pool` then allocates
tensors to these geometries.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, cast

from ayaka.core.types import KVLayoutKind
from ayaka.ops.kvcache import FP8_E4M3_CODEC, RAW_CODEC, SUPPORTED_KV_CODECS
from ayaka.utils.dtypes import COMPUTE_DTYPES, DType
from ayaka.utils.validation import require_int, require_number

__all__ = [
    "PADDING_BLOCK_ID",
    "KVGroup",
    "KVSpec",
    "LatentKVGroup",
    "LatentKVSpec",
    "RecurrentStateGroup",
    "RecurrentStateSpec",
]

#: Block 0 of every pool is padding: it is never given to a request, so a block
#: table entry of 0 means "no block" and padded batch rows have somewhere harmless
#: to write.
PADDING_BLOCK_ID: Final[int] = 0


def _storage_dtype(compute_dtype: DType, codec: str) -> DType:
    if codec == FP8_E4M3_CODEC:
        return DType.FP8_E4M3
    return compute_dtype


def _validate_layer_ids(layer_ids: tuple[int, ...], num_layers: int) -> None:
    if not isinstance(cast("object", layer_ids), tuple):
        raise TypeError("layer_ids must be a tuple of ints")
    for layer in layer_ids:
        require_int(layer, "layer id")
    if len(set(layer_ids)) != len(layer_ids):
        raise ValueError("layer_ids must be distinct")
    if len(layer_ids) != num_layers:
        raise ValueError(f"layer_ids has {len(layer_ids)} layers, but the spec says {num_layers}")


def _layer_index(layer_ids: tuple[int, ...], layer_id: int) -> int:
    try:
        return layer_ids.index(layer_id)
    except ValueError:
        raise KeyError(f"layer {layer_id} is not in this group {layer_ids}") from None


@dataclass(frozen=True, slots=True)
class KVSpec:
    """Geometry of the KV cache of a group of layers.

    Attributes:
        num_layers: Layers that share this spec.
        num_kv_heads: Key/value heads per layer.
        head_dim: Elements per head.
        block_size: Tokens per block.
        dtype: Compute dtype of the model (fp16, bf16 or fp32).
        layout: ``NHD`` or ``HND``; asked from the ops backend.
        codec: How values are encoded in storage (``"raw"`` or ``"fp8_e4m3"``).
        window: Sliding-window width in tokens, or ``None`` for full attention.
        kv_scale: Static dequantization scale when ``codec`` is ``"fp8_e4m3"``.
    """

    num_layers: int
    num_kv_heads: int
    head_dim: int
    block_size: int
    dtype: DType
    layout: KVLayoutKind
    codec: str = RAW_CODEC
    window: int | None = None
    kv_scale: float = 1.0

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
        if self.codec not in SUPPORTED_KV_CODECS:
            raise ValueError(
                f"codec must be one of {list(SUPPORTED_KV_CODECS)}, got {self.codec!r}"
            )
        if self.window is not None:
            require_int(self.window, "window", minimum=1)
        scale = require_number(self.kv_scale, "kv_scale")
        if scale <= 0:
            raise ValueError(f"kv_scale must be positive, got {scale}")
        object.__setattr__(self, "kv_scale", float(scale))

    @property
    def storage_dtype(self) -> DType:
        """DType stored in the physical KV pool tensors."""
        return _storage_dtype(self.dtype, self.codec)

    def bytes_per_token(self) -> int:
        """Bytes one token occupies across every layer of the group, keys and values."""
        return self.storage_dtype.nbytes(2 * self.num_layers * self.num_kv_heads * self.head_dim)

    def bytes_per_block(self) -> int:
        """Bytes one block occupies across every layer of the group, keys and values."""
        return self.bytes_per_token() * self.block_size

    def fixed_bytes(self) -> int:
        """Bytes that do not scale with blocks; zero for attention KV."""
        return 0


@dataclass(frozen=True, slots=True)
class KVGroup:
    """The layers that share one :class:`KVSpec`, and so one pool and one block pool."""

    layer_ids: tuple[int, ...]
    spec: KVSpec

    def __post_init__(self) -> None:
        _validate_layer_ids(self.layer_ids, self.spec.num_layers)

    def layer_index(self, layer_id: int) -> int:
        """Position of a model layer inside the group's tensors."""
        return _layer_index(self.layer_ids, layer_id)


@dataclass(frozen=True, slots=True)
class LatentKVSpec:
    """Geometry of a compressed Multi-Head Latent Attention (MLA) KV cache group.

    Attributes:
        num_layers: Layers sharing this latent spec.
        kv_lora_rank: Compressed KV latent width per token per layer.
        qk_rope_head_dim: Decoupled rotary key width per token per layer.
        block_size: Tokens per block.
        dtype: Compute dtype of the model.
        codec: Storage codec (``"raw"`` or ``"fp8_e4m3"``).
        kv_scale: Static dequantization scale for FP8 latent storage.
    """

    num_layers: int
    kv_lora_rank: int
    qk_rope_head_dim: int
    block_size: int
    dtype: DType
    codec: str = RAW_CODEC
    kv_scale: float = 1.0

    def __post_init__(self) -> None:
        require_int(self.num_layers, "num_layers", minimum=1)
        require_int(self.kv_lora_rank, "kv_lora_rank", minimum=1)
        require_int(self.qk_rope_head_dim, "qk_rope_head_dim", minimum=1)
        require_int(self.block_size, "block_size", minimum=1)
        if not isinstance(cast("object", self.dtype), DType):
            raise TypeError("dtype must be a DType member")
        if self.dtype not in COMPUTE_DTYPES:
            raise ValueError(f"dtype must be a compute dtype, got {self.dtype.label}")
        if self.codec not in SUPPORTED_KV_CODECS:
            raise ValueError(
                f"codec must be one of {list(SUPPORTED_KV_CODECS)}, got {self.codec!r}"
            )
        scale = require_number(self.kv_scale, "kv_scale")
        if scale <= 0:
            raise ValueError(f"kv_scale must be positive, got {scale}")
        object.__setattr__(self, "kv_scale", float(scale))

    @property
    def storage_dtype(self) -> DType:
        """DType stored in the physical latent KV pool tensors."""
        return _storage_dtype(self.dtype, self.codec)

    def bytes_per_token(self) -> int:
        """Bytes one token occupies across all layers in compressed ckv + kpe."""
        return self.storage_dtype.nbytes(
            self.num_layers * (self.kv_lora_rank + self.qk_rope_head_dim)
        )

    def bytes_per_block(self) -> int:
        """Bytes one block occupies across all layers in compressed ckv + kpe."""
        return self.bytes_per_token() * self.block_size

    def fixed_bytes(self) -> int:
        """Fixed per-engine bytes; zero for paged latent blocks."""
        return 0

    def compression_ratio_vs_kv(self, num_kv_heads: int, head_dim: int) -> float:
        """Ratio of latent KV bytes per token to standard GQA/MHA KV bytes per token."""
        require_int(num_kv_heads, "num_kv_heads", minimum=1)
        require_int(head_dim, "head_dim", minimum=1)
        standard = self.dtype.nbytes(2 * self.num_layers * num_kv_heads * head_dim)
        return self.bytes_per_token() / standard


@dataclass(frozen=True, slots=True)
class LatentKVGroup:
    """Layers sharing one :class:`LatentKVSpec`."""

    layer_ids: tuple[int, ...]
    spec: LatentKVSpec

    def __post_init__(self) -> None:
        _validate_layer_ids(self.layer_ids, self.spec.num_layers)

    def layer_index(self, layer_id: int) -> int:
        """Position of a model layer inside the group's latent tensors."""
        return _layer_index(self.layer_ids, layer_id)


@dataclass(frozen=True, slots=True)
class RecurrentStateSpec:
    """Geometry of per-request recurrent state slots for GDN/KDA linear-attention layers.

    Unlike paged KV caches, recurrent state size per active request is fixed and
    independent of sequence length: each live request holds one SSM matrix and
    one causal conv1d ring buffer per linear-attention layer.
    """

    num_layers: int
    num_v_heads: int
    key_head_dim: int
    value_head_dim: int
    conv_dim: int
    conv_kernel_size: int
    state_dtype: DType = DType.FP32
    conv_dtype: DType = DType.BF16

    def __post_init__(self) -> None:
        require_int(self.num_layers, "num_layers", minimum=1)
        require_int(self.num_v_heads, "num_v_heads", minimum=1)
        require_int(self.key_head_dim, "key_head_dim", minimum=1)
        require_int(self.value_head_dim, "value_head_dim", minimum=1)
        require_int(self.conv_dim, "conv_dim", minimum=1)
        require_int(self.conv_kernel_size, "conv_kernel_size", minimum=1)
        for attr, val in (("state_dtype", self.state_dtype), ("conv_dtype", self.conv_dtype)):
            if not isinstance(cast("object", val), DType):
                raise TypeError(f"{attr} must be a DType member")
            if val not in COMPUTE_DTYPES:
                raise ValueError(f"{attr} must be a compute dtype, got {val.label}")

    def ssm_bytes_per_slot(self) -> int:
        """Bytes of the recurrent matrix state for one request slot across all layers."""
        return self.state_dtype.nbytes(
            self.num_layers * self.num_v_heads * self.key_head_dim * self.value_head_dim
        )

    def conv_bytes_per_slot(self) -> int:
        """Bytes of the causal conv1d state for one request slot across all layers."""
        return self.conv_dtype.nbytes(self.num_layers * self.conv_dim * self.conv_kernel_size)

    def bytes_per_slot(self) -> int:
        """Total recurrent state bytes per request slot across all layers."""
        return self.ssm_bytes_per_slot() + self.conv_bytes_per_slot()

    def total_bytes(self, max_slots: int) -> int:
        """Total pool bytes for ``max_slots`` slots (including padding slot 0)."""
        require_int(max_slots, "max_slots", minimum=2)
        return max_slots * self.bytes_per_slot()


@dataclass(frozen=True, slots=True)
class RecurrentStateGroup:
    """Linear-attention layers sharing one :class:`RecurrentStateSpec`."""

    layer_ids: tuple[int, ...]
    spec: RecurrentStateSpec

    def __post_init__(self) -> None:
        _validate_layer_ids(self.layer_ids, self.spec.num_layers)

    def layer_index(self, layer_id: int) -> int:
        """Position of a model layer inside the group's state tensors."""
        return _layer_index(self.layer_ids, layer_id)
