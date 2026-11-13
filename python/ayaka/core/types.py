"""Dependency-free identifiers and enums shared by every Ayaka layer.

Nothing here imports ``ayaka.*`` or an optional package. These are the values
that request, engine, execution, cache, serving and metrics can all name
without importing one another.
"""

from __future__ import annotations

from enum import StrEnum
from typing import NewType

__all__ = [
    "FinishReason",
    "ForwardMode",
    "Iteration",
    "KVLayoutKind",
    "Layout",
    "RequestId",
    "SequenceIndex",
    "TokenId",
]

#: Identity assigned at the API/serving door. It is never mutated later.
RequestId = NewType("RequestId", str)
#: A vocabulary token id. Kept as an alias so signatures document intent.
type TokenId = int
#: Index of an output sequence inside one request (``n > 1``).
type SequenceIndex = int
#: Engine step / cache epoch number.
type Iteration = int


class Layout(StrEnum):
    """Physical memory layout hints for memory planning and kernel dispatching.

    Layout serves as a performance hint and does not alter the mathematical
    semantics of tensor data.

    Attributes:
        ROW_MAJOR: Standard C-contiguous row-major layout.
        COL_MAJOR: Fortran-contiguous column-major layout.
        PACKED_SUB_BYTE: Packed layout for sub-byte quantized types.
        BLOCKED: Block/Tiled layout optimized for Tensor Cores or block-quantization.
    """

    ROW_MAJOR = "row_major"
    COL_MAJOR = "col_major"
    PACKED_SUB_BYTE = "packed_sub_byte"
    BLOCKED = "blocked"


class KVLayoutKind(StrEnum):
    """Physical layout arrangement of a single Key-Value cache block.

    This is an engine-global configuration. Modifying this layout at runtime
    invalidates all active block tables.

    Attributes:
        NHD: Shape [block_size, num_kv_heads, head_dim] (FlashInfer default).
        HND: Shape [num_kv_heads, block_size, head_dim] (vLLM / FlashAttention default).
        NLD: Shape [(page, slot, latent_dim)] MLA latent and RoPE planes
    """

    NHD = "nhd"
    HND = "hnd"
    NLD = "NLD"


class ForwardMode(StrEnum):
    """How a model forward pass consumes its query rows."""

    PREFILL = "prefill"
    DECODE = "decode"
    MIXED = "mixed"
    IDLE = "idle"
    VERIFY = "verify"

    @property
    def is_decode_like(self) -> bool:
        """Whether the batch contains decode rows."""

        return self in (ForwardMode.DECODE, ForwardMode.VERIFY)

    @property
    def is_prefill_like(self) -> bool:
        """Whether the batch contains prompt/extension rows."""

        return self in (ForwardMode.PREFILL, ForwardMode.MIXED)


class FinishReason(StrEnum):
    """Why a generation stopped.

    This enum is terminal-only. In-flight request state belongs to the engine,
    not to the public result contract; a streaming chunk simply has no finish
    reason yet.
    """

    STOP = "stop"
    LENGTH = "length"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"
    ERROR = "error"
