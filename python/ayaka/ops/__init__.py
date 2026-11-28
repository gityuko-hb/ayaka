"""The kernel boundary: op contracts owned by Ayaka, implemented by backends.

Importing this package does not import torch; a backend's library is imported
when ``load_backend`` selects it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ayaka.utils.lazy_exports import build_lazy_exports

if TYPE_CHECKING:
    from ayaka.ops.activation import ActivationOps
    from ayaka.ops.attention import AttentionMetadata, AttentionOps
    from ayaka.ops.backend import (
        GRAPH_FAMILIES,
        BackendCaps,
        BackendFamilies,
        FamilyName,
        OpsBackend,
        available_backends,
        load_backend,
        register_backend,
    )
    from ayaka.ops.embedding import EmbeddingOps
    from ayaka.ops.kvcache import (
        FP8_E4M3_CODEC,
        RAW_CODEC,
        SUPPORTED_KV_CODECS,
        KVBuffers,
        KVCacheOps,
        LatentKVBuffers,
    )
    from ayaka.ops.linear import LinearOps
    from ayaka.ops.linear_attention import (
        LinearAttentionMetadata,
        LinearAttentionOps,
        RecurrentStateBuffers,
    )
    from ayaka.ops.mla import MLAOps
    from ayaka.ops.moe import MoeOps
    from ayaka.ops.norm import NormOps
    from ayaka.ops.rope import RopeOps
    from ayaka.ops.sampling import DrawStream, SamplingOps

__all__ = [
    "FP8_E4M3_CODEC",
    "GRAPH_FAMILIES",
    "RAW_CODEC",
    "SUPPORTED_KV_CODECS",
    "ActivationOps",
    "AttentionMetadata",
    "AttentionOps",
    "BackendCaps",
    "BackendFamilies",
    "DrawStream",
    "EmbeddingOps",
    "FamilyName",
    "KVBuffers",
    "KVCacheOps",
    "LatentKVBuffers",
    "LinearAttentionMetadata",
    "LinearAttentionOps",
    "LinearOps",
    "MLAOps",
    "MoeOps",
    "NormOps",
    "OpsBackend",
    "RecurrentStateBuffers",
    "RopeOps",
    "SamplingOps",
    "available_backends",
    "load_backend",
    "register_backend",
]

_EXPORTS = {
    "activation": ("ActivationOps",),
    "attention": ("AttentionMetadata", "AttentionOps"),
    "backend": (
        "GRAPH_FAMILIES",
        "BackendCaps",
        "BackendFamilies",
        "FamilyName",
        "OpsBackend",
        "available_backends",
        "load_backend",
        "register_backend",
    ),
    "embedding": ("EmbeddingOps",),
    "kvcache": (
        "FP8_E4M3_CODEC",
        "RAW_CODEC",
        "SUPPORTED_KV_CODECS",
        "KVBuffers",
        "KVCacheOps",
        "LatentKVBuffers",
    ),
    "linear": ("LinearOps",),
    "linear_attention": (
        "LinearAttentionMetadata",
        "LinearAttentionOps",
        "RecurrentStateBuffers",
    ),
    "mla": ("MLAOps",),
    "moe": ("MoeOps",),
    "norm": ("NormOps",),
    "rope": ("RopeOps",),
    "sampling": ("DrawStream", "SamplingOps"),
}

__getattr__, __dir__ = build_lazy_exports(__name__, _EXPORTS, globals())
