"""Inference-only building blocks of a model: every computation goes through ``ayaka.ops``.

A layer is a :class:`torch.nn.Module` that holds its parameters, is built with
the :class:`~ayaka.ops.backend.OpsBackend` it will call, and allocates its
output on every call. Parameters are created uninitialised and without
gradients; ``ayaka.model_loader`` fills them. No layer touches the KV cache
or a block manager: attention receives its cache through the forward context.

Public names are resolved from their submodule on first access, and a
submodule imports torch, so importing this package does not.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ayaka.utils.lazy_exports import build_lazy_exports

if TYPE_CHECKING:
    from ayaka.layers.activation import GELU, GELUAndMul, SiLUAndMul
    from ayaka.layers.attention.dense import Attention
    from ayaka.layers.attention.linear import GatedDeltaNetAttention, KimiDeltaAttention
    from ayaka.layers.attention.mla import MultiHeadLatentAttention
    from ayaka.layers.context import ForwardContext, StateCheckpointCapture
    from ayaka.layers.embedding.token import LMHead, TokenEmbedding
    from ayaka.layers.ffn.dense import MLP, GatedMLP
    from ayaka.layers.ffn.moe import MoEBlock, StackedExperts
    from ayaka.layers.linear import Linear
    from ayaka.layers.norm import LayerNorm, RMSNorm, RMSNormGated, ScalelessRMSNorm
    from ayaka.layers.rope import RotaryEmbedding, rope_inverse_frequencies

__all__ = [
    "GELU",
    "MLP",
    "Attention",
    "ForwardContext",
    "GELUAndMul",
    "GatedDeltaNetAttention",
    "GatedMLP",
    "KimiDeltaAttention",
    "LMHead",
    "LayerNorm",
    "Linear",
    "MoEBlock",
    "MultiHeadLatentAttention",
    "RMSNorm",
    "RMSNormGated",
    "RotaryEmbedding",
    "ScalelessRMSNorm",
    "SiLUAndMul",
    "StackedExperts",
    "StateCheckpointCapture",
    "TokenEmbedding",
    "rope_inverse_frequencies",
]

_EXPORTS = {
    "activation": ("GELU", "GELUAndMul", "SiLUAndMul"),
    "attention.dense": ("Attention",),
    "attention.linear": ("GatedDeltaNetAttention", "KimiDeltaAttention"),
    "attention.mla": ("MultiHeadLatentAttention",),
    "context": ("ForwardContext", "StateCheckpointCapture"),
    "embedding.token": ("LMHead", "TokenEmbedding"),
    "ffn.dense": ("MLP", "GatedMLP"),
    "ffn.moe": ("MoEBlock", "StackedExperts"),
    "linear": ("Linear",),
    "norm": ("LayerNorm", "RMSNorm", "RMSNormGated", "ScalelessRMSNorm"),
    "rope": ("RotaryEmbedding", "rope_inverse_frequencies"),
}

__getattr__, __dir__ = build_lazy_exports(__name__, _EXPORTS, globals())
