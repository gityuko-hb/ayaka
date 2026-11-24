"""Phi-3: the Llama block, with checkpoints that store the fused projections.

Phi-3, Phi-3.5 and Phi-4 checkpoints (``Phi3ForCausalLM``) compute what Llama computes: rotary
positions, grouped-query attention, RMSNorm and a SiLU-gated feed-forward. They differ in their
layout on disk: ``self_attn.qkv_proj`` and ``mlp.gate_up_proj`` are stored already fused, rows
in the order Ayaka keeps them (q, k, v; gate, up), so nothing is split when they are loaded.

What is not implemented is refused when ``config.json`` is read
(:meth:`~ayaka.configs.model.ModelConfig.from_hf_config`): ``longrope`` scaling (Phi-3-mini-128k,
Phi-3.5-mini, Phi-4-mini) and a rotary embedding over part of each head
(``partial_rotary_factor``, Phi-4-mini). Phi-3-mini-4k declares a 2,047-token sliding window,
shorter than its 4,096-token context; it is served up to the window
(:attr:`~ayaka.configs.model.ModelConfig.context_length`).
"""

from __future__ import annotations

from typing import ClassVar, Final

from ayaka.models.llama import LlamaForCausalLM

__all__ = ["Phi3ForCausalLM"]

_RENAMES: Final = (
    (r"model\.embed_tokens\.weight", r"embed_tokens.weight"),
    (r"model\.norm\.weight", r"norm.weight"),
    (
        r"model\.layers\.(\d+)\.(input_layernorm|post_attention_layernorm)\.weight",
        r"layers.\1.\2.weight",
    ),
    (r"model\.layers\.(\d+)\.self_attn\.(qkv|o)_proj\.(weight|bias)", r"layers.\1.attn.\2_proj.\3"),
    (r"model\.layers\.(\d+)\.mlp\.(gate_up|down)_proj\.(weight|bias)", r"layers.\1.mlp.\2_proj.\3"),
)


class Phi3ForCausalLM(LlamaForCausalLM):
    """Phi-3 with a language-model head; the parameters are those of :class:`LlamaForCausalLM`.

    ``qkv_proj`` and ``gate_up_proj`` are read whole from the checkpoint.
    """

    checkpoint_renames: ClassVar[tuple[tuple[str, str], ...]] = _RENAMES

    def stored_parts(self, name: str) -> tuple[tuple[str, int], ...]:
        """Nothing is split: the checkpoint stores every parameter whole."""
        del name
        return ()
