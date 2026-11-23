"""Qwen2: the Llama block with a bias on the query, key and value projections.

Qwen2 and Qwen2.5 checkpoints (``Qwen2ForCausalLM``) differ from Llama in one place that
changes the arithmetic: ``q_proj``, ``k_proj`` and ``v_proj`` have a bias, ``o_proj`` has
none, and ``config.json`` has no ``attention_bias`` key to say so. The tensor names, the
fused ``qkv_proj`` and ``gate_up_proj``, the rotary embedding and the norms are Llama's
(:mod:`ayaka.models.llama`).

``sliding_window`` is read by :meth:`~ayaka.configs.model.ModelConfig.from_hf_config`:
the Qwen2.5 checkpoints switch it off (``use_sliding_window: false``); one that switches it
on is served up to the window (:attr:`~ayaka.configs.model.ModelConfig.context_length`).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ayaka.models.llama import LlamaForCausalLM

if TYPE_CHECKING:
    from ayaka.configs.model import ModelConfig

__all__ = ["Qwen2ForCausalLM"]


class Qwen2ForCausalLM(LlamaForCausalLM):
    """Qwen2 with a language-model head; the parameters are those of :class:`LlamaForCausalLM`.

    Every ``layers.N.attn.qkv_proj`` has a bias, stored as ``q_proj.bias``, ``k_proj.bias``
    and ``v_proj.bias``; ``o_proj`` has none.
    """

    @classmethod
    def projection_biases(cls, config: ModelConfig) -> tuple[bool, bool]:
        """``qkv_proj`` always has a bias and ``o_proj`` never does."""
        del config
        return True, False
