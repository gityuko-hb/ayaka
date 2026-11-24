"""MiMo-7B: the Qwen2 model, with a multi-token-prediction layer the engine does not use yet.

MiMo-7B checkpoints (``MiMoForCausalLM``, such as ``XiaomiMiMo/MiMo-7B-RL``) are Qwen2 checkpoints
with one addition. The authors' ``modeling_mimo.py`` defines ``MiMoModel(Qwen2Model)`` and
``MiMoForCausalLM(Qwen2ForCausalLM)`` and overrides no forward method: the main path is exactly
:mod:`ayaka.models.qwen2` (biases on the query, key and value projections, none on the output
projection, an untied head, no query/key norm). What they add is ``mtp_layers``, a list of
``num_nextn_predict_layers`` (1) layers that guess the token after the next one from the main
model's final hidden state, for speculative decoding.

Those tensors (``model.mtp_layers.N.*``, 16 for one layer) are **set aside**: the weight mapping
ignores them, so the model is built, sized in ``ayaka plan`` and run as the plain Qwen2 model it is.
They are not read into memory. Using them is the multi-token-prediction drafter of the speculative
decoding work. Every other tensor that no parameter claims is still an error.
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, Final

from ayaka.models.qwen2 import Qwen2ForCausalLM

if TYPE_CHECKING:
    from ayaka.configs.model import ModelConfig
    from ayaka.weights.mapping import WeightMapping

__all__ = ["MiMoForCausalLM"]

#: The multi-token-prediction layers of the checkpoint, which no parameter of this model claims.
_MTP_TENSORS: Final = (r"model\.mtp_layers\..*",)


class MiMoForCausalLM(Qwen2ForCausalLM):
    """MiMo-7B with a language-model head; the parameters are those of :class:`Qwen2ForCausalLM`."""

    @classmethod
    def weight_mapping(cls, config: ModelConfig) -> WeightMapping:
        """Qwen2's names, with the checkpoint's multi-token-prediction tensors ignored."""
        mapping = super().weight_mapping(config)
        return dataclasses.replace(mapping, ignore=(*mapping.ignore, *_MTP_TENSORS))
