"""Model architectures: the :class:`CausalLM` contract, the registry and the models themselves.

A model is built from a ``ModelConfig`` and the ops backend, runs a step through
``ayaka.layers`` and owns the translation from its checkpoint's tensors to its
parameters. Reading the checkpoint is ``ayaka.model_loader``'s job.

Public names are resolved from their submodule on first access, and the
submodules that define models import torch, so importing this package does not.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ayaka.utils.lazy_exports import build_lazy_exports

if TYPE_CHECKING:
    from ayaka.models.base import CausalLM
    from ayaka.models.gpt2 import GPT2LMHeadModel
    from ayaka.models.registry import register_model, resolve_model

__all__ = ["CausalLM", "GPT2LMHeadModel", "register_model", "resolve_model"]

_EXPORTS = {
    "base": ("CausalLM",),
    "gpt2": ("GPT2LMHeadModel",),
    "registry": ("register_model", "resolve_model"),
}

__getattr__, __dir__ = build_lazy_exports(__name__, _EXPORTS, globals())
