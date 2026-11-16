"""From logits to tokens: counter-based random numbers, per-step parameters and the sampler.

Public names are resolved from the module that defines them on first access, so
importing this package never loads torch. ``ayaka.sampling.rng`` is plain Python;
``batch`` and ``sampler`` import torch when they are first used.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ayaka.utils.lazy_exports import build_lazy_exports

if TYPE_CHECKING:
    from ayaka.sampling.batch import SamplingBatch, SamplingRow
    from ayaka.sampling.rng import (
        MAX_SEED,
        MAX_SEQUENCE,
        draw_uniform,
        philox4x32_10,
        resolve_seed,
    )
    from ayaka.sampling.sampler import Sampler

__all__ = [
    "MAX_SEED",
    "MAX_SEQUENCE",
    "Sampler",
    "SamplingBatch",
    "SamplingRow",
    "draw_uniform",
    "philox4x32_10",
    "resolve_seed",
]

_EXPORTS = {
    "batch": ("SamplingBatch", "SamplingRow"),
    "rng": ("MAX_SEED", "MAX_SEQUENCE", "draw_uniform", "philox4x32_10", "resolve_seed"),
    "sampler": ("Sampler",),
}

__getattr__, __dir__ = build_lazy_exports(__name__, _EXPORTS, globals())
