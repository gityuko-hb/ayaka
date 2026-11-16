"""Counter-based random numbers for sampling, as pure functions of the position.

``ayaka.sampling.rng`` is plain Python: it is the host-side reference the torch sampling
ops are checked against, and it never imports torch. The per-step parameter batch and the
sampler that use it land with the execution layer.

Public names are resolved from the module that defines them on first access.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ayaka.utils.lazy_exports import build_lazy_exports

if TYPE_CHECKING:
    from ayaka.sampling.rng import (
        MAX_SEED,
        MAX_SEQUENCE,
        draw_uniform,
        philox4x32_10,
        resolve_seed,
    )

__all__ = [
    "MAX_SEED",
    "MAX_SEQUENCE",
    "draw_uniform",
    "philox4x32_10",
    "resolve_seed",
]

_EXPORTS = {
    "rng": ("MAX_SEED", "MAX_SEQUENCE", "draw_uniform", "philox4x32_10", "resolve_seed"),
}

__getattr__, __dir__ = build_lazy_exports(__name__, _EXPORTS, globals())
