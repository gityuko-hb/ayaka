"""The kernel boundary: op contracts owned by Ayaka, implemented by backends.

Each op family is a :class:`typing.Protocol` in ``ayaka.ops.<family>``, written
from what the layers need. A kernel library is an implementation under
``ayaka.ops.backends.<name>``; only that directory imports the library.
:func:`~ayaka.ops.backend.load_backend` assembles one
:class:`~ayaka.ops.backend.OpsBackend` per engine, and layers receive it when
they are built. There is no global backend.

Rules every contract follows:

* Arguments are ``torch.Tensor`` objects on one device. No type of a kernel
  library crosses the boundary.
* The caller allocates the output and passes it as ``out``. An op returns
  nothing and never reallocates or resizes ``out``.
* ``out`` must not overlap an input unless the op's docstring allows it.
* Floating-point tensors use a compute dtype: fp16, bf16 or fp32.
* A backend that cannot run a call as specified raises
  :class:`~ayaka.core.errors.BackendError`. It never substitutes a different
  computation.

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
    from ayaka.ops.kvcache import KVBuffers, KVCacheOps
    from ayaka.ops.linear import LinearOps
    from ayaka.ops.norm import NormOps
    from ayaka.ops.sampling import DrawStream, SamplingOps

__all__ = [
    "GRAPH_FAMILIES",
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
    "LinearOps",
    "NormOps",
    "OpsBackend",
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
    "kvcache": ("KVBuffers", "KVCacheOps"),
    "linear": ("LinearOps",),
    "norm": ("NormOps",),
    "sampling": ("DrawStream", "SamplingOps"),
}

__getattr__, __dir__ = build_lazy_exports(__name__, _EXPORTS, globals())
