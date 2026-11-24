"""The flashinfer backend: paged attention on NVIDIA GPUs under Linux and WSL2.

It implements one family, ``attention``; ``load_backend`` takes every other
family from the next backend in the list (``torch``). The checks that decided
this were measured on an RTX 3050 Laptop under WSL2 with flashinfer 0.7.4:

* sampling stays on ``torch``: flashinfer's samplers draw from a generator of
  their own (a seed and an offset) and take no random number from the caller,
  so they cannot follow :mod:`ayaka.ops.sampling`;
* the cache needs no adapter: flashinfer reads the ``(k, v)`` pair of
  ``KVPool`` in both the NHD and HND layouts, so ``kvcache`` stays on ``torch``;
* attention over several rows per request (prefill chunks, verify windows) and
  the decode wrapper inside a CUDA graph both match the ``torch`` backend.

The library is imported by :func:`load`, which ``load_backend`` calls only when
it picks this backend; importing this package imports neither torch nor
flashinfer. flashinfer compiles its kernels the first time a shape is used
(about 40 seconds per module on the dev machine) and keeps them in its own
cache directory.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

from ayaka.utils.import_utils import require_module

if TYPE_CHECKING:
    from ayaka.ops.backend import BackendFamilies

__all__ = ["HEAD_DIMS", "load"]

#: Head widths flashinfer has attention kernels for. ``load_backend`` takes attention from the
#: next backend for a model with another width (Phi-3-mini: 96).
HEAD_DIMS: Final = frozenset({64, 128, 256})


def load() -> BackendFamilies:
    """Import flashinfer and return the families this backend implements.

    Raises:
        CapabilityError: If flashinfer (or torch) is not installed.
    """
    require_module(
        "torch",
        capability="ops.flashinfer",
        remedy="install PyTorch with `uv sync --extra cu130`",
    )
    require_module(
        "flashinfer",
        capability="ops.flashinfer",
        remedy="install it on Linux or WSL2 with `uv sync --extra cu130 --extra flashinfer`",
    )
    from ayaka.ops.backends.flashinfer import attention

    return {"attention": attention}
