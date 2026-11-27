"""The flashinfer backend: paged attention, MLA, and linear-attention kernels on NVIDIA GPUs."""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Final

from ayaka.utils.import_utils import require_module

if TYPE_CHECKING:
    from ayaka.ops.backend import BackendFamilies

__all__ = ["HEAD_DIMS", "load"]

#: Head widths flashinfer has attention kernels for.
HEAD_DIMS: Final = frozenset({64, 128, 256})


def load() -> BackendFamilies:
    """Import flashinfer and return the families this backend implements."""
    require_module(
        "torch",
        capability="ops.flashinfer",
        remedy="install PyTorch with `uv sync --extra cu130`",
    )
    with warnings.catch_warnings():
        # FlashInfer imports cutlass_dsl utilities that emit cutlass.utils DeprecationWarnings.
        warnings.filterwarnings("ignore", category=DeprecationWarning, message=r"cutlass\.utils\.")
        require_module(
            "flashinfer",
            capability="ops.flashinfer",
            remedy="install it on Linux or WSL2 with `uv sync --extra cu130 --extra flashinfer`",
        )
    from ayaka.ops.backends.flashinfer import attention, linear_attention, mla

    return {
        "attention": attention,
        "linear_attention": linear_attention,
        "mla": mla,
    }
