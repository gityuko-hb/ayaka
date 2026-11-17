"""Argument checks of the torch reference ops.

PyTorch would accept each case refused here and compute something else: an
integer input is promoted, a wrongly shaped ``out`` is broadcast into, a
wrongly typed one is cast into.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from ayaka.core.errors import BackendError
from ayaka.utils.torch_utils import compute_torch_dtypes, dtype_name

if TYPE_CHECKING:
    from collections.abc import Sequence


def capturing(tensor: torch.Tensor) -> bool:
    """Whether ``tensor`` is on a GPU whose current stream is capturing a CUDA graph.

    A check that reads the device cannot run then: the read would end the capture.
    """
    return tensor.is_cuda and torch.cuda.is_current_stream_capturing()


def require_compute_dtype(op: str, name: str, tensor: torch.Tensor) -> None:
    """Refuse a tensor whose dtype is not fp16, bf16 or fp32."""
    if tensor.dtype not in compute_torch_dtypes():
        supported = ", ".join(dtype_name(dtype) for dtype in compute_torch_dtypes())
        raise BackendError(
            f"{op}: {name} is {dtype_name(tensor.dtype)}; "
            f"the torch backend computes in {supported}",
            context={"op": op, "argument": name},
        )


def require_like(
    op: str, name: str, tensor: torch.Tensor, shape: Sequence[int], dtype: torch.dtype
) -> None:
    """Refuse a tensor that does not have exactly this shape and dtype."""
    if tuple(tensor.shape) != tuple(shape) or tensor.dtype != dtype:
        raise BackendError(
            f"{op}: {name} must be {dtype_name(dtype)} {list(shape)}, "
            f"got {dtype_name(tensor.dtype)} {list(tensor.shape)}",
            context={"op": op, "argument": name},
        )


def require_index_tensor(op: str, name: str, tensor: torch.Tensor, *, ndim: int) -> None:
    """Refuse an index tensor that is not int32 or int64 with ``ndim`` dimensions."""
    if tensor.dtype not in (torch.int32, torch.int64) or tensor.ndim != ndim:
        raise BackendError(
            f"{op}: {name} must be an int32 or int64 tensor with {ndim} dimension(s), "
            f"got {dtype_name(tensor.dtype)} {list(tensor.shape)}",
            context={"op": op, "argument": name},
        )
