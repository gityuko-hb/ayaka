"""Reference linear ops; implements :class:`ayaka.ops.linear.LinearOps`."""

from __future__ import annotations

from typing import Final

import torch
from torch.nn import functional

from ayaka.core.errors import BackendError
from ayaka.ops.backends.torch._checks import require_compute_dtype, require_like

__all__ = ["SLICE_BYTES", "dequantize_rows", "linear", "scaled_linear", "slice_rows"]


def linear(
    x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None, out: torch.Tensor
) -> None:
    op = "linear.linear"
    require_compute_dtype(op, "x", x)
    if weight.ndim != 2:
        raise BackendError(
            f"{op}: weight must be [out_features, in_features], got {list(weight.shape)}",
            context={"op": op, "argument": "weight"},
        )
    out_features = weight.shape[0]
    require_like(op, "weight", weight, (out_features, x.shape[-1]), x.dtype)
    if bias is not None:
        require_like(op, "bias", bias, (out_features,), x.dtype)
    require_like(op, "out", out, (*x.shape[:-1], out_features), x.dtype)
    # Computed in the dtype of the inputs: widening would copy the whole weight.
    out.copy_(functional.linear(x, weight, bias))


#: Bytes of float32 weight one slice of :func:`scaled_linear` dequantizes at a time. Every slice of
#: every layer is written into the same two buffers (:func:`_workspace`), so this bounds the extra
#: memory of all 8-bit linear layers together.
SLICE_BYTES: Final = 64 * 2**20
_FP8_DTYPES: Final = (torch.float8_e4m3fn, torch.float8_e5m2)
#: One float32 buffer and one compute-dtype buffer per device, compute dtype and size.
_workspaces: dict[tuple[torch.device, torch.dtype, int], tuple[torch.Tensor, torch.Tensor]] = {}


def _workspace(
    device: torch.device, dtype: torch.dtype, elements: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """The two buffers every slice is dequantized into: float32, then the compute dtype.

    They are allocated on the first call and kept, so the allocator does not see a new tensor of
    a new size for every slice of every layer (which fragments its cache next to a KV cache sized
    to fill the device), and a CUDA graph captured later finds them at the same address. The
    start-up run makes the first call, so the start-up profile counts them.
    """
    key = (device, dtype, elements)
    found = _workspaces.get(key)
    if found is None:
        found = (
            torch.empty(elements, dtype=torch.float32, device=device),
            torch.empty(elements, dtype=dtype, device=device),
        )
        _workspaces[key] = found
    return found


def scaled_linear(
    x: torch.Tensor,
    weight: torch.Tensor,
    scale: torch.Tensor,
    bias: torch.Tensor | None,
    out: torch.Tensor,
    *,
    block: tuple[int, int] | None,
) -> None:
    """Dequantize a bounded slice of rows at a time into the compute dtype and multiply.

    No PyTorch op multiplies an 8-bit weight by a 16-bit activation on every device, so the
    weight stays 8 bit in memory and each slice of at most :data:`SLICE_BYTES` (in float32) is
    turned into the compute dtype for its product. Every slice starts on a block boundary, so
    a row is dequantized with the same scale whichever slice holds it: the dequantized weight
    does not depend on the slicing, and the product only by the rounding of the matrix kernel.
    """
    op = "linear.scaled_linear"
    require_compute_dtype(op, "x", x)
    if weight.ndim != 2 or weight.dtype not in _FP8_DTYPES:
        raise BackendError(
            f"{op}: weight must be a 2-d 8-bit float tensor, got {list(weight.shape)} "
            f"{weight.dtype}",
            context={"op": op, "argument": "weight"},
        )
    out_features, in_features = weight.shape
    if in_features != x.shape[-1]:
        raise BackendError(
            f"{op}: x has {x.shape[-1]} features but weight takes {in_features}",
            context={"op": op, "argument": "x"},
        )
    require_like(op, "weight", weight, (out_features, in_features), weight.dtype)
    if block is None:
        require_like(op, "scale", scale, (out_features, 1), torch.float32)
    else:
        rows, columns = block
        require_like(
            op,
            "scale",
            scale,
            (-(-out_features // rows), -(-in_features // columns)),
            torch.float32,
        )
    if bias is not None:
        require_like(op, "bias", bias, (out_features,), x.dtype)
    require_like(op, "out", out, (*x.shape[:-1], out_features), x.dtype)
    if not out.is_contiguous():
        raise BackendError(f"{op}: out must be contiguous", context={"op": op, "argument": "out"})

    flat_x = x.reshape(-1, in_features)
    flat_out = out.view(-1, out_features)
    step = slice_rows(out_features, in_features, block)
    exact, rounded = _workspace(x.device, x.dtype, max(step * in_features, SLICE_BYTES // 4))
    for start in range(0, out_features, step):
        stop = min(start + step, out_features)
        rows = dequantize_rows(weight, scale, start, stop, block, out=exact)
        dense = rounded[: rows.numel()].view_as(rows)
        dense.copy_(rows)
        part_bias = None if bias is None else bias[start:stop]
        flat_out[:, start:stop] = functional.linear(flat_x, dense, part_bias)


def slice_rows(out_features: int, in_features: int, block: tuple[int, int] | None) -> int:
    """Rows of one slice: as many as fit :data:`SLICE_BYTES` in float32, whole blocks only."""
    rows = max(1, SLICE_BYTES // (4 * max(in_features, 1)))
    if block is not None:
        rows = max(block[0], rows - rows % block[0])
    return min(rows, out_features)


def dequantize_rows(
    weight: torch.Tensor,
    scale: torch.Tensor,
    start: int,
    stop: int,
    block: tuple[int, int] | None,
    *,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Rows ``start:stop`` of ``scale * weight`` in float32; ``start`` is on a block boundary.

    With ``out``, a float32 buffer of at least that many elements, the rows are written into its
    first elements and scaled in place; otherwise a new tensor is returned.
    """
    in_features = weight.shape[1]
    count = stop - start
    if out is None:
        rows = weight[start:stop].to(torch.float32)
    else:
        rows = out[: count * in_features].view(count, in_features)
        rows.copy_(weight[start:stop])
    if block is None:
        return rows.mul_(scale[start:stop])
    block_rows, block_columns = block
    first = start // block_rows
    last = -(-stop // block_rows)
    scales = scale[first:last]
    if count % block_rows == 0 and in_features % block_columns == 0:
        shaped = rows.view(count // block_rows, block_rows, -1, block_columns)
        shaped.mul_(scales[:, None, :, None])
        return rows
    expanded = scales.repeat_interleave(block_rows, dim=0)[:count]
    expanded = expanded.repeat_interleave(block_columns, dim=1)[:, :in_features]
    return rows.mul_(expanded)
