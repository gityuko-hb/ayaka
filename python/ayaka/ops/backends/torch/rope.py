"""Reference rotary embedding ops; implements :class:`ayaka.ops.rope.RopeOps`.

The rotation gathers the cache rows of the positions and computes in fp32. It
reads nothing back to the host, so it can run inside a CUDA graph.
"""

from __future__ import annotations

import torch

from ayaka.core.errors import BackendError
from ayaka.ops.backends.torch._checks import (
    require_compute_dtype,
    require_index_tensor,
    require_like,
)

__all__ = ["apply", "build_cache"]


def build_cache(inv_freq: torch.Tensor, out: torch.Tensor, *, attention_factor: float) -> None:
    op = "rope.build_cache"
    if inv_freq.ndim != 1 or inv_freq.dtype != torch.float32:
        raise BackendError(
            f"{op}: inv_freq must be a float32 vector, got {inv_freq.dtype} {list(inv_freq.shape)}",
            context={"op": op, "argument": "inv_freq"},
        )
    half = inv_freq.shape[0]
    positions = out.shape[0] if out.ndim == 2 else 0
    require_like(op, "out", out, (positions, 2 * half), torch.float32)
    angles = torch.outer(
        torch.arange(positions, dtype=torch.float32, device=inv_freq.device), inv_freq
    )
    out[:, :half] = torch.cos(angles) * attention_factor
    out[:, half:] = torch.sin(angles) * attention_factor


def apply(
    positions: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    cache: torch.Tensor,
    *,
    interleaved: bool,
) -> None:
    op = "rope.apply"
    require_index_tensor(op, "positions", positions, ndim=1)
    require_compute_dtype(op, "q", q)
    rows = positions.shape[0]
    if q.ndim != 3 or k.ndim != 3 or q.shape[0] != rows or k.shape[0] != rows:
        raise BackendError(
            f"{op}: q and k must be [{rows}, heads, head_dim], got {list(q.shape)} and "
            f"{list(k.shape)}",
            context={"op": op, "argument": "q"},
        )
    require_like(op, "k", k, (rows, k.shape[1], q.shape[2]), q.dtype)
    if cache.dtype != torch.float32 or cache.ndim != 2:
        raise BackendError(
            f"{op}: cache must be a float32 [positions, rotary_dim] table, got "
            f"{cache.dtype} {list(cache.shape)}",
            context={"op": op, "argument": "cache"},
        )
    rotary = cache.shape[1]
    if rotary % 2 or rotary > q.shape[2] or rotary == 0:
        raise BackendError(
            f"{op}: the cache rotates {rotary} elements, which must be even, positive and at "
            f"most head_dim {q.shape[2]}",
            context={"op": op, "argument": "cache"},
        )
    if not (q.is_contiguous() and k.is_contiguous()):
        raise BackendError(f"{op}: q and k must be contiguous", context={"op": op})
    table = cache.index_select(0, positions.long())  # [N, rotary]
    cos, sin = table[:, None, : rotary // 2], table[:, None, rotary // 2 :]
    for x in (q, k):
        part = x[..., :rotary]
        wide = part.float()
        if interleaved:
            first, second = wide[..., 0::2], wide[..., 1::2]
            rotated = torch.stack(
                (first * cos - second * sin, second * cos + first * sin), dim=-1
            ).flatten(-2)
        else:
            first, second = wide[..., : rotary // 2], wide[..., rotary // 2 :]
            rotated = torch.cat((first * cos - second * sin, second * cos + first * sin), dim=-1)
        part.copy_(rotated)
