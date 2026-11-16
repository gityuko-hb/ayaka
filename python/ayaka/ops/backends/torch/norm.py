"""Reference normalisation ops; implements :class:`ayaka.ops.norm.NormOps`."""

from __future__ import annotations

import torch
from torch.nn import functional

from ayaka.ops.backends.torch._checks import require_compute_dtype, require_like

__all__ = ["fused_add_rms_norm", "layer_norm", "rms_norm"]


def rms_norm(x: torch.Tensor, weight: torch.Tensor, out: torch.Tensor, *, eps: float) -> None:
    _check("norm.rms_norm", x, weight, out)
    out.copy_(_rms_normalised(x.float(), weight, eps))


def layer_norm(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    out: torch.Tensor,
    *,
    eps: float,
) -> None:
    op = "norm.layer_norm"
    _check(op, x, weight, out)
    if bias is not None:
        require_like(op, "bias", bias, x.shape[-1:], x.dtype)
    out.copy_(
        functional.layer_norm(
            x.float(),
            x.shape[-1:],
            weight.float(),
            None if bias is None else bias.float(),
            eps,
        )
    )


def fused_add_rms_norm(
    x: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor, *, eps: float
) -> None:
    _check("norm.fused_add_rms_norm", x, weight, residual, out_name="residual")
    total = x.float() + residual.float()
    residual.copy_(total)
    # Normalise the fp32 sum, not the copy just rounded into ``residual``.
    x.copy_(_rms_normalised(total, weight, eps))


def _rms_normalised(wide: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    # fp32 on purpose: the square of an fp16 value above 255 is already infinite.
    scale = torch.rsqrt(wide.square().mean(dim=-1, keepdim=True) + eps)
    return wide * scale * weight.float()


def _check(
    op: str, x: torch.Tensor, weight: torch.Tensor, out: torch.Tensor, *, out_name: str = "out"
) -> None:
    require_compute_dtype(op, "x", x)
    require_like(op, "weight", weight, x.shape[-1:], x.dtype)
    require_like(op, out_name, out, x.shape, x.dtype)
