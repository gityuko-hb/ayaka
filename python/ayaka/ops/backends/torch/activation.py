"""Reference activation ops; implements :class:`ayaka.ops.activation.ActivationOps`."""

from __future__ import annotations

from typing import TYPE_CHECKING

from torch.nn import functional

from ayaka.core.errors import BackendError
from ayaka.ops.backends.torch._checks import require_compute_dtype, require_like

if TYPE_CHECKING:
    import torch

__all__ = ["gelu", "gelu_and_mul", "silu_and_mul"]


def gelu(x: torch.Tensor, out: torch.Tensor, *, tanh: bool) -> None:
    op = "activation.gelu"
    require_compute_dtype(op, "x", x)
    require_like(op, "out", out, x.shape, x.dtype)
    out.copy_(_gelu(x.float(), tanh))


def silu_and_mul(x: torch.Tensor, out: torch.Tensor) -> None:
    activated, gate = _halves("activation.silu_and_mul", x, out)
    out.copy_(functional.silu(activated) * gate)


def gelu_and_mul(x: torch.Tensor, out: torch.Tensor, *, tanh: bool) -> None:
    activated, gate = _halves("activation.gelu_and_mul", x, out)
    out.copy_(_gelu(activated, tanh) * gate)


def _gelu(wide: torch.Tensor, tanh: bool) -> torch.Tensor:
    return functional.gelu(wide, approximate="tanh" if tanh else "none")


def _halves(op: str, x: torch.Tensor, out: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Check a packed ``[a, b]`` input and return both halves in fp32."""
    require_compute_dtype(op, "x", x)
    width, odd = divmod(x.shape[-1], 2)
    if odd:
        raise BackendError(
            f"{op}: x packs two halves along its last dimension, which is {x.shape[-1]}",
            context={"op": op, "argument": "x"},
        )
    require_like(op, "out", out, (*x.shape[:-1], width), x.dtype)
    activated, gate = x.float().chunk(2, dim=-1)
    return activated, gate
