"""Normalisation ops: the contract a backend implements for ``ayaka.layers.norm``."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from torch import Tensor

__all__ = ["NormOps"]


class NormOps(Protocol):
    """Normalisation over the last dimension.

    ``x`` is ``[..., hidden]``; ``weight`` and ``bias`` are ``[hidden]``. All
    tensors of a call share one compute dtype. Statistics are accumulated in at
    least fp32, whatever that dtype is, and the result is rounded to it once.
    """

    def rms_norm(self, x: Tensor, weight: Tensor, out: Tensor, *, eps: float) -> None:
        """Write ``x / sqrt(mean(x**2) + eps) * weight`` into ``out``.

        ``out`` has the shape and dtype of ``x`` and may be ``x`` itself.
        """
        ...

    def layer_norm(
        self, x: Tensor, weight: Tensor, bias: Tensor | None, out: Tensor, *, eps: float
    ) -> None:
        """Write ``(x - mean(x)) / sqrt(var(x) + eps) * weight + bias`` into ``out``.

        The variance is the biased one (divided by ``hidden``). ``out`` has the
        shape and dtype of ``x`` and may be ``x`` itself.
        """
        ...

    def fused_add_rms_norm(
        self, x: Tensor, residual: Tensor, weight: Tensor, *, eps: float
    ) -> None:
        """Replace ``residual`` by ``x + residual`` and ``x`` by the RMS norm of that sum.

        Both tensors are overwritten. They have the same shape and must not
        overlap.
        """
        ...
