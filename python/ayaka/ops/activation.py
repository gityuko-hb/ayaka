"""Activation ops: the contract a backend implements for the feed-forward layers."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from torch import Tensor

__all__ = ["ActivationOps"]


class ActivationOps(Protocol):
    """Element-wise activations, plain and fused with the gate of a gated MLP.

    ``x`` and ``out`` share one compute dtype. The activation is evaluated in at
    least fp32 and the result is rounded to that dtype once. ``tanh`` selects
    the tanh approximation of GELU instead of the exact erf form.

    The ``*_and_mul`` ops take the two halves of a gated MLP packed along the
    last dimension, ``x = [a, b]`` with shape ``[..., 2 * d]``, and write
    ``act(a) * b`` into an ``out`` of shape ``[..., d]`` that must not overlap
    ``x``.
    """

    def gelu(self, x: Tensor, out: Tensor, *, tanh: bool) -> None:
        """Write ``gelu(x)`` into ``out``, which has the shape of ``x`` and may be ``x``."""
        ...

    def silu_and_mul(self, x: Tensor, out: Tensor) -> None:
        """Write ``silu(a) * b`` into ``out``."""
        ...

    def gelu_and_mul(self, x: Tensor, out: Tensor, *, tanh: bool) -> None:
        """Write ``gelu(a) * b`` into ``out``."""
        ...
