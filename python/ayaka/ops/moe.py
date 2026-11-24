"""Mixture-of-experts ops: the contract a backend implements for ``ayaka.layers.ffn.moe``.

Two calls per MoE layer and step, on the ``N`` rows of the step::

    route(logits [N, E])             -> weights [N, k] float32, ids [N, k] int32
    experts(x [N, H], experts, ...)  -> out [N, H]

Every expert is a SiLU-gated MLP. The experts of a layer are two stacked tensors, as
``transformers`` keeps them: ``gate_up`` ``[E, 2I, H]`` (each expert's gate rows, then its up
rows) and ``down`` ``[E, H, I]``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from torch import Tensor

__all__ = ["MoeOps"]


class MoeOps(Protocol):
    """Top-k routing and the gated expert MLPs."""

    def route(self, logits: Tensor, weights: Tensor, ids: Tensor, *, renormalize: bool) -> None:
        """Choose ``k`` experts per row: float32 softmax, the ``k`` largest, ties to the lower id.

        ``logits`` is ``[N, E]`` in a compute dtype; ``weights`` ``[N, k]`` float32 and ``ids``
        ``[N, k]`` int32 receive the chosen probabilities and expert ids, largest first. With
        ``renormalize`` the ``k`` weights of a row are divided by their sum.
        """
        ...

    def experts(
        self,
        x: Tensor,
        gate_up: Tensor,
        down: Tensor,
        weights: Tensor,
        ids: Tensor,
        out: Tensor,
    ) -> None:
        """Write ``sum_j weights[n, j] * expert_{ids[n, j]}(x[n])`` into ``out[n]``.

        ``expert_e(v) = down[e] @ (silu(gate) * up)`` with ``gate, up = gate_up[e] @ v``. ``x``
        and ``out`` are ``[N, H]``, ``gate_up`` ``[E, 2I, H]`` and ``down`` ``[E, H, I]`` in the
        compute dtype; ``weights`` and ``ids`` are as :meth:`route` writes them. The weighted
        sum is accumulated in float32. The result must not depend on how a backend groups the
        rows, beyond the rounding of that sum.
        """
        ...
