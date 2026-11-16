"""Linear ops: the contract a backend implements for ``ayaka.layers.linear``."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from torch import Tensor

__all__ = ["LinearOps"]


class LinearOps(Protocol):
    """Affine projection of the last dimension."""

    def linear(self, x: Tensor, weight: Tensor, bias: Tensor | None, out: Tensor) -> None:
        """Write ``x @ weight.T + bias`` into ``out``.

        ``x`` is ``[..., in_features]``, ``weight`` is
        ``[out_features, in_features]``, ``bias`` is ``[out_features]`` or
        ``None`` and ``out`` is ``[..., out_features]``. All of them share one
        compute dtype. ``out`` must not overlap an input.
        """
        ...
