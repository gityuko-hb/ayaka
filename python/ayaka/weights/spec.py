"""What a model parameter needs, independent of any file."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import cast

from ayaka.utils.dtypes import DType
from ayaka.utils.validation import require_int, require_text

__all__ = ["TensorSpec"]


@dataclass(frozen=True, slots=True)
class TensorSpec:
    """A named tensor with a shape and a dtype.

    A model declares one per parameter; a checkpoint header yields one per
    stored tensor. Neither says where the bytes live (see
    :class:`~ayaka.weights.tensor.TensorRef`).

    Attributes:
        name: Parameter or checkpoint tensor name, e.g. ``"layers.0.attn.qkv.weight"``.
        shape: Dimensions; ``()`` is a scalar with one element.
        dtype: Element type. Sub-byte types count their packed bytes.

    Raises:
        ValueError: If ``name`` is blank or a dimension is negative.
        TypeError: If ``shape`` is not a tuple of ints or ``dtype`` is not a
            :class:`~ayaka.utils.dtypes.DType`.
    """

    name: str
    shape: tuple[int, ...]
    dtype: DType

    def __post_init__(self) -> None:
        require_text(self.name, "name")
        if not isinstance(cast("object", self.shape), tuple):
            raise TypeError("shape must be a tuple of ints")
        for dim in self.shape:
            require_int(dim, f"{self.name}: dimension", minimum=0)
        if not isinstance(cast("object", self.dtype), DType):
            raise TypeError("dtype must be a DType member")

    @property
    def numel(self) -> int:
        """Number of elements."""
        return math.prod(self.shape)

    @property
    def nbytes(self) -> int:
        """Bytes the tensor occupies in storage."""
        return self.dtype.nbytes(self.numel)
