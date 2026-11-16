"""Where one tensor's bytes are stored."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, cast

from ayaka.utils.validation import require_int

if TYPE_CHECKING:
    from ayaka.utils.dtypes import DType
    from ayaka.weights.spec import TensorSpec

__all__ = ["TensorRef"]


@dataclass(frozen=True, slots=True)
class TensorRef:
    """A tensor in a file: its description and the byte range that holds it.

    The bytes are ``spec.nbytes`` long and start at ``offset`` from the
    beginning of ``path``. A ref is a claim read from a header, not a reading;
    :meth:`ayaka.weights.safetensors.SafetensorsFile.read` is what checks the
    claim against the file.

    Attributes:
        spec: Name, shape and dtype of the stored tensor.
        path: File that holds it.
        offset: Absolute byte offset of the first byte, from the start of ``path``.
    """

    spec: TensorSpec
    path: Path
    offset: int

    def __post_init__(self) -> None:
        if not isinstance(cast("object", self.path), Path):
            raise TypeError("path must be a pathlib.Path")
        require_int(self.offset, "offset", minimum=0)

    @property
    def name(self) -> str:
        return self.spec.name

    @property
    def shape(self) -> tuple[int, ...]:
        return self.spec.shape

    @property
    def dtype(self) -> DType:
        return self.spec.dtype

    @property
    def nbytes(self) -> int:
        return self.spec.nbytes

    @property
    def end(self) -> int:
        """Offset one past the last byte."""
        return self.offset + self.nbytes
