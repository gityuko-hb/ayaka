"""Dtype registry and numeric limits shared by the runtime.

The registry is deliberately independent of PyTorch: the protocol layer must
remain importable on hosts without PyTorch or a CUDA driver. Conversion to
``torch.dtype`` lives in :mod:`ayaka.utils.torch_utils` and is loaded lazily
through :attr:`DType.torch_dtype`.
"""

from __future__ import annotations

import enum
from typing import Any, Final, cast

__all__ = [
    "COMPUTE_DTYPES",
    "FP8_E4M3_MAX",
    "FP8_E4M3_MIN",
    "INT8_MAX",
    "INT8_MIN",
    "LOCAL_ABSMAX_EPS",
    "MXFP4_BLOCK_SIZE",
    "NVFP4_BLOCK_SIZE",
    "DType",
]


class DType(enum.Enum):
    """Internal data type enumeration for the execution runtime.

    Deliberately decoupled from ``torch.dtype``: the protocol layer must remain
    importable on environments without PyTorch or CUDA drivers installed.
    Conversion to ``torch.dtype`` is provided by
    :func:`ayaka.utils.torch_utils.torch_dtype`.

    Attributes:
        label (str): Canonical lowercase string identifier (e.g., "fp16", "int4").
        bits (int): Bit width per element.
        sub_byte (bool): Flag indicating whether the type occupies less than 1 byte
            and requires packing.
        torch_name (str | None): PyTorch attribute name for the type, or None when
            PyTorch has no representation for it.
        safetensors_token (str): Safetensors ``dtype`` token for the type.
        max_finite (float | None): Largest finite magnitude representable by the
            type, or None for non-floating-point types.
    """

    # Floating Point Types
    FP64 = ("fp64", 64, False, "float64", "F64", None)
    FP32 = ("fp32", 32, False, "float32", "F32", 3.4028234663852886e38)
    FP16 = ("fp16", 16, False, "float16", "F16", 65504.0)
    BF16 = ("bf16", 16, False, "bfloat16", "BF16", 3.3895313892515355e38)
    FP8_E4M3 = ("fp8_e4m3", 8, False, "float8_e4m3fn", "F8_E4M3", 448.0)
    FP8_E5M2 = ("fp8_e5m2", 8, False, "float8_e5m2", "F8_E5M2", 57344.0)
    FP4_E2M1 = ("fp4_e2m1", 4, True, None, "F4", None)

    # Integer & Quantized Types
    INT64 = ("int64", 64, False, "int64", "I64", None)
    INT32 = ("int32", 32, False, "int32", "I32", None)
    INT8 = ("int8", 8, False, "int8", "I8", None)
    UINT8 = ("uint8", 8, False, "uint8", "U8", None)
    INT4 = ("int4", 4, True, None, "I4", None)

    # Boolean Type
    BOOL = ("bool", 8, False, "bool", "BOOL", None)

    def __init__(
        self,
        label: str,
        bits: int,
        sub_byte: bool,
        torch_name: str | None,
        safetensors_token: str,
        max_finite: float | None,
    ) -> None:
        self.label: Final[str] = label
        self.bits: Final[int] = bits
        self.sub_byte: Final[bool] = sub_byte
        self.torch_name: Final[str | None] = torch_name
        self.safetensors_token: Final[str] = safetensors_token
        self.max_finite: Final[float | None] = max_finite

    @classmethod
    def from_str(cls, name: str) -> DType:
        """Lookup a DType instance by any of its registered names (case-insensitive).

        Accepts the canonical label, the enum name, the PyTorch attribute name,
        and the safetensors token.

        Args:
            name: The dtype name (e.g., "fp16", "float16", "F16", "int4").

        Returns:
            The matching DType enum member.

        Raises:
            ValueError: If the identifier does not match any registered DType.
        """
        normalized = name.strip().lower()
        for member in cls:
            if normalized in (member.label, member.name.lower(), member.safetensors_token.lower()):
                return member
            if member.torch_name is not None and member.torch_name == normalized:
                return member
        raise ValueError(f"Unknown dtype: '{name}'. Valid options: {[m.label for m in cls]}")

    @classmethod
    def by_torch_name(cls, name: str) -> DType:
        """Lookup a DType by its PyTorch attribute name (case-insensitive).

        Raises:
            ValueError: If no registered type has a torch representation under
                ``name``.
        """
        normalized = name.strip().lower()
        for member in cls:
            if member.torch_name == normalized:
                return member
        raise ValueError(f"dtype {name!r} has no PyTorch representation")

    @classmethod
    def by_safetensors_token(cls, token: str) -> DType:
        """Lookup a DType by its safetensors ``dtype`` token (case-insensitive).

        Raises:
            ValueError: If no registered type carries ``token``.
        """
        normalized = token.strip().lower()
        for member in cls:
            if member.safetensors_token.lower() == normalized:
                return member
        raise ValueError(f"unknown safetensors dtype {token!r}")

    @property
    def torch_dtype(self) -> Any:
        """The matching ``torch.dtype``.

        Imported lazily so this module stays usable without PyTorch.

        Raises:
            CapabilityError: If PyTorch is unavailable, this build lacks the
                dtype, or the type has no torch representation.
        """
        from ayaka.utils.torch_utils import torch_dtype

        return torch_dtype(self, capability=f"dtype.{self.label}")

    @property
    def itemsize(self) -> float:
        """Bytes per element.

        Note:
            Fractional for sub-byte types (e.g., 0.5 for 4-bit types). Callers
            requiring exact byte allocation sizes must use :meth:`nbytes`.
        """
        return self.bits / 8.0

    @property
    def is_floating_point(self) -> bool:
        """Whether this dtype represents a standard or micro-scaled floating-point type."""
        return self in (
            DType.FP64,
            DType.FP32,
            DType.FP16,
            DType.BF16,
            DType.FP8_E4M3,
            DType.FP8_E5M2,
            DType.FP4_E2M1,
        )

    @property
    def is_integer(self) -> bool:
        """Whether this dtype represents a signed or unsigned integer type."""
        return self in (DType.INT64, DType.INT32, DType.INT8, DType.UINT8, DType.INT4)

    def nbytes(self, numel: int) -> int:
        """Calculate the exact byte storage required for a given number of elements.

        Sub-byte types are packed 2-per-byte along the contiguous dimension.
        An odd element count is rounded up to a whole byte boundary.

        Args:
            numel: Total number of logical elements in the tensor.

        Returns:
            Total bytes required for physical storage.

        Raises:
            ValueError: If ``numel`` is negative.
        """
        if numel < 0:
            raise ValueError(f"numel must be non-negative, got {numel}")
        total_bits = numel * self.bits
        return (total_bits + 7) // 8

    def __repr__(self) -> str:
        return f"DType.{self.name}"


#: The compute formats every torch-backed layer accepts. Single source for the
#: per-kernel "supported dtype" checks; resolve through
#: ``utils.torch_utils.compute_torch_dtypes`` on the torch side.
COMPUTE_DTYPES: Final[tuple[DType, ...]] = (DType.FP16, DType.BF16, DType.FP32)

#: Largest finite magnitude of ``torch.float8_e4m3fn``. Derived from the
#: registry so the two cannot drift; ``tests/utils/test_dtypes.py`` pins it.
FP8_E4M3_MAX: Final[float] = cast("float", DType.FP8_E4M3.max_finite)

FP8_E4M3_MIN: Final[float] = -FP8_E4M3_MAX

INT8_MAX: Final[int] = 2 ** (DType.INT8.bits - 1) - 1

INT8_MIN: Final[int] = -(2 ** (DType.INT8.bits - 1))

#: Floor for a per-tensor local absmax before its reciprocal is taken.
LOCAL_ABSMAX_EPS: Final[float] = 1.0e-10

#: Elements sharing one E8M0 scale in an MXFP4 quantization block.
MXFP4_BLOCK_SIZE: Final[int] = 32

#: Elements sharing one E4M3 scale in an NVFP4 quantization block.
NVFP4_BLOCK_SIZE: Final[int] = 16
