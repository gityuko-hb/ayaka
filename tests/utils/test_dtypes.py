from __future__ import annotations

import pytest

from ayaka.utils.dtypes import (
    COMPUTE_DTYPES,
    FP8_E4M3_MAX,
    FP8_E4M3_MIN,
    INT8_MAX,
    INT8_MIN,
    MXFP4_BLOCK_SIZE,
    NVFP4_BLOCK_SIZE,
    DType,
)

pytestmark = pytest.mark.unit


class TestDTypeLookup:
    def test_from_str_accepts_every_registered_name(self) -> None:
        assert DType.from_str("fp16") is DType.FP16
        assert DType.from_str("FP16") is DType.FP16
        assert DType.from_str("float16") is DType.FP16
        assert DType.from_str("f16") is DType.FP16

    def test_from_str_strips_whitespace(self) -> None:
        assert DType.from_str("  bf16  ") is DType.BF16

    def test_from_str_rejects_unknown_name(self) -> None:
        with pytest.raises(ValueError, match="Unknown dtype"):
            DType.from_str("float128")

    def test_by_torch_name(self) -> None:
        assert DType.by_torch_name("float8_e4m3fn") is DType.FP8_E4M3
        with pytest.raises(ValueError, match="has no PyTorch representation"):
            DType.by_torch_name("float128")

    def test_by_safetensors_token(self) -> None:
        assert DType.by_safetensors_token("F8_E4M3") is DType.FP8_E4M3
        assert DType.by_safetensors_token("bf16") is DType.BF16
        with pytest.raises(ValueError, match="unknown safetensors dtype"):
            DType.by_safetensors_token("F128")


class TestDTypeProperties:
    def test_itemsize_and_packed_nbytes(self) -> None:
        assert DType.FP16.itemsize == 2.0
        assert DType.INT4.itemsize == 0.5
        assert DType.FP16.nbytes(3) == 6
        assert DType.INT4.nbytes(3) == 2
        assert DType.INT4.nbytes(0) == 0

    def test_nbytes_rejects_negative_numel(self) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            DType.FP16.nbytes(-1)

    @pytest.mark.parametrize(
        "dtype",
        [DType.FP64, DType.FP32, DType.FP16, DType.BF16, DType.FP8_E4M3],
    )
    def test_floating_point_types(self, dtype: DType) -> None:
        assert dtype.is_floating_point

    @pytest.mark.parametrize(
        "dtype",
        [DType.INT64, DType.INT32, DType.INT8, DType.UINT8, DType.INT4],
    )
    def test_integer_types(self, dtype: DType) -> None:
        assert dtype.is_integer

    def test_bool_is_neither_float_nor_integer(self) -> None:
        assert not DType.BOOL.is_floating_point
        assert not DType.BOOL.is_integer

    def test_repr_is_stable(self) -> None:
        assert repr(DType.FP16) == "DType.FP16"


class TestLimits:
    def test_fp8_limit_matches_registry(self) -> None:
        assert DType.FP8_E4M3.max_finite == FP8_E4M3_MAX
        assert -FP8_E4M3_MAX == FP8_E4M3_MIN

    def test_int8_limits_match_registry(self) -> None:
        assert 2 ** (DType.INT8.bits - 1) - 1 == INT8_MAX
        assert -(2 ** (DType.INT8.bits - 1)) == INT8_MIN

    def test_quantization_block_sizes(self) -> None:
        assert MXFP4_BLOCK_SIZE == 32
        assert NVFP4_BLOCK_SIZE == 16

    def test_compute_dtypes_are_the_torch_compute_formats(self) -> None:
        assert COMPUTE_DTYPES == (DType.FP16, DType.BF16, DType.FP32)
