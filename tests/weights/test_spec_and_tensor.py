from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from ayaka.utils.dtypes import DType
from ayaka.weights.spec import TensorSpec
from ayaka.weights.tensor import TensorRef

pytestmark = pytest.mark.unit


class TestTensorSpec:
    def test_counts_elements_and_bytes(self) -> None:
        spec = TensorSpec("w", (4, 3), DType.FP16)

        assert (spec.numel, spec.nbytes) == (12, 24)

    def test_a_scalar_has_one_element(self) -> None:
        spec = TensorSpec("s", (), DType.FP32)

        assert (spec.numel, spec.nbytes) == (1, 4)

    def test_a_tensor_with_a_zero_dimension_is_empty(self) -> None:
        spec = TensorSpec("e", (0, 7), DType.BF16)

        assert (spec.numel, spec.nbytes) == (0, 0)

    def test_sub_byte_types_round_up_to_whole_bytes(self) -> None:
        assert TensorSpec("q", (3,), DType.INT4).nbytes == 2

    def test_is_hashable_and_compares_by_value(self) -> None:
        assert TensorSpec("w", (2,), DType.FP32) == TensorSpec("w", (2,), DType.FP32)
        assert len({TensorSpec("w", (2,), DType.FP32), TensorSpec("w", (2,), DType.FP32)}) == 1

    @pytest.mark.parametrize("name", ["", "  "])
    def test_rejects_a_blank_name(self, name: str) -> None:
        with pytest.raises(ValueError, match="name must be a non-empty string"):
            TensorSpec(name, (1,), DType.FP32)

    def test_rejects_a_negative_dimension(self) -> None:
        with pytest.raises(ValueError, match=r"w: dimension must be >= 0"):
            TensorSpec("w", (2, -1), DType.FP32)

    @pytest.mark.parametrize("shape", [[2, 3], 5, "23"])
    def test_rejects_a_shape_that_is_not_a_tuple(self, shape: Any) -> None:
        with pytest.raises(TypeError, match="shape must be a tuple"):
            TensorSpec("w", shape, DType.FP32)

    @pytest.mark.parametrize("dimension", [True, 2.0, "2"])
    def test_rejects_a_dimension_that_is_not_an_int(self, dimension: Any) -> None:
        with pytest.raises(TypeError, match="dimension must be an integer"):
            TensorSpec("w", (dimension,), DType.FP32)

    def test_rejects_a_dtype_given_as_text(self) -> None:
        with pytest.raises(TypeError, match="dtype must be a DType"):
            TensorSpec("w", (1,), "float32")  # type: ignore[arg-type]  # the check under test


class TestTensorRef:
    def test_exposes_the_spec_and_the_byte_range(self) -> None:
        ref = TensorRef(TensorSpec("w", (2, 3), DType.FP16), Path("a.safetensors"), 100)

        assert (ref.name, ref.shape, ref.dtype) == ("w", (2, 3), DType.FP16)
        assert (ref.offset, ref.nbytes, ref.end) == (100, 12, 112)

    def test_an_empty_tensor_ends_where_it_starts(self) -> None:
        ref = TensorRef(TensorSpec("e", (0,), DType.FP32), Path("a.safetensors"), 8)

        assert ref.end == ref.offset == 8

    def test_rejects_a_negative_offset(self) -> None:
        with pytest.raises(ValueError, match="offset must be >= 0"):
            TensorRef(TensorSpec("w", (1,), DType.FP32), Path("a.safetensors"), -1)

    def test_rejects_a_path_given_as_text(self) -> None:
        with pytest.raises(TypeError, match=r"path must be a pathlib.Path"):
            TensorRef(TensorSpec("w", (1,), DType.FP32), "a.safetensors", 0)  # type: ignore[arg-type]  # the check under test
