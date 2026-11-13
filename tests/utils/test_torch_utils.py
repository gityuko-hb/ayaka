from __future__ import annotations

import os
from typing import Any

import pytest

from ayaka.core.errors import CapabilityError
from ayaka.utils.dtypes import DType
from ayaka.utils.torch_utils import (
    compute_capability,
    compute_torch_dtypes,
    cuda_available,
    device_count,
    device_guard,
    device_index,
    device_profile,
    dtype_bytes,
    dtype_name,
    has_dtype,
    no_device_sync,
    require_cuda,
    require_cuda_contiguous,
    require_last_dim_contiguous,
    require_torch,
    require_torch_version,
    resolve_device,
    seed_everything,
    supports_bf16,
    synchronize,
    torch_available,
    torch_dtype,
    torch_version,
    validate_output_dtype,
)

torch: Any = pytest.importorskip("torch")

pytestmark = pytest.mark.unit


class TestAvailability:
    def test_require_torch_returns_the_real_module(self) -> None:
        assert require_torch() is torch

    def test_torch_available(self) -> None:
        assert torch_available() is True

    def test_cuda_available_matches_torch(self) -> None:
        assert cuda_available() == bool(torch.cuda.is_available())

    def test_device_count_matches_torch(self) -> None:
        expected = torch.cuda.device_count() if torch.cuda.is_available() else 0
        assert device_count() == expected

    def test_torch_version_is_a_three_tuple(self) -> None:
        version = torch_version()
        assert len(version) == 3
        assert all(isinstance(part, int) for part in version)

    def test_require_torch_version_accepts_an_old_floor(self) -> None:
        require_torch_version((0, 0), capability="test")

    def test_require_torch_version_rejects_a_future_floor(self) -> None:
        with pytest.raises(CapabilityError, match="requires torch >="):
            require_torch_version((999, 0), capability="test")


class TestDeviceResolution:
    def test_cpu_device(self) -> None:
        resolved = resolve_device("cpu")
        assert resolved.type == "cpu"
        assert device_index("cpu") is None

    def test_none_resolves_to_a_usable_device(self) -> None:
        resolved = resolve_device(None)
        assert resolved.type in {"cpu", "cuda"}

    def test_device_guard_is_a_noop_on_cpu(self) -> None:
        with device_guard("cpu") as resolved:
            assert resolved.type == "cpu"

    def test_cuda_request_without_cuda_raises(self) -> None:
        if torch.cuda.is_available():
            pytest.skip("requires a CPU-only host")
        with pytest.raises(CapabilityError):
            resolve_device("cuda")

    def test_compute_capability_is_none_without_cuda(self) -> None:
        if torch.cuda.is_available():
            pytest.skip("requires a CPU-only host")
        assert compute_capability() is None
        assert not supports_bf16()

    def test_cpu_profile_is_a_placeholder(self) -> None:
        if torch.cuda.is_available():
            pytest.skip("requires a CPU-only host")
        profile = device_profile()
        assert profile.index == -1
        assert profile.name == "cpu"
        assert profile.sm == "cpu"
        assert profile.describe() == "cpu"


class TestDtypes:
    def test_dtype_name_normalizes(self) -> None:
        assert dtype_name(" Torch.BFloat16 ") == "bfloat16"

    def test_torch_dtype_accepts_names_registry_and_dtype(self) -> None:
        assert torch_dtype("float16") is torch.float16
        assert torch_dtype("torch.float32") is torch.float32
        assert torch_dtype(DType.BF16) is torch.bfloat16
        assert torch_dtype(torch.float64) is torch.float64

    def test_registry_property_uses_the_adapter(self) -> None:
        assert DType.FP16.torch_dtype is torch.float16

    def test_torch_dtype_rejects_unrepresentable_types(self) -> None:
        with pytest.raises(CapabilityError, match="no PyTorch representation"):
            torch_dtype(DType.INT4)

    def test_has_dtype(self) -> None:
        assert has_dtype("float16")
        assert not has_dtype("not_a_dtype")

    def test_dtype_bytes(self) -> None:
        assert dtype_bytes(torch.float16) == 2
        assert dtype_bytes("float32") == 4
        assert dtype_bytes("int8") == 1

    def test_compute_torch_dtypes_follow_the_registry(self) -> None:
        assert compute_torch_dtypes() == (torch.float16, torch.bfloat16, torch.float32)


class TestTensorGuards:
    def test_require_cuda_rejects_cpu_tensor(self) -> None:
        with pytest.raises(ValueError, match="must be a CUDA tensor"):
            require_cuda(torch.zeros(1), "x")

    def test_contiguity_guards_reject_cpu_tensor(self) -> None:
        with pytest.raises(ValueError, match="must be a CUDA tensor"):
            require_cuda_contiguous(torch.zeros(1), "x")
        with pytest.raises(ValueError, match="must be a CUDA tensor"):
            require_last_dim_contiguous(torch.zeros(1), "x")

    def test_output_dtype_accepts_compute_formats(self) -> None:
        validate_output_dtype(torch.float16)
        validate_output_dtype(torch.bfloat16)
        validate_output_dtype(torch.float32)

    def test_output_dtype_rejects_other_dtypes(self) -> None:
        with pytest.raises(ValueError, match="out_dtype"):
            validate_output_dtype(torch.int8)

    @pytest.mark.gpu
    def test_cuda_guards_accept_a_cuda_tensor(self) -> None:
        if not torch.cuda.is_available():
            pytest.skip("CUDA is unavailable")
        tensor = torch.zeros(2, 2, device="cuda")
        require_cuda_contiguous(tensor, "x")
        require_last_dim_contiguous(tensor, "x")


class TestSynchronizationAndSeed:
    def test_synchronize_is_a_noop_on_cpu(self) -> None:
        synchronize("cpu")

    def test_no_device_sync_is_a_noop_without_cuda(self) -> None:
        with no_device_sync():
            pass

    def test_seed_everything_returns_the_seed(self) -> None:
        assert seed_everything(1234) == 1234
        assert torch.initial_seed() == 1234
        assert os.environ["PYTHONHASHSEED"] == "1234"
