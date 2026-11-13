from __future__ import annotations

from typing import Any

import pytest

from ayaka.core.errors import CapabilityError
from ayaka.utils.torch_memory import (
    DeviceMemory,
    device_memory,
    empty_cache,
    empty_host_tensor,
    peak_memory_bytes,
)

torch: Any = pytest.importorskip("torch")

pytestmark = pytest.mark.unit


class TestDeviceMemory:
    def test_cpu_is_not_a_cuda_device(self) -> None:
        with pytest.raises(CapabilityError, match="not a CUDA device"):
            device_memory("cpu")

    def test_derived_fields(self) -> None:
        memory = DeviceMemory(
            device_index=0,
            driver_free=6 * 2**20,
            driver_total=10 * 2**20,
            allocated=3 * 2**20,
            reserved=4 * 2**20,
        )
        assert memory.driver_used == 4 * 2**20
        assert memory.allocator_overhead == 1 * 2**20
        description = memory.describe()
        assert "driver 4/10 MiB" in description
        assert "allocated 3 MiB" in description
        assert "+1 MiB overhead" in description


class TestEmptyCache:
    def test_empty_cache_is_a_noop_without_cuda(self) -> None:
        empty_cache()


class TestPeakMemory:
    def test_cpu_block_reports_zero_peak(self) -> None:
        with peak_memory_bytes("cpu") as peak:
            assert peak == [0]
        assert peak == [0]


class TestEmptyHostTensor:
    def test_pageable_allocation(self) -> None:
        tensor, pinned = empty_host_tensor((2, 3), torch.float16, pinned=False)
        assert tuple(tensor.shape) == (2, 3)
        assert tensor.dtype is torch.float16
        assert pinned is False

    def test_dtype_name_is_accepted(self) -> None:
        tensor, pinned = empty_host_tensor(4, "int8", pinned=False)
        assert tuple(tensor.shape) == (4,)
        assert tensor.dtype is torch.int8
        assert pinned is False

    def test_pinned_request_without_cuda_raises(self) -> None:
        if torch.cuda.is_available():
            pytest.skip("requires a CPU-only host")
        with pytest.raises(CapabilityError, match="requires CUDA"):
            empty_host_tensor((2, 3), torch.float16, pinned=True)

    def test_pageable_fallback_is_explicit(self) -> None:
        if torch.cuda.is_available():
            pytest.skip("requires a CPU-only host")
        tensor, pinned = empty_host_tensor((2, 3), torch.float16, pinned=True, allow_pageable=True)
        assert tuple(tensor.shape) == (2, 3)
        assert pinned is False

    @pytest.mark.gpu
    def test_pinned_allocation_with_cuda(self) -> None:
        if not torch.cuda.is_available():
            pytest.skip("CUDA is unavailable")
        tensor, pinned = empty_host_tensor((2, 3), torch.float16, pinned=True)
        assert pinned is True
        assert bool(tensor.is_pinned())
