"""Unit tests for KV cache pressure watermarks and pool utilization."""

from __future__ import annotations

import pytest

from ayaka.core.errors import ConfigError
from ayaka.core.types import KVLayoutKind, RequestId
from ayaka.kvcache.manager import KVCacheManager
from ayaka.kvcache.pressure import PressureLevel, PressureThresholds, evaluate_pressure
from ayaka.kvcache.storage.spec import KVGroup, KVSpec
from ayaka.utils.dtypes import DType

pytestmark = pytest.mark.unit


def _manager(num_blocks: int = 11) -> KVCacheManager:
    # 11 total blocks = 10 usable blocks (block 0 is padding).
    spec = KVSpec(
        num_layers=2,
        num_kv_heads=2,
        head_dim=4,
        block_size=4,
        dtype=DType.FP32,
        layout=KVLayoutKind.NHD,
    )
    group = KVGroup(layer_ids=(0, 1), spec=spec)
    return KVCacheManager([group], [num_blocks], max_requests=4, max_blocks_per_request=10)


def test_pressure_thresholds_validation() -> None:
    defaults = PressureThresholds()
    assert (defaults.high, defaults.critical) == (0.80, 0.92)
    assert PressureThresholds(high=0.5, critical=1.0).critical == 1.0

    for high, critical in ((0.0, 0.8), (0.8, 0.8), (0.9, 0.8), (0.7, 1.1), (-0.1, 0.5)):
        with pytest.raises(ConfigError):
            PressureThresholds(high=high, critical=critical)


def test_evaluate_pressure_boundaries() -> None:
    thresholds = PressureThresholds(high=0.8, critical=0.9)
    assert evaluate_pressure(0.0, thresholds) is PressureLevel.NORMAL
    assert evaluate_pressure(0.79, thresholds) is PressureLevel.NORMAL
    assert evaluate_pressure(0.80, thresholds) is PressureLevel.HIGH
    assert evaluate_pressure(0.89, thresholds) is PressureLevel.HIGH
    assert evaluate_pressure(0.90, thresholds) is PressureLevel.CRITICAL
    assert evaluate_pressure(1.0, thresholds) is PressureLevel.CRITICAL


def test_kv_manager_usable_counts_and_utilization() -> None:
    kv = _manager(num_blocks=11)  # 10 usable blocks
    assert kv.usable_counts == (10,)
    assert kv.utilization == pytest.approx(0.0)

    assert kv.allocate(RequestId("r1"), 32)  # 8 blocks
    assert kv.utilization == pytest.approx(0.8)

    assert kv.allocate(RequestId("r2"), 4)  # +1 block -> 9 / 10
    assert kv.utilization == pytest.approx(0.9)

    kv.free(RequestId("r1"))
    kv.free(RequestId("r2"))
    assert kv.utilization == pytest.approx(0.0)
    kv.check_integrity()
