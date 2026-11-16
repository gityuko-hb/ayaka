"""KV geometry: bytes per token and per block, and the checks on a spec and a group."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import pytest

from ayaka.core.types import KVLayoutKind
from ayaka.kvcache.storage.spec import PADDING_BLOCK_ID, KVGroup, KVSpec
from ayaka.utils.dtypes import DType

pytestmark = pytest.mark.unit


def _spec(**overrides: Any) -> KVSpec:
    fields: dict[str, Any] = {
        "num_layers": 28,
        "num_kv_heads": 2,
        "head_dim": 128,
        "block_size": 16,
        "dtype": DType.FP16,
        "layout": KVLayoutKind.NHD,
    }
    fields.update(overrides)
    return KVSpec(**fields)


def test_bytes_per_token_and_block_match_the_strategy_example() -> None:
    spec = _spec()

    # 2 (K and V) x 28 layers x 2 heads x 128 dim x 2 bytes.
    assert spec.bytes_per_token() == 28672
    assert spec.bytes_per_block() == 28672 * 16
    assert spec.fixed_bytes() == 0


@pytest.mark.parametrize(
    ("dtype", "itemsize"),
    [(DType.FP16, 2), (DType.BF16, 2), (DType.FP32, 4)],
)
def test_bytes_per_block_follows_the_dtype(dtype: DType, itemsize: int) -> None:
    spec = _spec(num_layers=3, num_kv_heads=4, head_dim=8, block_size=5, dtype=dtype)

    assert spec.bytes_per_block() == 2 * 3 * 4 * 8 * 5 * itemsize


def test_bytes_per_block_scales_linearly_with_every_factor() -> None:
    base = _spec(num_layers=2, num_kv_heads=2, head_dim=4, block_size=8).bytes_per_block()

    assert (
        _spec(num_layers=4, num_kv_heads=2, head_dim=4, block_size=8).bytes_per_block() == 2 * base
    )
    assert (
        _spec(num_layers=2, num_kv_heads=4, head_dim=4, block_size=8).bytes_per_block() == 2 * base
    )
    assert (
        _spec(num_layers=2, num_kv_heads=2, head_dim=8, block_size=8).bytes_per_block() == 2 * base
    )
    assert (
        _spec(num_layers=2, num_kv_heads=2, head_dim=4, block_size=16).bytes_per_block() == 2 * base
    )


def test_block_size_is_a_value_not_a_constant() -> None:
    assert _spec(block_size=1).bytes_per_block() == _spec().bytes_per_token()
    assert _spec(block_size=128).bytes_per_block() == _spec().bytes_per_token() * 128


def test_block_zero_is_the_padding_block() -> None:
    assert PADDING_BLOCK_ID == 0


@pytest.mark.parametrize(
    ("overrides", "error"),
    [
        ({"num_layers": 0}, ValueError),
        ({"num_kv_heads": 0}, ValueError),
        ({"head_dim": -1}, ValueError),
        ({"block_size": 0}, ValueError),
        ({"block_size": 16.0}, TypeError),
        ({"dtype": DType.INT8}, ValueError),
        ({"dtype": "float16"}, TypeError),
        ({"layout": "nhd"}, TypeError),
        ({"layout": KVLayoutKind.NLD}, ValueError),
        ({"codec": "fp8"}, ValueError),
        ({"window": 0}, ValueError),
    ],
)
def test_a_bad_spec_is_refused(overrides: dict[str, Any], error: type[Exception]) -> None:
    with pytest.raises(error):
        _spec(**overrides)


def test_a_window_is_carried_and_does_not_change_the_cost() -> None:
    assert _spec(window=4096).window == 4096
    assert _spec(window=4096).bytes_per_block() == _spec().bytes_per_block()


def test_a_spec_is_immutable_and_hashable() -> None:
    spec = _spec()

    with pytest.raises(AttributeError):
        spec.block_size = 32  # type: ignore[misc]  # frozen dataclass: the point of the test
    assert hash(spec) == hash(replace(spec))


def test_a_group_maps_model_layers_to_positions() -> None:
    group = KVGroup(layer_ids=(3, 5, 9), spec=_spec(num_layers=3))

    assert [group.layer_index(layer) for layer in (3, 5, 9)] == [0, 1, 2]
    with pytest.raises(KeyError, match="layer 4"):
        group.layer_index(4)


@pytest.mark.parametrize(
    ("layer_ids", "error"),
    [
        ((), ValueError),
        ((0, 0, 1), ValueError),
        ((0, -1, 1), ValueError),
        ((0, 1), ValueError),
        ([0, 1, 2], TypeError),
    ],
)
def test_a_bad_group_is_refused(layer_ids: Any, error: type[Exception]) -> None:
    with pytest.raises(error):
        KVGroup(layer_ids=layer_ids, spec=_spec(num_layers=3))
