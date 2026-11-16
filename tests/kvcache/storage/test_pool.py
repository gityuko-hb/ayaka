"""The KV pool allocates to its spec, keeps block 0 as padding and hands out layer views."""

from __future__ import annotations

import subprocess
import sys
from typing import Any

import pytest

from ayaka.core.types import KVLayoutKind
from ayaka.kvcache.storage.pool import KVPool
from ayaka.kvcache.storage.spec import PADDING_BLOCK_ID, KVGroup, KVSpec
from ayaka.ops.kvcache import RAW_CODEC
from ayaka.utils.dtypes import DType

torch: Any = pytest.importorskip("torch")

pytestmark = pytest.mark.unit


def _group(layout: KVLayoutKind = KVLayoutKind.NHD, **overrides: Any) -> KVGroup:
    fields: dict[str, Any] = {
        "num_layers": 3,
        "num_kv_heads": 2,
        "head_dim": 8,
        "block_size": 4,
        "dtype": DType.FP32,
        "layout": layout,
    }
    fields.update(overrides)
    spec = KVSpec(**fields)
    return KVGroup(layer_ids=tuple(range(10, 10 + spec.num_layers)), spec=spec)


@pytest.mark.parametrize("layout", [KVLayoutKind.NHD, KVLayoutKind.HND])
@pytest.mark.parametrize("dtype", [DType.FP16, DType.BF16, DType.FP32])
def test_allocated_bytes_are_bytes_per_block_times_blocks(
    layout: KVLayoutKind, dtype: DType
) -> None:
    group = _group(layout, dtype=dtype)

    pool = KVPool(group, 7, device="cpu")

    assert pool.nbytes == 7 * group.spec.bytes_per_block()


@pytest.mark.parametrize(
    ("layout", "shape"),
    [
        (KVLayoutKind.NHD, (6, 4, 2, 8)),
        (KVLayoutKind.HND, (6, 2, 4, 8)),
    ],
)
def test_a_layer_view_has_the_layout_the_ops_contract_defines(
    layout: KVLayoutKind, shape: tuple[int, ...]
) -> None:
    pool = KVPool(_group(layout), 6, device="cpu")

    buffers = pool.buffers(11)

    assert tuple(buffers.k.shape) == shape
    assert tuple(buffers.v.shape) == shape
    assert buffers.k.dtype == buffers.v.dtype == torch.float32
    assert buffers.k.is_contiguous()
    assert buffers.v.is_contiguous()
    assert buffers.layout is layout
    assert buffers.codec == RAW_CODEC
    assert (buffers.num_blocks, buffers.block_size) == (6, 4)
    assert (buffers.num_kv_heads, buffers.head_dim) == (2, 8)


def test_block_zero_is_padding_and_never_counted_as_usable() -> None:
    pool = KVPool(_group(), 5, device="cpu")

    assert PADDING_BLOCK_ID == 0
    assert pool.num_blocks == 5
    assert pool.usable_blocks == 4
    for layer in pool.group.layer_ids:
        buffers = pool.buffers(layer)
        assert not buffers.k[PADDING_BLOCK_ID].any()
        assert not buffers.v[PADDING_BLOCK_ID].any()


def test_the_smallest_pool_is_the_padding_block_and_one_usable_block() -> None:
    pool = KVPool(_group(), 2, device="cpu")

    assert pool.usable_blocks == 1
    with pytest.raises(ValueError, match="num_blocks"):
        KVPool(_group(), 1, device="cpu")


def test_layer_views_alias_the_pool_and_do_not_overlap() -> None:
    pool = KVPool(_group(), 4, device="cpu")

    pool.buffers(10).k[2].fill_(1.0)
    pool.buffers(12).v[3].fill_(2.0)

    assert pool.buffers(10).k[2].eq(1.0).all()
    assert not pool.buffers(11).k[2].any()
    assert not pool.buffers(10).v[2].any()
    assert pool.buffers(12).v[3].eq(2.0).all()
    assert not pool.buffers(12).k[3].any()


def test_a_layer_outside_the_group_is_refused() -> None:
    pool = KVPool(_group(), 4, device="cpu")

    with pytest.raises(KeyError):
        pool.buffers(0)


def test_the_pool_lives_on_the_requested_device() -> None:
    pool = KVPool(_group(), 3, device="cpu")

    assert pool.device == torch.device("cpu")
    assert pool.buffers(10).k.device == torch.device("cpu")


def test_importing_the_storage_modules_does_not_import_torch() -> None:
    program = (
        "import sys\n"
        "import ayaka.kvcache\n"
        "import ayaka.kvcache.storage.spec\n"
        "import ayaka.kvcache.storage.pool\n"
        "assert 'torch' not in sys.modules, 'torch was imported'\n"
    )

    result = subprocess.run(  # noqa: S603  # fixed argv, no input from outside
        [sys.executable, "-I", "-c", program], capture_output=True, text=True, check=False
    )

    assert result.returncode == 0, result.stderr
