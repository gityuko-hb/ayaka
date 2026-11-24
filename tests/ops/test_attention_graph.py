"""Attention on graph metadata (torch backend, CPU): the fixed record a captured decode step reads.

A decode step written with ``update_graph_metadata`` must store to the same slots and attend to
the same result as the same step planned with ``plan``. The cache is filled with NaN outside the
written positions, so reading a slot past a request's length, block 0 or an unused block shows up.
The GPU capture itself is tested in ``tests/execution/test_graph.py``.
"""

from __future__ import annotations

from typing import Any

import pytest

from ayaka.core.errors import BackendError
from ayaka.core.types import ForwardMode, KVLayoutKind
from ayaka.ops.backend import load_backend
from ayaka.ops.kvcache import RAW_CODEC, KVBuffers
from ayaka.utils.dtypes import DType

torch: Any = pytest.importorskip("torch")

pytestmark = pytest.mark.unit

BLOCK = 4
MAX_BLOCKS = 5


def _ops() -> Any:
    return load_backend(["torch"], device="cpu", dtype=DType.FP32)


def _cache(layout: KVLayoutKind, blocks: int = 16) -> KVBuffers:
    shape = (blocks, BLOCK, 2, 8) if layout is KVLayoutKind.NHD else (blocks, 2, BLOCK, 8)
    generator = torch.Generator().manual_seed(0)
    k = torch.randn(shape, generator=generator)
    v = torch.randn(shape, generator=generator)
    return KVBuffers(k, v, layout, RAW_CODEC)


def _step(lengths: list[int]) -> tuple[Any, Any]:
    """Lengths and a table of distinct blocks (never 0) for each real request."""
    tables = []
    next_block = 1
    for length in lengths:
        needed = -(-length // BLOCK)
        tables.append([*range(next_block, next_block + needed), *([0] * (MAX_BLOCKS - needed))])
        next_block += needed
    return (
        torch.tensor(lengths, dtype=torch.int32),
        torch.tensor(tables, dtype=torch.int32).reshape(len(lengths), MAX_BLOCKS),
    )


@pytest.mark.parametrize("layout", [KVLayoutKind.NHD, KVLayoutKind.HND], ids=["nhd", "hnd"])
def test_a_graph_step_equals_the_planned_step(layout: KVLayoutKind) -> None:
    ops = _ops()
    buffers = _cache(layout)
    lengths, table = _step([5, 1, 12, 20])
    q = torch.randn(4, 4, 8, generator=torch.Generator().manual_seed(1))

    planned = ops.attention.plan(
        ForwardMode.DECODE, torch.ones(4, dtype=torch.int32), lengths, table, BLOCK
    )
    graph = ops.attention.graph_metadata(4, MAX_BLOCKS, BLOCK, torch.device("cpu"))
    ops.attention.update_graph_metadata(graph, lengths, table)
    expected, out = torch.empty_like(q), torch.empty_like(q)
    ops.attention.attend(q, buffers, planned, expected, scale=0.3)
    ops.attention.attend(q, buffers, graph, out, scale=0.3)

    assert graph.positions.tolist() == planned.positions.tolist()
    assert graph.slot_mapping.tolist() == planned.slot_mapping.tolist()
    torch.testing.assert_close(out, expected, rtol=1e-5, atol=1e-6)


def test_padding_requests_read_nothing_and_get_zeros() -> None:
    ops = _ops()
    buffers = _cache(KVLayoutKind.NHD)
    _, table = _step([6, 3])  # request 0 in blocks 1-2, request 1 in block 3
    # NaN everywhere but the written positions: 0..5 of request 0 and 0..2 of request 1.
    for blocks in (buffers.k, buffers.v):
        real = blocks.clone()
        blocks.fill_(float("nan"))
        blocks[1] = real[1]
        blocks[2, :2] = real[2, :2]
        blocks[3, :3] = real[3, :3]
    graph = ops.attention.graph_metadata(4, MAX_BLOCKS, BLOCK, torch.device("cpu"))
    padded_lengths = torch.tensor([6, 3, 0, 0], dtype=torch.int32)
    padded_table = torch.cat([table, torch.zeros(2, MAX_BLOCKS, dtype=torch.int32)])
    ops.attention.update_graph_metadata(graph, padded_lengths, padded_table)
    q = torch.randn(4, 4, 8)
    out = torch.full_like(q, 7.0)

    ops.attention.attend(q, buffers, graph, out, scale=0.3)

    assert torch.isfinite(out).all()
    assert torch.count_nonzero(out[2:]) == 0
    assert graph.slot_mapping.tolist()[2:] == [0, 0]


def test_store_writes_the_slot_of_each_rows_position() -> None:
    ops = _ops()
    buffers = _cache(KVLayoutKind.NHD)
    lengths, table = _step([5, 8])
    graph = ops.attention.graph_metadata(2, MAX_BLOCKS, BLOCK, torch.device("cpu"))
    ops.attention.update_graph_metadata(graph, lengths, table)
    k = torch.full((2, 2, 8), 3.0)
    v = torch.full((2, 2, 8), 4.0)

    ops.kvcache.store(k, v, buffers, graph)

    # request 0: position 4 is offset 0 of its second block (2); request 1: position 7 of block 4
    assert torch.equal(buffers.k[2, 0], k[0])
    assert torch.equal(buffers.v[4, 3], v[1])


def test_a_new_step_rewrites_the_same_tensors() -> None:
    ops = _ops()
    graph = ops.attention.graph_metadata(2, MAX_BLOCKS, BLOCK, torch.device("cpu"))
    addresses = [t.data_ptr() for t in (graph.positions, graph.slot_mapping, graph.kv_lengths)]

    for lengths in ([1, 2], [9, 17]):
        lengths_t, table = _step(lengths)
        ops.attention.update_graph_metadata(graph, lengths_t, table)

    assert graph.kv_lengths.tolist() == [9, 17]
    assert [t.data_ptr() for t in (graph.positions, graph.slot_mapping, graph.kv_lengths)] == (
        addresses
    )


class TestRejections:
    def test_a_length_beyond_the_table_is_refused(self) -> None:
        ops = _ops()
        graph = ops.attention.graph_metadata(1, MAX_BLOCKS, BLOCK, torch.device("cpu"))

        with pytest.raises(BackendError, match=r"in \[0, 20\]"):
            ops.attention.update_graph_metadata(
                graph,
                torch.tensor([21], dtype=torch.int32),
                torch.ones(1, MAX_BLOCKS, dtype=torch.int32),
            )

    def test_inputs_of_the_wrong_shape_or_place_are_refused(self) -> None:
        ops = _ops()
        graph = ops.attention.graph_metadata(2, MAX_BLOCKS, BLOCK, torch.device("cpu"))

        with pytest.raises(
            BackendError, match=r"kv_lengths must be an int32 CPU tensor of shape \[2\]"
        ):
            ops.attention.update_graph_metadata(
                graph,
                torch.ones(3, dtype=torch.int32),
                torch.ones(2, MAX_BLOCKS, dtype=torch.int32),
            )

    def test_planned_metadata_is_not_graph_metadata(self) -> None:
        ops = _ops()
        lengths, table = _step([3])
        planned = ops.attention.plan(
            ForwardMode.DECODE, torch.ones(1, dtype=torch.int32), lengths, table, BLOCK
        )

        with pytest.raises(BackendError, match="not made by this backend's graph_metadata"):
            ops.attention.update_graph_metadata(planned, lengths, table)

    def test_sizes_must_be_positive(self) -> None:
        with pytest.raises(BackendError, match="must be positive"):
            _ops().attention.graph_metadata(0, MAX_BLOCKS, BLOCK, torch.device("cpu"))


@pytest.mark.parametrize("chunk_blocks", [1, 2, 3, MAX_BLOCKS])
def test_a_decode_step_walked_in_chunks_equals_one_pass(
    monkeypatch: pytest.MonkeyPatch, chunk_blocks: int
) -> None:
    from ayaka.ops.backends.torch import attention

    ops = _ops()
    buffers = _cache(KVLayoutKind.NHD)
    lengths, table = _step([5, 1, 12, 20])
    padded_lengths = torch.cat([lengths, torch.zeros(2, dtype=torch.int32)])
    padded_table = torch.cat([table, torch.zeros(2, MAX_BLOCKS, dtype=torch.int32)])
    # NaN in every slot no request has written: blocks 0 and 12-15, and the tails of 2, 3 and 9.
    for blocks in (buffers.k, buffers.v):
        blocks[0] = float("nan")
        blocks[12:] = float("nan")
        blocks[2, 1:] = float("nan")
        blocks[3, 1:] = float("nan")
    graph = ops.attention.graph_metadata(6, MAX_BLOCKS, BLOCK, torch.device("cpu"))
    ops.attention.update_graph_metadata(graph, padded_lengths, padded_table)
    q = torch.randn(6, 4, 8, generator=torch.Generator().manual_seed(2))
    whole = torch.empty_like(q)
    ops.attention.attend(q, buffers, graph, whole, scale=0.3)
    # 6 requests x 4 positions x 2 kv heads x 8 dims x 4 bytes = 1,536 bytes per block.
    monkeypatch.setattr(attention, "DECODE_CHUNK_BYTES", 1536 * chunk_blocks)
    chunked = torch.empty_like(q)

    ops.attention.attend(q, buffers, graph, chunked, scale=0.3)

    assert torch.isfinite(chunked).all()
    assert torch.count_nonzero(chunked[4:]) == 0
    torch.testing.assert_close(chunked, whole, rtol=1e-6, atol=1e-6)


@pytest.mark.gpu
def test_a_decode_step_over_a_long_table_stays_within_the_chunk_memory() -> None:
    """#122: 4 requests whose table covers 131,072 tokens, as Llama-3.2-1B's does.

    Gathering the whole table in float32 peaked at 3,364,814,848 bytes on an NVIDIA L4. The
    table points at a few real blocks again and again, so the cache itself is small.
    """
    if not torch.cuda.is_available():
        pytest.skip("needs a CUDA device")
    from ayaka.ops.backends.torch import attention

    device = torch.device("cuda")
    ops = load_backend(["torch"], device="cuda", dtype=DType.FP16)
    block, kv_heads, dim, width = 16, 8, 64, 131_072 // 16
    keys = torch.randn(8, block, kv_heads, dim, device=device, dtype=torch.float16)
    buffers = KVBuffers(keys, torch.randn_like(keys), KVLayoutKind.NHD, RAW_CODEC)
    table = (torch.arange(4 * width, dtype=torch.int32) % 7 + 1).reshape(4, width)
    lengths = torch.tensor([131_072, 70_000, 17, 0], dtype=torch.int32)  # the host's copy
    graph = ops.attention.graph_metadata(4, width, block, device)
    ops.attention.update_graph_metadata(graph, lengths, table)
    q = torch.randn(4, 32, dim, device=device, dtype=torch.float16)
    out = torch.empty_like(q)
    torch.cuda.synchronize()
    before = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()

    ops.attention.attend(q, buffers, graph, out, scale=dim**-0.5)
    torch.cuda.synchronize()

    peak = torch.cuda.max_memory_allocated() - before
    assert peak < 8 * attention.DECODE_CHUNK_BYTES, f"decode scratch peaked at {peak} bytes"
    assert torch.isfinite(out).all()
    assert torch.count_nonzero(out[3]) == 0
