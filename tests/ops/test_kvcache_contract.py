"""KV-cache ops on the torch backend, and the store-then-attend path through the paged cache.

``store`` is checked by reading the cache back through direct indexing, not through ``gather``.
The last class runs a whole generation, a prefill chunk and then decode and verify steps, with
only ``plan``, ``store`` and ``attend`` touching the cache, and compares each step with plain
contiguous attention over the same keys and values.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

from ayaka.core.errors import BackendError
from ayaka.core.types import ForwardMode, KVLayoutKind
from ayaka.ops.backend import load_backend
from ayaka.ops.kvcache import RAW_CODEC, KVBuffers
from ayaka.utils.dtypes import DType

torch: Any = pytest.importorskip("torch")

if TYPE_CHECKING:
    from ayaka.ops.backend import OpsBackend

pytestmark = pytest.mark.unit

DTYPES = [torch.float16, torch.bfloat16, torch.float32]
DTYPE_IDS = ["fp16", "bf16", "fp32"]
LAYOUTS = [KVLayoutKind.NHD, KVLayoutKind.HND]
_REGISTRY_DTYPE = {torch.float16: DType.FP16, torch.bfloat16: DType.BF16, torch.float32: DType.FP32}

BLOCK, BLOCKS, KV_HEADS, HEAD_DIM = 4, 8, 2, 8


def _ops(dtype: Any = torch.float32) -> OpsBackend:
    return load_backend(["torch"], device="cpu", dtype=_REGISTRY_DTYPE[dtype])


def _ints(*values: int) -> Any:
    return torch.tensor(values, dtype=torch.int32)


def _cache(
    dtype: Any, layout: KVLayoutKind, *, blocks: int = BLOCKS, fill: float = 0.0
) -> KVBuffers:
    shape = (
        (blocks, BLOCK, KV_HEADS, HEAD_DIM)
        if layout is KVLayoutKind.NHD
        else (blocks, KV_HEADS, BLOCK, HEAD_DIM)
    )
    return KVBuffers(
        torch.full(shape, fill, dtype=dtype),
        torch.full(shape, fill, dtype=dtype),
        layout,
        RAW_CODEC,
    )


def _nhd(tensor: Any, layout: KVLayoutKind) -> Any:
    return tensor if layout is KVLayoutKind.NHD else tensor.permute(0, 2, 1, 3)


def _rows(count: int, dtype: Any, seed: int) -> Any:
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(count, KV_HEADS, HEAD_DIM, generator=generator).to(dtype)


def test_the_reference_backend_prefers_nhd() -> None:
    assert _ops().kvcache.layout() is KVLayoutKind.NHD


class TestKVBuffers:
    @pytest.mark.parametrize("layout", LAYOUTS, ids=["nhd", "hnd"])
    def test_dimensions_follow_the_layout(self, layout: KVLayoutKind) -> None:
        buffers = _cache(torch.float32, layout, blocks=5)

        assert (buffers.num_blocks, buffers.block_size) == (5, BLOCK)
        assert (buffers.num_kv_heads, buffers.head_dim) == (KV_HEADS, HEAD_DIM)


@pytest.mark.parametrize("dtype", DTYPES, ids=DTYPE_IDS)
@pytest.mark.parametrize("layout", LAYOUTS, ids=["nhd", "hnd"])
class TestStore:
    def test_rows_land_in_their_slots_and_nothing_else_changes(
        self, dtype: Any, layout: KVLayoutKind
    ) -> None:
        ops, buffers = _ops(dtype), _cache(dtype, layout, fill=-1.0)
        # Request 0 has 9 tokens in blocks 5, 2, 6 and 3 rows; request 1 has 2 tokens in block 7.
        meta = ops.attention.plan(
            ForwardMode.MIXED, _ints(3, 1), _ints(9, 2), torch.tensor([[5, 2, 6], [7, 0, 0]]), BLOCK
        )
        k, v = _rows(4, dtype, seed=1), _rows(4, dtype, seed=2)

        ops.kvcache.store(k, v, buffers, meta)

        nhd_k, nhd_v = _nhd(buffers.k, layout), _nhd(buffers.v, layout)
        # Request 0 holds 9 tokens and its three rows are positions 6, 7, 8: (block 2, 2),
        # (block 2, 3), (block 6, 0). Request 1's row is position 1: (block 7, 1).
        for row, (block, offset) in enumerate([(2, 2), (2, 3), (6, 0), (7, 1)]):
            assert torch.equal(nhd_k[block, offset], k[row])
            assert torch.equal(nhd_v[block, offset], v[row])
        written = {(2, 2), (2, 3), (6, 0), (7, 1)}
        for block in range(BLOCKS):
            for offset in range(BLOCK):
                if (block, offset) not in written:
                    assert (nhd_k[block, offset] == -1.0).all()
                    assert (nhd_v[block, offset] == -1.0).all()

    def test_padding_rows_only_touch_the_padding_block(
        self, dtype: Any, layout: KVLayoutKind
    ) -> None:
        ops, buffers = _ops(dtype), _cache(dtype, layout, fill=-1.0)
        meta = ops.attention.plan(
            ForwardMode.MIXED, _ints(1, 3, 1), _ints(2, 0, 0), torch.tensor([[3], [0], [0]]), BLOCK
        )
        k, v = _rows(5, dtype, seed=3), _rows(5, dtype, seed=4)

        ops.kvcache.store(k, v, buffers, meta)

        nhd_k = _nhd(buffers.k, layout)
        assert torch.equal(nhd_k[3, 1], k[0])
        assert (nhd_k[1:3] == -1.0).all()
        assert (nhd_k[4:] == -1.0).all()
        assert (nhd_k[3, [0, 2, 3]] == -1.0).all()

    def test_gather_copies_whole_blocks_in_the_requested_order(
        self, dtype: Any, layout: KVLayoutKind
    ) -> None:
        ops, buffers = _ops(dtype), _cache(dtype, layout)
        buffers.k.copy_(torch.randn(buffers.k.shape).to(dtype))
        buffers.v.copy_(torch.randn(buffers.v.shape).to(dtype))
        out_k, out_v = (
            torch.empty(3, *buffers.k.shape[1:], dtype=dtype),
            torch.empty(3, *buffers.v.shape[1:], dtype=dtype),
        )

        ops.kvcache.gather(buffers, torch.tensor([5, 1, 5]), out_k, out_v)

        assert torch.equal(out_k, buffers.k[[5, 1, 5]])
        assert torch.equal(out_v, buffers.v[[5, 1, 5]])

    def test_copy_blocks_copies_keys_and_values(self, dtype: Any, layout: KVLayoutKind) -> None:
        ops, buffers = _ops(dtype), _cache(dtype, layout)
        buffers.k.copy_(torch.randn(buffers.k.shape).to(dtype))
        buffers.v.copy_(torch.randn(buffers.v.shape).to(dtype))
        before_k, before_v = buffers.k.clone(), buffers.v.clone()

        ops.kvcache.copy_blocks(buffers, torch.tensor([1, 2]), torch.tensor([6, 7]))

        assert torch.equal(buffers.k[[6, 7]], before_k[[1, 2]])
        assert torch.equal(buffers.v[[6, 7]], before_v[[1, 2]])
        untouched = [0, 1, 2, 3, 4, 5]
        assert torch.equal(buffers.k[untouched], before_k[untouched])
        assert torch.equal(buffers.v[untouched], before_v[untouched])

    def test_zero_blocks_clears_only_the_listed_blocks(
        self, dtype: Any, layout: KVLayoutKind
    ) -> None:
        ops, buffers = _ops(dtype), _cache(dtype, layout, fill=3.0)

        ops.kvcache.zero_blocks(buffers, torch.tensor([2, 4], dtype=torch.int32))

        for tensor in (buffers.k, buffers.v):
            assert (tensor[[2, 4]] == 0).all()
            assert (tensor[[0, 1, 3, 5, 6, 7]] == 3.0).all()


class TestRejections:
    def _plan(self, ops: OpsBackend) -> Any:
        return ops.attention.plan(
            ForwardMode.DECODE, _ints(1), _ints(2), torch.tensor([[1]]), BLOCK
        )

    def test_the_layout_must_be_one_the_backend_serves(self) -> None:
        ops = _ops()
        buffers = _cache(torch.float32, KVLayoutKind.NHD)._replace(layout=KVLayoutKind.NLD)
        rows = _rows(1, torch.float32, seed=0)

        with pytest.raises(BackendError, match="serves the NHD and HND layouts"):
            ops.kvcache.store(rows, rows, buffers, self._plan(ops))

    def test_the_codec_must_be_raw(self) -> None:
        ops = _ops()
        buffers = _cache(torch.float32, KVLayoutKind.NHD)._replace(codec="fp8")
        rows = _rows(1, torch.float32, seed=0)

        with pytest.raises(BackendError, match=r"stores \['raw', 'fp8_e4m3'\] codecs, not 'fp8'"):
            ops.kvcache.store(rows, rows, buffers, self._plan(ops))

    def test_values_must_match_the_keys_buffer(self) -> None:
        ops = _ops()
        buffers = _cache(torch.float32, KVLayoutKind.NHD)
        buffers = buffers._replace(v=torch.zeros(BLOCKS, BLOCK, KV_HEADS, HEAD_DIM + 1))
        rows = _rows(1, torch.float32, seed=0)

        with pytest.raises(BackendError, match=r"buffers.v must be float32 \[8, 4, 2, 8\]"):
            ops.kvcache.store(rows, rows, buffers, self._plan(ops))

    def test_rows_must_match_the_slot_mapping_and_the_cache_dtype(self) -> None:
        ops = _ops()
        buffers = _cache(torch.float32, KVLayoutKind.NHD)
        meta = self._plan(ops)

        with pytest.raises(
            BackendError, match=r"k must be float32 \[1, 2, 8\], got float32 \[2, 2, 8\]"
        ):
            ops.kvcache.store(_rows(2, torch.float32, 0), _rows(1, torch.float32, 0), buffers, meta)
        with pytest.raises(BackendError, match=r"v must be float32 \[1, 2, 8\], got float16"):
            ops.kvcache.store(_rows(1, torch.float32, 0), _rows(1, torch.float16, 0), buffers, meta)

    def test_the_step_must_be_planned_for_the_block_size_of_the_cache(self) -> None:
        ops = _ops()
        buffers = _cache(torch.float32, KVLayoutKind.NHD)
        meta = ops.attention.plan(ForwardMode.DECODE, _ints(1), _ints(2), torch.tensor([[1]]), 2)
        rows = _rows(1, torch.float32, seed=0)

        with pytest.raises(BackendError, match="planned for blocks of 2 positions"):
            ops.kvcache.store(rows, rows, buffers, meta)

    def test_block_ids_must_be_integers(self) -> None:
        buffers = _cache(torch.float32, KVLayoutKind.NHD)

        with pytest.raises(BackendError, match="block_ids must be an int32 or int64 tensor"):
            _ops().kvcache.zero_blocks(buffers, torch.tensor([1.0]))

    def test_copy_needs_as_many_destinations_as_sources(self) -> None:
        buffers = _cache(torch.float32, KVLayoutKind.NHD)

        with pytest.raises(BackendError, match="src has 2 blocks but dst has 1"):
            _ops().kvcache.copy_blocks(buffers, torch.tensor([1, 2]), torch.tensor([3]))

    def test_gather_output_must_have_one_block_per_id(self) -> None:
        buffers = _cache(torch.float32, KVLayoutKind.NHD)
        wrong = torch.empty(1, BLOCK, KV_HEADS, HEAD_DIM)

        with pytest.raises(BackendError, match=r"out_k must be float32 \[2, 4, 2, 8\]"):
            _ops().kvcache.gather(buffers, torch.tensor([1, 2]), wrong, wrong.clone())


def _contiguous(q: Any, keys: Any, values: Any, scale: float) -> Any:
    """Plain causal attention of the last ``len(q)`` positions, over a flat sequence, in float64."""
    rows, heads, _ = q.shape
    length = keys.shape[0]
    group = heads // keys.shape[1]
    k, v = (
        keys.double().repeat_interleave(group, dim=1),
        values.double().repeat_interleave(group, dim=1),
    )
    scores = torch.einsum("rhd,thd->hrt", q.double(), k) * scale
    positions = torch.arange(length - rows, length)[:, None]
    scores = scores.masked_fill(torch.arange(length)[None, :] > positions, float("-inf"))
    return torch.einsum("hrt,thd->rhd", torch.softmax(scores, dim=-1), v)


@pytest.mark.parametrize("layout", LAYOUTS, ids=["nhd", "hnd"])
def test_paged_attention_over_a_whole_generation_equals_contiguous_attention(
    layout: KVLayoutKind,
) -> None:
    """Prefill a prompt in chunks, decode, then verify a window, for two requests that interleave.

    Request A has a 10-token prompt prefilled as 6 + 4 rows, so its second chunk starts in the
    middle of a block. Request B's prompt is 4 tokens, exactly one block. Blocks are handed out
    in a scrambled order and the batch is padded with a padding request in every step.
    """
    ops, dtype, heads = _ops(), torch.float32, 4
    generator = torch.Generator().manual_seed(11)
    total = 16
    sequences = {
        name: (
            torch.randn(total, KV_HEADS, HEAD_DIM, generator=generator),
            torch.randn(total, KV_HEADS, HEAD_DIM, generator=generator),
        )
        for name in "AB"
    }
    queries = {name: torch.randn(total, heads, HEAD_DIM, generator=generator) for name in "AB"}
    tables = {"A": [6, 2, 7, 3], "B": [5, 1, 4, 8]}
    buffers = _cache(dtype, layout, blocks=9, fill=float("nan"))
    scale = HEAD_DIM**-0.5

    # (request, first position, rows): two prefill chunks, then a decode row next to a verify
    # window of 1 + 3 rows, then a window of 1 + 4 rows alone.
    steps = [
        [("A", 0, 6), ("B", 0, 4)],
        [("A", 6, 4), ("B", 4, 1)],
        [("A", 10, 1), ("B", 5, 4)],
        [("A", 11, 5)],
    ]
    for step in steps:
        rows = [*(rows for _, _, rows in step), 2]  # a padding request of two rows
        lengths = [*(first + count for _, first, count in step), 0]
        width = max(len(tables[name]) for name, *_ in step)
        block_rows = torch.tensor(
            [*([*tables[name], 0, 0, 0, 0][:width] for name, *_ in step), [0] * width],
            dtype=torch.int32,
        )
        meta = ops.attention.plan(
            ForwardMode.MIXED, _ints(*rows), _ints(*lengths), block_rows, BLOCK
        )

        keys, values = (
            torch.cat([sequences[name][i][first : first + count] for name, first, count in step])
            for i in range(2)
        )
        q = torch.cat([queries[name][first : first + count] for name, first, count in step])
        padding = torch.randn(2, KV_HEADS, HEAD_DIM, generator=generator)
        ops.kvcache.store(torch.cat([keys, padding]), torch.cat([values, padding]), buffers, meta)
        out = torch.empty(q.shape[0] + 2, heads, HEAD_DIM)

        ops.attention.attend(
            torch.cat([q, torch.randn(2, heads, HEAD_DIM)]), buffers, meta, out, scale=scale
        )

        start = 0
        for name, first, count in step:
            context_k, context_v = (sequences[name][i][: first + count] for i in range(2))
            expected = _contiguous(
                queries[name][first : first + count], context_k, context_v, scale
            )
            torch.testing.assert_close(
                out[start : start + count].double(), expected, rtol=0.0, atol=1e-5
            )
            start += count
        assert torch.count_nonzero(out[start:]) == 0
