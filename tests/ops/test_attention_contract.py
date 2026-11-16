"""Attention ops on the torch backend: planning a step and attending over the paged cache.

The oracle is plain contiguous attention in float64, written for these tests, over the keys
and values of each request as a flat ``[length, kv_heads, head_dim]`` sequence. The paged
cache is filled by direct indexing, not by ``store``, so a bug in ``store`` cannot hide a bug
in ``attend``. Every slot nobody wrote holds NaN: reading block 0, an unused block or the
unwritten tail of a request's last block turns the result into NaN.

Tolerance. ``attend`` computes in fp32 and rounds the result once, so against the float64
oracle on the same inputs the error is that rounding plus fp32 accumulation: at most
``4 * eps * max|y|`` for the dtype's ``eps``, with a small absolute floor for fp32.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import pytest

from ayaka.core.errors import BackendError
from ayaka.core.types import ForwardMode, KVLayoutKind
from ayaka.ops.backend import load_backend
from ayaka.ops.kvcache import RAW_CODEC, KVBuffers
from ayaka.utils.dtypes import DType

torch: Any = pytest.importorskip("torch")

if TYPE_CHECKING:
    from collections.abc import Sequence

    from ayaka.ops.attention import AttentionMetadata
    from ayaka.ops.backend import OpsBackend

pytestmark = pytest.mark.unit

DTYPES = [torch.float16, torch.bfloat16, torch.float32]
DTYPE_IDS = ["fp16", "bf16", "fp32"]
LAYOUTS = [KVLayoutKind.NHD, KVLayoutKind.HND]
_REGISTRY_DTYPE = {torch.float16: DType.FP16, torch.bfloat16: DType.BF16, torch.float32: DType.FP32}

BLOCK = 4


def _ops(dtype: Any = torch.float32) -> OpsBackend:
    return load_backend(["torch"], device="cpu", dtype=_REGISTRY_DTYPE[dtype])


def _ints(*values: int) -> Any:
    return torch.tensor(values, dtype=torch.int32)


def _table(rows: Sequence[Sequence[int]]) -> Any:
    width = max((len(row) for row in rows), default=1)
    return torch.tensor([[*row, *([0] * (width - len(row)))] for row in rows], dtype=torch.int32)


@dataclass
class Step:
    """One step of one request: how many query rows, and the context length after them."""

    rows: int
    length: int  # 0 marks a padding request


@dataclass
class Scenario:
    buffers: KVBuffers
    meta: AttentionMetadata
    q: Any
    expected: Any
    keys: list[Any]
    values: list[Any]
    tables: list[list[int]]


def _build(
    steps: Sequence[Step],
    *,
    dtype: Any = torch.float32,
    layout: KVLayoutKind = KVLayoutKind.NHD,
    heads: int = 4,
    kv_heads: int = 2,
    head_dim: int = 8,
    mode: ForwardMode = ForwardMode.MIXED,
    seed: int = 0,
) -> Scenario:
    """Fill a paged cache with random contexts and compute the float64 oracle."""
    generator = torch.Generator().manual_seed(seed)

    def randn(*shape: int) -> Any:
        return torch.randn(*shape, generator=generator).to(dtype)

    blocks_needed = [-(-step.length // BLOCK) for step in steps]
    num_blocks = 1 + sum(blocks_needed) + 3  # block 0, the blocks in use and some spare
    shape = (
        (num_blocks, BLOCK, kv_heads, head_dim)
        if layout is KVLayoutKind.NHD
        else (num_blocks, kv_heads, BLOCK, head_dim)
    )
    cache_k = torch.full(shape, float("nan"), dtype=dtype)
    cache_v = torch.full(shape, float("nan"), dtype=dtype)
    buffers = KVBuffers(cache_k, cache_v, layout, RAW_CODEC)
    nhd_k = cache_k if layout is KVLayoutKind.NHD else cache_k.permute(0, 2, 1, 3)
    nhd_v = cache_v if layout is KVLayoutKind.NHD else cache_v.permute(0, 2, 1, 3)

    free = (torch.randperm(num_blocks - 1, generator=generator) + 1).tolist()  # never block 0
    tables: list[list[int]] = []
    keys: list[Any] = []
    values: list[Any] = []
    for step, needed in zip(steps, blocks_needed, strict=True):
        table = [free.pop() for _ in range(needed)]
        tables.append(table)
        context_k, context_v = (
            randn(step.length, kv_heads, head_dim),
            randn(step.length, kv_heads, head_dim),
        )
        keys.append(context_k)
        values.append(context_v)
        for position in range(step.length):
            block, offset = table[position // BLOCK], position % BLOCK
            nhd_k[block, offset] = context_k[position]
            nhd_v[block, offset] = context_v[position]

    q = randn(sum(step.rows for step in steps), heads, head_dim)
    meta = _ops(dtype).attention.plan(
        mode,
        _ints(*(step.rows for step in steps)),
        _ints(*(step.length for step in steps)),
        _table(tables),
        BLOCK,
    )
    expected = torch.zeros(q.shape, dtype=torch.float64)
    first = 0
    for step, context_k, context_v in zip(steps, keys, values, strict=True):
        if step.length:
            expected[first : first + step.rows] = _oracle(
                q[first : first + step.rows], context_k, context_v, scale=head_dim**-0.5
            )
        first += step.rows
    return Scenario(buffers, meta, q, expected, keys, values, tables)


def _oracle(q: Any, keys: Any, values: Any, *, scale: float) -> Any:
    """Causal attention of the last ``len(q)`` positions of one sequence, in float64."""
    rows, heads, _ = q.shape
    length, kv_heads, _ = keys.shape
    group = heads // kv_heads
    result = torch.empty(q.shape, dtype=torch.float64)
    for row in range(rows):
        visible = length - rows + row + 1  # this row's position, plus one
        k = keys[:visible].double().repeat_interleave(group, dim=1)  # [visible, heads, dim]
        v = values[:visible].double().repeat_interleave(group, dim=1)
        scores = torch.einsum("hd,thd->ht", q[row].double(), k) * scale
        result[row] = torch.einsum("ht,thd->hd", torch.softmax(scores, dim=-1), v)
    return result


def _attend(scenario: Scenario, dtype: Any = torch.float32) -> Any:
    out = torch.full_like(scenario.q, 7.0)
    head_dim = scenario.q.shape[-1]
    _ops(dtype).attention.attend(
        scenario.q, scenario.buffers, scenario.meta, out, scale=head_dim**-0.5
    )
    return out


def _assert_matches(out: Any, expected: Any, dtype: Any) -> None:
    eps = torch.finfo(dtype).eps
    atol = 4 * eps * float(expected.abs().max()) + (1e-6 if dtype is torch.float32 else 0.0)
    torch.testing.assert_close(out.double(), expected, rtol=0.0, atol=atol)


@pytest.mark.parametrize("dtype", DTYPES, ids=DTYPE_IDS)
@pytest.mark.parametrize("layout", LAYOUTS, ids=["nhd", "hnd"])
class TestAttend:
    def test_decode_with_a_context_of_exactly_one_block(
        self, dtype: Any, layout: KVLayoutKind
    ) -> None:
        scenario = _build(
            [Step(1, BLOCK), Step(1, BLOCK)], dtype=dtype, layout=layout, mode=ForwardMode.DECODE
        )

        _assert_matches(_attend(scenario, dtype), scenario.expected, dtype)

    @pytest.mark.parametrize("blocks", [2, 3])
    def test_decode_with_a_context_that_is_a_multiple_of_the_block_size(
        self, dtype: Any, layout: KVLayoutKind, blocks: int
    ) -> None:
        scenario = _build(
            [Step(1, blocks * BLOCK)], dtype=dtype, layout=layout, mode=ForwardMode.DECODE
        )

        _assert_matches(_attend(scenario, dtype), scenario.expected, dtype)

    @pytest.mark.parametrize("length", [1, BLOCK - 1, BLOCK + 1, 2 * BLOCK - 1, 2 * BLOCK + 1])
    def test_decode_with_a_context_that_ends_inside_a_block(
        self, dtype: Any, layout: KVLayoutKind, length: int
    ) -> None:
        scenario = _build([Step(1, length)], dtype=dtype, layout=layout, mode=ForwardMode.DECODE)

        _assert_matches(_attend(scenario, dtype), scenario.expected, dtype)

    @pytest.mark.parametrize(
        ("rows", "length"),
        [
            (BLOCK, BLOCK),  # a whole prompt that is exactly one block
            (5, 5),  # a whole prompt that spills into a second block
            (3, 11),  # a chunk after 8 cached tokens: the context is a multiple of the block
            (4, 12),  # a chunk of exactly one block, on a context of three
            (1 + 3, 9),  # a verify window: one sampled token and three drafts
        ],
    )
    def test_several_rows_per_request_are_causal(
        self, dtype: Any, layout: KVLayoutKind, rows: int, length: int
    ) -> None:
        scenario = _build([Step(rows, length)], dtype=dtype, layout=layout)

        _assert_matches(_attend(scenario, dtype), scenario.expected, dtype)

    def test_a_batch_mixes_prefill_chunks_decode_rows_and_verify_windows(
        self, dtype: Any, layout: KVLayoutKind
    ) -> None:
        steps = [Step(7, 7), Step(1, 9), Step(4, 16), Step(2, 2), Step(1, 1), Step(3, 6)]
        scenario = _build(steps, dtype=dtype, layout=layout)

        _assert_matches(_attend(scenario, dtype), scenario.expected, dtype)

    def test_padding_requests_get_zero_rows_and_leave_the_others_alone(
        self, dtype: Any, layout: KVLayoutKind
    ) -> None:
        steps = [Step(2, 6), Step(1, 0), Step(3, 0), Step(1, 5), Step(1, 0)]
        scenario = _build(steps, dtype=dtype, layout=layout)

        out = _attend(scenario, dtype)

        _assert_matches(out, scenario.expected, dtype)
        padding_rows = [2, 3, 4, 5, 7]
        assert torch.count_nonzero(out[padding_rows]) == 0
        assert torch.isfinite(out).all()


class TestGroupedQueryAttention:
    @pytest.mark.parametrize(("heads", "kv_heads"), [(4, 4), (4, 2), (6, 1), (8, 4)])
    def test_query_heads_share_the_key_value_head_they_divide_into(
        self, heads: int, kv_heads: int
    ) -> None:
        scenario = _build([Step(3, 10), Step(1, 6)], heads=heads, kv_heads=kv_heads)

        _assert_matches(_attend(scenario), scenario.expected, torch.float32)

    def test_a_head_count_that_is_not_a_multiple_is_refused(self) -> None:
        scenario = _build([Step(1, 3)], heads=4, kv_heads=2)
        q = torch.zeros(1, 3, 8)

        with pytest.raises(
            BackendError, match=r"3 heads, which is not a positive multiple of .* 2"
        ):
            _ops().attention.attend(q, scenario.buffers, scenario.meta, q.clone(), scale=1.0)


class TestCausality:
    def test_a_row_ignores_keys_after_its_position(self) -> None:
        scenario = _build([Step(4, 4)], mode=ForwardMode.PREFILL)
        before = _attend(scenario)
        # Overwrite the key and value of the last position: only the last row may change.
        block, offset = scenario.tables[0][3 // BLOCK], 3 % BLOCK
        scenario.buffers.k[block, offset] = 100.0
        scenario.buffers.v[block, offset] = -100.0

        after = _attend(scenario)

        assert torch.equal(after[:3], before[:3])
        assert not torch.equal(after[3], before[3])

    def test_a_single_position_attends_to_itself_exactly(self) -> None:
        scenario = _build([Step(1, 1)], heads=2, kv_heads=2)

        out = _attend(scenario)

        # softmax over one key is 1, so the output is that value.
        torch.testing.assert_close(out[0], scenario.values[0][0], rtol=0.0, atol=1e-6)


class TestEmptyAndOutput:
    def test_an_empty_step_is_a_no_op(self) -> None:
        buffers = KVBuffers(
            torch.zeros(3, BLOCK, 2, 8), torch.zeros(3, BLOCK, 2, 8), KVLayoutKind.NHD, RAW_CODEC
        )
        ops = _ops().attention
        meta = ops.plan(
            ForwardMode.IDLE, _ints(), _ints(), torch.zeros(0, 1, dtype=torch.int32), BLOCK
        )
        out = torch.empty(0, 4, 8)

        ops.attend(torch.empty(0, 4, 8), buffers, meta, out, scale=1.0)

        assert meta.slot_mapping.shape == (0,)
        assert out.shape == (0, 4, 8)

    def test_the_cache_is_only_read(self) -> None:
        scenario = _build([Step(3, 9)])
        k_before, v_before = scenario.buffers.k.clone(), scenario.buffers.v.clone()

        _attend(scenario)

        assert torch.equal(scenario.buffers.k.nan_to_num(), k_before.nan_to_num())
        assert torch.equal(scenario.buffers.v.nan_to_num(), v_before.nan_to_num())

    def test_the_scale_multiplies_the_scores(self) -> None:
        scenario = _build([Step(2, 6)])
        out = torch.empty_like(scenario.q)

        _ops().attention.attend(scenario.q, scenario.buffers, scenario.meta, out, scale=0.0)

        # Equal scores give a uniform average over the visible positions.
        for row, visible in enumerate([5, 6]):
            mean = scenario.values[0][:visible].mean(dim=0)  # [kv_heads, head_dim]
            torch.testing.assert_close(
                out[row], mean.repeat_interleave(2, dim=0), rtol=0.0, atol=1e-6
            )


class TestAttendRejections:
    def test_metadata_from_another_backend_is_refused(self) -> None:
        scenario = _build([Step(1, 3)])
        foreign = type(scenario.meta)(
            mode=scenario.meta.mode,
            block_size=BLOCK,
            rows_per_request=scenario.meta.rows_per_request,
            request_of_row=scenario.meta.request_of_row,
            positions=scenario.meta.positions,
            slot_mapping=scenario.meta.slot_mapping,
            kv_lengths=scenario.meta.kv_lengths,
            block_table=object(),
            plan=object(),
        )

        with pytest.raises(BackendError, match="planned by another backend"):
            _ops().attention.attend(
                scenario.q, scenario.buffers, foreign, torch.empty_like(scenario.q), scale=1.0
            )

    def test_the_block_size_must_match_the_cache(self) -> None:
        scenario = _build([Step(1, 3)])
        other = KVBuffers(
            torch.zeros(4, 2 * BLOCK, 2, 8),
            torch.zeros(4, 2 * BLOCK, 2, 8),
            KVLayoutKind.NHD,
            RAW_CODEC,
        )

        with pytest.raises(
            BackendError, match=r"blocks of 4 positions but the cache has blocks of 8"
        ):
            _ops().attention.attend(
                scenario.q, other, scenario.meta, torch.empty_like(scenario.q), scale=1.0
            )

    def test_q_must_have_one_row_per_planned_row(self) -> None:
        scenario = _build([Step(2, 5)])

        with pytest.raises(
            BackendError, match=r"q must be float32 \[2, 4, 8\], got float32 \[1, 4, 8\]"
        ):
            _ops().attention.attend(
                scenario.q[:1], scenario.buffers, scenario.meta, torch.empty(1, 4, 8), scale=1.0
            )

    def test_out_is_not_cast_into(self) -> None:
        scenario = _build([Step(1, 3)])

        with pytest.raises(BackendError, match=r"out must be float32 \[1, 4, 8\]"):
            _ops().attention.attend(
                scenario.q,
                scenario.buffers,
                scenario.meta,
                torch.empty(1, 4, 8, dtype=torch.float16),
                scale=1.0,
            )

    def test_a_cache_in_another_codec_is_refused(self) -> None:
        scenario = _build([Step(1, 3)])
        encoded = scenario.buffers._replace(codec="fp8")

        with pytest.raises(BackendError, match="stores the 'raw' codec, not 'fp8'"):
            _ops().attention.attend(
                scenario.q, encoded, scenario.meta, torch.empty_like(scenario.q), scale=1.0
            )


class TestPlan:
    def test_it_numbers_rows_positions_and_slots(self) -> None:
        # Request 0: a chunk of 2 rows after 3 cached tokens (length 5) in blocks 3 and 1.
        # Request 1: a padding request with 2 rows.   Request 2: one decode row, length 6.
        meta = _ops().attention.plan(
            ForwardMode.MIXED,
            _ints(2, 2, 1),
            _ints(5, 0, 6),
            _table([[3, 1], [0], [2, 4]]),
            BLOCK,
        )

        assert meta.request_of_row.tolist() == [0, 0, 1, 1, 2]
        assert meta.positions.tolist() == [3, 4, 0, 0, 5]
        # position 3 is slot 3 of block 3; position 4 is slot 0 of block 1; padding is slot 0;
        # position 5 is slot 1 of block 4.
        assert meta.slot_mapping.tolist() == [3 * 4 + 3, 1 * 4 + 0, 0, 0, 4 * 4 + 1]
        assert meta.rows_per_request.tolist() == [2, 2, 1]
        assert meta.kv_lengths.tolist() == [5, 0, 6]
        assert meta.block_size == BLOCK
        assert (meta.rows_per_request.dtype, meta.slot_mapping.dtype) == (torch.int32, torch.int64)

    def test_int64_inputs_are_accepted(self) -> None:
        meta = _ops().attention.plan(
            ForwardMode.DECODE,
            torch.tensor([1]),
            torch.tensor([3]),
            torch.tensor([[2]]),
            BLOCK,
        )

        assert meta.slot_mapping.tolist() == [2 * 4 + 2]

    def test_it_is_a_frozen_snapshot(self) -> None:
        meta = _ops().attention.plan(ForwardMode.DECODE, _ints(1), _ints(3), _table([[1]]), BLOCK)

        with pytest.raises(AttributeError):
            meta.mode = ForwardMode.PREFILL  # type: ignore[misc]  # frozen on purpose

    @pytest.mark.parametrize(
        ("mode", "rows", "lengths", "table", "message"),
        [
            (ForwardMode.DECODE, [2], [5], [[1, 2]], "DECODE step has one row per request"),
            (ForwardMode.IDLE, [1], [1], [[1]], "IDLE step has no rows"),
            (ForwardMode.MIXED, [4], [3], [[1]], "4 query rows but only 3 tokens"),
            (ForwardMode.MIXED, [-1], [3], [[1]], "neither can be negative"),
            (ForwardMode.MIXED, [1], [9], [[1, 2]], "needs 3 blocks of 4"),
            (ForwardMode.MIXED, [1], [5], [[1, 0]], "padding block 0 inside"),
        ],
    )
    def test_an_inconsistent_step_is_refused(
        self,
        mode: ForwardMode,
        rows: list[int],
        lengths: list[int],
        table: list[list[int]],
        message: str,
    ) -> None:
        with pytest.raises(BackendError, match=message):
            _ops().attention.plan(
                mode, torch.tensor(rows), torch.tensor(lengths), torch.tensor(table), BLOCK
            )

    def test_requests_must_agree_on_the_batch_size(self) -> None:
        with pytest.raises(BackendError, match=r"same requests, got 2, 1 and 2"):
            _ops().attention.plan(
                ForwardMode.MIXED, _ints(1, 1), _ints(2), _table([[1], [2]]), BLOCK
            )

    def test_the_indices_must_be_integers(self) -> None:
        with pytest.raises(BackendError, match="rows_per_request must be an int32 or int64"):
            _ops().attention.plan(ForwardMode.MIXED, torch.ones(1), _ints(2), _table([[1]]), BLOCK)

    def test_the_block_size_must_be_positive(self) -> None:
        with pytest.raises(BackendError, match="block_size must be positive"):
            _ops().attention.plan(ForwardMode.MIXED, _ints(1), _ints(1), _table([[1]]), 0)
