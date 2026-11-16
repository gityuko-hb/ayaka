"""The attention layer over the paged KV cache, and the forward context it reads.

The oracle runs plain contiguous causal attention over each request's whole sequence at once
(``scaled_dot_product_attention(..., is_causal=True)`` on float32 copies of the layer's
parameters). The layer sees the same sequences in pieces, a prefill chunk, then decode rows and
verify windows, with its keys and values stored in blocks handed out in scrambled order. Every
slot nobody wrote holds NaN, so reading a wrong block shows up as NaN. The two must agree.

Tolerance: ``factor * eps * max|y|`` for the dtype's ``eps``. The layer rounds after each of its
four stages (projection, attention, projection) and the oracle does not, so ``factor`` is the
number of stages with some room for accumulation.
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, Any

import pytest

from ayaka.core.errors import EngineError
from ayaka.core.types import ForwardMode, KVLayoutKind
from ayaka.ops.backend import load_backend
from ayaka.ops.kvcache import RAW_CODEC, KVBuffers
from ayaka.utils.dtypes import DType

torch: Any = pytest.importorskip("torch")

from torch.nn import functional  # noqa: E402  # needs the importorskip above

from ayaka.layers.attention.dense import Attention  # noqa: E402
from ayaka.layers.context import ForwardContext  # noqa: E402

if TYPE_CHECKING:
    from collections.abc import Sequence

    from ayaka.ops.backend import OpsBackend

pytestmark = pytest.mark.unit

DTYPES = [torch.float16, torch.bfloat16, torch.float32]
DTYPE_IDS = ["fp16", "bf16", "fp32"]
LAYOUTS = [KVLayoutKind.NHD, KVLayoutKind.HND]
_REGISTRY_DTYPE = {torch.float16: DType.FP16, torch.bfloat16: DType.BF16, torch.float32: DType.FP32}

BLOCK, HEAD_DIM, HIDDEN = 4, 8, 32
LAYER_ID = 3


def _ops(dtype: Any = torch.float32) -> OpsBackend:
    return load_backend(["torch"], device="cpu", dtype=_REGISTRY_DTYPE[dtype])


def _ints(*values: int) -> Any:
    return torch.tensor(values, dtype=torch.int32)


def _layer(
    dtype: Any, *, heads: int = 4, kv_heads: int = 2, bias: bool = True, ops: Any = None
) -> Attention:
    layer = Attention(
        HIDDEN,
        heads,
        kv_heads,
        HEAD_DIM,
        layer_id=LAYER_ID,
        bias=bias,
        ops=ops or _ops(dtype),
        dtype=dtype,
    )
    generator = torch.Generator().manual_seed(5)
    for parameter in layer.parameters():
        parameter.copy_(torch.randn(parameter.shape, generator=generator) * 0.3)
    return layer


def _cache(dtype: Any, layout: KVLayoutKind, blocks: int, kv_heads: int) -> KVBuffers:
    shape = (
        (blocks, BLOCK, kv_heads, HEAD_DIM)
        if layout is KVLayoutKind.NHD
        else (blocks, kv_heads, BLOCK, HEAD_DIM)
    )
    nan = float("nan")
    return KVBuffers(
        torch.full(shape, nan, dtype=dtype), torch.full(shape, nan, dtype=dtype), layout, RAW_CODEC
    )


def _reference(layer: Attention, sequence: Any) -> Any:
    """Contiguous causal attention over a whole ``[T, hidden]`` sequence, in float32."""
    heads, kv_heads, steps = layer.num_heads, layer.num_kv_heads, sequence.shape[0]
    qkv = functional.linear(
        sequence.float(),
        layer.qkv_proj.weight.float(),
        None if layer.qkv_proj.bias is None else layer.qkv_proj.bias.float(),
    )
    q, k, v = qkv.split([heads * HEAD_DIM, kv_heads * HEAD_DIM, kv_heads * HEAD_DIM], dim=-1)

    def as_heads(x: Any, count: int) -> Any:
        return x.reshape(1, steps, count, HEAD_DIM).transpose(1, 2)

    attended = functional.scaled_dot_product_attention(
        as_heads(q, heads),
        as_heads(k, kv_heads),
        as_heads(v, kv_heads),
        is_causal=True,
        scale=layer.scale,
        enable_gqa=True,
    )
    return functional.linear(
        attended.transpose(1, 2).reshape(steps, heads * HEAD_DIM),
        layer.o_proj.weight.float(),
        None if layer.o_proj.bias is None else layer.o_proj.bias.float(),
    )


def _close(actual: Any, expected: Any, dtype: Any, factor: float) -> None:
    eps = torch.finfo(dtype).eps
    tolerance = factor * eps * max(float(expected.abs().max()), 1.0)
    torch.testing.assert_close(actual.float(), expected, rtol=0.0, atol=tolerance)


def _run_steps(
    layer: Attention,
    dtype: Any,
    layout: KVLayoutKind,
    steps: Sequence[Sequence[tuple[str, int, int]]],
    *,
    lengths: dict[str, int],
    tables: dict[str, list[int]],
    padding_rows: int = 2,
) -> dict[str, Any]:
    """Run the steps; each entry is (request, first position, rows).

    Returns ``(layer output, contiguous reference)`` for the whole sequence of each request.
    """
    ops = _ops(dtype)
    generator = torch.Generator().manual_seed(9)
    sequences = {
        name: torch.randn(length, HIDDEN, generator=generator).to(dtype)
        for name, length in lengths.items()
    }
    blocks = 1 + sum(len(table) for table in tables.values())
    buffers = _cache(dtype, layout, blocks + 2, layer.num_kv_heads)
    outputs: dict[str, list[Any]] = {name: [] for name in lengths}
    for step in steps:
        width = max(len(tables[name]) for name, _, _ in step)
        block_rows = torch.tensor(
            [
                *([*tables[name], *([0] * width)][:width] for name, _, _ in step),
                [0] * width,  # the padding request
            ],
            dtype=torch.int32,
        )
        meta = ops.attention.plan(
            ForwardMode.MIXED,
            _ints(*(count for _, _, count in step), padding_rows),
            _ints(*(first + count for _, first, count in step), 0),
            block_rows,
            BLOCK,
        )
        context = ForwardContext(
            attention=meta, kv_buffers={0: _cache(dtype, layout, 2, 1), LAYER_ID: buffers}
        )
        hidden = torch.cat(
            [
                *(sequences[name][first : first + count] for name, first, count in step),
                torch.randn(padding_rows, HIDDEN, generator=generator).to(dtype),
            ]
        )

        out = layer(hidden, context)

        start = 0
        for name, _, count in step:
            outputs[name].append(out[start : start + count])
            start += count
        assert torch.isfinite(out[:start]).all()
        # Padding rows still pass through the projections, but attention contributes zeros.
        expected_padding = layer.o_proj(
            torch.zeros(padding_rows, layer.num_heads * HEAD_DIM, dtype=dtype)
        )
        assert torch.equal(out[start:], expected_padding)
    layer_outputs = {name: torch.cat(parts) for name, parts in outputs.items()}
    reference = {name: _reference(layer, sequences[name]) for name in lengths}
    # Pair each request's output with the reference rows it covers: the steps cover every position.
    return {name: (layer_outputs[name], reference[name]) for name in lengths}


@pytest.mark.parametrize("layout", LAYOUTS, ids=["nhd", "hnd"])
@pytest.mark.parametrize(
    ("dtype", "factor"),
    [(torch.float32, 8.0), (torch.bfloat16, 32.0), (torch.float16, 32.0)],
    ids=["fp32", "bf16", "fp16"],
)
class TestPagedAttentionEqualsContiguousAttention:
    def test_prefill_chunks_decode_and_verify_windows(
        self, dtype: Any, factor: float, layout: KVLayoutKind
    ) -> None:
        layer = _layer(dtype)
        # A: a 6-row chunk, a 4-row chunk that starts inside a block, a decode row, a window of
        # 1 + 4 rows. B: a prompt of exactly one block, then a decode row and a window of 1 + 3.
        steps = [
            [("A", 0, 6), ("B", 0, 4)],
            [("A", 6, 4), ("B", 4, 1)],
            [("A", 10, 1), ("B", 5, 4)],
            [("A", 11, 5)],
        ]

        result = _run_steps(
            layer,
            dtype,
            layout,
            steps,
            lengths={"A": 16, "B": 9},
            tables={"A": [6, 2, 7, 3], "B": [5, 1, 4]},
        )

        for name, (paged, contiguous) in result.items():
            assert paged.shape == contiguous.shape, name
            _close(paged, contiguous, dtype, factor)


@pytest.mark.parametrize(("heads", "kv_heads"), [(4, 4), (4, 2), (6, 1)])
def test_multi_head_grouped_query_and_multi_query_attention(heads: int, kv_heads: int) -> None:
    layer = _layer(torch.float32, heads=heads, kv_heads=kv_heads, bias=False)

    result = _run_steps(
        layer,
        torch.float32,
        KVLayoutKind.NHD,
        [[("A", 0, 7)], [("A", 7, 1)], [("A", 8, 4)]],
        lengths={"A": 12},
        tables={"A": [4, 1, 3]},
    )

    paged, contiguous = result["A"]
    _close(paged, contiguous, torch.float32, 8.0)


def test_decode_with_contexts_that_are_exactly_a_block_and_a_multiple_of_blocks() -> None:
    layer = _layer(torch.float32)
    # Build the caches by prefilling, then decode one row each: lengths 5 and 9 after the step.
    result = _run_steps(
        layer,
        torch.float32,
        KVLayoutKind.NHD,
        [[("A", 0, 4), ("B", 0, 8)], [("A", 4, 1), ("B", 8, 1)]],
        lengths={"A": 5, "B": 9},
        tables={"A": [2, 1], "B": [3, 4, 5]},
    )

    for paged, contiguous in result.values():
        _close(paged, contiguous, torch.float32, 8.0)


class TestForwardContext:
    def _meta(self) -> Any:
        return _ops().attention.plan(
            ForwardMode.DECODE, _ints(1), _ints(3), torch.tensor([[1]]), BLOCK
        )

    def test_it_exposes_the_step_metadata(self) -> None:
        meta = self._meta()
        context = ForwardContext(attention=meta, kv_buffers={})

        assert context.mode is ForwardMode.DECODE
        assert context.positions is meta.positions
        assert context.attention is meta

    def test_it_is_frozen(self) -> None:
        context = ForwardContext(attention=self._meta(), kv_buffers={})

        with pytest.raises(dataclasses.FrozenInstanceError):
            context.attention = self._meta()  # type: ignore[misc]  # frozen on purpose

    def test_the_buffers_mapping_is_a_read_only_copy(self) -> None:
        buffers = _cache(torch.float32, KVLayoutKind.NHD, 3, 2)
        source = {1: buffers}
        context = ForwardContext(attention=self._meta(), kv_buffers=source)

        source[2] = buffers
        source.pop(1)

        assert list(context.kv_buffers) == [1]
        assert context.buffers(1) is buffers
        with pytest.raises(TypeError):
            context.kv_buffers[5] = buffers  # type: ignore[index]  # read-only on purpose

    def test_a_layer_without_buffers_is_an_engine_error_that_lists_the_ones_it_has(self) -> None:
        context = ForwardContext(
            attention=self._meta(), kv_buffers={0: _cache(torch.float32, KVLayoutKind.NHD, 3, 2)}
        )

        with pytest.raises(EngineError, match=r"no KV buffers for layer 7.*layers \[0\]"):
            context.buffers(7)


class TestAttentionLayer:
    def test_parameters_are_the_fused_projection_and_the_output_projection(self) -> None:
        layer = _layer(torch.float32, heads=6, kv_heads=2)

        shapes = {name: tuple(p.shape) for name, p in layer.named_parameters()}

        assert shapes == {
            "qkv_proj.weight": ((6 + 2 * 2) * HEAD_DIM, HIDDEN),
            "qkv_proj.bias": ((6 + 2 * 2) * HEAD_DIM,),
            "o_proj.weight": (HIDDEN, 6 * HEAD_DIM),
            "o_proj.bias": (HIDDEN,),
        }

    def test_kv_heads_must_divide_the_query_heads(self) -> None:
        with pytest.raises(ValueError, match=r"num_kv_heads \(4\) must divide num_heads \(6\)"):
            _layer(torch.float32, heads=6, kv_heads=4)

    def test_a_context_without_this_layers_buffers_is_refused(self) -> None:
        layer = _layer(torch.float32)
        meta = _ops().attention.plan(
            ForwardMode.DECODE, _ints(1), _ints(1), torch.tensor([[1]]), BLOCK
        )
        context = ForwardContext(attention=meta, kv_buffers={})

        with pytest.raises(EngineError, match="no KV buffers for layer 3"):
            layer(torch.zeros(1, HIDDEN), context)

    def test_it_stores_this_steps_keys_and_values_before_it_attends(self) -> None:
        calls: list[str] = []
        real = _ops()

        class Spy:
            def __init__(self, name: str, inner: Any) -> None:
                self.name, self.inner = name, inner

            def __getattr__(self, op: str) -> Any:
                target = getattr(self.inner, op)

                def call(*args: Any, **kwargs: Any) -> Any:
                    calls.append(f"{self.name}.{op}")
                    return target(*args, **kwargs)

                return call

        ops = dataclasses.replace(
            real,
            kvcache=Spy("kvcache", real.kvcache),
            attention=Spy("attention", real.attention),
        )
        layer = _layer(torch.float32, ops=ops)
        meta = real.attention.plan(
            ForwardMode.DECODE, _ints(1), _ints(2), torch.tensor([[1]]), BLOCK
        )
        context = ForwardContext(
            attention=meta,
            kv_buffers={LAYER_ID: _cache(torch.float32, KVLayoutKind.NHD, 3, 2)},
        )
        context.buffers(LAYER_ID).k[1].zero_()
        context.buffers(LAYER_ID).v[1].zero_()

        layer(torch.ones(1, HIDDEN), context)

        assert calls == ["kvcache.store", "attention.attend"]

    def test_it_does_not_modify_its_input(self) -> None:
        layer = _layer(torch.float32)
        ops = _ops()
        meta = ops.attention.plan(
            ForwardMode.PREFILL, _ints(3), _ints(3), torch.tensor([[1]]), BLOCK
        )
        context = ForwardContext(
            attention=meta,
            kv_buffers={LAYER_ID: _cache(torch.float32, KVLayoutKind.NHD, 3, 2)},
        )
        hidden = torch.randn(3, HIDDEN)
        before = hidden.clone()

        out = layer(hidden, context)

        assert torch.equal(hidden, before)
        assert out.shape == (3, HIDDEN)

    def test_the_context_module_does_not_import_torch(self) -> None:
        import subprocess
        import sys

        program = (
            "import sys\n"
            "from ayaka.layers.context import ForwardContext\n"
            "assert 'torch' not in sys.modules, 'torch was imported'\n"
        )

        result = subprocess.run(  # noqa: S603  # fixed argv, no input from outside
            [sys.executable, "-I", "-c", program], capture_output=True, text=True, check=False
        )

        assert result.returncode == 0, result.stderr
