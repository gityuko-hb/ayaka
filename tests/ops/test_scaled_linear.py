"""The sliced 8-bit linear of the torch backend against "dequantize in float32, then multiply".

Tolerances fixed in #112 before any run: within ``1e-5`` in float32; in fp16 and bf16 within the
tolerance the unquantized linear already meets against float32. Block, channel and tensor
scales (a tensor scale reaches the op as one scale per row), and slices that do not divide the
rows.
"""

from __future__ import annotations

from typing import Any

import pytest

torch: Any = pytest.importorskip("torch")

from ayaka.core.errors import BackendError  # noqa: E402
from ayaka.ops.backends.torch import linear as ops  # noqa: E402
from fp8_support import dequantize, quantize  # noqa: E402  # needs the importorskip above

pytestmark = pytest.mark.unit

_LAYOUTS = [
    ("channel", None),
    ("tensor", None),
    ("block", (16, 16)),
    ("block", (8, 32)),
]
_IDS = [f"{layout}-{block}" for layout, block in _LAYOUTS]


def _case(
    out_features: int, in_features: int, layout: str, block: tuple[int, int] | None, seed: int
) -> tuple[Any, Any, Any, Any]:
    generator = torch.Generator().manual_seed(seed)
    weight = torch.randn(out_features, in_features, generator=generator)
    stored, scale = quantize(weight, layout, block or (16, 16))
    if layout == "tensor":  # the method repeats a tensor scale over its rows
        scale = scale.reshape(1, 1).expand(out_features, 1).contiguous()
    x = torch.randn(3, 5, in_features, generator=generator)
    bias = torch.randn(out_features, generator=generator)
    return stored, scale, x, bias


def _reference(stored: Any, scale: Any, block: tuple[int, int] | None, x: Any, bias: Any) -> Any:
    dense = dequantize(stored, scale, block)
    return torch.nn.functional.linear(x.to(torch.float64), dense.to(torch.float64), bias.double())


@pytest.mark.parametrize(("layout", "block"), _LAYOUTS, ids=_IDS)
@pytest.mark.parametrize(("out_features", "in_features"), [(64, 48), (50, 40), (7, 33)])
def test_float32_matches_dequantize_then_multiply(
    layout: str, block: tuple[int, int] | None, out_features: int, in_features: int
) -> None:
    stored, scale, x, bias = _case(out_features, in_features, layout, block, seed=1)
    out = x.new_empty((3, 5, out_features))

    ops.scaled_linear(x, stored, scale, bias, out, block=block)

    torch.testing.assert_close(
        out.double(), _reference(stored, scale, block, x, bias), rtol=1e-5, atol=1e-5
    )


@pytest.mark.parametrize(("layout", "block"), _LAYOUTS, ids=_IDS)
def test_slices_that_do_not_divide_the_rows_give_the_same_result(
    monkeypatch: pytest.MonkeyPatch, layout: str, block: tuple[int, int] | None
) -> None:
    stored, scale, x, bias = _case(72, 32, layout, block, seed=2)
    whole = x.new_empty((3, 5, 72))
    ops.scaled_linear(x, stored, scale, bias, whole, block=block)
    # 4 bytes x 32 columns x 20 rows: slices of 20 rows, or 16 / 8 for whole blocks.
    monkeypatch.setattr(ops, "SLICE_BYTES", 4 * 32 * 20)
    sliced = x.new_empty((3, 5, 72))

    ops.scaled_linear(x, stored, scale, bias, sliced, block=block)

    step = ops.slice_rows(72, 32, block)
    assert step < 72
    assert step % (block[0] if block else 1) == 0
    # Every row is dequantized with the same scales whichever slice holds it, bit for bit.
    pieces = [
        ops.dequantize_rows(stored, scale, start, min(start + step, 72), block)
        for start in range(0, 72, step)
    ]
    whole_rows = ops.dequantize_rows(stored, scale, 0, 72, block)
    torch.testing.assert_close(torch.cat(pieces), whole_rows, rtol=0, atol=0)
    # The products agree within #112's float32 tolerance: a BLAS may block fewer rows otherwise.
    torch.testing.assert_close(sliced, whole, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
@pytest.mark.parametrize(("layout", "block"), _LAYOUTS, ids=_IDS)
def test_half_precision_is_as_close_as_the_unquantized_linear(
    dtype: str, layout: str, block: tuple[int, int] | None
) -> None:
    compute = getattr(torch, dtype)
    stored, scale, x, bias = _case(64, 48, layout, block, seed=3)
    dense = dequantize(stored, scale, block)
    exact = _reference(stored, scale, block, x, bias)
    plain = x.new_empty((3, 5, 64), dtype=compute)
    ops.linear(x.to(compute), dense.to(compute), bias.to(compute), plain)
    tolerance = (plain.double() - exact).abs().max().item()
    out = x.new_empty((3, 5, 64), dtype=compute)

    ops.scaled_linear(x.to(compute), stored, scale, bias.to(compute), out, block=block)

    assert (out.double() - exact).abs().max().item() <= tolerance * 1.0 + 1e-12


def test_no_bias_and_a_two_dimensional_input() -> None:
    stored, scale, x, _ = _case(32, 16, "channel", None, seed=4)
    flat = x.reshape(-1, 16)
    out = flat.new_empty((15, 32))

    ops.scaled_linear(flat, stored, scale, None, out, block=None)

    expected = flat.double() @ dequantize(stored, scale).double().t()
    torch.testing.assert_close(out.double(), expected, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ("weight-dtype", "8-bit float"),
        ("scale-shape", "scale must be float32"),
        ("scale-dtype", "scale must be float32"),
        ("x-width", "features but weight takes"),
        ("out-strided", "contiguous"),
    ],
)
def test_bad_arguments_are_refused(change: str, message: str) -> None:
    stored, scale, x, bias = _case(32, 16, "block", (16, 16), seed=5)
    out = x.new_empty((3, 5, 32))
    match change:
        case "weight-dtype":
            stored = stored.to(torch.float16)
        case "scale-shape":
            scale = scale[:1]
        case "scale-dtype":
            scale = scale.double()
        case "x-width":
            x = x[..., :8]
        case "out-strided":
            out = x.new_empty((3, 5, 64))[..., ::2]
        case _:
            raise AssertionError(change)

    with pytest.raises(BackendError, match=message):
        ops.scaled_linear(x, stored, scale, bias, out, block=(16, 16))


def test_every_slice_reuses_the_same_two_buffers(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ops, "_workspaces", {})
    monkeypatch.setattr(ops, "SLICE_BYTES", 4 * 32 * 16)
    stored, scale, x, bias = _case(72, 32, "block", (16, 16), seed=6)

    for _ in range(3):
        ops.scaled_linear(x, stored, scale, bias, x.new_empty((3, 5, 72)), block=(16, 16))

    assert len(ops._workspaces) == 1  # one pair for this device, dtype and size
    [(exact, rounded)] = ops._workspaces.values()
    assert (exact.dtype, rounded.dtype) == (torch.float32, x.dtype)
