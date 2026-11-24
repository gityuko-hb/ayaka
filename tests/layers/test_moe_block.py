"""MoEBlock: the router, routing and the stacked experts, against the formulas in float32."""

from __future__ import annotations

from typing import Any

import pytest

torch: Any = pytest.importorskip("torch")

from ayaka.layers.ffn.moe import MoEBlock  # noqa: E402
from ayaka.ops.backend import load_backend  # noqa: E402
from ayaka.utils.dtypes import DType  # noqa: E402

pytestmark = pytest.mark.unit

_HIDDEN, _INNER, _EXPERTS = 8, 6, 5


def _block(top_k: int, *, renormalize: bool) -> Any:
    ops = load_backend(["torch"], device="cpu", dtype=DType.FP32)
    block = MoEBlock(
        _HIDDEN, _INNER, _EXPERTS, top_k, renormalize=renormalize, ops=ops, dtype=torch.float32
    )
    generator = torch.Generator().manual_seed(0)
    with torch.no_grad():
        for parameter in block.parameters():
            parameter.copy_(torch.randn(parameter.shape, generator=generator) / 3)
    return block


def _formula(block: Any, x: Any, top_k: int, *, renormalize: bool) -> Any:
    gate_up = block.experts.gate_up_proj.view(_EXPERTS, 2 * _INNER, _HIDDEN)
    down = block.experts.down_proj.view(_EXPERTS, _HIDDEN, _INNER)
    probabilities = torch.softmax(x @ block.gate.weight.t(), dim=-1)
    out = torch.zeros_like(x)
    for row in range(x.shape[0]):
        weights, ids = probabilities[row].topk(top_k)
        if renormalize:
            weights = weights / weights.sum()
        for weight, expert in zip(weights, ids.tolist(), strict=True):
            gate, up = (gate_up[expert] @ x[row]).split(_INNER)
            out[row] += weight * (down[expert] @ (torch.nn.functional.silu(gate) * up))
    return out


@pytest.mark.parametrize("renormalize", [False, True], ids=["softmax", "renormalised"])
@pytest.mark.parametrize("rows", [1, 7, 50])
def test_the_block_computes_the_routed_experts(renormalize: bool, rows: int) -> None:
    block = _block(2, renormalize=renormalize)
    x = torch.randn(rows, _HIDDEN, generator=torch.Generator().manual_seed(1))

    out = block(x)

    expected = _formula(block, x, 2, renormalize=renormalize)
    torch.testing.assert_close(out, expected, rtol=1e-5, atol=1e-5)


def test_leading_dimensions_are_kept() -> None:
    block = _block(3, renormalize=True)
    x = torch.randn(2, 3, _HIDDEN)

    assert block(x).shape == (2, 3, _HIDDEN)


def test_the_stacked_experts_are_views_of_the_matrices() -> None:
    block = _block(1, renormalize=False)
    gate_up, down = block.experts.stacked()

    assert gate_up.shape == (_EXPERTS, 2 * _INNER, _HIDDEN)
    assert down.shape == (_EXPERTS, _HIDDEN, _INNER)
    assert gate_up.data_ptr() == block.experts.gate_up_proj.data_ptr()
    assert block.experts.gate_up_proj.shape == (_EXPERTS * 2 * _INNER, _HIDDEN)


def test_a_top_k_outside_the_experts_is_refused() -> None:
    with pytest.raises(ValueError, match="top_k must be between 1 and 5"):
        _block(6, renormalize=False)
