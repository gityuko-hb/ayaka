"""The torch MoE ops against a naive loop over rows and experts, in float32 within ``1e-5``.

Each of the three strategies of :func:`experts` (gathered, dense, grouped) is reached by its row
count and compared with the loop; routing ties go to the lower expert id. Thresholds fixed in
#113 before any run.
"""

from __future__ import annotations

from typing import Any

import pytest

torch: Any = pytest.importorskip("torch")

from ayaka.core.errors import BackendError  # noqa: E402
from ayaka.ops.backends.torch import moe  # noqa: E402

pytestmark = pytest.mark.unit

_EXPERTS, _HIDDEN, _INNER, _TOP_K = 6, 16, 12, 2


def _weights(seed: int, dtype: Any = torch.float32) -> tuple[Any, Any]:
    generator = torch.Generator().manual_seed(seed)
    gate_up = torch.randn(_EXPERTS, 2 * _INNER, _HIDDEN, generator=generator) / 4
    down = torch.randn(_EXPERTS, _HIDDEN, _INNER, generator=generator) / 4
    return gate_up.to(dtype), down.to(dtype)


def _routing(rows: int, seed: int, top_k: int = _TOP_K) -> tuple[Any, Any]:
    generator = torch.Generator().manual_seed(seed)
    logits = torch.randn(rows, _EXPERTS, generator=generator)
    weights = torch.empty(rows, top_k)
    ids = torch.empty(rows, top_k, dtype=torch.int32)
    moe.route(logits, weights, ids, renormalize=True)
    return weights, ids


def _naive(x: Any, gate_up: Any, down: Any, weights: Any, ids: Any) -> Any:
    out = torch.zeros(x.shape, dtype=torch.float64)
    for row in range(x.shape[0]):
        for slot in range(ids.shape[1]):
            expert = int(ids[row, slot])
            hidden = gate_up[expert].double() @ x[row].double()
            gate, up = hidden[:_INNER], hidden[_INNER:]
            value = down[expert].double() @ (torch.nn.functional.silu(gate) * up)
            out[row] += float(weights[row, slot]) * value
    return out


@pytest.mark.parametrize(
    "rows",
    [
        1,
        moe.GATHER_MAX_ROWS,
        moe.GATHER_MAX_ROWS + 1,
        moe.DENSE_MAX_ROWS,
        moe.DENSE_MAX_ROWS + 1,
        97,
    ],
    ids=["gather-1", "gather-max", "dense-min", "dense-max", "grouped-min", "grouped-97"],
)
def test_every_strategy_equals_the_naive_loop(rows: int) -> None:
    gate_up, down = _weights(1)
    x = torch.randn(rows, _HIDDEN, generator=torch.Generator().manual_seed(2))
    weights, ids = _routing(rows, seed=3)
    out = torch.empty_like(x)

    moe.experts(x, gate_up, down, weights, ids, out)

    torch.testing.assert_close(
        out.double(), _naive(x, gate_up, down, weights, ids), rtol=1e-5, atol=1e-5
    )


def test_the_strategies_agree_on_the_same_rows(monkeypatch: pytest.MonkeyPatch) -> None:
    gate_up, down = _weights(4)
    x = torch.randn(12, _HIDDEN, generator=torch.Generator().manual_seed(5))
    weights, ids = _routing(12, seed=6)
    results = []
    for gather, dense in [(12, 12), (0, 12), (0, 0)]:
        monkeypatch.setattr(moe, "GATHER_MAX_ROWS", gather)
        monkeypatch.setattr(moe, "DENSE_MAX_ROWS", dense)
        out = torch.empty_like(x)
        moe.experts(x, gate_up, down, weights, ids, out)
        results.append(out)

    torch.testing.assert_close(results[0], results[1], rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(results[1], results[2], rtol=1e-5, atol=1e-5)


def test_an_expert_no_row_chose_is_skipped_and_rows_may_repeat_an_expert() -> None:
    gate_up, down = _weights(7)
    x = torch.randn(40, _HIDDEN, generator=torch.Generator().manual_seed(8))
    ids = torch.tensor([[1, 1]] * 20 + [[4, 1]] * 20, dtype=torch.int32)
    weights = torch.full((40, 2), 0.5)
    out = torch.empty_like(x)

    moe.experts(x, gate_up, down, weights, ids, out)

    torch.testing.assert_close(
        out.double(), _naive(x, gate_up, down, weights, ids), rtol=1e-5, atol=1e-5
    )


@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
def test_half_precision_stays_close_to_float32(dtype: str) -> None:
    compute = getattr(torch, dtype)
    gate_up, down = _weights(9)
    x = torch.randn(6, _HIDDEN, generator=torch.Generator().manual_seed(10))
    weights, ids = _routing(6, seed=11)
    exact = _naive(x, gate_up, down, weights, ids)
    out = torch.empty(6, _HIDDEN, dtype=compute)

    moe.experts(x.to(compute), gate_up.to(compute), down.to(compute), weights, ids, out)

    torch.testing.assert_close(out.double(), exact, rtol=5e-2, atol=5e-2)


def test_route_is_a_float32_softmax_top_k_with_or_without_renormalisation() -> None:
    logits = torch.randn(5, _EXPERTS, generator=torch.Generator().manual_seed(12)).to(
        torch.bfloat16
    )
    probabilities = torch.softmax(logits.float(), dim=-1)
    for renormalize in (False, True):
        weights = torch.empty(5, 3)
        ids = torch.empty(5, 3, dtype=torch.int32)

        moe.route(logits, weights, ids, renormalize=renormalize)

        expected, expected_ids = probabilities.topk(3, dim=-1)
        if renormalize:
            expected = expected / expected.sum(-1, keepdim=True)
        assert torch.equal(ids.long(), expected_ids)
        torch.testing.assert_close(weights, expected, rtol=1e-6, atol=1e-7)


def test_route_breaks_ties_by_the_lower_expert_id() -> None:
    logits = torch.tensor([[0.0, 2.0, 1.0, 2.0, 2.0, 0.0], [1.0] * _EXPERTS])
    weights = torch.empty(2, 2)
    ids = torch.empty(2, 2, dtype=torch.int32)

    moe.route(logits, weights, ids, renormalize=False)

    assert ids.tolist() == [[1, 3], [0, 1]]
    assert torch.allclose(weights[1], torch.full((2,), 1 / _EXPERTS))


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ("top-k", "1 <= k <= 6"),
        ("ids-dtype", "ids must be int32"),
        ("weights-dtype", "weights must be float32"),
        ("down-shape", "down must be"),
    ],
)
def test_bad_arguments_are_refused(change: str, message: str) -> None:
    gate_up, down = _weights(13)
    x = torch.randn(3, _HIDDEN)
    weights, ids = _routing(3, seed=14)
    logits = torch.randn(3, _EXPERTS)
    match change:
        case "top-k":
            with pytest.raises(BackendError, match=message):
                moe.route(logits, torch.empty(3, 7), ids, renormalize=False)
            return
        case "ids-dtype":
            ids = ids.long()
        case "weights-dtype":
            weights = weights.double()
        case "down-shape":
            down = down[:, :, :-1]
        case _:
            raise AssertionError(change)

    with pytest.raises(BackendError, match=message):
        moe.experts(x, gate_up, down, weights, ids, torch.empty_like(x))


def test_no_rows_is_nothing_to_do() -> None:
    gate_up, down = _weights(15)
    out = torch.empty(0, _HIDDEN)

    moe.experts(
        torch.empty(0, _HIDDEN),
        gate_up,
        down,
        torch.empty(0, 2),
        torch.empty(0, 2, dtype=torch.int32),
        out,
    )

    assert out.shape == (0, _HIDDEN)


@pytest.mark.gpu
@pytest.mark.parametrize("rows", [1, 3, moe.GATHER_MAX_ROWS + 2, moe.DENSE_MAX_ROWS])
def test_a_cuda_graph_replay_equals_eager(rows: int) -> None:
    if not torch.cuda.is_available():
        pytest.skip("needs a CUDA device")
    gate_up, down = (t.cuda().to(torch.bfloat16) for t in _weights(16))
    x = torch.randn(rows, _HIDDEN, device="cuda").to(torch.bfloat16)
    logits = torch.randn(rows, _EXPERTS, device="cuda").to(torch.bfloat16)
    weights = torch.empty(rows, _TOP_K, device="cuda")
    ids = torch.empty(rows, _TOP_K, dtype=torch.int32, device="cuda")
    eager = torch.empty_like(x)
    moe.route(logits, weights, ids, renormalize=True)
    moe.experts(x, gate_up, down, weights, ids, eager)
    replayed = torch.empty_like(x)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):  # warm-up outside the capture, as the engine does
        moe.route(logits, weights, ids, renormalize=True)
        moe.experts(x, gate_up, down, weights, ids, replayed)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        moe.route(logits, weights, ids, renormalize=True)
        moe.experts(x, gate_up, down, weights, ids, replayed)
    replayed.zero_()

    graph.replay()
    torch.cuda.synchronize()

    assert torch.equal(replayed, eager)


@pytest.mark.gpu
@pytest.mark.filterwarnings("ignore:The CUDA Graph is empty")  # the refusal leaves nothing
def test_a_capture_above_the_dense_rows_is_refused() -> None:
    if not torch.cuda.is_available():
        pytest.skip("needs a CUDA device")
    rows = moe.DENSE_MAX_ROWS + 1
    gate_up, down = (t.cuda() for t in _weights(17))
    x = torch.randn(rows, _HIDDEN, device="cuda")
    weights = torch.full((rows, _TOP_K), 0.5, device="cuda")
    ids = torch.zeros(rows, _TOP_K, dtype=torch.int32, device="cuda")
    graph = torch.cuda.CUDAGraph()

    with pytest.raises(BackendError, match="cannot be captured"), torch.cuda.graph(graph):
        moe.experts(x, gate_up, down, weights, ids, torch.empty_like(x))
