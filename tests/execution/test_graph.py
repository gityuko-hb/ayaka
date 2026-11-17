"""Decode steps through CUDA graphs: staging on the host, refusals, and replay against eager.

The CPU tests check what ``InputBuilder.stage_decode`` writes and what ``GraphRunner`` refuses.
The ``gpu`` tests run the tiny GPT-2 twice on the same requests, once eagerly and once with decode
graphs, and require the same tokens; and check that a replayed step is where the decode steps go.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

from ayaka.core.errors import ConfigError
from ayaka.core.params import SamplingParams
from ayaka.core.request import TokenizedRequest
from ayaka.core.types import ForwardMode, RequestId
from ayaka.utils.dtypes import DType

torch: Any = pytest.importorskip("torch")

from ayaka.execution.graph import GraphRunner  # noqa: E402
from tiny_model import TinyStack, build_tiny_stack, greedy_reference  # noqa: E402

if TYPE_CHECKING:
    from collections.abc import Sequence

    from ayaka.core.types import TokenId
    from ayaka.scheduler.step import StepInput


def _request(name: str, prompt: Sequence[int], *, max_tokens: int, **sampling: Any) -> Any:
    options: dict[str, Any] = {"temperature": 0.0, **sampling}
    params = SamplingParams(max_tokens=max_tokens, **options)
    return TokenizedRequest(
        request_id=RequestId(name), prompt_token_ids=tuple(prompt), sampling=params
    )


def _prompt(seed: int, length: int) -> list[int]:
    generator = torch.Generator().manual_seed(seed)
    return [int(t) for t in torch.randint(32, 127, (length,), generator=generator)]


def _run(stack: TinyStack) -> tuple[dict[str, list[TokenId]], list[StepInput]]:
    emitted: dict[str, list[TokenId]] = {}
    steps: list[StepInput] = []
    while (step := stack.scheduler.schedule()) is not None:
        steps.append(step)
        stack.scheduler.update(stack.runner.execute(step))
        for event in stack.scheduler.take_events():
            emitted.setdefault(event.request_id, []).extend(event.token_ids)
    return emitted, steps


def _first_decode_step(stack: TinyStack) -> StepInput:
    while (step := stack.scheduler.schedule()) is not None:
        if step.mode is ForwardMode.DECODE:
            return step
        stack.scheduler.update(stack.runner.execute(step))
    raise AssertionError("no decode step")


@pytest.mark.unit
class TestStaging:
    def test_a_decode_step_is_padded_to_the_bucket(self) -> None:
        stack = build_tiny_stack(max_tokens_per_step=64)
        stack.scheduler.add_request(_request("a", _prompt(1, 6), max_tokens=4))
        stack.scheduler.add_request(_request("b", _prompt(2, 9), max_tokens=4, temperature=0.7))
        step = _first_decode_step(stack)
        builder = stack.runner.builder

        staged = builder.stage_decode(step, 4)

        assert (staged.bucket, staged.count) == (4, 2)
        assert staged.kv_lengths.tolist() == [7, 10, 0, 0]
        table = stack.kv.table.rows[0]
        assert staged.block_rows[:2].tolist() == table[list(step.batch_rows)].tolist()
        assert staged.block_rows[2:].count_nonzero() == 0
        assert not staged.all_greedy
        tokens, batch, _ = builder.graph_views(4)
        assert tokens.tolist()[:2] == [known[0] for known in step.tokens]
        assert tokens.tolist()[2:] == [0, 0]
        assert batch.temperature.tolist()[2:] == [0.0, 0.0]  # padding rows are greedy
        assert batch.positions.tolist() == [7, 10, 0, 0]

    def test_a_prefill_step_cannot_be_staged(self) -> None:
        stack = build_tiny_stack(max_tokens_per_step=64)
        stack.scheduler.add_request(_request("a", _prompt(1, 6), max_tokens=2))
        step = stack.scheduler.schedule()
        assert step is not None

        with pytest.raises(Exception, match="not a decode step"):
            stack.runner.builder.stage_decode(step, 4)


@pytest.mark.unit
class TestRefusals:
    def test_ops_loaded_without_graphs_are_refused(self) -> None:
        stack = build_tiny_stack()

        with pytest.raises(ConfigError) as caught:
            GraphRunner(stack.model, stack.runner.builder, stack.runner.kv_buffers, buckets=(1,))

        assert caught.value.code == "GRAPH_NEEDS_GRAPH_SAFE_OPS"

    def test_a_cache_on_the_cpu_is_refused(self) -> None:
        stack = build_tiny_stack(cuda_graph=True)

        with pytest.raises(ConfigError) as caught:
            GraphRunner(stack.model, stack.runner.builder, stack.runner.kv_buffers, buckets=(1,))

        assert caught.value.code == "GRAPH_NEEDS_CUDA"


GPU_CASES = [
    pytest.param(("torch",), DType.FP32, id="torch-fp32"),
    pytest.param(("torch",), DType.FP16, id="torch-fp16"),
    pytest.param(("flashinfer", "torch"), DType.FP16, id="flashinfer-fp16"),
]


def _gpu_stack(backends: tuple[str, ...], dtype: DType, *, cuda_graph: bool) -> TinyStack:
    if "flashinfer" in backends:
        pytest.importorskip("flashinfer", reason="the flashinfer extra is not installed")
    # flashinfer's kernels need a head of 64; the fixture's 8 is for the CPU reference.
    from tiny_model import tiny_gpt2_config

    config = tiny_gpt2_config(hidden_size=128, num_attention_heads=2, num_kv_heads=2, head_dim=64)
    return build_tiny_stack(
        max_tokens_per_step=64,
        device="cuda",
        dtype=dtype,
        backends=backends,
        cuda_graph=cuda_graph,
        config=config,
        block_size=16,
    )


@pytest.mark.gpu
@pytest.mark.parametrize(("backends", "dtype"), GPU_CASES)
def test_graph_replay_draws_the_tokens_of_eager_decode(
    backends: tuple[str, ...], dtype: DType
) -> None:
    prompts = {"a": _prompt(1, 7), "b": _prompt(2, 13), "c": _prompt(3, 4)}
    results = []
    for graphs in (False, True):
        stack = _gpu_stack(backends, dtype, cuda_graph=graphs)
        if graphs:
            captured = stack.runner.capture_graphs((1, 2, 4))
            assert [item.bucket for item in captured] == [1, 2, 4]
        for name, prompt in prompts.items():
            stack.scheduler.add_request(_request(name, prompt, max_tokens=12))
        sampled = _request("s", _prompt(4, 5), max_tokens=12, temperature=0.8, seed=7)
        stack.scheduler.add_request(sampled)
        emitted, steps = _run(stack)
        stack.kv.check_integrity()
        results.append(emitted)
        if graphs:
            graph = stack.runner.graph
            assert graph is not None
            decode_steps = sum(step.mode is ForwardMode.DECODE for step in steps)
            assert graph.replays == decode_steps > 0

    eager, graphed = results
    assert graphed == eager


@pytest.mark.gpu
def test_graph_replay_matches_the_formulas_in_fp32() -> None:
    stack = _gpu_stack(("torch",), DType.FP32, cuda_graph=True)
    stack.runner.capture_graphs((1, 2))
    prompt = _prompt(5, 11)
    stack.scheduler.add_request(_request("a", prompt, max_tokens=10))

    emitted, _ = _run(stack)

    state = {name: tensor.cpu() for name, tensor in stack.state.items()}
    assert emitted["a"] == greedy_reference(state, stack.config, prompt, 10)
