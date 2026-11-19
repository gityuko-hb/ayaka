"""Decode steps through CUDA graphs: staging on the host, refusals, and replay against eager.

The CPU tests check what ``InputBuilder.stage_decode`` writes and what ``GraphRunner`` refuses.
The ``gpu`` tests run the tiny GPT-2 twice on the same requests, once eagerly and once with decode
graphs, and require the same tokens; and check that a replayed step is where the decode steps go.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

from ayaka.core.errors import ConfigError, EngineError
from ayaka.core.params import SamplingParams
from ayaka.core.request import TokenizedRequest
from ayaka.core.types import ForwardMode, RequestId
from ayaka.sampling.structured.matcher import ConstraintMatcher
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


class _Cycle:
    """A stand-in for llguidance: position ``i`` allows three ids that move with ``i``.

    The set is tiny next to the vocabulary and changes every token, so a draw that ignores the
    mask, or applies the one of the step before, leaves it at once.
    """

    def __init__(self, vocab: int, position: int = 0) -> None:
        self._vocab = vocab
        self._position = position

    def allowed(self) -> set[int]:
        first = 40 + (self._position * 7) % 20
        return {first, first + 1, first + 2}

    def compute_bitmask(self) -> bytes:
        words = np.zeros((self._vocab + 31) // 32, dtype="<u4")
        for token in self.allowed():
            words[token // 32] |= 1 << (token % 32)
        return words.tobytes()

    def consume_token(self, sampled_token: int) -> bool:
        allowed = sampled_token in self.allowed()
        self._position += 1
        return allowed

    def is_stopped(self) -> bool:
        return False

    def is_error(self) -> bool:
        return False

    def get_error(self) -> str:
        return ""

    def deep_copy(self) -> _Cycle:
        return _Cycle(self._vocab, self._position)


def _sync_debug(mode: str) -> None:
    """torch's debug mode for implicit synchronisation, minus its "prototype feature" warning."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        torch.cuda.set_sync_debug_mode(mode)


def _constrain(stack: TinyStack, *names: str) -> None:
    vocab = stack.config.vocab_size
    for name in names:
        matcher = ConstraintMatcher(_Cycle(vocab), vocab_size=vocab, eos_token_ids=())
        stack.runner.constraints.admit(RequestId(name), matcher)


def _run_constrained(stack: TinyStack) -> tuple[dict[str, list[TokenId]], list[StepInput]]:
    """:func:`_run`, feeding every committed token to the constraints as the engine does."""
    emitted: dict[str, list[TokenId]] = {}
    steps: list[StepInput] = []
    while (step := stack.scheduler.schedule()) is not None:
        steps.append(step)
        stack.scheduler.update(stack.runner.execute(step))
        for event in stack.scheduler.take_events():
            emitted.setdefault(event.request_id, []).extend(event.token_ids)
            stack.runner.constraints.accept(event.request_id, event.token_ids)
    return emitted, steps


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
class TestMasks:
    def test_a_step_writes_the_masks_of_its_constrained_rows_and_leaves_the_rest_open(self) -> None:
        builder = build_tiny_stack().runner.builder
        view = builder.graph_mask(4, 3)
        assert view.dtype == torch.int32
        assert view.tolist() == [[-1, -1, -1]] * 4  # every token allowed until a step says more

        # A longer array is cut to the width; bit 31 of a word is the sign bit.
        words = np.array([1, 0x80000000, 7, 99, 99], dtype=np.uint32)
        builder.stage_mask({1: words}, 4)

        assert view.tolist() == [[-1] * 3, [1, -(2**31), 7], [-1] * 3, [-1] * 3]

        builder.stage_mask({}, 4)  # the next step has no constraint: nothing stays masked

        assert view.tolist() == [[-1, -1, -1]] * 4

    def test_every_bucket_reads_the_same_buffer(self) -> None:
        builder = build_tiny_stack().runner.builder

        small, large = builder.graph_mask(2, 3), builder.graph_mask(4, 3)

        assert small.data_ptr() == large.data_ptr()
        assert builder.buffers["token_masks"].data_ptr() == large.data_ptr()

    def test_a_mask_of_another_width_is_refused(self) -> None:
        builder = build_tiny_stack().runner.builder
        builder.graph_mask(4, 3)

        with pytest.raises(EngineError, match="3 words wide"):
            builder.graph_mask(4, 5)

    def test_masks_need_a_buffer_and_a_row_of_the_bucket(self) -> None:
        builder = build_tiny_stack().runner.builder
        words = np.zeros(3, dtype=np.uint32)

        with pytest.raises(EngineError, match="no token mask buffer"):
            builder.stage_mask({0: words}, 4)

        builder.graph_mask(4, 3)
        with pytest.raises(EngineError, match="outside the bucket"):
            builder.stage_mask({2: words}, 2)

    def test_an_active_constraint_alone_does_not_make_a_step_eager(self) -> None:
        stack = build_tiny_stack(max_tokens_per_step=64)
        _constrain(stack, "a")
        stack.scheduler.add_request(_request("a", _prompt(1, 6), max_tokens=4))
        stack.scheduler.add_request(_request("b", _prompt(2, 9), max_tokens=4))
        step = _first_decode_step(stack)

        masks = stack.runner._structural_masks(step)

        assert not stack.runner._needs_controls(step)
        assert list(masks) == [step.request_ids.index("a")]  # only the constrained row has one
        mask = masks[step.request_ids.index("a")]
        vocab = stack.config.vocab_size
        assert mask.dtype == np.uint32
        allowed = {t for t in range(vocab) if mask[t // 32] >> (t % 32) & 1}
        assert allowed == _Cycle(vocab).allowed()

    def test_a_row_that_asks_for_more_than_a_mask_still_runs_eagerly(self) -> None:
        stack = build_tiny_stack(max_tokens_per_step=64)
        _constrain(stack, "a")
        stack.scheduler.add_request(_request("a", _prompt(1, 6), max_tokens=4))
        stack.scheduler.add_request(_request("b", _prompt(2, 9), max_tokens=4, logit_bias={3: 1.0}))
        step = _first_decode_step(stack)

        assert stack.runner._needs_controls(step)


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
@pytest.mark.parametrize(("backends", "dtype"), GPU_CASES)
def test_constrained_decode_replays_graphs_and_draws_the_tokens_of_eager_decode(
    backends: tuple[str, ...], dtype: DType
) -> None:
    """Two constrained requests (one greedy, one sampled) beside a plain one, as buckets shrink."""
    results = []
    for graphs in (False, True):
        stack = _gpu_stack(backends, dtype, cuda_graph=graphs)
        if graphs:
            stack.runner.capture_graphs((1, 2, 4))
        _constrain(stack, "greedy", "sampled")
        stack.scheduler.add_request(_request("greedy", _prompt(1, 7), max_tokens=12))
        sampled = _request("sampled", _prompt(2, 11), max_tokens=9, temperature=0.8, seed=5)
        stack.scheduler.add_request(sampled)
        stack.scheduler.add_request(_request("plain", _prompt(3, 5), max_tokens=12))
        emitted, steps = _run_constrained(stack)
        stack.kv.check_integrity()
        results.append(emitted)
        if graphs:
            graph = stack.runner.graph
            assert graph is not None
            decode_steps = sum(step.mode is ForwardMode.DECODE for step in steps)
            # Every decode step replays, the constrained ones too.
            assert graph.replays == decode_steps > 0

    eager, graphed = results
    assert graphed == eager
    for name in ("greedy", "sampled"):  # the mask held, step after step
        cycle = _Cycle(stack.config.vocab_size)
        assert len(graphed[name]) > 1
        for token in graphed[name]:
            assert token in cycle.allowed()
            cycle.consume_token(token)


@pytest.mark.gpu
@pytest.mark.parametrize(("backends", "dtype"), GPU_CASES)
def test_launching_a_masked_step_waits_for_nothing(backends: tuple[str, ...], dtype: DType) -> None:
    """The mask is one asynchronous copy: only reading the sampled tokens waits for the device."""
    stack = _gpu_stack(backends, dtype, cuda_graph=True)
    assert stack.runner.capture_graphs((1, 2, 4))
    _constrain(stack, "a")
    stack.scheduler.add_request(_request("a", _prompt(1, 7), max_tokens=6))
    step = _first_decode_step(stack)
    masks = stack.runner._structural_masks(step)
    assert masks
    runner = stack.runner.graph
    assert runner is not None

    _sync_debug("error")  # an implicit synchronisation raises
    try:
        runner.launch(step, masks)
    finally:
        _sync_debug("default")

    [token] = stack.runner.builder.read_sampled(1)
    assert token in _Cycle(stack.config.vocab_size).allowed()


@pytest.mark.gpu
@pytest.mark.parametrize(("backends", "dtype"), GPU_CASES)
def test_a_step_without_a_constrained_row_copies_no_mask(
    backends: tuple[str, ...], dtype: DType, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The mask is staged for a step that has a constrained row, once, and for no other."""
    stack = _gpu_stack(backends, dtype, cuda_graph=True)
    assert stack.runner.capture_graphs((1, 2, 4))
    runner = stack.runner.graph
    assert runner is not None
    builder = stack.runner.builder
    staged: list[int] = []
    stage_mask = builder.stage_mask

    def spy(masks: Any, bucket: int) -> None:
        staged.append(len(masks))
        stage_mask(masks, bucket)

    monkeypatch.setattr(builder, "stage_mask", spy)
    stack.scheduler.add_request(_request("a", _prompt(1, 7), max_tokens=6))
    stack.scheduler.add_request(_request("b", _prompt(2, 9), max_tokens=6))
    step = _first_decode_step(stack)

    runner.run(step, stack.runner._structural_masks(step))
    assert staged == []  # no row has a constraint: no copy

    _constrain(stack, "a")
    runner.run(step, stack.runner._structural_masks(step))
    assert staged == [1]  # one constrained row: one staging of its mask, one copy


@pytest.mark.gpu
def test_graph_replay_matches_the_formulas_in_fp32() -> None:
    stack = _gpu_stack(("torch",), DType.FP32, cuda_graph=True)
    stack.runner.capture_graphs((1, 2))
    prompt = _prompt(5, 11)
    stack.scheduler.add_request(_request("a", prompt, max_tokens=10))

    emitted, _ = _run(stack)

    state = {name: tensor.cpu() for name, tensor in stack.state.items()}
    assert emitted["a"] == greedy_reference(state, stack.config, prompt, 10)
