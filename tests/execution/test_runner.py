"""The runner and the scheduler together, on the tiny GPT-2, against the plain GPT-2 formulas.

The oracle is ``greedy_reference``: the formulas of GPT-2 over the whole sequence, with no cache,
re-run for every token. The runner sees the same requests in chunks and one token per step through
a paged cache, with two requests interleaved, and must choose exactly the same tokens.
"""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING, Any

import pytest

from ayaka.core.errors import EngineError
from ayaka.core.params import SamplingParams
from ayaka.core.request import TokenizedRequest
from ayaka.core.types import FinishReason, ForwardMode, RequestId

torch: Any = pytest.importorskip("torch")

from ayaka.execution.runner import ModelRunner  # noqa: E402
from tiny_model import (  # noqa: E402  # needs the importorskip above
    MAX_POSITIONS,
    TinyStack,
    build_tiny_stack,
    greedy_reference,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from ayaka.core.types import TokenId
    from ayaka.scheduler.step import StepInput

pytestmark = pytest.mark.unit


def _request(
    name: str, prompt: Sequence[int], *, max_tokens: int, stop: frozenset[int] = frozenset()
) -> TokenizedRequest:
    return TokenizedRequest(
        request_id=RequestId(name),
        prompt_token_ids=tuple(prompt),
        sampling=SamplingParams(temperature=0.0, max_tokens=max_tokens, stop_token_ids=stop),
    )


def _prompt(seed: int, length: int) -> list[int]:
    generator = torch.Generator().manual_seed(seed)
    return [int(t) for t in torch.randint(32, 127, (length,), generator=generator)]


def _run(stack: TinyStack, *, max_steps: int = 500) -> dict[str, list[TokenId]]:
    """The engine loop in miniature; returns the tokens each request emitted."""
    emitted: dict[str, list[TokenId]] = {}
    for _ in range(max_steps):
        step = stack.scheduler.schedule()
        for event in stack.scheduler.take_events():
            emitted.setdefault(event.request_id, []).extend(event.token_ids)
        if step is None:
            return emitted
        stack.scheduler.update(stack.runner.execute(step))
        for event in stack.scheduler.take_events():
            emitted.setdefault(event.request_id, []).extend(event.token_ids)
        stack.kv.check_integrity()
    raise AssertionError("the loop did not finish")


class TestRequiredRows:
    def test_a_prefill_budget_larger_than_the_decode_window_wins(self) -> None:
        assert ModelRunner.required_rows(token_budget=64, max_running=4, lookahead=0) == 64

    def test_lookahead_gives_each_running_request_its_extra_rows(self) -> None:
        assert ModelRunner.required_rows(token_budget=4, max_running=4, lookahead=3) == 16

    def test_a_negative_lookahead_is_refused(self) -> None:
        with pytest.raises(ValueError, match="lookahead"):
            ModelRunner.required_rows(token_budget=64, max_running=4, lookahead=-1)


class TestAgainstTheReference:
    def test_one_request_with_one_chunk(self) -> None:
        stack = build_tiny_stack(max_tokens_per_step=64)
        prompt = _prompt(1, 9)
        stack.scheduler.add_request(_request("a", prompt, max_tokens=6))

        emitted = _run(stack)

        assert emitted["a"] == greedy_reference(stack.state, stack.config, prompt, 6)

    def test_chunked_prefill_matches_the_whole_prompt(self) -> None:
        stack = build_tiny_stack(max_tokens_per_step=5)  # 23 prompt tokens in five chunks
        prompt = _prompt(2, 23)
        stack.scheduler.add_request(_request("a", prompt, max_tokens=7))

        emitted = _run(stack)

        assert emitted["a"] == greedy_reference(stack.state, stack.config, prompt, 7)

    def test_two_interleaved_requests_each_match_their_own_reference(self) -> None:
        stack = build_tiny_stack(max_tokens_per_step=6)
        first, second = _prompt(3, 14), _prompt(4, 19)
        stack.scheduler.add_request(_request("a", first, max_tokens=9))
        stack.scheduler.add_request(_request("b", second, max_tokens=5))

        steps: list[StepInput] = []
        emitted: dict[str, list[TokenId]] = {}
        while (step := stack.scheduler.schedule()) is not None:
            steps.append(step)
            stack.scheduler.update(stack.runner.execute(step))
            for event in stack.scheduler.take_events():
                emitted.setdefault(event.request_id, []).extend(event.token_ids)

        assert emitted["a"] == greedy_reference(stack.state, stack.config, first, 9)
        assert emitted["b"] == greedy_reference(stack.state, stack.config, second, 5)
        # The point of the test: some step really carried both requests, and prefill was chunked.
        assert any(step.num_requests == 2 and step.mode is ForwardMode.DECODE for step in steps)
        assert sum(step.mode is ForwardMode.PREFILL for step in steps) > 2

    def test_a_request_that_stops_on_a_token_ends_with_that_token(self) -> None:
        stack = build_tiny_stack(max_tokens_per_step=8)
        prompt = _prompt(5, 6)
        expected = greedy_reference(stack.state, stack.config, prompt, 8)
        stop = expected[3]
        stack.scheduler.add_request(_request("a", prompt, max_tokens=8, stop=frozenset({stop})))

        emitted = _run(stack)

        assert emitted["a"] == expected[: expected.index(stop) + 1]
        stack.kv.check_integrity()

    def test_a_request_that_reuses_a_freed_row_and_blocks_is_still_right(self) -> None:
        stack = build_tiny_stack(max_running=1, max_tokens_per_step=8)
        first, second = _prompt(6, 11), _prompt(7, 13)
        stack.scheduler.add_request(_request("a", first, max_tokens=4))
        stack.scheduler.add_request(_request("b", second, max_tokens=4))

        emitted = _run(stack)

        assert emitted["a"] == greedy_reference(stack.state, stack.config, first, 4)
        assert emitted["b"] == greedy_reference(stack.state, stack.config, second, 4)


class TestFixedBuffers:
    def test_no_input_buffer_moves_across_steps(self) -> None:
        stack = build_tiny_stack(max_tokens_per_step=5)
        stack.scheduler.add_request(_request("a", _prompt(8, 13), max_tokens=5))
        stack.scheduler.add_request(_request("b", _prompt(9, 8), max_tokens=5))
        buffers = stack.runner.builder.buffers
        before = {name: tensor.data_ptr() for name, tensor in buffers.items()}

        steps = 0
        while (step := stack.scheduler.schedule()) is not None:
            stack.scheduler.update(stack.runner.execute(step))
            stack.scheduler.take_events()
            steps += 1
            assert {n: t.data_ptr() for n, t in stack.runner.builder.buffers.items()} == before

        assert steps > 6
        assert before.keys() >= {"tokens", "block_table", "block_rows", "seeds", "sampled"}

    def test_the_inputs_of_a_step_are_views_of_the_buffers(self) -> None:
        stack = build_tiny_stack(max_tokens_per_step=64)
        stack.scheduler.add_request(_request("a", _prompt(10, 7), max_tokens=2))
        step = stack.scheduler.schedule()
        assert step is not None

        inputs = stack.runner.builder.build(step)

        buffers = stack.runner.builder.buffers
        assert inputs.token_ids.data_ptr() == buffers["tokens"].data_ptr()
        assert inputs.sampled.data_ptr() == buffers["sampled"].data_ptr()
        assert inputs.sampling.seeds.data_ptr() == buffers["seeds"].data_ptr()
        assert inputs.token_ids.tolist() == list(step.tokens[0])

    def test_a_step_larger_than_the_buffers_is_refused(self) -> None:
        stack = build_tiny_stack(max_tokens_per_step=64)
        stack.scheduler.add_request(_request("a", _prompt(11, 20), max_tokens=2))
        step = stack.scheduler.schedule()
        assert step is not None
        small = build_tiny_stack(max_tokens_per_step=4, max_running=1)

        with pytest.raises(EngineError, match="buffers hold"):
            small.runner.builder.build(step)


class TestBlockTableMirror:
    def test_the_device_mirror_follows_the_host_table(self) -> None:
        stack = build_tiny_stack(max_tokens_per_step=5, max_running=3)
        for index, length in enumerate((9, 14, 6)):
            stack.scheduler.add_request(
                _request(str(index), _prompt(20 + index, length), max_tokens=6)
            )
        mirror = stack.runner.builder.buffers["block_table"]

        while (step := stack.scheduler.schedule()) is not None:
            stack.scheduler.update(stack.runner.execute(step))
            stack.scheduler.take_events()
            for row in step.batch_rows:
                host = torch.from_numpy(stack.kv.table.rows[0, row])
                # The step just ran with this mirror; a request that is still running must match.
                if stack.kv.table.lengths[0, row]:
                    assert mirror[row].tolist() == host.tolist()

    def test_a_failed_build_makes_the_next_step_copy_the_whole_table(self) -> None:
        stack = build_tiny_stack(max_tokens_per_step=8)
        prompt = _prompt(30, 6)
        stack.scheduler.add_request(_request("a", prompt, max_tokens=3))
        step = stack.scheduler.schedule()
        assert step is not None
        with pytest.raises(EngineError):
            stack.runner.builder.build(step, drafts=[(1, 2)])  # the step has no draft rows
        # Pretend the updates of the failed build never reached the device.
        stack.runner.builder.buffers["block_table"].zero_()

        output = stack.runner.execute(step)

        assert output.error is None
        assert output.tokens == ((greedy_reference(stack.state, stack.config, prompt, 1)[0],),)


class TestMultiRowWindows:
    """A decode request with draft rows, built by hand: the shape a draft proposer will use."""

    def _decoding(self) -> tuple[TinyStack, list[int], list[int]]:
        stack = build_tiny_stack(max_tokens_per_step=64)
        prompt = _prompt(40, 10)
        expected = greedy_reference(stack.state, stack.config, prompt, 8)
        stack.scheduler.add_request(_request("a", prompt, max_tokens=8))
        first = stack.scheduler.schedule()
        assert first is not None
        stack.scheduler.update(stack.runner.execute(first))
        stack.scheduler.take_events()
        return stack, prompt, expected

    def test_correct_drafts_are_all_accepted(self) -> None:
        stack, _, expected = self._decoding()
        step = stack.scheduler.schedule({RequestId("a"): 3})
        assert step is not None
        assert step.mode is ForwardMode.VERIFY
        assert step.rows_per_request == (4,)

        output = stack.runner.execute(step, drafts=[expected[1:4]])

        # The pending token is expected[0]; row j draws expected[j + 1], and every draft matched.
        assert output.tokens == (tuple(expected[1:5]),)
        assert output.accepted == (3,)

    def test_a_wrong_draft_stops_acceptance_at_the_first_mismatch(self) -> None:
        stack, _, expected = self._decoding()
        step = stack.scheduler.schedule({RequestId("a"): 3})
        assert step is not None
        wrong = (expected[1], (expected[2] + 1) % 300, expected[3])

        output = stack.runner.execute(step, drafts=[wrong])

        assert output.tokens == (tuple(expected[1:3]),)
        assert output.accepted == (1,)

    def test_a_missing_draft_is_filled_and_the_output_is_still_the_plain_one(self) -> None:
        stack, _, expected = self._decoding()
        step = stack.scheduler.schedule({RequestId("a"): 2})
        assert step is not None

        output = stack.runner.execute(step)

        assert output.tokens[0][: 1 + output.accepted[0]] == tuple(
            expected[1 : 2 + output.accepted[0]]
        )

    def test_verified_steps_finish_with_the_plain_decoding_tokens(self) -> None:
        stack, prompt, expected = self._decoding()
        emitted: list[int] = [expected[0]]
        reason: FinishReason | None = None
        while reason is None:
            step = stack.scheduler.schedule({RequestId("a"): 2})
            assert step is not None
            # Drafts taken from the reference: the best a proposer could do.
            have = len(emitted)
            drafts = [expected[have : have + step.draft_slots[0]]]
            stack.scheduler.update(stack.runner.execute(step, drafts=drafts))
            for event in stack.scheduler.take_events():
                emitted.extend(event.token_ids)
                reason = event.finish_reason

        assert emitted == expected
        assert reason is FinishReason.LENGTH
        assert len(prompt) + len(emitted) <= MAX_POSITIONS


class TestStepErrors:
    def test_a_backend_error_becomes_a_step_error_and_the_engine_goes_on(self) -> None:
        stack = build_tiny_stack(max_tokens_per_step=64)
        prompt = _prompt(50, 6)
        stack.scheduler.add_request(_request("a", prompt, max_tokens=3))
        step = stack.scheduler.schedule()
        assert step is not None
        # The step reaches the runner without its table updates, so the mirror is empty and the
        # backend refuses a request whose blocks are the padding block.

        output = stack.runner.execute(replace(step, table_updates=()))

        assert output.error is not None
        assert output.error.code == "backend_error"
        stack.scheduler.update(output)
        [event] = stack.scheduler.take_events()
        assert event.finish_reason is FinishReason.ERROR
        stack.kv.check_integrity()
