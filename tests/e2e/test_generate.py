"""Generation end to end on the tiny GPT-2, through the public ``LLM`` and the command line.

Every request goes the whole way: text in, tokenizer, scheduler, paged KV cache, the model on the
CPU, sampler, detokenizer, text out. Nothing here is a golden string written by hand: the oracle
is ``greedy_reference``, GPT-2's formulas rerun over the whole sequence with no cache and no
shared code with the engine, decoded by the byte tokenizer's own ``decode``.

After every run the engine must be idle and its KV pool entirely free, whatever way the requests
ended (length, stop, abort, error).
"""

from __future__ import annotations

import subprocess
import sys
from typing import TYPE_CHECKING, Any

import pytest

from ayaka.core.errors import InvalidRequestError
from ayaka.core.params import SamplingParams
from ayaka.core.request import GenerateRequest
from ayaka.core.types import FinishReason, RequestId
from ayaka.entrypoints.cli import main
from tokenizer_fakes import FakeTokenizer

torch: Any = pytest.importorskip("torch")

from ayaka.engine.llm import LLM  # noqa: E402
from tiny_model import (  # noqa: E402  # needs the importorskip above
    MAX_POSITIONS,
    TinyCheckpoint,
    assert_cache_released,
    greedy_reference,
    patch_tokenizer_loading,
    prompt_ids,
    tiny_engine_config,
    tiny_tokenizer,
    write_tiny_gpt2,
)

if TYPE_CHECKING:
    from collections.abc import Iterator

    from ayaka.core.outputs import RequestOutput

pytestmark = pytest.mark.unit

_FAKE = FakeTokenizer()
_PROMPTS = ("hello", "The capital of France is", "x", "a prompt that is longer than several blocks")


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory: pytest.TempPathFactory) -> TinyCheckpoint:
    return write_tiny_gpt2(tmp_path_factory.mktemp("e2e-generate"))


def _build(checkpoint: TinyCheckpoint, **options: Any) -> LLM:
    return LLM.from_config(
        tiny_engine_config(checkpoint.directory, **options), tokenizer=tiny_tokenizer()
    )


@pytest.fixture
def llm(checkpoint: TinyCheckpoint) -> Iterator[LLM]:
    with _build(checkpoint) as engine:
        yield engine


def _greedy(max_tokens: int, **more: Any) -> SamplingParams:
    return SamplingParams(temperature=0.0, max_tokens=max_tokens, **more)


def _request(request_id: RequestId, prompt: str, sampling: SamplingParams) -> GenerateRequest:
    return GenerateRequest(request_id=request_id, prompt=prompt, sampling=sampling)


def _tokens(checkpoint: TinyCheckpoint, prompt: str, count: int) -> list[int]:
    return greedy_reference(checkpoint.state, checkpoint.config, prompt_ids(prompt), count)


def _text(tokens: list[int]) -> str:
    return _FAKE.decode(tokens, skip_special_tokens=True)


def _one(result: RequestOutput | list[RequestOutput]) -> RequestOutput:
    assert not isinstance(result, list)
    return result


def _drain(llm: LLM) -> dict[RequestId, list[RequestOutput]]:
    """Step the engine until it is idle; the outputs it returned, by request."""
    seen: dict[RequestId, list[RequestOutput]] = {}
    while llm.engine.has_unfinished:
        for output in llm.engine.step():
            seen.setdefault(output.request_id, []).append(output)
    return seen


class TestSingleAndBatched:
    def test_one_prompt_gives_the_reference_tokens_and_text(
        self, llm: LLM, checkpoint: TinyCheckpoint
    ) -> None:
        tokens = _tokens(checkpoint, "hello", 12)

        out = _one(llm.generate("hello", _greedy(12)))

        [completion] = out.outputs
        assert completion.token_ids == tuple(tokens)
        assert completion.text == _text(tokens)
        assert completion.finish_reason is FinishReason.LENGTH
        assert out.finished
        assert out.error is None
        assert out.prompt_token_ids == tuple(prompt_ids("hello"))
        assert (out.usage.prompt_tokens, out.usage.completion_tokens) == (6, 12)
        assert_cache_released(llm.engine)

    @pytest.mark.parametrize("max_running", [1, 2, 4], ids=lambda n: f"{n}-at-a-time")
    def test_a_batch_comes_back_in_order_each_with_its_own_reference_tokens(
        self, checkpoint: TinyCheckpoint, max_running: int
    ) -> None:
        with _build(checkpoint, max_running=max_running) as llm:
            outs = llm.generate(list(_PROMPTS), [_greedy(5 + n) for n in range(len(_PROMPTS))])

            assert isinstance(outs, list)
            assert len(outs) == len(_PROMPTS)
            for n, (prompt, out) in enumerate(zip(_PROMPTS, outs, strict=True)):
                assert out.outputs[0].token_ids == tuple(_tokens(checkpoint, prompt, 5 + n)), prompt
                assert out.prompt_token_ids == tuple(prompt_ids(prompt))
            assert_cache_released(llm.engine)

    def test_token_id_prompts_and_text_prompts_are_the_same_request(
        self, llm: LLM, checkpoint: TinyCheckpoint
    ) -> None:
        by_text, by_ids = llm.generate(["abc", tuple(prompt_ids("abc"))], _greedy(6))  # type: ignore[misc]

        assert by_text.outputs[0].token_ids == by_ids.outputs[0].token_ids
        assert by_text.outputs[0].token_ids == tuple(_tokens(checkpoint, "abc", 6))

    def test_a_prompt_that_fills_the_context_but_one_still_generates(
        self, llm: LLM, checkpoint: TinyCheckpoint
    ) -> None:
        ids = tuple(prompt_ids("y" * (MAX_POSITIONS - 2)))
        assert len(ids) == MAX_POSITIONS - 1

        out = _one(llm.generate(ids, _greedy(1)))

        expected = greedy_reference(checkpoint.state, checkpoint.config, list(ids), 1)
        assert out.outputs[0].token_ids == tuple(expected)
        assert_cache_released(llm.engine)

    def test_a_long_run_across_many_block_boundaries(
        self, llm: LLM, checkpoint: TinyCheckpoint
    ) -> None:
        out = _one(llm.generate("go", _greedy(60)))

        assert out.outputs[0].token_ids == tuple(_tokens(checkpoint, "go", 60))
        assert_cache_released(llm.engine)


class TestChunkedPrefill:
    PROMPT = "a prompt of thirty characters"  # 29 bytes and the BOS: 30 tokens

    @pytest.mark.parametrize("budget", [4, 5, 7, 30, 64], ids=lambda n: f"budget-{n}")
    def test_a_prompt_in_pieces_gives_the_same_tokens(
        self, checkpoint: TinyCheckpoint, budget: int
    ) -> None:
        with _build(checkpoint, max_tokens_per_step=budget) as llm:
            out = _one(llm.generate(self.PROMPT, _greedy(8)))

            assert out.outputs[0].token_ids == tuple(_tokens(checkpoint, self.PROMPT, 8))
            assert_cache_released(llm.engine)

    def test_a_small_budget_really_takes_more_steps(self, checkpoint: TinyCheckpoint) -> None:
        steps: dict[int, int] = {}
        for budget in (64, 4):
            with _build(checkpoint, max_tokens_per_step=budget) as llm:
                llm.engine.add_request(_request(RequestId("p"), self.PROMPT, _greedy(3)))
                steps[budget] = 0
                while llm.engine.has_unfinished:
                    llm.engine.step()
                    steps[budget] += 1

        # 30 prompt tokens in pieces of 4 need 8 prefill steps; in one piece, 1.
        assert steps[4] - steps[64] >= 6

    def test_several_long_prompts_share_a_small_budget(self, checkpoint: TinyCheckpoint) -> None:
        prompts = [self.PROMPT, self.PROMPT[::-1], "short", self.PROMPT.upper()]
        with _build(checkpoint, max_tokens_per_step=5) as llm:
            outs = llm.generate(prompts, _greedy(6))

            assert isinstance(outs, list)
            for prompt, out in zip(prompts, outs, strict=True):
                assert out.outputs[0].token_ids == tuple(_tokens(checkpoint, prompt, 6))
            assert_cache_released(llm.engine)


class TestStops:
    def test_each_way_to_stop_in_one_batch(self, llm: LLM, checkpoint: TinyCheckpoint) -> None:
        by_id = _tokens(checkpoint, "stop by id", 14)
        stop_id = by_id[5]
        by_text = _tokens(checkpoint, "stop by text", 16)
        stop_text = _text(by_text)[4:8]
        requests = [
            _request(RequestId("length"), "stop by length", _greedy(9)),
            _request(
                RequestId("id"), "stop by id", _greedy(14, stop_token_ids=frozenset({stop_id}))
            ),
            _request(RequestId("text"), "stop by text", _greedy(16, stop=(stop_text,))),
            _request(RequestId("one"), "stop by one", _greedy(1)),
        ]

        length, by_token, by_string, one = llm.generate(requests)  # type: ignore[misc]

        assert length.outputs[0].token_ids == tuple(_tokens(checkpoint, "stop by length", 9))
        assert length.outputs[0].finish_reason is FinishReason.LENGTH

        cut = by_id.index(stop_id) + 1
        assert by_token.outputs[0].token_ids == tuple(by_id[:cut])
        assert by_token.outputs[0].finish_reason is FinishReason.STOP
        assert by_token.outputs[0].stop_reason == stop_id

        at = _text(by_text).index(stop_text)
        assert by_string.outputs[0].text == _text(by_text)[:at]
        assert by_string.outputs[0].finish_reason is FinishReason.STOP
        assert by_string.outputs[0].stop_reason == stop_text
        assert by_string.outputs[0].token_ids == tuple(by_text[: at + len(stop_text)])

        assert one.outputs[0].token_ids == tuple(_tokens(checkpoint, "stop by one", 1))
        assert one.usage.completion_tokens == 1
        assert_cache_released(llm.engine)

    def test_a_stop_that_ends_a_request_early_frees_its_place_for_the_next(
        self, checkpoint: TinyCheckpoint
    ) -> None:
        tokens = _tokens(checkpoint, "first", 20)
        stop = frozenset({tokens[2]})
        with _build(checkpoint, max_running=1) as llm:
            first, second = llm.generate(  # type: ignore[misc]
                ["first", "second"], [_greedy(20, stop_token_ids=stop), _greedy(5)]
            )

            assert first.outputs[0].finish_reason is FinishReason.STOP
            assert len(first.outputs[0].token_ids) < 20
            assert second.outputs[0].token_ids == tuple(_tokens(checkpoint, "second", 5))
            assert_cache_released(llm.engine)


class TestStreaming:
    def test_the_chunks_of_one_stream_add_up_to_the_whole_answer(
        self, llm: LLM, checkpoint: TinyCheckpoint
    ) -> None:
        tokens = _tokens(checkpoint, "stream me", 10)

        chunks = list(llm.stream("stream me", _greedy(10)))

        assert [t for c in chunks for t in c.outputs[0].token_ids] == tokens
        assert "".join(c.outputs[0].text for c in chunks) == _text(tokens)
        assert [c.finished for c in chunks] == [False] * (len(chunks) - 1) + [True]
        assert chunks[-1].outputs[0].finish_reason is FinishReason.LENGTH
        assert chunks[-1].usage.completion_tokens == 10
        assert_cache_released(llm.engine)

    def test_requests_streamed_together_each_add_up_to_their_own_answer(
        self, llm: LLM, checkpoint: TinyCheckpoint
    ) -> None:
        prompts = {RequestId(f"s{n}"): prompt for n, prompt in enumerate(_PROMPTS)}
        for request_id, prompt in prompts.items():
            llm.engine.add_request(_request(request_id, prompt, _greedy(7)), stream=True)

        seen = _drain(llm)

        assert set(seen) == set(prompts)
        for request_id, prompt in prompts.items():
            chunks = seen[request_id]
            tokens = _tokens(checkpoint, prompt, 7)
            assert [t for c in chunks for t in c.outputs[0].token_ids] == tokens
            assert "".join(c.outputs[0].text for c in chunks) == _text(tokens)
            assert [c.finished for c in chunks] == [False] * (len(chunks) - 1) + [True]
        assert_cache_released(llm.engine)

    def test_a_streamed_stop_string_never_leaks_past_the_stop(
        self, llm: LLM, checkpoint: TinyCheckpoint
    ) -> None:
        text = _text(_tokens(checkpoint, "leak", 16))
        stop = text[6:10]

        chunks = list(llm.stream("leak", _greedy(16, stop=(stop,))))

        assert "".join(c.outputs[0].text for c in chunks) == text[: text.index(stop)]
        assert chunks[-1].outputs[0].stop_reason == stop
        assert_cache_released(llm.engine)

    def test_a_stream_walked_away_from_cancels_its_request(self, llm: LLM) -> None:
        stream = llm.stream("walk away", _greedy(50))
        next(stream)

        stream.close()

        assert_cache_released(llm.engine)


class TestAbort:
    def test_an_aborted_request_ends_as_cancelled_and_the_others_are_untouched(
        self, llm: LLM, checkpoint: TinyCheckpoint
    ) -> None:
        engine = llm.engine
        for name, prompt in (("keep-1", "alpha"), ("drop", "beta"), ("keep-2", "gamma")):
            engine.add_request(_request(RequestId(name), prompt, _greedy(20)))
        for _ in range(4):
            assert engine.step() == []  # nothing finishes yet and a plain request waits for the end

        assert engine.abort(RequestId("drop"))
        seen = _drain(llm)

        [dropped] = seen[RequestId("drop")]
        assert dropped.outputs[0].finish_reason is FinishReason.CANCELLED
        kept = dropped.outputs[0].token_ids
        assert 0 < len(kept) < 20
        assert kept == tuple(_tokens(checkpoint, "beta", len(kept)))  # what it had is right
        for name, prompt in (("keep-1", "alpha"), ("keep-2", "gamma")):
            [done] = seen[RequestId(name)]
            assert done.outputs[0].finish_reason is FinishReason.LENGTH
            assert done.outputs[0].token_ids == tuple(_tokens(checkpoint, prompt, 20))
        assert_cache_released(engine)

    def test_a_request_aborted_while_it_waits_never_runs(self, checkpoint: TinyCheckpoint) -> None:
        with _build(checkpoint, max_running=1) as llm:
            engine = llm.engine
            engine.add_request(_request(RequestId("runs"), "first", _greedy(6)))
            engine.add_request(_request(RequestId("waits"), "second", _greedy(6)))
            engine.step()
            assert engine.scheduler.num_waiting == 1

            assert engine.abort(RequestId("waits"))
            seen = _drain(llm)

            [cancelled] = seen[RequestId("waits")]
            assert cancelled.outputs[0].finish_reason is FinishReason.CANCELLED
            assert cancelled.outputs[0].token_ids == ()
            [done] = seen[RequestId("runs")]
            assert done.outputs[0].token_ids == tuple(_tokens(checkpoint, "first", 6))
            assert_cache_released(engine)

    def test_aborting_a_finished_or_unknown_request_does_nothing(self, llm: LLM) -> None:
        llm.generate("done", _greedy(2))

        assert not llm.engine.abort(RequestId("ayaka-1"))
        assert not llm.engine.abort(RequestId("never-existed"))
        assert_cache_released(llm.engine)


class TestMoreRequestsThanFit:
    PROMPTS = tuple(f"request number {n} " + "z" * n for n in range(12))

    def _check(self, llm: LLM, checkpoint: TinyCheckpoint, count: int = 8) -> None:
        outs = llm.generate(list(self.PROMPTS), _greedy(count))

        assert isinstance(outs, list)
        assert len(outs) == len(self.PROMPTS)
        for prompt, out in zip(self.PROMPTS, outs, strict=True):
            assert out.finished
            assert out.outputs[0].finish_reason is FinishReason.LENGTH
            assert out.outputs[0].token_ids == tuple(_tokens(checkpoint, prompt, count)), prompt
        assert_cache_released(llm.engine)

    def test_twelve_requests_two_at_a_time_all_finish(self, checkpoint: TinyCheckpoint) -> None:
        with _build(checkpoint, max_running=2, max_tokens_per_step=6) as llm:
            self._check(llm, checkpoint)

    def test_a_cache_smaller_than_the_crowd_never_gives_wrong_tokens_or_leaks_blocks(
        self, checkpoint: TinyCheckpoint
    ) -> None:
        # The smallest cache the plan accepts holds one full context (32 blocks of 4 tokens) and
        # these requests need about 15 blocks each, so the cache, not max_running, limits how
        # many can run. A request the cache cannot carry to its end must say so with a
        # capacity error and keep the tokens it did make; one that finishes must be exact.
        prompts = [f"request number {n} " + "z" * (40 + n) for n in range(12)]
        with _build(checkpoint, max_running=4, num_blocks=1 + MAX_POSITIONS // 4) as llm:
            outs = llm.generate(prompts, _greedy(8))

            assert isinstance(outs, list)
            complete = 0
            for prompt, out in zip(prompts, outs, strict=True):
                tokens = out.outputs[0].token_ids
                assert tokens == tuple(_tokens(checkpoint, prompt, len(tokens))), prompt
                if out.error is None:
                    assert len(tokens) == 8, prompt
                    complete += 1
                else:
                    assert out.error.code == "capacity_exceeded", prompt
                    assert len(tokens) < 8, prompt
            assert complete >= 1
            assert_cache_released(llm.engine)
            # The engine is still healthy: a request that fits runs to the end.
            again = _one(llm.generate("after the crowd", _greedy(8)))
            assert again.error is None
            assert again.outputs[0].token_ids == tuple(_tokens(checkpoint, "after the crowd", 8))

    def test_a_waiting_queue_shorter_than_the_batch_is_fed_as_it_drains(
        self, checkpoint: TinyCheckpoint
    ) -> None:
        with _build(checkpoint, max_running=2, max_waiting=2) as llm:
            self._check(llm, checkpoint)

    def test_the_engine_serves_again_after_a_crowd(self, checkpoint: TinyCheckpoint) -> None:
        with _build(checkpoint, max_running=2) as llm:
            self._check(llm, checkpoint, count=4)
            self._check(llm, checkpoint, count=9)


class TestRefusals:
    def test_a_prompt_longer_than_the_context_is_rejected_with_the_typed_error(
        self, llm: LLM
    ) -> None:
        with pytest.raises(InvalidRequestError):
            llm.generate("x" * MAX_POSITIONS, _greedy(4))

        assert_cache_released(llm.engine)

    def test_a_prompt_that_leaves_no_room_for_the_answer_is_rejected(self, llm: LLM) -> None:
        ids = tuple(prompt_ids("y" * (MAX_POSITIONS - 2)))  # one short of the context

        with pytest.raises(InvalidRequestError):
            llm.generate(ids, _greedy(2))

        assert_cache_released(llm.engine)

    def test_a_bad_prompt_in_a_batch_leaves_nothing_running_and_the_engine_usable(
        self, llm: LLM, checkpoint: TinyCheckpoint
    ) -> None:
        with pytest.raises(InvalidRequestError):
            llm.generate(["fine", "x" * 500, "also fine"], _greedy(3))
        assert_cache_released(llm.engine)

        again = _one(llm.generate("after", _greedy(3)))

        assert again.outputs[0].token_ids == tuple(_tokens(checkpoint, "after", 3))
        assert_cache_released(llm.engine)


class TestCommandLine:
    """``ayaka run`` through ``main(argv)``: the flags to the printed text, on the same oracle."""

    @pytest.fixture(autouse=True)
    def tiny_tokenizer_everywhere(self, monkeypatch: pytest.MonkeyPatch) -> None:
        patch_tokenizer_loading(monkeypatch)

    @staticmethod
    def _run(checkpoint: TinyCheckpoint, *flags: str) -> list[str]:
        return ["run", str(checkpoint.directory), "--offline", "-q", *flags]

    @pytest.mark.parametrize(("prompt", "count"), [("hello", 8), ("x" * 40, 5), ("ab", 30)])
    def test_the_printed_text_is_the_reference_text(
        self,
        checkpoint: TinyCheckpoint,
        capsys: pytest.CaptureFixture[str],
        prompt: str,
        count: int,
    ) -> None:
        code = main(self._run(checkpoint, "-p", prompt, "--max-tokens", str(count)))

        captured = capsys.readouterr()
        assert code == 0
        assert captured.out == _text(_tokens(checkpoint, prompt, count)) + "\n"
        assert captured.err == ""

    def test_a_stop_text_cuts_the_printed_text(
        self, checkpoint: TinyCheckpoint, capsys: pytest.CaptureFixture[str]
    ) -> None:
        text = _text(_tokens(checkpoint, "stop", 16))
        stop = text[5:9]

        code = main(self._run(checkpoint, "-p", "stop", "--max-tokens", "16", "--stop", stop))

        assert code == 0
        assert capsys.readouterr().out == text[: text.index(stop)] + "\n"

    def test_a_prompt_longer_than_the_context_exits_2_with_one_line(
        self, checkpoint: TinyCheckpoint, capsys: pytest.CaptureFixture[str]
    ) -> None:
        code = main(self._run(checkpoint, "-p", "x" * MAX_POSITIONS, "--max-tokens", "4"))

        captured = capsys.readouterr()
        assert code == 2
        assert captured.out == ""
        assert captured.err.startswith("ayaka: error: ")
        assert "Traceback" not in captured.err
        assert len(captured.err.splitlines()) == 1


class TestImportsStayLight:
    def test_the_package_the_engine_and_the_help_of_the_command_import_no_torch(self) -> None:
        code = (
            "import runpy, sys\n"
            "import ayaka, ayaka.engine, ayaka.entrypoints\n"
            "assert 'torch' not in sys.modules, 'importing the packages loaded torch'\n"
            "sys.argv = ['ayaka', 'run', '--help']\n"
            "try:\n"
            "    runpy.run_module('ayaka', run_name='__main__', alter_sys=True)\n"
            "except SystemExit as stop:\n"
            "    assert stop.code == 0, stop.code\n"
            "else:\n"
            "    raise AssertionError('--help did not exit')\n"
            "assert 'torch' not in sys.modules, 'ayaka run --help loaded torch'\n"
        )

        result = subprocess.run(  # noqa: S603
            [sys.executable, "-I", "-c", code], capture_output=True, text=True, check=False
        )

        assert result.returncode == 0, result.stderr
        assert "--max-tokens" in result.stdout
