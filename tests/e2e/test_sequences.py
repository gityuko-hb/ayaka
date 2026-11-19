"""Several sequences per request and the per-step statistics, end to end on the tiny GPT-2.

A request with ``n > 1`` runs as sibling requests that share the prompt and the seed and differ in
their sequence index. The oracle for sequence ``i`` is not a hand-written string: sequence 0 must
equal an ``n = 1`` request with the same seed, and the result must not depend on whether the
prefix cache attached the prompt or the siblings computed it again.
"""

from __future__ import annotations

import dataclasses
import logging
from typing import TYPE_CHECKING, Any

import pytest

from ayaka.core.errors import InvalidRequestError
from ayaka.core.params import SamplingParams
from ayaka.core.request import GenerateRequest
from ayaka.core.types import FinishReason, RequestId

torch: Any = pytest.importorskip("torch")

from ayaka.engine.build import build_engine  # noqa: E402
from ayaka.engine.llm import LLM  # noqa: E402
from ayaka.metrics.stats import IterationStats  # noqa: E402
from tiny_model import (  # noqa: E402  # needs the importorskip above
    TINY_BLOCK_SIZE,
    TinyCheckpoint,
    assert_cache_released,
    tiny_engine_config,
    tiny_tokenizer,
    write_tiny_gpt2,
)

if TYPE_CHECKING:
    from collections.abc import Iterator

    from ayaka.core.outputs import RequestOutput
    from ayaka.engine.core import EngineCore

pytestmark = pytest.mark.unit

_PROMPT = "the shared prompt of several sequences"


class Recorder:
    """A stat logger that keeps every step."""

    def __init__(self) -> None:
        self.steps: list[IterationStats] = []
        self.logs = 0

    def record(self, stats: IterationStats) -> None:
        self.steps.append(stats)

    def log(self) -> None:
        self.logs += 1


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory: pytest.TempPathFactory) -> TinyCheckpoint:
    return write_tiny_gpt2(tmp_path_factory.mktemp("e2e-sequences"))


def _engine(
    checkpoint: TinyCheckpoint, recorder: Recorder | None = None, *, prefix_cache: bool = True
) -> EngineCore:
    config = tiny_engine_config(checkpoint.directory)
    config = dataclasses.replace(
        config, cache=dataclasses.replace(config.cache, prefix_cache=prefix_cache)
    )
    return build_engine(
        config, tokenizer=tiny_tokenizer(), stat_loggers=(recorder,) if recorder else ()
    )


@pytest.fixture
def recorder() -> Recorder:
    return Recorder()


@pytest.fixture
def llm(checkpoint: TinyCheckpoint, recorder: Recorder) -> Iterator[LLM]:
    with LLM(_engine(checkpoint, recorder)) as engine:
        yield engine


def _one(result: RequestOutput | list[RequestOutput]) -> RequestOutput:
    assert not isinstance(result, list)
    return result


def _sampled(n: int, count: int = 8, seed: int = 7) -> SamplingParams:
    # top_k keeps the draws among the printable tokens the fixture favours, which decode.
    return SamplingParams(
        temperature=1.0, top_k=4, n=n, seed=seed, max_tokens=count, min_tokens=count
    )


def _drain(engine: EngineCore) -> list[RequestOutput]:
    outputs: list[RequestOutput] = []
    for _ in range(500):
        if not engine.has_unfinished:
            return outputs
        outputs += engine.step()
    raise AssertionError("the engine did not finish")


class TestSequences:
    def test_two_sequences_come_back_in_order_with_one_prompt_in_the_usage(self, llm: LLM) -> None:
        out = _one(llm.generate(_PROMPT, _sampled(2)))

        assert out.finished
        assert [c.index for c in out.outputs] == [0, 1]
        assert all(len(c.token_ids) == 8 for c in out.outputs)
        assert all(c.finish_reason is FinishReason.LENGTH for c in out.outputs)
        assert out.outputs[0].token_ids != out.outputs[1].token_ids
        assert out.usage.prompt_tokens == len(out.prompt_token_ids or ())
        assert out.usage.completion_tokens == 16
        assert out.usage.cached_tokens > 0
        assert_cache_released(llm.engine)

    def test_sequence_zero_is_the_single_request_with_the_same_seed(self, llm: LLM) -> None:
        pair = _one(llm.generate(_PROMPT, _sampled(2)))
        single = _one(llm.generate(_PROMPT, _sampled(1)))

        assert pair.outputs[0].token_ids == single.outputs[0].token_ids

    def test_the_prompt_is_prefilled_once_and_the_siblings_attach_it(
        self, llm: LLM, recorder: Recorder
    ) -> None:
        before = llm.engine.kv.stats().hit_tokens
        recorder.steps.clear()

        out = _one(llm.generate("a different prompt that is long enough", _sampled(3, count=2)))

        length = len(out.prompt_token_ids or ())
        matched = ((length - 1) // TINY_BLOCK_SIZE) * TINY_BLOCK_SIZE  # whole blocks, never all
        rows = sum(step.num_scheduled_tokens for step in recorder.steps)
        # One full prefill, two prefills of what the cache did not hold, one decode row each.
        assert rows == length + 2 * (length - matched) + 3
        assert llm.engine.kv.stats().hit_tokens - before == 2 * matched
        assert out.usage.cached_tokens == matched

    def test_the_cache_changes_the_work_but_not_the_tokens(
        self, checkpoint: TinyCheckpoint
    ) -> None:
        with LLM(_engine(checkpoint, prefix_cache=False)) as cold:
            without = _one(cold.generate(_PROMPT, _sampled(3)))
        with LLM(_engine(checkpoint)) as warm:
            with_cache = _one(warm.generate(_PROMPT, _sampled(3)))

        assert [c.token_ids for c in without.outputs] == [c.token_ids for c in with_cache.outputs]
        assert without.usage.cached_tokens == 0
        assert with_cache.usage.cached_tokens > 0

    def test_streaming_fragments_name_their_sequence_and_add_up(self, llm: LLM) -> None:
        whole = _one(llm.generate(_PROMPT, _sampled(2)))
        pieces: dict[int, list[int]] = {0: [], 1: []}
        finished = []

        for fragment in llm.stream(_PROMPT, _sampled(2)):
            for completion in fragment.outputs:
                pieces[completion.index] += completion.token_ids
            finished.append(fragment.finished)

        assert finished[-1]
        assert not any(finished[:-1])
        assert tuple(pieces[0]) == whole.outputs[0].token_ids
        assert tuple(pieces[1]) == whole.outputs[1].token_ids

    def test_aborting_the_parent_aborts_every_sequence(self, checkpoint: TinyCheckpoint) -> None:
        engine = _engine(checkpoint)
        parent = RequestId("p")
        engine.add_request(GenerateRequest(request_id=parent, prompt=_PROMPT, sampling=_sampled(2)))
        engine.step()

        assert engine.abort(parent)
        outputs = _drain(engine)

        [final] = [out for out in outputs if out.finished]
        assert final.request_id == parent
        assert [c.finish_reason for c in final.outputs] == [FinishReason.CANCELLED] * 2
        assert_cache_released(engine)
        engine.shutdown()

    def test_cancelling_one_sequence_leaves_the_other_running(
        self, checkpoint: TinyCheckpoint
    ) -> None:
        engine = _engine(checkpoint)
        parent = RequestId("p")
        engine.add_request(GenerateRequest(request_id=parent, prompt=_PROMPT, sampling=_sampled(2)))
        engine.step()

        engine.scheduler.abort(RequestId("p:0"))
        outputs = _drain(engine)

        [final] = [out for out in outputs if out.finished]
        assert final.outputs[0].finish_reason is FinishReason.CANCELLED
        assert final.outputs[1].finish_reason is FinishReason.LENGTH
        assert len(final.outputs[1].token_ids) == 8
        assert final.error is None
        assert_cache_released(engine)
        engine.shutdown()

    def test_one_failed_sequence_fails_the_request(self, checkpoint: TinyCheckpoint) -> None:
        engine = _engine(checkpoint)
        parent = RequestId("p")
        engine.add_request(GenerateRequest(request_id=parent, prompt=_PROMPT, sampling=_sampled(2)))
        engine.step()

        engine.scheduler.abort(RequestId("p:1"), FinishReason.ERROR)
        outputs = _drain(engine)

        [final] = [out for out in outputs if out.finished]
        reasons = {c.index: c.finish_reason for c in final.outputs}
        assert reasons[1] is FinishReason.ERROR
        assert reasons[0] is FinishReason.CANCELLED
        assert not engine.has_unfinished
        assert_cache_released(engine)
        engine.shutdown()

    def test_a_used_parent_id_is_refused_without_queueing(self, checkpoint: TinyCheckpoint) -> None:
        engine = _engine(checkpoint)
        parent = RequestId("p")
        engine.add_request(GenerateRequest(request_id=parent, prompt=_PROMPT, sampling=_sampled(2)))
        waiting = engine.scheduler.num_waiting

        with pytest.raises(InvalidRequestError, match="already in use"):
            engine.add_request(
                GenerateRequest(request_id=parent, prompt=_PROMPT, sampling=_sampled(2))
            )

        assert engine.scheduler.num_waiting == waiting
        _drain(engine)
        engine.shutdown()


class TestStats:
    def test_every_step_is_recorded_and_an_idle_engine_records_nothing(
        self, llm: LLM, recorder: Recorder
    ) -> None:
        llm.generate(_PROMPT, SamplingParams(temperature=0.0, max_tokens=4))
        steps = len(recorder.steps)

        llm.engine.step()  # idle

        assert steps >= 4
        assert len(recorder.steps) == steps
        assert recorder.steps[0].num_scheduled_tokens > 0
        assert sum(step.num_new_tokens for step in recorder.steps) == 4
        assert sum(step.num_finished for step in recorder.steps) == 1
        assert all(step.step_seconds >= 0 for step in recorder.steps)

    def test_the_interval_line_shows_scheduled_tokens(
        self, llm: LLM, caplog: pytest.LogCaptureFixture
    ) -> None:
        llm.generate(_PROMPT, SamplingParams(temperature=0.0, max_tokens=3))
        stats = llm.engine.stats
        assert stats is not None

        with caplog.at_level(logging.INFO, logger="ayaka.engine.stats"):
            stats.log()

        [line] = [r.getMessage() for r in caplog.records if r.name == "ayaka.engine.stats"]
        assert "scheduled_tokens=0 " not in line
        assert "output_tokens=3 " in line

    def test_a_chunked_prompt_records_a_step_with_no_new_tokens(
        self, checkpoint: TinyCheckpoint
    ) -> None:
        recorder = Recorder()
        config = tiny_engine_config(checkpoint.directory, max_tokens_per_step=8)
        engine = build_engine(config, tokenizer=tiny_tokenizer(), stat_loggers=(recorder,))
        engine.add_request(
            GenerateRequest(
                request_id=RequestId("long"),
                prompt="a prompt long enough for three chunks of eight",
                sampling=SamplingParams(temperature=0.0, max_tokens=1),
            )
        )

        _drain(engine)

        assert recorder.steps[0].num_scheduled_tokens == 8
        assert recorder.steps[0].num_new_tokens == 0
        assert engine.scheduler.take_stats() is None
        engine.shutdown()

    def test_no_request_id_reaches_a_metric(self) -> None:
        names = {field.name for field in dataclasses.fields(IterationStats)}
        assert not any("request" in name for name in names)
