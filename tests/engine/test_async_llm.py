"""The asynchronous facade on the tiny GPT-2: streams, cancellation, slow readers and shutdown.

Every test runs its own event loop with ``asyncio.run``. The oracle is ``greedy_reference`` and the
synchronous facade: a greedy request must give the same tokens through either.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

import pytest

from ayaka.core.errors import EngineDeadError, EngineError, UnsupportedFeatureError
from ayaka.core.outputs import CompletionOutput, RequestOutput, Usage
from ayaka.core.params import SamplingParams
from ayaka.core.request import GenerateRequest
from ayaka.core.types import FinishReason, RequestId
from tokenizer_fakes import BOS

torch: Any = pytest.importorskip("torch")

from ayaka.engine.async_llm import AsyncLLM, RequestStream, merge_outputs  # noqa: E402
from ayaka.engine.build import build_engine  # noqa: E402
from ayaka.engine.seam import EngineState  # noqa: E402
from tiny_model import (  # noqa: E402  # needs the importorskip above
    TinyCheckpoint,
    assert_cache_released,
    greedy_reference,
    tiny_engine_config,
    tiny_tokenizer,
    write_tiny_gpt2,
)

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from ayaka.scheduler.step import StepOutput

pytestmark = pytest.mark.unit


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory: pytest.TempPathFactory) -> TinyCheckpoint:
    return write_tiny_gpt2(tmp_path_factory.mktemp("tiny-gpt2"))


def _request(name: str, prompt: str, max_tokens: int, **more: Any) -> GenerateRequest:
    return GenerateRequest(
        request_id=RequestId(name),
        prompt=prompt,
        sampling=SamplingParams(temperature=0.0, max_tokens=max_tokens, **more),
    )


def _expected(checkpoint: TinyCheckpoint, prompt: str, count: int) -> list[int]:
    return greedy_reference(checkpoint.state, checkpoint.config, [BOS, *prompt.encode()], count)


def _run(
    checkpoint: TinyCheckpoint,
    body: Callable[[AsyncLLM], Awaitable[None]],
    *,
    max_pending: int = 64,
    grace: float = 0.0,
) -> AsyncLLM:
    """Build an engine, start the facade in a fresh loop, run ``body``, then shut it down."""
    engine = build_engine(tiny_engine_config(checkpoint.directory), tokenizer=tiny_tokenizer())
    llm = AsyncLLM(engine, max_pending=max_pending)

    async def main() -> None:
        llm.start()
        try:
            await body(llm)
        finally:
            await llm.shutdown(grace)

    asyncio.run(main())
    return llm


async def _collect(stream: RequestStream) -> list[RequestOutput]:
    return [output async for output in stream]


def _tokens(outputs: list[RequestOutput]) -> list[int]:
    return [token for output in outputs for token in output.outputs[0].token_ids]


class TestGenerate:
    def test_a_stream_gives_the_reference_tokens_and_ends_once(
        self, checkpoint: TinyCheckpoint
    ) -> None:
        seen: list[RequestOutput] = []

        async def body(llm: AsyncLLM) -> None:
            seen.extend(await _collect(await llm.add(_request("a", "hello", 8))))

        llm = _run(checkpoint, body)

        assert _tokens(seen) == _expected(checkpoint, "hello", 8)
        assert [output.finished for output in seen] == [False] * (len(seen) - 1) + [True]
        assert seen[-1].outputs[0].finish_reason is FinishReason.LENGTH
        assert seen[-1].usage.completion_tokens == 8
        assert llm.num_streams == 0

    def test_concurrent_requests_each_get_their_own_tokens(
        self, checkpoint: TinyCheckpoint
    ) -> None:
        prompts = {"one": ("first prompt", 6), "two": ("a second, longer prompt", 9), "3": ("x", 4)}
        results: dict[str, list[int]] = {}

        async def body(llm: AsyncLLM) -> None:
            async def one(name: str, prompt: str, count: int) -> None:
                outputs = [o async for o in llm.generate(_request(name, prompt, count))]
                results[name] = _tokens(outputs)

            await asyncio.gather(*(one(name, *spec) for name, spec in prompts.items()))

        _run(checkpoint, body)

        for name, (prompt, count) in prompts.items():
            assert results[name] == _expected(checkpoint, prompt, count), name

    def test_an_unsupported_feature_is_refused_at_admission(
        self, checkpoint: TinyCheckpoint
    ) -> None:
        async def body(llm: AsyncLLM) -> None:
            with pytest.raises(UnsupportedFeatureError, match="prompt_logprobs"):
                await llm.add(_request("lp", "x", 2, prompt_logprobs=1))
            assert llm.num_streams == 0
            # The engine is still fine.
            assert len(await _collect(await llm.add(_request("ok", "x", 2)))) >= 1

        _run(checkpoint, body)

    def test_a_running_id_cannot_be_used_twice(self, checkpoint: TinyCheckpoint) -> None:
        async def body(llm: AsyncLLM) -> None:
            stream = await llm.add(_request("same", "x", 20))
            with pytest.raises(EngineError, match="already running"):
                await llm.add(_request("same", "y", 2))
            await _collect(stream)

        _run(checkpoint, body)

    def test_a_request_before_start_is_refused(self, checkpoint: TinyCheckpoint) -> None:
        engine = build_engine(tiny_engine_config(checkpoint.directory), tokenizer=tiny_tokenizer())
        llm = AsyncLLM(engine)

        async def main() -> None:
            with pytest.raises(EngineError, match="not started"):
                await llm.add(_request("early", "x", 2))
            await llm.shutdown()

        asyncio.run(main())


class TestCancellation:
    def test_abort_ends_the_stream_cancelled_and_frees_the_cache(
        self, checkpoint: TinyCheckpoint
    ) -> None:
        async def body(llm: AsyncLLM) -> None:
            stream = await llm.add(_request("long", "go on", 100))
            await stream.wait()
            llm.abort(RequestId("long"))
            assert llm.num_streams == 0
            assert await llm.drain(5.0)
            assert_cache_released(llm.engine)

        _run(checkpoint, body)

    def test_leaving_a_generator_cancels_only_that_request(
        self, checkpoint: TinyCheckpoint
    ) -> None:
        other: list[RequestOutput] = []

        async def body(llm: AsyncLLM) -> None:
            async def leave_early() -> None:
                generator = llm.generate(_request("leaver", "leave", 100))
                async for _ in generator:
                    break
                await generator.aclose()

            async def stay() -> None:
                other.extend([o async for o in llm.generate(_request("stayer", "stay", 12))])

            await asyncio.gather(leave_early(), stay())
            assert await llm.drain(5.0)
            assert_cache_released(llm.engine)

        _run(checkpoint, body)

        assert _tokens(other) == _expected(checkpoint, "stay", 12)
        assert other[-1].outputs[0].finish_reason is FinishReason.LENGTH

    def test_a_caller_cancelled_during_admission_leaves_nothing_behind(
        self, checkpoint: TinyCheckpoint
    ) -> None:
        async def body(llm: AsyncLLM) -> None:
            task = asyncio.create_task(llm.add(_request("gone", "bye", 100)))
            await asyncio.sleep(0)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            assert llm.num_streams == 0
            assert await llm.drain(5.0)
            assert_cache_released(llm.engine)

        _run(checkpoint, body)


class TestSlowReader:
    def test_a_reader_that_falls_behind_gets_merged_outputs_and_loses_nothing(
        self, checkpoint: TinyCheckpoint
    ) -> None:
        outputs: list[RequestOutput] = []
        merged: list[int] = []

        async def body(llm: AsyncLLM) -> None:
            stream = await llm.add(_request("slow", "slow reader", 20))
            # Let the engine run the whole request before reading anything.
            assert await llm.drain(10.0)
            merged.append(stream.merged)
            outputs.extend(await _collect(stream))

        _run(checkpoint, body, max_pending=2)

        assert merged[0] > 0
        assert len(outputs) <= 2
        assert _tokens(outputs) == _expected(checkpoint, "slow reader", 20)
        assert outputs[-1].finished


class TestFailureAndShutdown:
    def test_shutdown_with_no_grace_cancels_what_is_running(
        self, checkpoint: TinyCheckpoint
    ) -> None:
        last: list[RequestOutput] = []

        async def body(llm: AsyncLLM) -> None:
            stream = await llm.add(_request("cut", "cut short", 100))
            await stream.wait()
            await llm.shutdown(0.0)
            last.extend(await _collect(stream))
            with pytest.raises(EngineError, match="shutting down"):
                await llm.add(_request("late", "x", 2))

        llm = _run(checkpoint, body)

        assert last[-1].finished
        assert last[-1].outputs[0].finish_reason is FinishReason.CANCELLED
        assert llm.health().state is EngineState.STOPPED

    def test_shutdown_with_grace_lets_a_request_finish(self, checkpoint: TinyCheckpoint) -> None:
        last: list[RequestOutput] = []

        async def body(llm: AsyncLLM) -> None:
            stream = await llm.add(_request("short", "short", 5))
            await llm.shutdown(30.0)
            last.extend(await _collect(stream))

        _run(checkpoint, body)

        assert last[-1].outputs[0].finish_reason is FinishReason.LENGTH
        assert _tokens(last) == _expected(checkpoint, "short", 5)

    def test_a_dead_engine_ends_every_stream_with_an_error_and_refuses_more(
        self, checkpoint: TinyCheckpoint
    ) -> None:
        last: list[RequestOutput] = []

        async def body(llm: AsyncLLM) -> None:
            def explode(step: Any, drafts: Any = None) -> StepOutput:
                del step, drafts
                raise RuntimeError("kernel on fire")

            llm.engine.runner.execute = explode  # type: ignore[method-assign]  # inject a failure
            stream = await llm.add(_request("doomed", "x", 5))
            last.extend(await _collect(stream))
            assert llm.health().state is EngineState.DEAD
            with pytest.raises(EngineDeadError):
                await llm.add(_request("after", "x", 2))

        llm = _run(checkpoint, body)

        [final] = last
        assert final.finished
        assert final.outputs[0].finish_reason is FinishReason.ERROR
        assert final.error is not None
        assert final.error.code == "engine_dead"
        assert llm.health().state is EngineState.DEAD


class TestMerge:
    def test_merging_concatenates_text_and_tokens_and_keeps_the_newer_end(self) -> None:
        first = RequestOutput(
            request_id=RequestId("r"),
            outputs=(CompletionOutput(index=0, token_ids=(1, 2), text="ab"),),
        )
        second = RequestOutput(
            request_id=RequestId("r"),
            outputs=(
                CompletionOutput(
                    index=0, token_ids=(3,), text="c", finish_reason=FinishReason.STOP
                ),
            ),
            finished=True,
            usage=Usage(prompt_tokens=4, completion_tokens=3),
        )

        merged = merge_outputs(first, second)

        assert merged.outputs[0].token_ids == (1, 2, 3)
        assert merged.outputs[0].text == "abc"
        assert merged.outputs[0].finish_reason is FinishReason.STOP
        assert merged.finished
        assert merged.usage.completion_tokens == 3

    def test_a_stream_merges_beyond_its_bound_and_drains_in_one(self) -> None:
        async def main() -> None:
            stream = RequestStream(RequestId("r"), max_pending=2)
            for i, text in enumerate("abcde"):
                stream.put(
                    RequestOutput(
                        request_id=RequestId("r"),
                        outputs=(CompletionOutput(index=0, token_ids=(i,), text=text),),
                        finished=text == "e",
                    )
                )
            assert stream.pending == 2
            assert stream.merged == 3
            output = stream.drain_nowait()
            assert output is not None
            assert output.outputs[0].text == "abcde"
            assert output.finished
            assert [o async for o in stream] == []

        asyncio.run(main())

    def test_a_stream_ignores_outputs_after_its_final_one(self) -> None:
        stream = RequestStream(RequestId("r"))
        final = RequestOutput(request_id=RequestId("r"), finished=True)
        stream.put(final)
        stream.put(RequestOutput(request_id=RequestId("r")))

        assert stream.pending == 1
        assert stream.finished
