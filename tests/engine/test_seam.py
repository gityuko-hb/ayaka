"""The seam between the server and the engine: who satisfies it and what it guarantees.

``AsyncLLM`` and the fake engine of the server tests are both ``EngineClient``; pyright checks that
the arguments of ``_client`` satisfy the protocol and ``isinstance`` checks it at run time. The
guarantees are tested on the real engine (the tiny GPT-2 on the CPU): every admitted request ends
with exactly one final output on every path, and the caller that aborts gets nothing more.
"""

from __future__ import annotations

import asyncio
import threading
import time
from typing import TYPE_CHECKING, Any

import pytest

from ayaka.core.errors import (
    EngineDeadError,
    ErrorInfo,
    OverloadedError,
    UnsupportedFeatureError,
)
from ayaka.core.outputs import RequestOutput
from ayaka.core.params import SamplingParams
from ayaka.core.request import GenerateRequest
from ayaka.core.types import FinishReason, RequestId
from ayaka.engine.async_llm import AsyncLLM
from ayaka.engine.seam import EngineClient, EngineInfo, EngineState
from fake_engine import FakeEngine, script

torch: Any = pytest.importorskip("torch")

from ayaka.engine.build import build_engine  # noqa: E402
from ayaka.scheduler.step import StepOutput  # noqa: E402
from tiny_model import (  # noqa: E402  # needs the importorskip above
    TinyCheckpoint,
    assert_cache_released,
    tiny_engine_config,
    tiny_tokenizer,
    write_tiny_gpt2,
)

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from ayaka.engine.async_llm import RequestStream

pytestmark = pytest.mark.unit

_INFO = EngineInfo(max_model_len=128, max_context_tokens=100, prefix_caching=True)


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory: pytest.TempPathFactory) -> TinyCheckpoint:
    return write_tiny_gpt2(tmp_path_factory.mktemp("tiny-gpt2"))


def _client(engine: EngineClient) -> EngineClient:
    """Pyright checks that whatever is passed here satisfies the protocol."""
    return engine


def _request(name: str, max_tokens: int, prompt: str = "go on") -> GenerateRequest:
    return GenerateRequest(
        request_id=RequestId(name),
        prompt=prompt,
        sampling=SamplingParams(temperature=0.0, max_tokens=max_tokens),
    )


def _run(checkpoint: TinyCheckpoint, body: Callable[[AsyncLLM], Awaitable[None]]) -> None:
    """Build an engine, start the facade in a fresh loop, run ``body``, then shut it down."""
    llm = AsyncLLM(
        build_engine(tiny_engine_config(checkpoint.directory), tokenizer=tiny_tokenizer())
    )

    async def main() -> None:
        llm.start()
        try:
            await body(llm)
        finally:
            await llm.shutdown()

    asyncio.run(main())


async def _all(stream: RequestStream) -> list[RequestOutput]:
    return [output async for output in stream]


def _ended_once(outputs: list[RequestOutput]) -> RequestOutput:
    """The final output, after checking that it is the only one and that nothing follows it."""
    assert outputs, "the request ended with no output at all"
    assert [output.finished for output in outputs] == [False] * (len(outputs) - 1) + [True]
    return outputs[-1]


def _nothing_follows(stream: RequestStream) -> None:
    assert stream.get_nowait() is None
    stream.put(RequestOutput(request_id=stream.request_id, finished=True))  # a second end
    assert stream.pending == 0


class TestContract:
    def test_async_llm_is_an_engine_client(self, checkpoint: TinyCheckpoint) -> None:
        engine = build_engine(tiny_engine_config(checkpoint.directory), tokenizer=tiny_tokenizer())
        llm = AsyncLLM(engine)

        assert isinstance(_client(llm), EngineClient)
        assert llm.info == engine.info
        assert llm.info.max_model_len == 128
        assert llm.info.max_context_tokens <= 128
        assert llm.info.prefix_caching is True
        asyncio.run(llm.shutdown())

    def test_the_fake_engine_is_an_engine_client(self) -> None:
        assert isinstance(_client(FakeEngine(_INFO)), EngineClient)

    def test_health_follows_the_life_of_the_engine(self, checkpoint: TinyCheckpoint) -> None:
        llm = AsyncLLM(
            build_engine(tiny_engine_config(checkpoint.directory), tokenizer=tiny_tokenizer())
        )
        assert llm.health().state is EngineState.STOPPED

        async def main() -> None:
            llm.start()
            first = llm.health()
            assert first.state is EngineState.RUNNING
            await _all(await llm.add(_request("a", 3)))
            assert await llm.drain(5.0)
            assert llm.health().progress_at > first.progress_at  # the engine thread made progress
            await llm.shutdown()
            assert llm.health().state is EngineState.STOPPED

        asyncio.run(main())


class TestProgress:
    """``health()`` tells a stuck step from an idle engine: busy, with a clock that stands still."""

    def test_a_blocked_step_is_busy_and_its_progress_stands_still(
        self, checkpoint: TinyCheckpoint
    ) -> None:
        release = threading.Event()

        async def body(llm: AsyncLLM) -> None:
            runner = llm.engine.runner
            real = runner.execute

            def blocked(step: Any, drafts: Any = None) -> StepOutput:
                release.wait(10)
                return real(step, drafts)

            assert not llm.health().busy
            await asyncio.sleep(0.2)  # idle long enough for an old timestamp to show
            runner.execute = blocked  # type: ignore[method-assign]  # a step that hangs
            added = time.monotonic()
            stream = await llm.add(_request("stuck", 3))
            first = llm.health()
            # Work that ends an idle period counts as progress: not the idle timestamp.
            assert first.busy
            assert first.progress_at >= added
            await asyncio.sleep(0.2)
            second = llm.health()
            assert second.busy
            assert second.progress_at == first.progress_at
            release.set()
            _ended_once(await _all(stream))
            assert await llm.drain(5.0)
            done = llm.health()
            assert not done.busy
            assert done.progress_at > second.progress_at

        try:
            _run(checkpoint, body)
        finally:
            release.set()

    def test_the_fake_engine_stalls_and_resumes(self) -> None:
        fake = FakeEngine(_INFO)
        assert not fake.health().busy
        fake.stall()
        stalled = fake.health()
        assert stalled.busy
        assert fake.health().progress_at == stalled.progress_at
        fake.resume()
        assert not fake.health().busy
        assert fake.health().progress_at >= stalled.progress_at


class TestEveryAdmittedRequestEndsOnce:
    def test_a_request_that_finishes(self, checkpoint: TinyCheckpoint) -> None:
        async def body(llm: AsyncLLM) -> None:
            stream = await llm.add(_request("done", 6))
            final = _ended_once(await _all(stream))
            assert final.outputs[0].finish_reason is FinishReason.LENGTH
            _nothing_follows(stream)
            assert llm.num_streams == 0

        _run(checkpoint, body)

    def test_requests_the_engine_cancels(self, checkpoint: TinyCheckpoint) -> None:
        async def body(llm: AsyncLLM) -> None:
            streams = [await llm.add(_request(name, 100)) for name in ("one", "two")]
            await streams[0].wait()
            llm.abort_all()
            for stream in streams:
                final = _ended_once(await _all(stream))
                assert final.outputs[0].finish_reason is FinishReason.CANCELLED
                _nothing_follows(stream)
            assert await llm.drain(5.0)
            assert_cache_released(llm.engine)

        _run(checkpoint, body)

    def test_a_request_that_fails_ends_with_its_error_and_the_engine_goes_on(
        self, checkpoint: TinyCheckpoint
    ) -> None:
        async def body(llm: AsyncLLM) -> None:
            runner = llm.engine.runner
            real = runner.execute
            calls = 0

            def fail_once(step: Any, drafts: Any = None) -> StepOutput:
                nonlocal calls
                calls += 1
                if calls == 1:
                    return StepOutput(
                        step_id=step.step_id,
                        request_ids=step.request_ids,
                        error=ErrorInfo("backend_error", "injected"),
                    )
                return real(step, drafts)

            runner.execute = fail_once  # type: ignore[method-assign]  # inject a failure
            stream = await llm.add(_request("broken", 5))
            final = _ended_once(await _all(stream))
            assert final.outputs[0].finish_reason is FinishReason.ERROR
            assert final.error is not None
            assert final.error.code == "backend_error"
            _nothing_follows(stream)
            assert llm.health().state is EngineState.RUNNING
            assert _ended_once(await _all(await llm.add(_request("after", 3)))).error is None

        _run(checkpoint, body)

    def test_every_open_stream_ends_when_the_engine_dies(self, checkpoint: TinyCheckpoint) -> None:
        async def body(llm: AsyncLLM) -> None:
            runner = llm.engine.runner
            real = runner.execute
            armed = False

            def flaky(step: Any, drafts: Any = None) -> StepOutput:
                if armed:
                    raise RuntimeError("kernel on fire")
                return real(step, drafts)

            runner.execute = flaky  # type: ignore[method-assign]  # inject a failure
            streams = [await llm.add(_request(name, 100)) for name in ("one", "two")]
            armed = True
            for stream in streams:
                final = _ended_once(await _all(stream))
                assert final.outputs[0].finish_reason is FinishReason.ERROR
                assert final.error is not None
                assert final.error.code == "engine_dead"
                _nothing_follows(stream)
            assert llm.health().state is EngineState.DEAD
            with pytest.raises(EngineDeadError):
                await llm.add(_request("late", 2))

        _run(checkpoint, body)

    def test_an_admission_error_leaves_no_trace(self, checkpoint: TinyCheckpoint) -> None:
        async def body(llm: AsyncLLM) -> None:
            refused = GenerateRequest(
                request_id=RequestId("refused"),
                prompt="x",
                sampling=SamplingParams(max_tokens=2, prompt_logprobs=1),
            )
            with pytest.raises(UnsupportedFeatureError, match="prompt_logprobs"):
                await llm.add(refused)
            assert llm.num_streams == 0
            assert await llm.drain(5.0)
            assert_cache_released(llm.engine)

        _run(checkpoint, body)


class TestAbort:
    def test_the_caller_that_aborts_gets_nothing_more(self, checkpoint: TinyCheckpoint) -> None:
        async def body(llm: AsyncLLM) -> None:
            stream = await llm.add(_request("leaving", 100))
            await stream.wait()
            assert stream.pending > 0
            llm.abort(RequestId("leaving"))

            assert stream.pending == 0  # what was waiting is dropped
            assert await _all(stream) == []
            # The engine ends the request at its next step and makes a cancelled output: it is
            # discarded, not delivered.
            assert await llm.drain(5.0)
            assert stream.pending == 0
            assert not stream.finished
            assert llm.num_streams == 0
            assert_cache_released(llm.engine)

        _run(checkpoint, body)

    def test_an_abort_of_an_unknown_or_finished_request_is_ignored(
        self, checkpoint: TinyCheckpoint
    ) -> None:
        async def body(llm: AsyncLLM) -> None:
            stream = await llm.add(_request("over", 3))
            final = _ended_once(await _all(stream))
            llm.abort(RequestId("over"))
            llm.abort(RequestId("never-was"))

            assert final.outputs[0].finish_reason is FinishReason.LENGTH
            assert await llm.drain(5.0)
            assert_cache_released(llm.engine)

        _run(checkpoint, body)


class TestFakeEngine:
    """The fake keeps the same guarantees, so a server test on it means what it says."""

    def test_a_script_is_delivered_with_its_final_output_once(self) -> None:
        async def main() -> None:
            fake = FakeEngine(_INFO, [script("a", "b")])
            stream = await fake.add(_request("r", 5))
            final = _ended_once(await _all(stream))
            assert final.outputs[0].finish_reason is FinishReason.STOP
            _nothing_follows(stream)

        asyncio.run(main())

    def test_an_admission_error_is_raised_and_leaves_no_trace(self) -> None:
        async def main() -> None:
            fake = FakeEngine(_INFO, [OverloadedError("full")])
            with pytest.raises(OverloadedError):
                await fake.add(_request("r", 5))
            assert fake.health().state is EngineState.RUNNING

        asyncio.run(main())

    def test_an_abort_closes_the_stream_with_nothing_more(self) -> None:
        async def main() -> None:
            fake = FakeEngine(_INFO, [script("a", "b", "c")])
            stream = await fake.add(_request("r", 5))
            await stream.wait()
            fake.abort(RequestId("r"))
            await asyncio.sleep(0.05)  # the script would have run to its end by now
            assert fake.aborted == [RequestId("r")]
            assert stream.pending == 0
            assert await _all(stream) == []

        asyncio.run(main())

    def test_shutdown_cancels_and_death_ends_open_streams_once(self) -> None:
        hanging = script("a", "b")[:-1]  # no final output: the request hangs

        async def main() -> None:
            fake = FakeEngine(_INFO, [hanging, hanging])
            cancelled = await fake.add(_request("one", 5))
            await fake.shutdown()
            final = _ended_once(await _all(cancelled))
            assert final.outputs[0].finish_reason is FinishReason.CANCELLED
            assert fake.health().state is EngineState.STOPPED

            other = FakeEngine(_INFO, [hanging])
            doomed = await other.add(_request("two", 5))
            other.die()
            final = _ended_once(await _all(doomed))
            assert final.error is not None
            assert final.error.code == "engine_dead"
            assert other.health().state is EngineState.DEAD
            with pytest.raises(EngineDeadError):
                await other.add(_request("three", 5))

        asyncio.run(main())
