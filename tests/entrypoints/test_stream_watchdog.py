"""One watchdog for every stream: keepalives, blocked writes and bytes a client never takes."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

import pytest

from ayaka.core.outputs import CompletionOutput, RequestOutput
from ayaka.core.types import FinishReason, RequestId
from ayaka.engine.async_llm import RequestStream
from ayaka.entrypoints.conversation import ErrorEvent, FinishEvent
from ayaka.entrypoints.parsers.output import OutputFormat
from ayaka.entrypoints.stream import SlowClientError, StreamWatchdog, pump

if TYPE_CHECKING:
    from ayaka.entrypoints.conversation import OutputEvent

pytestmark = pytest.mark.unit


class Frames:
    """A minimal encoder: one tagged frame per event."""

    def start(self) -> bytes:
        return b"start;"

    def encode(self, event: OutputEvent) -> bytes:
        return f"{type(event).__name__};".encode()

    def keepalive(self) -> bytes:
        return b"keepalive;"

    def end(self) -> bytes:
        return b"end;"


class Sink:
    def __init__(self) -> None:
        self.data = b""

    async def write(self, data: bytes) -> None:
        self.data += data


class Stuck:
    """Takes the first write, then blocks forever, like a socket nobody reads."""

    def __init__(self) -> None:
        self.writes = 0

    async def write(self, data: bytes) -> None:
        del data
        self.writes += 1
        if self.writes > 1:
            await asyncio.Event().wait()


class Transport:
    """The part of a transport the watchdog reads: bytes still waiting to reach the client."""

    def __init__(self) -> None:
        self.waiting = 0

    def get_write_buffer_size(self) -> int:
        return self.waiting

    def is_closing(self) -> bool:
        return False

    def get_extra_info(self, name: str, default: object = None) -> object:
        del name
        return default


class Clock:
    def __init__(self) -> None:
        self.now = 100.0

    def __call__(self) -> float:
        return self.now


def _output(text: str, *, finished: bool = False) -> RequestOutput:
    reason = FinishReason.STOP if finished else None
    return RequestOutput(
        request_id=RequestId("r"),
        outputs=(CompletionOutput(index=0, token_ids=(1,), text=text, finish_reason=reason),),
        finished=finished,
    )


def _parser() -> object:
    return OutputFormat().parser(call_id_prefix="c", tools=False)


def test_an_idle_stream_gets_keepalives_from_the_watchdog_and_none_while_busy() -> None:
    async def main() -> tuple[bytes, FinishEvent | ErrorEvent]:
        stream = RequestStream(RequestId("r"))
        watchdog = StreamWatchdog(tick=0.02)
        sink = Sink()
        task = asyncio.create_task(
            pump(
                stream,
                _parser(),  # type: ignore[arg-type]  # an OutputParser
                Frames(),
                sink,
                watch=watchdog.watch(stream, keepalive=0.1, slow_after=5),
            )
        )
        await asyncio.sleep(0.35)
        stream.put(_output("hi", finished=True))
        terminal = await asyncio.wait_for(task, 2.0)
        return sink.data, terminal

    data, terminal = asyncio.run(main())

    assert isinstance(terminal, FinishEvent)
    assert data.startswith(b"start;keepalive;")
    assert 2 <= data.count(b"keepalive;") <= 4
    assert data.endswith(b"end;")


def test_a_write_blocked_past_the_limit_gives_up_on_the_client() -> None:
    async def main() -> float:
        stream = RequestStream(RequestId("r"))
        watchdog = StreamWatchdog(tick=0.02)
        loop = asyncio.get_running_loop()
        started = loop.time()
        stream.put(_output("never read"))
        with pytest.raises(SlowClientError, match="write blocked"):
            await pump(
                stream,
                _parser(),  # type: ignore[arg-type]  # an OutputParser
                Frames(),
                Stuck(),
                watch=watchdog.watch(stream, keepalive=5, slow_after=0.2),
            )
        return loop.time() - started

    waited = asyncio.run(main())

    assert 0.2 <= waited < 1.0


def test_bytes_the_client_never_takes_give_up_without_a_blocked_write() -> None:
    async def main() -> None:
        clock = Clock()
        stream = RequestStream(RequestId("r"))
        watchdog = StreamWatchdog(tick=60, clock=clock)
        transport = Transport()
        watch = watchdog.watch(stream, keepalive=60, slow_after=10, transport=transport)  # type: ignore[arg-type]  # a WriteTransport
        watch.begin_write()
        watch.end_write(500)  # the write returned: the bytes are in the socket's queues
        transport.waiting = 400  # 100 bytes reached the client, then nothing more

        clock.now += 9
        watchdog.check()
        assert watch.slow is None

        clock.now += 2
        watchdog.check()
        assert watch.slow is not None
        assert "took nothing" in watch.slow
        await asyncio.wait_for(stream.wait(), 1.0)  # the pump is woken to raise
        with pytest.raises(SlowClientError):
            watch.raise_if_slow()
        watch.close()

    asyncio.run(main())


def test_a_client_that_keeps_taking_bytes_is_never_given_up() -> None:
    async def main() -> None:
        clock = Clock()
        stream = RequestStream(RequestId("r"))
        watchdog = StreamWatchdog(tick=60, clock=clock)
        transport = Transport()
        watch = watchdog.watch(stream, keepalive=600, slow_after=10, transport=transport)  # type: ignore[arg-type]  # a WriteTransport
        for _ in range(100):  # a slow link: something is always in flight, but it moves
            watch.begin_write()
            watch.end_write(100)
            transport.waiting = 150
            clock.now += 1
            watchdog.check()
        assert watch.slow is None
        watch.close()

    asyncio.run(main())


def test_the_ticking_task_ends_with_the_last_watched_stream() -> None:
    async def main() -> None:
        watchdog = StreamWatchdog(tick=0.01)
        watch = watchdog.watch(RequestStream(RequestId("r")), keepalive=5, slow_after=5)
        task = watchdog._task  # pyright: ignore[reportPrivateUsage]
        assert task is not None
        assert not task.done()
        watch.close()
        await asyncio.wait_for(task, 1.0)
        assert len(watchdog) == 0

    asyncio.run(main())
