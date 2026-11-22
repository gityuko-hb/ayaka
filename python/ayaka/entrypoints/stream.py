"""Server-sent events: framing, and the loop that moves one request's events to its client.

A streamed answer goes::

    RequestStream (engine side, bounded, merges when the reader lags)
        -> OutputParser (text, reasoning, tool calls, the end)
        -> StreamEncoder (one protocol's frames)
        -> the client socket

:func:`pump` owns that loop; it can wait a few milliseconds after a token to put more tokens in
the same frame. Time is kept by one :class:`StreamWatchdog` for every open stream, which ticks about
once a second instead of arming a timer per frame. It asks a stream for a keepalive frame when
nothing was written for a while, so a proxy does not cut the connection during a long prompt, and it
gives up on a slow client, so the caller cancels the request (:class:`SlowClientError`):

* when the oldest byte written to the client has not been taken for ``slow_after`` seconds. Taken
  means acknowledged by the client's TCP: what waits in the transport's buffer and, on Linux, in
  the kernel's send queue (``SIOCOUTQ``) is not. The clock therefore starts when the client stops
  reading, not once the kernel's send buffer (it grows to megabytes) has filled up;
* or when one write has been blocked for ``slow_after`` seconds (elsewhere the only signal).

The engine itself never waits for a client: the stream in front of it merges instead.

Every stream ends with exactly one terminal event (a finish or an error) followed by the
protocol's closing frames. If something fails after the response started, the pump still writes
an error event in the protocol's shape before it gives up.
"""

from __future__ import annotations

import asyncio
import json
import logging
import struct
import sys
import time
from collections import deque
from typing import TYPE_CHECKING, Final, Protocol

from ayaka.entrypoints.conversation import ErrorEvent, FinishEvent

if TYPE_CHECKING:
    from asyncio import WriteTransport
    from collections.abc import Callable, Sequence

    from ayaka.engine.async_llm import RequestStream
    from ayaka.entrypoints.conversation import OutputEvent
    from ayaka.entrypoints.parsers.output import EventParser

__all__ = [
    "KEEPALIVE_COMMENT",
    "WRITE_BUFFER_HIGH",
    "SlowClientError",
    "StreamEncoder",
    "StreamWatch",
    "StreamWatchdog",
    "Writer",
    "collect_events",
    "pump",
    "sse_comment",
    "sse_json",
]

_LOG = logging.getLogger("ayaka.serve")


def sse_json(payload: object, *, event: str | None = None) -> bytes:
    """One SSE frame whose data is ``payload`` as compact JSON."""
    data = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    head = f"event: {event}\n" if event is not None else ""
    return f"{head}data: {data}\n\n".encode()


def sse_comment(text: str) -> bytes:
    """An SSE comment line; clients ignore it, proxies see traffic."""
    if "\n" in text or "\r" in text:
        raise ValueError("an SSE comment is one line")
    return f": {text}\n\n".encode()


#: The keepalive frame of the protocols that use a comment for it.
KEEPALIVE_COMMENT: Final = sse_comment("keepalive")

#: Bytes a stream's transport buffers before a write waits for the client: the server's own share
#: of what a client that stopped reading holds up.
WRITE_BUFFER_HIGH: Final = 64 * 1024


class StreamEncoder(Protocol):
    """Turns events into one protocol's SSE frames."""

    def start(self) -> bytes:
        """Frames sent before the first event."""
        ...

    def encode(self, event: OutputEvent) -> bytes:
        """Frames for one event; may be empty."""
        ...

    def keepalive(self) -> bytes:
        """A frame that carries nothing but keeps the connection busy."""
        ...

    def end(self) -> bytes:
        """Frames sent after the terminal event."""
        ...


class Writer(Protocol):
    """Where frames go: an aiohttp ``StreamResponse`` in the server."""

    async def write(self, data: bytes) -> None: ...


class SlowClientError(Exception):
    """The client did not take what was written to it in time."""


def _kernel_unsent(transport: WriteTransport) -> int:
    """Bytes in the socket's send queue the client has not acknowledged; 0 where unknown."""
    if sys.platform != "linux":
        return 0
    import fcntl
    import termios

    sock = transport.get_extra_info("socket")
    if sock is None:
        return 0
    try:
        raw = fcntl.ioctl(sock.fileno(), termios.TIOCOUTQ, b"\0\0\0\0")  # SIOCOUTQ
    except (OSError, ValueError):
        return 0
    return struct.unpack("i", raw)[0]


class StreamWatch:
    """The watchdog's record of one stream; made by :meth:`StreamWatchdog.watch`.

    The pump reports every write to it; the watchdog reads it on each tick.
    """

    def __init__(
        self,
        owner: StreamWatchdog,
        stream: RequestStream,
        *,
        keepalive: float,
        slow_after: float,
        transport: WriteTransport | None,
    ) -> None:
        self._owner = owner
        self.stream = stream
        self.keepalive = keepalive
        self.slow_after = slow_after
        self.transport = transport
        now = owner.clock()
        self.slow: str | None = None
        """Why the client was given up on, once it was."""
        self._last_write = now
        self._keepalive_due = False
        self._task: asyncio.Task[object] | None = None  # the pump, while it is in a write
        self._write_started = now
        self._writing = False
        # Bytes written so far and, per write, (bytes written up to its end, when): the oldest
        # entry not yet taken by the client tells how long the client has held up the stream.
        pending = self._pending()
        self._total = pending or 0
        self._sent: deque[tuple[int, float]] = deque(((self._total, now),) if pending else ())

    def close(self) -> None:
        """Stop watching the stream."""
        self._owner.forget(self)

    # -- the pump's side ----------------------------------------------------------------------

    def begin_write(self) -> None:
        """A write starts; the watchdog may cancel it if it blocks too long."""
        self._writing = True
        self._write_started = self._owner.clock()
        self._task = asyncio.current_task()

    def end_write(self, size: int) -> None:
        """A write of ``size`` bytes handed them to the transport (or failed)."""
        now = self._owner.clock()
        self._writing = False
        self._task = None
        self._last_write = now
        self._keepalive_due = False
        if size and self.transport is not None:
            self._total += size
            self._sent.append((self._total, now))

    def take_keepalive(self) -> bool:
        """Whether a keepalive frame is due; asking clears it."""
        due, self._keepalive_due = self._keepalive_due, False
        return due

    def raise_if_slow(self) -> None:
        """Raise :class:`SlowClientError` once the watchdog gave up on the client."""
        if self.slow is not None:
            raise SlowClientError(self.slow)

    # -- the watchdog's side ------------------------------------------------------------------

    def _pending(self) -> int | None:
        """Bytes written and not yet taken by the client; ``None`` without a transport."""
        transport = self.transport
        if transport is None or transport.is_closing():
            return None
        return transport.get_write_buffer_size() + _kernel_unsent(transport)

    def check(self, now: float) -> None:
        """One tick: give up on a slow client, or ask for a keepalive."""
        if self.slow is not None:
            return
        if self._writing and now - self._write_started > self.slow_after:
            self._give_up(f"a write blocked for more than {self.slow_after:g} s")
            return
        pending = self._pending()
        if pending is not None:
            taken = self._total - pending
            sent = self._sent
            while sent and sent[0][0] <= taken:
                sent.popleft()
            if sent and now - sent[0][1] > self.slow_after:
                self._give_up(f"the client took nothing for more than {self.slow_after:g} s")
                return
        if not self._writing and now - self._last_write >= self.keepalive:
            self._keepalive_due = True
            self.stream.wake()

    def _give_up(self, reason: str) -> None:
        self.slow = reason
        if self._task is not None:
            self._task.cancel()  # a write that may never return; _write turns it into the error
        self.stream.wake()


class StreamWatchdog:
    """One timer for every open stream: keepalives and slow clients.

    The ticking task starts with the first stream watched and ends with the last one closed.

    Args:
        tick: Seconds between two checks.
        clock: Monotonic seconds.
    """

    def __init__(self, *, tick: float = 1.0, clock: Callable[[], float] = time.monotonic) -> None:
        if tick <= 0:
            raise ValueError("tick must be > 0")
        self.tick = tick
        self.clock = clock
        self._watches: set[StreamWatch] = set()
        self._task: asyncio.Task[None] | None = None

    def __len__(self) -> int:
        return len(self._watches)

    def watch(
        self,
        stream: RequestStream,
        *,
        keepalive: float,
        slow_after: float,
        transport: WriteTransport | None = None,
    ) -> StreamWatch:
        """Start watching a stream; call from the event loop.

        Args:
            stream: The request's outputs, woken for a keepalive or a give-up.
            keepalive: Idle seconds before a keepalive frame.
            slow_after: Seconds a client may leave written bytes untaken, or a write blocked.
            transport: The client's connection, to see what it has not taken; without it only
                blocked writes count.
        """
        watch = StreamWatch(
            self, stream, keepalive=keepalive, slow_after=slow_after, transport=transport
        )
        self._watches.add(watch)
        if self._task is None or self._task.done():
            self._task = asyncio.get_running_loop().create_task(self._run())
        return watch

    def forget(self, watch: StreamWatch) -> None:
        """Stop watching; :meth:`StreamWatch.close` calls it."""
        self._watches.discard(watch)

    def check(self) -> None:
        """Check every stream once (what each tick does)."""
        now = self.clock()
        for watch in tuple(self._watches):
            watch.check(now)

    async def _run(self) -> None:
        while self._watches:
            await asyncio.sleep(self.tick)
            try:
                self.check()
            except Exception:
                _LOG.exception("the stream watchdog failed a check")


async def _write(watch: StreamWatch, writer: Writer, data: bytes) -> None:
    if not data:
        return
    watch.begin_write()
    try:
        await writer.write(data)
    except asyncio.CancelledError:
        task = asyncio.current_task()
        if watch.slow is not None and task is not None and task.uncancel() == 0:
            raise SlowClientError(watch.slow) from None
        raise
    except ConnectionError:
        watch.raise_if_slow()
        raise
    finally:
        watch.end_write(len(data))
    watch.raise_if_slow()


async def pump(
    stream: RequestStream,
    parser: EventParser,
    encoder: StreamEncoder,
    writer: Writer,
    *,
    watch: StreamWatch,
    coalesce: float = 0.0,
    observe: Callable[[Sequence[OutputEvent]], None] | None = None,
) -> FinishEvent | ErrorEvent:
    """Stream one request to its client until the terminal event.

    Args:
        stream: The request's outputs.
        parser: Its output parser.
        encoder: The protocol's frames.
        writer: The client.
        watch: The watchdog's record of this stream: keepalives and the slow-client rules.
        coalesce: Seconds to wait after an output for more, to share a frame.
        observe: Called with each batch of events (metrics, logs).

    Returns:
        The terminal event that was sent.

    Raises:
        SlowClientError: If the watchdog gave up on the client; the caller cancels the request.
        ConnectionError: If the client went away.
    """
    await _write(watch, writer, encoder.start())
    try:
        while True:
            await stream.wait()
            watch.raise_if_slow()
            if coalesce > 0 and not stream.finished:
                await asyncio.sleep(coalesce)
            output = stream.drain_nowait()
            if output is None:
                if watch.take_keepalive():
                    await _write(watch, writer, encoder.keepalive())
                continue
            events = parser.feed(output)
            if observe is not None:
                observe(events)
            frames = b"".join(encoder.encode(event) for event in events)
            terminal = _terminal(events)
            if terminal is not None:
                await _write(watch, writer, frames + encoder.end())
                return terminal
            await _write(watch, writer, frames)
    except (SlowClientError, ConnectionError, asyncio.CancelledError):
        raise
    except Exception:
        _LOG.exception("a stream failed after it started")
        failure = ErrorEvent("internal_error", "the server failed while streaming the answer")
        await _write(watch, writer, encoder.encode(failure) + encoder.end())
        return failure


async def collect_events(
    stream: RequestStream,
    parser: EventParser,
    *,
    observe: Callable[[Sequence[OutputEvent]], None] | None = None,
) -> list[OutputEvent]:
    """Every event of a request that is answered in one response, up to the terminal one."""
    events: list[OutputEvent] = []
    async for output in stream:
        batch = parser.feed(output)
        if observe is not None:
            observe(batch)
        events += batch
        if _terminal(batch) is not None:
            break
    return events


def _terminal(events: Sequence[OutputEvent]) -> FinishEvent | ErrorEvent | None:
    if events and isinstance(events[-1], (FinishEvent, ErrorEvent)):
        return events[-1]
    return None
