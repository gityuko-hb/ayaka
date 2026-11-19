"""Server-sent events: framing, and the loop that moves one request's events to its client.

A streamed answer goes::

    RequestStream (engine side, bounded, merges when the reader lags)
        -> OutputParser (text, reasoning, tool calls, the end)
        -> StreamEncoder (one protocol's frames)
        -> the client socket, each write with a deadline

:func:`pump` owns that loop. It sends a keepalive frame whenever nothing happened for a while, so
a proxy does not cut the connection during a long prompt; it can wait a few milliseconds after a
token to put more tokens in the same frame; and it gives up on a client whose socket has not taken
a write within the deadline, raising :class:`SlowClientError` so the caller cancels the request.
The engine itself never waits for a client: the stream in front of it merges instead.

Every stream ends with exactly one terminal event (a finish or an error) followed by the
protocol's closing frames. If something fails after the response started, the pump still writes
an error event in the protocol's shape before it gives up.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import TYPE_CHECKING, Final, Protocol

from ayaka.entrypoints.conversation import ErrorEvent, FinishEvent

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from ayaka.engine.async_llm import RequestStream
    from ayaka.entrypoints.conversation import OutputEvent
    from ayaka.entrypoints.parsers.output import EventParser

__all__ = [
    "KEEPALIVE_COMMENT",
    "SlowClientError",
    "StreamEncoder",
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
    """The client did not take a write within the deadline."""


async def _write(writer: Writer, data: bytes, deadline: float) -> None:
    if not data:
        return
    try:
        async with asyncio.timeout(deadline):
            await writer.write(data)
    except TimeoutError:
        raise SlowClientError(f"a write blocked for more than {deadline:g} s") from None


async def pump(
    stream: RequestStream,
    parser: EventParser,
    encoder: StreamEncoder,
    writer: Writer,
    *,
    keepalive: float,
    coalesce: float = 0.0,
    write_deadline: float,
    observe: Callable[[Sequence[OutputEvent]], None] | None = None,
) -> FinishEvent | ErrorEvent:
    """Stream one request to its client until the terminal event.

    Args:
        stream: The request's outputs.
        parser: Its output parser.
        encoder: The protocol's frames.
        writer: The client.
        keepalive: Idle seconds before a keepalive frame.
        coalesce: Seconds to wait after an output for more, to share a frame.
        write_deadline: Seconds a write may block.
        observe: Called with each batch of events (metrics, logs).

    Returns:
        The terminal event that was sent.

    Raises:
        SlowClientError: If a write blocked past the deadline; the caller cancels the request.
        ConnectionError: If the client went away.
    """
    await _write(writer, encoder.start(), write_deadline)
    try:
        while True:
            try:
                async with asyncio.timeout(keepalive):
                    await stream.wait()
            except TimeoutError:
                await _write(writer, encoder.keepalive(), write_deadline)
                continue
            if coalesce > 0 and not stream.finished:
                await asyncio.sleep(coalesce)
            output = stream.drain_nowait()
            if output is None:
                continue
            events = parser.feed(output)
            if observe is not None:
                observe(events)
            frames = b"".join(encoder.encode(event) for event in events)
            terminal = _terminal(events)
            if terminal is not None:
                await _write(writer, frames + encoder.end(), write_deadline)
                return terminal
            await _write(writer, frames, write_deadline)
    except (SlowClientError, ConnectionError, asyncio.CancelledError):
        raise
    except Exception:
        _LOG.exception("a stream failed after it started")
        failure = ErrorEvent("internal_error", "the server failed while streaming the answer")
        await _write(writer, encoder.encode(failure) + encoder.end(), write_deadline)
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
