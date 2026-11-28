"""Response streaming, non-streaming collection, and active-stream shutdown for :class:`Serving`."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from typing import TYPE_CHECKING

from aiohttp import web

from ayaka.entrypoints.conversation import collect, collect_choices
from ayaka.entrypoints.failures import AnswerFailedError
from ayaka.entrypoints.stream import (
    WRITE_BUFFER_HIGH,
    SlowClientError,
    StreamWatchdog,
    collect_events,
    pump,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from ayaka.configs.server import ServerConfig
    from ayaka.engine.seam import EngineClient
    from ayaka.entrypoints.conversation import Collected, OutputEvent
    from ayaka.entrypoints.metrics import ServingMetrics
    from ayaka.entrypoints.serving import Generation
    from ayaka.entrypoints.stream import StreamEncoder

__all__ = ["ResponseDelivery"]

_LOG = logging.getLogger("ayaka.serve")


class ResponseDelivery:
    """Delivers admitted :class:`~ayaka.entrypoints.serving.Generation` answers to HTTP clients."""

    def __init__(self, config: ServerConfig, metrics: ServingMetrics) -> None:
        self._config = config
        self._metrics = metrics
        # Requests whose event stream is open, by id(): an aiohttp request is not hashable.
        self._streaming: dict[int, web.Request] = {}
        self._no_streams = asyncio.Event()
        self._no_streams.set()
        # One timer for every event stream; it ticks at least twice per keepalive interval.
        self.watchdog = StreamWatchdog(
            tick=max(0.01, min(1.0, config.keepalive_seconds / 2, config.slow_client_seconds / 4))
        )

    async def cut_stuck_streams(self, within: float) -> int:
        """Give open event streams ``within`` seconds to end, then abort the ones still writing.

        Aborting a transport drops what it has buffered and fails the blocked write, so the
        handler ends at once; closing it would wait for the buffer to drain, which never happens.
        """
        with contextlib.suppress(TimeoutError):
            await asyncio.wait_for(self._no_streams.wait(), within)
        stuck = [request.transport for request in self._streaming.values()]
        for transport in stuck:
            if transport is not None:
                transport.abort()
        return len(stuck)

    async def stream(
        self,
        request: web.Request,
        generation: Generation,
        encoder: StreamEncoder,
        *,
        request_id: str,
    ) -> web.StreamResponse:
        """Answer with server-sent events until the terminal event.

        A client that stops taking what is written to it for ``slow_client_seconds``, or that went
        away, has its connection cut; :meth:`finish` then cancels the request in the engine.
        """
        config = self._config
        response = web.StreamResponse(
            headers={
                "Content-Type": "text/event-stream; charset=utf-8",
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
                "x-request-id": request_id,
            }
        )
        await response.prepare(request)
        self._streaming[id(request)] = request  # until the end of the body: cut_stuck_streams
        self._no_streams.clear()
        transport = request.transport
        if transport is not None:
            transport.set_write_buffer_limits(high=WRITE_BUFFER_HIGH)
        watch = self.watchdog.watch(
            generation.stream,
            keepalive=config.keepalive_seconds,
            slow_after=config.slow_client_seconds,
            transport=transport,
        )
        answered = False
        try:
            await pump(
                generation.stream,
                generation.parser,
                encoder,
                response,
                watch=watch,
                coalesce=config.sse_coalesce_ms / 1000,
                observe=self._observer(generation),
            )
            answered = True
            await response.write_eof()
        except SlowClientError as exc:
            generation.outcome = "slow_client"
            self._metrics.finished.labels("slow_client").inc()
            _LOG.info("request %s: cancelled, %s", request_id, exc)
            if transport is not None:
                transport.abort()  # what it holds would never drain: drop it and free the socket
            return response
        except ConnectionError:
            if not answered:  # once every frame is written, a lost end of body changes nothing
                generation.outcome = "client_gone"
                self._metrics.finished.labels("client_gone").inc()
        finally:
            watch.close()
            del self._streaming[id(request)]
            if not self._streaming:
                self._no_streams.set()
        return response

    async def collect(self, generation: Generation) -> Collected:
        """The whole answer of a request answered in one response.

        Raises:
            AnswerFailedError: If the request ended with an error.
        """
        events = await collect_events(
            generation.stream, generation.parser, observe=self._observer(generation)
        )
        answer = collect(events)
        if answer.error is not None:
            raise AnswerFailedError(answer.error)
        return answer

    async def collect_choices(self, generation: Generation, count: int) -> list[Collected]:
        """Every choice of a request with ``n > 1``, answered in one response.

        Raises:
            AnswerFailedError: If the request ended with an error.
        """
        events = await collect_events(
            generation.stream, generation.parser, observe=self._observer(generation)
        )
        answers = collect_choices(events, count)
        for answer in answers:
            if answer.error is not None:
                raise AnswerFailedError(answer.error)
        return answers

    def _observer(self, generation: Generation) -> Callable[[Sequence[OutputEvent]], None]:
        def observe(events: Sequence[OutputEvent]) -> None:
            generation.observe(events)
            self._metrics.observe(events)

        return observe

    def finish(self, generation: Generation, llm: EngineClient | None) -> None:
        """Release a generation: cancel it in the engine if it has not ended, and log it.

        The log line carries the request id, how it ended and token counts, never text.
        """
        if not generation.stream.finished and llm is not None:
            llm.abort(generation.request_id)
            if generation.outcome == "unfinished":
                generation.outcome = "client_gone"
        self._metrics.running.dec()
        seconds = time.monotonic() - generation.started
        self._metrics.duration.observe(seconds)
        usage = generation.usage
        _LOG.info(
            "request %s: %s, %d prompt and %d generated tokens in %.2f s",
            generation.public_id,
            generation.outcome,
            usage.prompt_tokens if usage is not None else generation.prompt_tokens,
            # A request that ended early (a client gone or too slow) has no usage event: count
            # what reached the parser. Tokens the engine made after that are not in it.
            usage.completion_tokens if usage is not None else generation.parser.generated_tokens,
            seconds,
        )
