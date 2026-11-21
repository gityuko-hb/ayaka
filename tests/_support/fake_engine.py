"""A fake engine for the server tests: an ``EngineClient`` that answers from scripts.

The server reaches the engine only through ``ayaka.engine.seam.EngineClient``, so a test that is
about the server (its routes, its error mapping, its streams) can put a ``FakeEngine`` in the
engine's place and decide what the engine does:

* ``script(...)`` writes the outputs of one request; ``FakeEngine(info, scripts)`` answers the
  requests it gets with the scripts in order (``script("ok")`` when they run out).
* A script that is an exception is raised from ``add``: an admission error (overloaded, invalid).
* A script without its final output leaves the request open, hanging, until the test calls
  ``abort``, ``abort_all``, ``shutdown`` or ``die``.
* ``die()`` kills the fake engine: every open stream ends with an ``engine_dead`` error output
  and every later ``add`` raises.
* ``stall()`` makes the engine look stuck in a step: ``health()`` reports it busy with a progress
  time that no longer moves, until ``resume()``.

The injection points are arguments of the test, never flags in the product code. ``requests``
and ``aborted`` record what the server sent. The fake keeps the contract of the seam: every
request that is not aborted by the caller ends with exactly one final output, and an aborted
one gets nothing more.

Import this module only from a test that can import ``ayaka.engine.async_llm``.
"""

from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING

from ayaka.core.errors import EngineDeadError, EngineError, ErrorInfo
from ayaka.core.outputs import CompletionOutput, RequestOutput, Usage
from ayaka.core.types import FinishReason
from ayaka.engine.async_llm import RequestStream
from ayaka.engine.seam import EngineHealth, EngineState

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from ayaka.core.request import GenerateRequest
    from ayaka.core.types import RequestId
    from ayaka.engine.seam import EngineInfo

__all__ = ["FakeEngine", "Script", "script"]

type Script = Sequence[RequestOutput] | Exception
"""What one request gets: its outputs, or the error ``add`` raises."""


def script(
    *pieces: str | tuple[int, str],
    reason: FinishReason = FinishReason.STOP,
    stop: str | int | None = None,
) -> list[RequestOutput]:
    """Outputs for a scripted request: one fragment per piece, then the final one.

    A piece is text (its token ids are its bytes) or ``(token id, text)``.
    """
    fragments: list[tuple[tuple[int, ...], str]] = []
    for piece in pieces:
        if isinstance(piece, tuple):
            fragments.append(((piece[0],), piece[1]))
        else:
            fragments.append((tuple(piece.encode()), piece))
    count = sum(len(ids) for ids, _ in fragments)
    outputs: list[RequestOutput] = []
    for i, (ids, text) in enumerate(fragments):
        last = i == len(fragments) - 1
        outputs.append(
            RequestOutput(
                request_id="scripted",  # type: ignore[arg-type]  # replaced per request
                outputs=(
                    CompletionOutput(
                        index=0,
                        token_ids=ids,
                        text=text,
                        finish_reason=reason if last else None,
                        stop_reason=stop if last else None,
                    ),
                ),
                finished=last,
                usage=Usage(prompt_tokens=0, completion_tokens=count if last else 0),
            )
        )
    return outputs


class FakeEngine:
    """Answers each request with the next script; see the module docstring.

    Args:
        info: What the engine reports about itself.
        scripts: One entry per expected request, in order.
    """

    def __init__(self, info: EngineInfo, scripts: Sequence[Script] = ()) -> None:
        self.info = info
        self.scripts = list(scripts)
        self.requests: list[GenerateRequest] = []
        self.aborted: list[RequestId] = []
        self._open: dict[RequestId, tuple[RequestStream, asyncio.Task[None]]] = {}
        self._accepting = True
        self._dead = False
        self._stalled_at: float | None = None

    async def add(self, request: GenerateRequest) -> RequestStream:
        if self._dead:
            raise EngineDeadError("the fake engine died")
        if not self._accepting:
            raise EngineError("the engine is shutting down")
        self.requests.append(request)
        outputs = self.scripts.pop(0) if self.scripts else script("ok")
        if isinstance(outputs, Exception):
            raise outputs
        stream = RequestStream(request.request_id)
        prompt = len(request.prompt) if isinstance(request.prompt, tuple) else 0

        async def feed() -> None:
            for output in outputs:
                await asyncio.sleep(0.005)
                stream.put(
                    RequestOutput(
                        request_id=request.request_id,
                        outputs=output.outputs,
                        finished=output.finished,
                        usage=Usage(
                            prompt_tokens=prompt, completion_tokens=output.usage.completion_tokens
                        ),
                        error=output.error,
                    )
                )
            if stream.finished:
                self._open.pop(request.request_id, None)

        self._open[request.request_id] = (stream, asyncio.get_running_loop().create_task(feed()))
        return stream

    def abort(self, request_id: RequestId) -> None:
        self.aborted.append(request_id)
        opened = self._open.pop(request_id, None)
        if opened is not None:
            stream, feeder = opened
            feeder.cancel()
            stream.close()

    def abort_all(self) -> None:
        self._end_all(
            lambda request_id: RequestOutput(
                request_id=request_id,
                outputs=(CompletionOutput(index=0, finish_reason=FinishReason.CANCELLED),),
                finished=True,
            )
        )

    async def shutdown(self, grace: float = 0.0) -> None:
        del grace  # the fake has no work to finish: what is open is cancelled
        self._accepting = False
        self.abort_all()

    def health(self) -> EngineHealth:
        if self._dead:
            state = EngineState.DEAD
        else:
            state = EngineState.RUNNING if self._accepting else EngineState.STOPPED
        if self._stalled_at is not None:
            return EngineHealth(state, self._stalled_at, busy=True)
        return EngineHealth(state, time.monotonic(), busy=bool(self._open))

    def stall(self) -> None:
        """Stop making progress, as an engine stuck in a step: busy, and the clock stands still."""
        self._stalled_at = time.monotonic()

    def resume(self) -> None:
        """Make progress again after :meth:`stall`."""
        self._stalled_at = None

    def die(self) -> None:
        """Kill the engine now: open streams end with ``engine_dead``, later calls raise."""
        self._dead = True
        error = EngineDeadError("the fake engine died")
        info = ErrorInfo(error.code, error.message)
        self._end_all(
            lambda request_id: RequestOutput(
                request_id=request_id,
                outputs=(CompletionOutput(index=0, finish_reason=FinishReason.ERROR),),
                finished=True,
                error=info,
            )
        )

    def _end_all(self, final: Callable[[RequestId], RequestOutput]) -> None:
        """End every open stream with the output ``final(request_id)`` makes."""
        for request_id, (stream, feeder) in tuple(self._open.items()):
            feeder.cancel()
            stream.put(final(request_id))
        self._open.clear()
