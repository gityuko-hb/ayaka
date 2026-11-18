"""The asynchronous face of the engine: many callers on an event loop, one engine thread.

::

    llm = AsyncLLM(build_engine(config))
    llm.start()                                    # inside a running event loop
    stream = await llm.add_request(request)        # admission errors are raised here
    async for output in stream:                    # one fragment per step, the last has finished
        ...

:class:`EngineCore` is not thread safe and is driven from exactly one thread. :class:`AsyncLLM`
owns that thread. Callers on the event loop never touch the engine: they send commands (add,
abort, stop) through a queue, and the engine thread hands the outputs of each step back to the
loop in one batch. Each request reads its outputs from its own :class:`RequestStream`.

The engine never waits for a caller. A stream holds at most ``max_pending`` undelivered outputs;
when a caller falls behind, a new output is merged into the newest one instead of queued, so a
slow reader costs memory in proportion to its own text, never a stalled step. Deciding that a
reader is too slow, and cancelling it, belongs to whoever writes to the client.

Failure and life cycle:

* An admission error (``InvalidRequestError``, ``UnsupportedFeatureError``, ``OverloadedError``)
  is raised from :meth:`AsyncLLM.add_request`; the request never existed.
* A request that ends in error ends its stream with an output that carries the error.
* If the engine dies, every open stream ends with an ``engine_dead`` error output, and every later
  call raises :class:`~ayaka.core.errors.EngineDeadError`.
* :meth:`AsyncLLM.abort` cancels at the next step boundary; the stream ends with
  ``FinishReason.CANCELLED`` unless its reader has already gone.
* :meth:`AsyncLLM.shutdown` stops admission, lets running requests finish within a timeout,
  cancels the rest at a step boundary, then stops the thread and the engine.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import threading
from collections import deque
from concurrent.futures import Future
from dataclasses import dataclass, replace
from queue import Empty, SimpleQueue
from typing import TYPE_CHECKING, Final, Self

from ayaka.core.errors import EngineDeadError, EngineError, ErrorInfo
from ayaka.core.outputs import CompletionOutput, RequestOutput
from ayaka.core.types import FinishReason

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Callable

    from ayaka.configs.engine import EngineConfig
    from ayaka.core.request import GenerateRequest
    from ayaka.core.types import RequestId
    from ayaka.engine.core import EngineCore
    from ayaka.tokenizers.contracts import LoadedTokenizer

__all__ = ["AsyncLLM", "RequestStream", "merge_outputs"]

_LOG = logging.getLogger("ayaka.engine.async")

#: Undelivered outputs a stream keeps before it merges new ones into the newest.
DEFAULT_MAX_PENDING: Final = 64


def merge_outputs(first: RequestOutput, second: RequestOutput) -> RequestOutput:
    """One streaming fragment that says what ``first`` then ``second`` said.

    The text and token ids of each sequence are concatenated, sequence by sequence (a request
    with ``n > 1`` has fragments of several); a sequence's finish reason comes from its newer
    fragment, and everything that describes the end of the request (finished, usage, metrics,
    error) comes from ``second``.
    """
    merged: dict[int, CompletionOutput] = {piece.index: piece for piece in first.outputs}
    for piece in second.outputs:
        earlier = merged.get(piece.index)
        merged[piece.index] = (
            piece
            if earlier is None
            else replace(
                piece,
                token_ids=earlier.token_ids + piece.token_ids,
                text=earlier.text + piece.text,
                logprobs=(
                    (earlier.logprobs or ()) + (piece.logprobs or ())
                    if earlier.logprobs or piece.logprobs
                    else None
                ),
            )
        )
    if not merged:
        merged[0] = CompletionOutput(index=0)
    return replace(second, outputs=tuple(merged[index] for index in sorted(merged)))


class RequestStream:
    """The outputs of one request, in order, as an asynchronous iterator.

    Made by :meth:`AsyncLLM.add_request`. Iteration ends after the output whose ``finished`` is
    true. Only the event loop thread touches a stream.

    Args:
        request_id: The request this stream belongs to.
        max_pending: Undelivered outputs kept before new ones are merged into the newest.
    """

    def __init__(self, request_id: RequestId, *, max_pending: int = DEFAULT_MAX_PENDING) -> None:
        if max_pending < 1:
            raise ValueError("max_pending must be >= 1")
        self.request_id = request_id
        self._max_pending = max_pending
        self._pending: deque[RequestOutput] = deque()
        self._ready = asyncio.Event()
        self._finished = False
        self._delivered_final = False
        self.merged = 0
        """How many outputs were merged into another because the reader fell behind."""

    @property
    def finished(self) -> bool:
        """Whether the final output has arrived (it may not have been read yet)."""
        return self._finished

    @property
    def pending(self) -> int:
        """Outputs that arrived and have not been read."""
        return len(self._pending)

    def put(self, output: RequestOutput) -> None:
        """Queue an output; merge it into the newest one when the stream is full."""
        if self._finished:
            return
        if len(self._pending) >= self._max_pending:
            self._pending[-1] = merge_outputs(self._pending[-1], output)
            self.merged += 1
        else:
            self._pending.append(output)
        self._finished = output.finished
        self._ready.set()

    def get_nowait(self) -> RequestOutput | None:
        """The next output if one is waiting; ``None`` otherwise."""
        if not self._pending:
            return None
        output = self._pending.popleft()
        if not self._pending:
            self._ready.clear()
        if output.finished:
            self._delivered_final = True
        return output

    def drain_nowait(self) -> RequestOutput | None:
        """Every waiting output merged into one; ``None`` if nothing is waiting."""
        output = self.get_nowait()
        while output is not None and not output.finished:
            more = self.get_nowait()
            if more is None:
                break
            output = merge_outputs(output, more)
        return output

    async def wait(self) -> None:
        """Return once an output is waiting."""
        await self._ready.wait()

    def __aiter__(self) -> Self:
        return self

    async def __anext__(self) -> RequestOutput:
        if self._delivered_final:
            raise StopAsyncIteration
        while not self._pending:
            await self._ready.wait()
        output = self.get_nowait()
        assert output is not None  # noqa: S101  # the loop above waited for one
        return output


@dataclass(frozen=True, slots=True)
class _Add:
    request: GenerateRequest
    future: Future[None]


@dataclass(frozen=True, slots=True)
class _Abort:
    request_id: RequestId


@dataclass(frozen=True, slots=True)
class _AbortAll:
    pass


@dataclass(frozen=True, slots=True)
class _Stop:
    pass


type _Command = _Add | _Abort | _AbortAll | _Stop


class AsyncLLM:
    """Runs an engine on its own thread and serves it to coroutines.

    Args:
        engine: A built engine; this object owns it from now on and shuts it down.
        max_pending: Undelivered outputs each stream keeps before merging (see
            :class:`RequestStream`).
    """

    def __init__(self, engine: EngineCore, *, max_pending: int = DEFAULT_MAX_PENDING) -> None:
        self._engine = engine
        self._max_pending = max_pending
        self._commands: SimpleQueue[_Command] = SimpleQueue()
        self._wake = threading.Event()
        self._idle = threading.Event()
        self._idle.set()
        self._idle_lock = threading.Lock()
        self._streams: dict[RequestId, RequestStream] = {}
        self._admitted: set[RequestId] = set()  # engine thread only: admitted, not yet finished
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._accepting = True
        self._dead: EngineDeadError | None = None
        self._stopped = threading.Event()

    @classmethod
    def from_config(
        cls, config: EngineConfig, *, tokenizer: LoadedTokenizer | None = None
    ) -> AsyncLLM:
        """Build an engine and wrap it; this blocks for as long as the build takes.

        Raises:
            ModelLoadError, ConfigError, CapabilityError, CapacityError: As ``build_engine``.
        """
        from ayaka.engine.build import build_engine

        return cls(build_engine(config, tokenizer=tokenizer))

    # -- state -----------------------------------------------------------------------------

    @property
    def engine(self) -> EngineCore:
        """The engine; read-only use only, since its thread owns it."""
        return self._engine

    @property
    def is_running(self) -> bool:
        """Whether the engine thread runs and accepts requests."""
        return (
            self._thread is not None
            and self._thread.is_alive()
            and self._accepting
            and self._dead is None
        )

    @property
    def is_dead(self) -> bool:
        """Whether the engine failed; nothing will run again."""
        return self._dead is not None

    @property
    def num_streams(self) -> int:
        """Requests whose stream is still open."""
        return len(self._streams)

    # -- life cycle ------------------------------------------------------------------------

    def start(self) -> None:
        """Start the engine thread, bound to the running event loop.

        Raises:
            RuntimeError: If no event loop is running or the thread was already started.
        """
        if self._thread is not None:
            raise RuntimeError("the engine thread was already started")
        self._loop = asyncio.get_running_loop()
        self._thread = threading.Thread(target=self._run, name="ayaka-engine", daemon=True)
        self._thread.start()

    async def drain(self, within: float | None = None) -> bool:
        """Wait until the engine has no unfinished request.

        Args:
            within: Seconds to wait at most; ``None`` waits as long as it takes.

        Returns:
            ``True`` if the engine became idle, ``False`` on timeout.
        """
        if self._dead is not None or self._thread is None:
            return True
        return await asyncio.to_thread(self._idle.wait, within)

    async def shutdown(self, grace: float = 0.0) -> None:
        """Stop admission, finish or cancel the requests in flight and stop the engine.

        Requests still running after ``grace`` seconds are cancelled at the next step
        boundary, so their streams end with ``FinishReason.CANCELLED``. Safe to call twice.

        Args:
            grace: Seconds the running requests get to finish on their own.
        """
        self._accepting = False
        thread = self._thread
        if thread is None:
            self._engine.shutdown()
            return
        if grace > 0:
            await self.drain(grace)
        self._send(_AbortAll())
        await self.drain()
        self._send(_Stop())
        await asyncio.to_thread(thread.join)

    # -- requests --------------------------------------------------------------------------

    async def add_request(self, request: GenerateRequest) -> RequestStream:
        """Admit a request and return the stream of its outputs.

        Raises:
            UnsupportedFeatureError, InvalidRequestError, OverloadedError: As
                ``EngineCore.add_request``; the request never ran.
            EngineDeadError: If the engine died.
            EngineError: If the engine is shutting down or was not started.
        """
        self._require_accepting()
        request_id = request.request_id
        if request_id in self._streams:
            raise EngineError(f"request {request_id!r} is already running")
        stream = RequestStream(request_id, max_pending=self._max_pending)
        self._streams[request_id] = stream
        future: Future[None] = Future()
        with self._idle_lock:
            self._idle.clear()
            self._send(_Add(request, future))
        try:
            await asyncio.wrap_future(future)
        except asyncio.CancelledError:
            # The caller left while the engine was admitting: whatever happened, end it.
            self.abort(request_id)
            raise
        except BaseException:
            self._streams.pop(request_id, None)
            raise
        return stream

    async def generate(self, request: GenerateRequest) -> AsyncGenerator[RequestOutput]:
        """Admit a request and yield its outputs; leaving early cancels it.

        Raises:
            As :meth:`add_request`.
        """
        stream = await self.add_request(request)
        try:
            async for output in stream:
                yield output
        finally:
            if not stream.finished:
                self.abort(request.request_id)

    def abort(self, request_id: RequestId) -> None:
        """Cancel a request at the next step boundary and stop delivering its outputs.

        Unknown or finished requests are ignored. Call from the event loop thread.
        """
        self._streams.pop(request_id, None)
        if self._dead is None and self._thread is not None:
            self._send(_Abort(request_id))

    # -- the event loop side ---------------------------------------------------------------

    def _require_accepting(self) -> None:
        if self._dead is not None:
            raise self._dead
        if self._thread is None:
            raise EngineError("the engine thread was not started")
        if not self._accepting or not self._thread.is_alive():
            raise EngineError("the engine is shutting down")

    def _send(self, command: _Command) -> None:
        self._commands.put(command)
        self._wake.set()

    def _deliver(self, outputs: list[RequestOutput]) -> None:
        """Hand the outputs of a step to their streams (on the loop thread)."""
        for output in outputs:
            stream = self._streams.get(output.request_id)
            if stream is None:
                continue
            stream.put(output)
            if output.finished:
                del self._streams[output.request_id]

    def _fail_streams(self, error: EngineDeadError) -> None:
        """End every open stream with the engine's death (on the loop thread)."""
        info = ErrorInfo(error.code, error.message)
        for request_id, stream in tuple(self._streams.items()):
            stream.put(
                RequestOutput(
                    request_id=request_id,
                    outputs=(CompletionOutput(index=0, finish_reason=FinishReason.ERROR),),
                    finished=True,
                    error=info,
                )
            )
        self._streams.clear()

    def _post(self, callback: Callable[..., object], *args: object) -> None:
        loop = self._loop
        if loop is None or loop.is_closed():
            return
        with contextlib.suppress(RuntimeError):  # the loop closed between the check and the call
            loop.call_soon_threadsafe(callback, *args)

    # -- the engine thread -----------------------------------------------------------------

    def _run(self) -> None:
        engine = self._engine
        stopping = False
        try:
            while True:
                stopping = self._execute_commands() or stopping
                if stopping and not engine.has_unfinished:
                    break
                if engine.has_unfinished:
                    outputs = engine.step()
                    if outputs:
                        for output in outputs:
                            if output.finished:
                                self._admitted.discard(output.request_id)
                        self._post(self._deliver, outputs)
                    continue
                with self._idle_lock:
                    if self._commands.empty():
                        self._idle.set()
                self._wake.wait()
                self._wake.clear()
        except BaseException as exc:
            dead = exc if isinstance(exc, EngineDeadError) else None
            if dead is None:
                _LOG.exception("the engine thread stopped on an unexpected error")
                dead = EngineDeadError(f"the engine thread stopped: {exc}")
            self._dead = dead
            self._post(self._fail_streams, dead)
            self._refuse_pending(dead)
        finally:
            self._idle.set()
            with contextlib.suppress(Exception):
                engine.shutdown()
            self._stopped.set()

    def _execute_commands(self) -> bool:
        """Run every queued command; return whether a stop was requested."""
        engine = self._engine
        stop = False
        while True:
            try:
                command = self._commands.get_nowait()
            except Empty:
                return stop
            match command:
                case _Add(request, future):
                    if not future.set_running_or_notify_cancel():
                        continue
                    try:
                        engine.add_request(request, stream=True)
                    except EngineDeadError:
                        future.set_exception(EngineDeadError("the engine died"))
                        raise
                    except Exception as exc:
                        future.set_exception(exc)
                    else:
                        self._admitted.add(request.request_id)
                        future.set_result(None)
                case _Abort(request_id):
                    engine.abort(request_id)
                case _AbortAll():
                    for request_id in tuple(self._admitted):
                        engine.abort(request_id)
                case _Stop():
                    stop = True

    def _refuse_pending(self, error: EngineDeadError) -> None:
        """Fail the admissions still queued when the engine died."""
        while True:
            try:
                command = self._commands.get_nowait()
            except Empty:
                return
            if isinstance(command, _Add) and command.future.set_running_or_notify_cancel():
                command.future.set_exception(error)
