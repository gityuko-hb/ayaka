"""The contract between the HTTP server and the engine: what one may ask of the other.

The server (``ayaka.entrypoints``) reaches the engine only through :class:`EngineClient`, which
:class:`~ayaka.engine.async_llm.AsyncLLM` implements and test doubles imitate. It never reads the
engine's insides (scheduler, KV cache, recorder): what it needs to know about the loaded engine is
in :class:`EngineInfo`, plain values fixed at start-up, and what it needs to know about the engine
now is in :class:`EngineHealth`. A test (``tests/entrypoints/test_seam_boundary.py``) checks that
nothing in ``ayaka.entrypoints`` reaches through an ``engine`` attribute.

The commands, which the engine runs in the order they were sent, so an abort sent after the add of
the same request takes effect::

    await add(request) -> RequestStream     admission errors are raised here
    abort(request_id)                       the caller leaves; the stream ends with nothing more
    abort_all()                             every admitted request ends CANCELLED
    await shutdown(grace)                   stop admission, let requests finish or cancel them

Commands to pause, resume and resize, and to read statistics, join the contract when the memory
milestone needs them, not before.

Guarantees:

* Every admitted request ends with **exactly one** output whose ``finished`` is true, and nothing
  follows it: a normal end, an error in the request's own output, a cancellation by
  ``abort_all`` or ``shutdown``, or the engine's death (an ``engine_dead`` error).
* The one exception is the caller's own :meth:`EngineClient.abort`: the stream is closed at once
  without a final output, what was waiting in it is dropped and nothing is delivered afterwards.
  Whoever aborts is the one that reads the stream, and has stopped reading it.
* An admission error leaves no trace: the request never existed.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from ayaka.core.request import GenerateRequest
    from ayaka.core.types import RequestId
    from ayaka.engine.async_llm import RequestStream

__all__ = ["EngineClient", "EngineHealth", "EngineInfo", "EngineState"]


@dataclass(frozen=True, slots=True)
class EngineInfo:
    """What the server needs to know about the loaded engine; fixed once it is built.

    Attributes:
        max_model_len: The longest sequence the model and its tokenizer take, in tokens.
        max_context_tokens: The longest sequence one request can get on this machine: the
            model's limit or the KV cache's, whichever is smaller.
        prefix_caching: Whether identical prompt prefixes are read from the cache.
    """

    max_model_len: int
    max_context_tokens: int
    prefix_caching: bool


class EngineState(StrEnum):
    """Where the engine is in its life."""

    RUNNING = "running"  # accepting requests
    STOPPED = "stopped"  # not started yet, or shut down
    DEAD = "dead"  # it failed; nothing will run again


@dataclass(frozen=True, slots=True)
class EngineHealth:
    """The engine now, as its thread last reported it.

    Attributes:
        state: Where the engine is in its life.
        progress_at: ``time.monotonic()`` when the engine last made progress: its thread finished
            a step or went idle, or work arrived while it was idle. While it is busy the value
            moves at every step; one that stops moving then means a step is stuck.
        busy: Whether work is queued or running. An idle engine makes no progress and is not
            stuck.
    """

    state: EngineState
    progress_at: float
    busy: bool


@runtime_checkable
class EngineClient(Protocol):
    """The engine as the server sees it; see the module docstring for the guarantees."""

    @property
    def info(self) -> EngineInfo:
        """The loaded engine's fixed facts."""
        ...

    async def add(self, request: GenerateRequest) -> RequestStream:
        """Admit a request and return the stream of its outputs.

        Raises:
            UnsupportedFeatureError, InvalidRequestError, OverloadedError: If the engine refuses
                the request; it never ran.
            EngineDeadError: If the engine died.
            EngineError: If the engine is not accepting requests.
        """
        ...

    def abort(self, request_id: RequestId) -> None:
        """Cancel a request; its stream ends and nothing more is delivered for it."""
        ...

    def abort_all(self) -> None:
        """Cancel every admitted request at the next step boundary; each stream ends CANCELLED."""
        ...

    async def shutdown(self, grace: float = 0.0) -> None:
        """Stop admission, give the requests in flight ``grace`` seconds, cancel the rest, stop."""
        ...

    def health(self) -> EngineHealth:
        """The engine now."""
        ...
