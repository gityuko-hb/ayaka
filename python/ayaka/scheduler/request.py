"""A request as the scheduler tracks it: live state, status and the events it emits.

:class:`Request` is the scheduler's own mutable record of one
:class:`~ayaka.core.request.TokenizedRequest`. Only the scheduler writes it. Nothing outside the
package holds one across a step: a :class:`~ayaka.scheduler.step.StepInput` copies what it needs,
so changing a ``Request`` after ``schedule()`` cannot change a step that is already out.

There are four statuses and a fixed table of moves between them. A move that is not in the table
is a programming error and raises, so a test that records every move proves the state machine.
Whether a request is prefilling or decoding is derived from ``computed``, not a status.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING, Final

from ayaka.core.errors import EngineError
from ayaka.core.types import FinishReason

if TYPE_CHECKING:
    from collections.abc import Collection

    from ayaka.core.errors import ErrorInfo
    from ayaka.core.outputs import TokenLogprob
    from ayaka.core.params import SamplingParams
    from ayaka.core.request import TokenizedRequest
    from ayaka.core.types import RequestId, TokenId

__all__ = ["LEGAL_TRANSITIONS", "Request", "RequestEvent", "RequestStatus"]


class RequestStatus(StrEnum):
    """Where a request is in the scheduler.

    Attributes:
        WAITING: In the queue, holds no KV.
        RUNNING: Holds KV; prefilling or decoding.
        PREEMPTED: Gave its KV back and waits to run again. Reserved: nothing enters it before
            preemption lands, but the status and its moves are part of the table.
        FINISHED: Done; carries a finish reason.
    """

    WAITING = "waiting"
    RUNNING = "running"
    PREEMPTED = "preempted"
    FINISHED = "finished"


#: Every move a request may make; anything else raises.
LEGAL_TRANSITIONS: Final[frozenset[tuple[RequestStatus, RequestStatus]]] = frozenset(
    {
        (RequestStatus.WAITING, RequestStatus.RUNNING),
        (RequestStatus.WAITING, RequestStatus.FINISHED),
        (RequestStatus.RUNNING, RequestStatus.PREEMPTED),
        (RequestStatus.RUNNING, RequestStatus.FINISHED),
        (RequestStatus.PREEMPTED, RequestStatus.RUNNING),
        (RequestStatus.PREEMPTED, RequestStatus.FINISHED),
    }
)


@dataclass(frozen=True, slots=True)
class RequestEvent:
    """What the engine must tell the outside about one request after a scheduler call.

    An event carries the tokens a step added and, on the last event of a request, why it ended.
    Events come out of :meth:`~ayaka.scheduler.scheduler.Scheduler.take_events` in the order they
    happened. A request that is cancelled, times out or is stopped by the engine ends without any
    step, so the engine learns of it only here.

    Attributes:
        request_id: The request.
        token_ids: Tokens emitted since the previous event; may be empty on a bare finish. When a
            stop token ends the request it is the last token here, and ``stop_reason`` is its id.
        finish_reason: ``None`` while the request goes on.
        stop_reason: The stop token id that ended the request, if one did.
        error: Why the request ended with ``ERROR``, ``TIMEOUT``, or ``LENGTH`` because of memory.
        logprobs: Logprobs of ``token_ids`` when the runner supplied them.
    """

    request_id: RequestId
    token_ids: tuple[TokenId, ...] = ()
    finish_reason: FinishReason | None = None
    stop_reason: TokenId | None = None
    error: ErrorInfo | None = None
    logprobs: tuple[TokenLogprob, ...] | None = None

    @property
    def finished(self) -> bool:
        """Whether this is the last event of the request."""
        return self.finish_reason is not None


@dataclass(slots=True, eq=False, kw_only=True)
class Request:
    """The live state of one request. Only the scheduler changes it.

    Attributes:
        request_id: Identity; never changes.
        prompt: Prompt token ids; never changes.
        params: Sampling parameters; frozen.
        priority: Higher runs first under the ``priority`` policy.
        arrival: Clock reading when the scheduler accepted the request.
        deadline: Absolute clock reading after which the request times out, or ``None``.
        output: Tokens emitted so far, the pending one included.
        row: The request's row in the block table and in the batch; ``-1`` until it runs. It is
            kept until the request finishes, so a request keeps its row between steps.
        computed: Tokens whose KV is real and committed.
        pending: The last sampled token: emitted, but not yet run through the model.
        abort_reason: Set by ``abort`` while the request is inside a step; applied at ``update``.
        finish_reason: Set when the request finishes.
    """

    request_id: RequestId
    prompt: tuple[TokenId, ...]
    params: SamplingParams
    priority: int = 0
    arrival: float = 0.0
    deadline: float | None = None
    output: list[TokenId] = field(default_factory=list[int])
    row: int = -1
    computed: int = 0
    pending: TokenId | None = None
    abort_reason: FinishReason | None = None
    finish_reason: FinishReason | None = None
    _status: RequestStatus = field(default=RequestStatus.WAITING, init=False)

    @classmethod
    def from_tokenized(
        cls, request: TokenizedRequest, *, arrival: float, deadline: float | None = None
    ) -> Request:
        """Start tracking a request that passed admission."""
        return cls(
            request_id=request.request_id,
            prompt=request.prompt_token_ids,
            params=request.sampling,
            priority=request.priority,
            arrival=arrival,
            deadline=deadline,
        )

    @property
    def status(self) -> RequestStatus:
        """The current status; change it only through :meth:`move_to`."""
        return self._status

    @property
    def prompt_length(self) -> int:
        """Number of prompt tokens."""
        return len(self.prompt)

    @property
    def is_prefilling(self) -> bool:
        """Whether part of the prompt still has no KV."""
        return self.computed < len(self.prompt)

    @property
    def abort_requested(self) -> bool:
        """Whether an abort is waiting for the step that holds this request to end."""
        return self.abort_reason is not None

    def move_to(self, status: RequestStatus) -> None:
        """Change the status along a move of :data:`LEGAL_TRANSITIONS`.

        Raises:
            EngineError: If the move is not in the table.
        """
        if (self._status, status) not in LEGAL_TRANSITIONS:
            raise EngineError(
                f"request {self.request_id!r} cannot move from {self._status} to {status}"
            )
        self._status = status

    def tokens_left(self, max_model_len: int | None) -> int:
        """Tokens the request may still emit before its own limits end it; at least 1."""
        limits = [sys.maxsize]
        if self.params.max_tokens is not None:
            limits.append(self.params.max_tokens - len(self.output))
        if max_model_len is not None:
            limits.append(max_model_len - len(self.prompt) - len(self.output))
        return max(1, min(limits))

    def accept(
        self, produced: tuple[TokenId, ...], eos: Collection[TokenId], max_model_len: int | None
    ) -> tuple[tuple[TokenId, ...], FinishReason | None, TokenId | None]:
        """Take the tokens a step produced: cut them at the first stop token and at the limits.

        A stop token ends the request and is kept as its last token. It counts only once at least
        ``min_tokens`` tokens exist before it. ``eos`` is ignored when the request sets
        ``ignore_eos``; ``stop_token_ids`` never is. The kept tokens become the output and the last
        one becomes ``pending``.

        Args:
            produced: Tokens the runner emitted for this request, at least one.
            eos: End-of-sequence ids of the model.
            max_model_len: The model context, or ``None``.

        Returns:
            ``(kept tokens, finish reason or None, stop token or None)``.
        """
        params = self.params
        left = self.tokens_left(max_model_len)
        stops = params.stop_token_ids if params.ignore_eos else params.stop_token_ids | set(eos)
        kept: list[TokenId] = []
        reason: FinishReason | None = None
        stop: TokenId | None = None
        for token in produced:
            kept.append(token)
            if token in stops and len(self.output) + len(kept) > params.min_tokens:
                reason, stop = FinishReason.STOP, token
                break
            if len(kept) == left:
                reason = FinishReason.LENGTH
                break
        self.output.extend(kept)
        self.pending = kept[-1]
        return tuple(kept), reason, stop
