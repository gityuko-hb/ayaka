"""From the tokens the scheduler emits to the text and results a caller sees.

:class:`OutputProcessor` is the last stage of the engine loop. It takes the
:class:`~ayaka.scheduler.request.RequestEvent` objects of a step and, per request, decodes the new
tokens to text with an incremental detokenizer, applies the stop rules the scheduler cannot, and
builds :class:`~ayaka.core.outputs.RequestOutput` values.

The rules about stopping are split between the two layers on purpose. The scheduler counts and
compares ids: the end-of-sequence token, ``stop_token_ids`` and ``max_tokens`` end a request there.
A **stop string** can only be found in decoded text, and it can straddle token boundaries, so it is
matched here by the detokenizer's stop checker, which holds back the tail of the text that might
still turn into a stop string. When it matches, the text is cut at the match (or just after it with
``include_stop_str_in_output``), the tokens after the one that completed the match are dropped from
the output, and the request is aborted in the scheduler with ``FinishReason.STOP``. A stop token id
is checked before any of this and does not depend on how the tokenizer joins characters.

The incremental decoder never emits half of a UTF-8 character: text for a token that ends inside a
character arrives with the token that completes it. At the end of a request whatever the decoder
and the stop checker still hold is flushed.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from ayaka.configs.tokenizer import DetokenizeParams
from ayaka.core.errors import ErrorInfo
from ayaka.core.outputs import CompletionOutput, RequestMetrics, RequestOutput, Usage
from ayaka.core.types import FinishReason

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

    from ayaka.core.params import SamplingParams
    from ayaka.core.request import TokenizedRequest
    from ayaka.core.types import RequestId, TokenId
    from ayaka.scheduler.request import RequestEvent
    from ayaka.tokenizers.ports import DetokenizerLike
    from ayaka.tokenizers.service import TokenizerService

__all__ = ["OutputProcessor"]


@dataclass(slots=True)
class _State:
    """What the processor remembers about one request that has not finished."""

    prompt: tuple[TokenId, ...]
    sampling: SamplingParams
    detokenizer: DetokenizerLike | None
    stream: bool
    arrival: float
    tokens: list[TokenId] = field(default_factory=list[int])
    text: list[str] = field(default_factory=list[str])
    first_token_at: float | None = None


class OutputProcessor:
    """Builds the outputs of requests from scheduler events.

    Args:
        service: The tokenizer service, which makes one detokenizer per request.
        abort: Ends a request in the scheduler with a reason and says whether it was alive; the
            engine passes ``Scheduler.abort``. Used to stop a request whose stop string matched.
        clock: A monotonic clock in seconds, for the metrics.
    """

    def __init__(
        self,
        service: TokenizerService,
        *,
        abort: Callable[[RequestId, FinishReason], bool],
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._service = service
        self._abort = abort
        self._clock = clock
        self._states: dict[RequestId, _State] = {}

    @property
    def num_active(self) -> int:
        """Requests registered and not yet finished."""
        return len(self._states)

    def register(self, request: TokenizedRequest, *, stream: bool = False) -> None:
        """Start tracking a request that the scheduler has accepted.

        Args:
            request: The tokenized request.
            stream: ``True`` to emit one output per step that carries only the new text and token
                ids (``finished`` false until the last). ``False`` to emit one output when the
                request ends, with the whole text and every token id.
        """
        sampling = request.sampling
        detokenizer: DetokenizerLike | None = None
        if self._service.tokenizer is not None:
            detokenizer = self._service.new_detokenizer(
                DetokenizeParams(
                    skip_special_tokens=True,
                    stop=sampling.stop,
                    include_stop_str_in_output=sampling.include_stop_str_in_output,
                ),
                prompt_token_ids=request.prompt_token_ids,
            )
        self._states[request.request_id] = _State(
            prompt=request.prompt_token_ids,
            sampling=sampling,
            detokenizer=detokenizer,
            stream=stream,
            arrival=self._clock(),
        )

    def discard(self, request_id: RequestId) -> None:
        """Forget a request without producing output, for one that never reached the scheduler."""
        self._states.pop(request_id, None)

    def process(self, events: Iterable[RequestEvent]) -> list[RequestOutput]:
        """Turn the events of a step into outputs, in event order.

        Events for a request that already finished here, such as the bare end event the scheduler
        queues after a stop-string abort, are ignored.
        """
        outputs: list[RequestOutput] = []
        for event in events:
            state = self._states.get(event.request_id)
            if state is None:
                continue
            output = self._handle(event, state)
            if output is not None:
                outputs.append(output)
        return outputs

    def fail_all(self, error: ErrorInfo) -> list[RequestOutput]:
        """End every unfinished request with an error, for an engine that has died."""
        outputs: list[RequestOutput] = []
        for request_id in tuple(self._states):
            state = self._states.pop(request_id)
            outputs.append(self._build(request_id, state, "", (), FinishReason.ERROR, None, error))
        return outputs

    # -- one event -------------------------------------------------------------------------

    def _handle(self, event: RequestEvent, state: _State) -> RequestOutput | None:
        now = self._clock()
        kept = event.token_ids
        text = ""
        stop_string: str | None = None
        reason = event.finish_reason
        stop_reason: TokenId | str | None = event.stop_reason
        error = event.error
        if kept and state.first_token_at is None:
            state.first_token_at = now
        detokenizer = state.detokenizer
        if detokenizer is not None:
            try:
                if kept:
                    # A stop string counts only once ``min_tokens`` tokens exist before it.
                    allowed = len(state.tokens) + len(kept) > state.sampling.min_tokens
                    update = detokenizer.update(kept, check_stops=allowed)
                    text = update.delta
                    if update.stop_matched is not None:
                        stop_string = update.stop_matched
                        kept = kept[: update.consumed_tokens]
                if reason is not None or stop_string is not None:
                    tail = detokenizer.finish()
                    text += tail.delta
                    if stop_string is None and tail.stop_matched is not None:
                        stop_string = tail.stop_matched
            except (ValueError, RuntimeError) as exc:
                self._abort(event.request_id, FinishReason.ERROR)
                del self._states[event.request_id]
                failure = ErrorInfo("detokenization_failed", f"cannot decode the output: {exc}")
                state.tokens.extend(kept)
                return self._build(
                    event.request_id, state, text, kept, FinishReason.ERROR, None, failure
                )
        if stop_string is not None:
            reason, stop_reason = FinishReason.STOP, stop_string
            if event.finish_reason is None:
                self._abort(event.request_id, FinishReason.STOP)
        state.tokens.extend(kept)
        state.text.append(text)
        if reason is None:
            if state.stream and (kept or text):
                return self._build(event.request_id, state, text, kept, None, None, None)
            return None
        del self._states[event.request_id]
        return self._build(event.request_id, state, text, kept, reason, stop_reason, error)

    def _build(
        self,
        request_id: RequestId,
        state: _State,
        text: str,
        kept: tuple[TokenId, ...],
        reason: FinishReason | None,
        stop_reason: TokenId | str | None,
        error: ErrorInfo | None,
    ) -> RequestOutput:
        """Assemble an output: a fragment for a streaming chunk, the whole sequence otherwise."""
        finished = reason is not None
        if state.stream:
            token_ids, body = kept, text
        else:
            token_ids, body = tuple(state.tokens), "".join(state.text)
        completion = CompletionOutput(
            index=0, token_ids=token_ids, text=body, finish_reason=reason, stop_reason=stop_reason
        )
        return RequestOutput(
            request_id=request_id,
            prompt_token_ids=state.prompt,
            outputs=(completion,),
            finished=finished,
            usage=Usage(prompt_tokens=len(state.prompt), completion_tokens=len(state.tokens)),
            metrics=self._metrics(state) if finished else None,
            error=error,
        )

    def _metrics(self, state: _State) -> RequestMetrics:
        """Timings from the engine clock; the queue and prefill times are not measured."""
        now = self._clock()
        first = state.first_token_at
        if first is None:
            return RequestMetrics(arrival_time=state.arrival)
        decode = max(0.0, now - first)
        count = len(state.tokens)
        return RequestMetrics(
            arrival_time=state.arrival,
            decode_time=decode,
            ttft=max(0.0, first - state.arrival),
            tpot=decode / (count - 1) if count > 1 else 0.0,
        )
