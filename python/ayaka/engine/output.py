"""From the tokens the scheduler emits to the text and results a caller sees.

:class:`OutputProcessor` is the last stage of the engine loop. It takes the
:class:`~ayaka.scheduler.request.RequestEvent` objects of a step and, per sequence, decodes the new
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

A request with ``n > 1`` runs as ``n`` sibling requests in the scheduler, one per sequence. Here
they form one **family** under the parent id: a streaming output carries the fragment of the
sequence whose event arrived (``CompletionOutput.index`` says which), and the request is finished
once every sequence is. The prompt is counted once in the usage, the generated tokens of every
sequence are summed. A sequence that fails (an error or a timeout) fails the whole request and
cancels its siblings, so a partial answer is never returned as a success; a sequence that is
cancelled alone leaves the others running.

The incremental decoder never emits half of a UTF-8 character: text for a token that ends inside a
character arrives with the token that completes it. At the end of a sequence whatever the decoder
and the stop checker still hold is flushed.

Logprobs arrive with the tokens; this is where each token and each listed alternative gets its text
(one token decoded alone), and where ``cumulative_logprob`` is summed over the tokens kept.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Final

from ayaka.configs.tokenizer import DetokenizeParams
from ayaka.core.errors import ErrorInfo
from ayaka.core.outputs import (
    CompletionOutput,
    RequestMetrics,
    RequestOutput,
    TokenLogprob,
    Usage,
)
from ayaka.core.types import FinishReason

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Sequence

    from ayaka.core.params import SamplingParams
    from ayaka.core.request import TokenizedRequest
    from ayaka.core.types import RequestId, TokenId
    from ayaka.scheduler.request import RequestEvent
    from ayaka.tokenizers.ports import DetokenizerLike
    from ayaka.tokenizers.service import TokenizerService

__all__ = ["OutputProcessor"]

#: A sequence ending this way ends its whole request with the error.
_FAILURES: Final = frozenset({FinishReason.ERROR, FinishReason.TIMEOUT})


@dataclass(slots=True, eq=False)
class _Family:
    """The sequences of one public request and what they share."""

    parent: RequestId
    prompt: tuple[TokenId, ...]
    stream: bool
    arrival: float
    sequences: list[_Sequence] = field(default_factory=list["_Sequence"])
    live: dict[RequestId, _Sequence] = field(default_factory=dict["RequestId", "_Sequence"])
    done: dict[int, CompletionOutput] = field(default_factory=dict[int, CompletionOutput])
    first_token_at: float | None = None
    completion_tokens: int = 0
    cached_tokens: int = 0
    cache_written_tokens: int = 0

    def usage(self) -> Usage:
        """The request's usage: its prompt counts once, however many sequences it has.

        The cache read is the largest of the sequences' (with ``n > 1`` the siblings attach the
        prompt the first sequence prefilled, and that attach counts as a read). The cache write
        is what the sequences registered, cut to what the read leaves of the prompt, so the
        read, the write and the rest always add up to the prompt.
        """
        prompt = len(self.prompt)
        cached = min(self.cached_tokens, prompt)
        return Usage(
            prompt_tokens=prompt,
            completion_tokens=self.completion_tokens,
            cached_tokens=cached,
            cache_written_tokens=min(self.cache_written_tokens, prompt - cached),
        )


@dataclass(slots=True, eq=False)
class _Sequence:
    """What the processor remembers about one sequence that has not finished."""

    request_id: RequestId
    index: int
    sampling: SamplingParams
    detokenizer: DetokenizerLike | None
    family: _Family
    tokens: list[TokenId] = field(default_factory=list[int])
    text: list[str] = field(default_factory=list[str])
    cached_tokens: int = 0
    cache_written_tokens: int = 0
    logprobs: list[TokenLogprob] = field(default_factory=list[TokenLogprob])

    def cumulative(self) -> float | None:
        if not self.logprobs:
            return None
        return sum(entry.logprob for entry in self.logprobs)


class OutputProcessor:
    """Builds the outputs of requests from scheduler events.

    Args:
        service: The tokenizer service, which makes one detokenizer per sequence.
        abort: Ends a request in the scheduler with a reason and says whether it was alive; the
            engine passes ``Scheduler.abort``. Used to stop a sequence whose stop string matched
            and the siblings of one that failed.
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
        self._sequences: dict[RequestId, _Sequence] = {}

    @property
    def num_active(self) -> int:
        """Sequences registered and not yet finished."""
        return len(self._sequences)

    def register(self, request: TokenizedRequest, *, stream: bool = False) -> None:
        """Start tracking a request that the scheduler has accepted.

        Args:
            request: The tokenized request.
            stream: ``True`` to emit one output per step that carries only the new text and token
                ids (``finished`` false until the last). ``False`` to emit one output when the
                request ends, with the whole text and every token id.
        """
        self.register_family(request.request_id, (request,), stream=stream)

    def register_family(
        self, parent: RequestId, sequences: Sequence[TokenizedRequest], *, stream: bool = False
    ) -> None:
        """Start tracking the sibling requests of one public request with ``n > 1``.

        Args:
            parent: The id the outputs carry.
            sequences: The siblings, in sequence order; each one's ``sequence_index`` is the
                ``CompletionOutput.index`` of its sequence.
            stream: As :meth:`register`.
        """
        first = sequences[0]
        family = _Family(
            parent=parent, prompt=first.prompt_token_ids, stream=stream, arrival=self._clock()
        )
        for request in sequences:
            sequence = _Sequence(
                request_id=request.request_id,
                index=request.sequence_index,
                sampling=request.sampling,
                detokenizer=self._detokenizer(request),
                family=family,
            )
            family.sequences.append(sequence)
            family.live[request.request_id] = sequence
            self._sequences[request.request_id] = sequence

    def _detokenizer(self, request: TokenizedRequest) -> DetokenizerLike | None:
        if self._service.tokenizer is None:
            return None
        sampling = request.sampling
        return self._service.new_detokenizer(
            DetokenizeParams(
                skip_special_tokens=True,
                stop=sampling.stop,
                include_stop_str_in_output=sampling.include_stop_str_in_output,
            ),
            prompt_token_ids=request.prompt_token_ids,
        )

    def discard(self, request_id: RequestId) -> None:
        """Forget a sequence without producing output, for one that never reached the scheduler.

        The other sequences of its request are forgotten with it.
        """
        sequence = self._sequences.get(request_id)
        if sequence is None:
            return
        for sibling in sequence.family.sequences:
            self._sequences.pop(sibling.request_id, None)

    def process(self, events: Iterable[RequestEvent]) -> list[RequestOutput]:
        """Turn the events of a step into outputs, in event order.

        Events for a sequence that already finished here, such as the bare end event the scheduler
        queues after a stop-string abort, are ignored.
        """
        outputs: list[RequestOutput] = []
        for event in events:
            sequence = self._sequences.get(event.request_id)
            if sequence is None:
                continue
            output = self._handle(event, sequence)
            if output is not None:
                outputs.append(output)
        return outputs

    def fail_all(self, error: ErrorInfo) -> list[RequestOutput]:
        """End every unfinished request with an error, for an engine that has died."""
        outputs: list[RequestOutput] = []
        families: dict[int, _Family] = {}
        for sequence in self._sequences.values():
            families.setdefault(id(sequence.family), sequence.family)
        self._sequences.clear()
        for family in families.values():
            for sequence in family.live.values():
                family.done[sequence.index] = CompletionOutput(
                    index=sequence.index,
                    token_ids=tuple(sequence.tokens),
                    text="".join(sequence.text),
                    finish_reason=FinishReason.ERROR,
                )
            family.live.clear()
            outputs.append(self._final(family, None, error))
        return outputs

    # -- one event -------------------------------------------------------------------------

    def _handle(self, event: RequestEvent, sequence: _Sequence) -> RequestOutput | None:
        now = self._clock()
        family = sequence.family
        kept = event.token_ids
        text = ""
        stop_string: str | None = None
        reason = event.finish_reason
        stop_reason: TokenId | str | None = event.stop_reason
        error = event.error
        sequence.cached_tokens = max(sequence.cached_tokens, event.cached_tokens)
        family.cached_tokens = max(family.cached_tokens, sequence.cached_tokens)
        written = max(sequence.cache_written_tokens, event.cache_written_tokens)
        family.cache_written_tokens += written - sequence.cache_written_tokens
        sequence.cache_written_tokens = written
        if kept and family.first_token_at is None:
            family.first_token_at = now
        detokenizer = sequence.detokenizer
        if detokenizer is not None:
            try:
                if kept:
                    # A stop string counts only once ``min_tokens`` tokens exist before it.
                    allowed = len(sequence.tokens) + len(kept) > sequence.sampling.min_tokens
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
                self._abort(sequence.request_id, FinishReason.ERROR)
                error = ErrorInfo("detokenization_failed", f"cannot decode the output: {exc}")
                reason, stop_reason, stop_string = FinishReason.ERROR, None, None
        if stop_string is not None:
            reason, stop_reason = FinishReason.STOP, stop_string
            if event.finish_reason is None:
                self._abort(sequence.request_id, FinishReason.STOP)
        sequence.tokens.extend(kept)
        family.completion_tokens += len(kept)
        sequence.text.append(text)
        fresh = self._named(event.logprobs[: len(kept)]) if event.logprobs else ()
        sequence.logprobs.extend(fresh)
        fragment = CompletionOutput(
            index=sequence.index,
            token_ids=kept,
            text=text,
            finish_reason=reason,
            stop_reason=stop_reason,
            logprobs=fresh or None,
            cumulative_logprob=sequence.cumulative(),
        )
        if reason is None:
            if family.stream and (kept or text):
                return self._output(family, (fragment,), finished=False)
            return None
        self._end_sequence(sequence, reason, stop_reason)
        if reason in _FAILURES:
            for sibling in tuple(family.live.values()):
                self._abort(sibling.request_id, FinishReason.CANCELLED)
                self._end_sequence(sibling, FinishReason.CANCELLED, None)
        if family.live:
            return self._output(family, (fragment,), finished=False) if family.stream else None
        return self._final(family, fragment, error)

    def _end_sequence(
        self, sequence: _Sequence, reason: FinishReason, stop_reason: TokenId | str | None
    ) -> None:
        family = sequence.family
        self._sequences.pop(sequence.request_id, None)
        family.live.pop(sequence.request_id, None)
        family.done[sequence.index] = CompletionOutput(
            index=sequence.index,
            token_ids=tuple(sequence.tokens),
            text="".join(sequence.text),
            finish_reason=reason,
            stop_reason=stop_reason,
            logprobs=tuple(sequence.logprobs) or None,
            cumulative_logprob=sequence.cumulative(),
        )

    def _named(self, entries: tuple[TokenLogprob, ...]) -> tuple[TokenLogprob, ...]:
        """The entries with the text of each token and of each alternative."""
        tokenizer = self._service.tokenizer
        if tokenizer is None:
            return entries

        def text(token: TokenId) -> str:
            try:
                return tokenizer.decode([token])
            except (ValueError, KeyError, IndexError):
                return ""

        return tuple(
            replace(
                entry,
                decoded=text(entry.token_id),
                top_decoded=tuple(text(token) for token, _ in entry.top),
            )
            for entry in entries
        )

    def _final(
        self, family: _Family, fragment: CompletionOutput | None, error: ErrorInfo | None
    ) -> RequestOutput:
        """The last output of a request: its last fragment when streaming, every sequence else."""
        if family.stream and fragment is not None:
            outputs: tuple[CompletionOutput, ...] = (fragment,)
        else:
            outputs = tuple(family.done[index] for index in sorted(family.done))
        return self._output(family, outputs, finished=True, error=error)

    def _output(
        self,
        family: _Family,
        outputs: tuple[CompletionOutput, ...],
        *,
        finished: bool,
        error: ErrorInfo | None = None,
    ) -> RequestOutput:
        return RequestOutput(
            request_id=family.parent,
            prompt_token_ids=family.prompt,
            outputs=outputs,
            finished=finished,
            usage=family.usage(),
            metrics=self._metrics(family) if finished else None,
            error=error,
        )

    def _metrics(self, family: _Family) -> RequestMetrics:
        """Timings from the engine clock; the queue and prefill times are not measured.

        The first token is the first of any sequence; the time per output token divides the
        decode time by every token after the first, across the sequences.
        """
        now = self._clock()
        first = family.first_token_at
        if first is None:
            return RequestMetrics(arrival_time=family.arrival)
        decode = max(0.0, now - first)
        count = sum(len(sequence.tokens) for sequence in family.sequences)
        return RequestMetrics(
            arrival_time=family.arrival,
            decode_time=decode,
            ttft=max(0.0, first - family.arrival),
            tpot=decode / (count - 1) if count > 1 else 0.0,
        )
