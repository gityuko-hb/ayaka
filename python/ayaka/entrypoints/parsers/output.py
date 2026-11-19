"""From one request's engine outputs to :data:`~ayaka.entrypoints.conversation.OutputEvent` values.

:class:`OutputFormat` is chosen once per server: which reasoning and tool-call formats the loaded
model uses, with their markers resolved against its tokenizer. Each request gets its own
:class:`OutputParser`, which takes the request's :class:`~ayaka.core.outputs.RequestOutput`
fragments in order and returns the events they complete::

    fragment text --> reasoning parser --+--> ReasoningDelta
                                         +--> tool-call parser --> TextDelta / ToolCall*
    final fragment --> flush both, then UsageEvent + FinishEvent, or ErrorEvent

The parsers see the decoded text after the engine applied the stop strings, so a stop string never
reaches them and the usage counts only the tokens that were emitted.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

from ayaka.core.outputs import RequestOutput
from ayaka.core.types import FinishReason
from ayaka.entrypoints.conversation import (
    ChoiceEvent,
    ErrorEvent,
    FinishEvent,
    ReasoningDelta,
    TextDelta,
    TokenLogprobs,
    ToolCallEnd,
    UsageEvent,
    merge_deltas,
)
from ayaka.entrypoints.parsers.reasoning import ReasoningParser, reasoning_format
from ayaka.entrypoints.parsers.tool_calls import ToolCallParser, tool_call_format

if TYPE_CHECKING:
    from collections.abc import Sequence

    from ayaka.entrypoints.conversation import OutputEvent
    from ayaka.entrypoints.parsers.markers import Marker
    from ayaka.entrypoints.parsers.reasoning import ReasoningFormat
    from ayaka.entrypoints.parsers.tool_calls import ToolCallFormat
    from ayaka.tokenizers.ports import TokenizerLike

__all__ = ["ChoicesParser", "EventParser", "OutputFormat", "OutputParser"]


class EventParser(Protocol):
    """Turns a request's outputs into events: :class:`OutputParser`, :class:`ChoicesParser`."""

    def feed(self, output: RequestOutput) -> list[OutputEvent]:
        """The events of the next fragment; the final fragment also ends the stream."""
        ...

    @property
    def generated_tokens(self) -> int:
        """Tokens of every sequence fed so far: the count when no usage event came."""
        ...


@dataclass(frozen=True, slots=True)
class OutputFormat:
    """The output formats of the loaded model, with their markers resolved.

    Attributes:
        reasoning: The reasoning format, or ``None``: all output is answer text.
        reasoning_markers: Its markers as the tokenizer spells them.
        tool_calls: The tool-call format, or ``None``: the model cannot be asked for tool calls.
        tool_markers: Its markers as the tokenizer spells them.
    """

    reasoning: ReasoningFormat | None = None
    reasoning_markers: tuple[Marker, Marker] | None = None
    tool_calls: ToolCallFormat | None = None
    tool_markers: tuple[Marker, Marker] | None = None

    @classmethod
    def resolve(
        cls,
        *,
        reasoning: str | None = None,
        tool_calls: str | None = None,
        tokenizer: TokenizerLike | None = None,
    ) -> OutputFormat:
        """Look the formats up by name and resolve their markers against ``tokenizer``.

        Raises:
            ConfigError: If a name is unknown or a marker is a special token.
        """
        reasoning_fmt = reasoning_format(reasoning) if reasoning is not None else None
        tool_fmt = tool_call_format(tool_calls) if tool_calls is not None else None
        return cls(
            reasoning=reasoning_fmt,
            reasoning_markers=reasoning_fmt.markers(tokenizer) if reasoning_fmt else None,
            tool_calls=tool_fmt,
            tool_markers=tool_fmt.markers(tokenizer) if tool_fmt else None,
        )

    def parser(
        self, *, call_id_prefix: str, tools: bool, parallel_tool_calls: bool = True
    ) -> OutputParser:
        """A parser for one request.

        Args:
            call_id_prefix: Prefix of the request's tool-call ids.
            tools: Whether the request offered tools; without them no call is looked for.
            parallel_tool_calls: Whether the answer may hold several calls. When not, the answer
                ends right after its first call (see :attr:`OutputParser.ended_early`).
        """
        reasoning = (
            ReasoningParser(self.reasoning, self.reasoning_markers)
            if self.reasoning is not None
            else None
        )
        calls = (
            ToolCallParser(
                self.tool_calls, call_id_prefix=call_id_prefix, markers=self.tool_markers
            )
            if self.tool_calls is not None and tools
            else None
        )
        return OutputParser(reasoning, calls, single_call=not parallel_tool_calls)


class OutputParser:
    """Turns the output fragments of one request into events.

    Args:
        reasoning: The reasoning parser, or ``None``.
        tool_calls: The tool-call parser, or ``None``.
        single_call: End the answer right after its first function call: the events stop there
            with a normal finish, and :attr:`ended_early` tells the caller to cancel the request
            in the engine. The usage then counts the tokens this parser was given.
    """

    def __init__(
        self,
        reasoning: ReasoningParser | None,
        tool_calls: ToolCallParser | None,
        *,
        single_call: bool = False,
    ) -> None:
        self._reasoning = reasoning
        self._calls = tool_calls
        self._single_call = single_call
        self._reasoning_tokens = 0
        self._tokens = 0
        self._done = False
        self.ended_early = False
        """Whether the answer ended before the engine finished the request."""

    @property
    def done(self) -> bool:
        """Whether the parser has emitted its terminal event."""
        return self._done

    @property
    def generated_tokens(self) -> int:
        """Tokens fed so far."""
        return self._tokens

    def feed(self, output: RequestOutput) -> list[OutputEvent]:
        """The events of the next fragment; the final fragment also ends the stream."""
        if self._done:
            return []
        completion = output.outputs[0] if output.outputs else None
        token_ids = completion.token_ids if completion is not None else ()
        text = completion.text if completion is not None else ""
        self._tokens += len(token_ids)
        events: list[OutputEvent] = []
        reasoning = self._reasoning
        if reasoning is not None:
            if reasoning.inside:
                self._reasoning_tokens += len(token_ids)
            reasoning.note_tokens(token_ids)
            pieces = reasoning.feed(text)
        else:
            pieces = [(False, text)] if text else []
        if self._calls is not None:
            # Only the ids of answer text: a marker token inside the reasoning is not a call.
            self._calls.note_tokens(reasoning.answer_tokens if reasoning is not None else token_ids)
        self._route(pieces, events)
        if completion is not None and completion.logprobs:
            events.append(TokenLogprobs(completion.logprobs))
        if self._single_call and not output.finished:
            for position, event in enumerate(events):
                if isinstance(event, ToolCallEnd):
                    return self._end_early(output, events[: position + 1])
        if output.finished:
            self._done = True
            if reasoning is not None:
                self._route(reasoning.finish(), events)
            if self._calls is not None:
                events += self._calls.finish()
            terminal = self._terminal(output)
            if isinstance(terminal, FinishEvent):
                events.append(self._usage(output))
            events.append(terminal)
        return merge_deltas(events)

    def _end_early(self, output: RequestOutput, events: list[OutputEvent]) -> list[OutputEvent]:
        self._done = True
        self.ended_early = True
        usage = output.usage
        events.append(
            UsageEvent(
                prompt_tokens=usage.prompt_tokens,
                completion_tokens=self._tokens,
                cached_tokens=usage.cached_tokens,
                reasoning_tokens=min(self._reasoning_tokens, self._tokens),
            )
        )
        events.append(FinishEvent(FinishReason.STOP, tool_calls=True))
        return merge_deltas(events)

    def _route(self, pieces: list[tuple[bool, str]], events: list[OutputEvent]) -> None:
        for is_reasoning, text in pieces:
            if is_reasoning:
                events.append(ReasoningDelta(text))
            elif self._calls is not None:
                events += self._calls.feed(text)
            else:
                events.append(TextDelta(text))

    def _usage(self, output: RequestOutput) -> UsageEvent:
        usage = output.usage
        return UsageEvent(
            prompt_tokens=usage.prompt_tokens,
            completion_tokens=usage.completion_tokens,
            cached_tokens=usage.cached_tokens,
            reasoning_tokens=min(self._reasoning_tokens, usage.completion_tokens),
        )

    def _terminal(self, output: RequestOutput) -> FinishEvent | ErrorEvent:
        """How the request ended; a finished answer may carry a warning and is still an answer."""
        completion = output.outputs[0] if output.outputs else None
        reason = completion.finish_reason if completion is not None else None
        if reason is FinishReason.STOP or reason is FinishReason.LENGTH:
            stop = completion.stop_reason if completion is not None else None
            return FinishEvent(
                reason=reason,
                stop_sequence=stop if isinstance(stop, str) else None,
                tool_calls=self._calls is not None and self._calls.num_calls > 0,
            )
        error = output.error
        if error is not None:
            return ErrorEvent(error.code, error.message, retryable=error.retryable)
        if reason is FinishReason.TIMEOUT:
            return ErrorEvent("request_timeout", "the request ran out of time", retryable=True)
        if reason is FinishReason.CANCELLED:
            return ErrorEvent("cancelled", "the request was cancelled")
        return ErrorEvent("engine_error", "the request failed")


class ChoicesParser:
    """The parser of a request with several sequences: one :class:`OutputParser` per choice.

    Each choice's events come wrapped in :class:`~ayaka.entrypoints.conversation.ChoiceEvent`,
    its own finish included. The request's usage and terminal event follow, unwrapped, once the
    engine finished the request, or as soon as every choice has ended (a choice that ends right
    after its first tool call does not wait for the engine).

    Args:
        parsers: One parser per choice, in index order.
    """

    def __init__(self, parsers: Sequence[OutputParser]) -> None:
        self._parsers = tuple(parsers)
        self._done = False
        self._tool_calls = False
        self._cancelled = False

    @property
    def count(self) -> int:
        """Choices of the request."""
        return len(self._parsers)

    @property
    def generated_tokens(self) -> int:
        """Tokens fed so far, over every choice."""
        return sum(parser.generated_tokens for parser in self._parsers)

    def feed(self, output: RequestOutput) -> list[OutputEvent]:
        """The events of the next fragment, per choice, then the request's end."""
        if self._done:
            return []
        events: list[OutputEvent] = []
        for completion in output.outputs:
            parser = self._parsers[completion.index]
            ended = completion.finish_reason is not None
            if completion.finish_reason is FinishReason.CANCELLED:
                self._cancelled = True
            view = RequestOutput(
                request_id=output.request_id,
                outputs=(completion,),
                finished=ended or output.finished,
                usage=output.usage,
                error=output.error if output.finished else None,
            )
            for event in parser.feed(view):
                if isinstance(event, UsageEvent):
                    continue
                if isinstance(event, FinishEvent) and event.tool_calls:
                    self._tool_calls = True
                events.append(ChoiceEvent(completion.index, event))
        if output.finished or all(parser.done for parser in self._parsers):
            self._done = True
            usage = output.usage
            events.append(
                UsageEvent(
                    prompt_tokens=usage.prompt_tokens,
                    completion_tokens=usage.completion_tokens,
                    cached_tokens=usage.cached_tokens,
                )
            )
            error = output.error if output.finished else None
            if error is not None:
                events.append(ErrorEvent(error.code, error.message, retryable=error.retryable))
            elif self._cancelled:
                events.append(ErrorEvent("cancelled", "the request was cancelled"))
            else:
                events.append(FinishEvent(FinishReason.STOP, tool_calls=self._tool_calls))
        return events
