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

from ayaka.core.constraints import ToolCallSyntax
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
from ayaka.entrypoints.parsers.formats import reasoning_format, tool_call_format
from ayaka.entrypoints.parsers.reasoning import ReasoningParser
from ayaka.entrypoints.parsers.tool_calls import ToolCallParser
from ayaka.entrypoints.parsers.unified import ChatResponseProcessor

if TYPE_CHECKING:
    from collections.abc import Sequence

    from ayaka.entrypoints.conversation import OutputEvent
    from ayaka.entrypoints.parsers.formats import ReasoningFormat, ToolCallFormat
    from ayaka.entrypoints.parsers.markers import Marker
    from ayaka.entrypoints.parsers.unified import OutputInterpretation
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

    @property
    def reasoning_tokens(self) -> int:
        """Reasoning tokens attributed so far."""
        ...

    @property
    def done(self) -> bool:
        """Whether the parser has emitted its terminal event."""
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

    def constraint_start(
        self,
        *,
        thinking_enabled: bool | None = None,
        starts_inside_reasoning: bool | None = None,
    ) -> tuple[int | None, int | None]:
        """Where an output constraint starts: ``(start_after, reasoning_open)``.

        When ``thinking_enabled`` is ``False`` for the current request, constraints start
        immediately at token 0 regardless of the model's default reasoning format. Otherwise,
        ``starts_inside_reasoning`` (when provided by the render layer) overrides
        ``self.reasoning.starts_inside`` so prompt-opened and model-opened reasoning stay
        synchronized with the parser.
        """
        if thinking_enabled is False:
            return None, None
        markers = self.reasoning_markers
        if self.reasoning is None or markers is None:
            return None, None
        opening, closing = markers
        starts_inside = (
            self.reasoning.starts_inside
            if starts_inside_reasoning is None
            else starts_inside_reasoning
        )
        if starts_inside or opening.token_id is None:
            return closing.token_id, None
        return closing.token_id, opening.token_id

    def call_syntax(self, spelled_open: tuple[int, ...] = ()) -> ToolCallSyntax | None:
        """The tool-call syntax a grammar enforces: this format's markers and arguments key.

        Args:
            spelled_open: The tokens of the opening marker when it is not one token.

        Returns:
            ``None`` without a tool-call format. A format written without markers gets a syntax
            with empty markers: its grammar can only serve a forced call.
        """
        fmt, markers = self.tool_calls, self.tool_markers
        if fmt is None:
            return None
        if markers is None:
            separator = f"{fmt.separator} " if fmt.separator else ""
            return ToolCallSyntax(
                "",
                "",
                arguments_key=fmt.arguments,
                separator=separator,
            )
        opening, closing = markers
        return ToolCallSyntax(
            opening.text,
            closing.text,
            opening.token_id,
            closing.token_id,
            spelled_open if opening.token_id is None else (),
            arguments_key=fmt.arguments,
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

    def processor(
        self,
        interpretation: OutputInterpretation,
        *,
        call_id_prefix: str,
    ) -> ChatResponseProcessor:
        """Build a request-scoped :class:`ChatResponseProcessor` from ``interpretation``."""
        reasoning = (
            ReasoningParser(
                self.reasoning,
                self.reasoning_markers,
                starts_inside_override=interpretation.force_reasoning,
                previous_content=interpretation.previous_content,
            )
            if self.reasoning is not None and interpretation.force_reasoning is not False
            else None
        )
        calls = (
            ToolCallParser(
                self.tool_calls, call_id_prefix=call_id_prefix, markers=self.tool_markers
            )
            if self.tool_calls is not None and interpretation.tools
            else None
        )
        return ChatResponseProcessor(reasoning, calls, interpretation=interpretation)


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

    @property
    def reasoning_tokens(self) -> int:
        """Tokens fed so far inside the reasoning block, at most :attr:`generated_tokens`."""
        return min(self._reasoning_tokens, self._tokens)

    @property
    def in_reasoning(self) -> bool:
        """Whether the output is inside the reasoning block now."""
        return self._reasoning is not None and self._reasoning.inside

    @property
    def in_call(self) -> bool:
        """Whether the answer is inside the body of a call now."""
        return self._calls is not None and self._calls.in_call

    @property
    def opened_calls(self) -> int:
        """Call bodies the answer entered so far, with or without a name."""
        return self._calls.opened_calls if self._calls is not None else 0

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
                cache_written_tokens=usage.cache_written_tokens,
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
            cache_written_tokens=usage.cache_written_tokens,
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

    def __init__(self, parsers: Sequence[EventParser]) -> None:
        self._parsers = tuple(parsers)
        self._done = False
        self._tool_calls = False
        self._cancelled = False

    @property
    def done(self) -> bool:
        """Whether every choice has finished."""
        return self._done

    @property
    def count(self) -> int:
        """Choices of the request."""
        return len(self._parsers)

    @property
    def generated_tokens(self) -> int:
        """Tokens fed so far, over every choice."""
        return sum(parser.generated_tokens for parser in self._parsers)

    @property
    def reasoning_tokens(self) -> int:
        """Reasoning tokens fed so far, over every choice."""
        return sum(parser.reasoning_tokens for parser in self._parsers)

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
            reasoning = sum(parser.reasoning_tokens for parser in self._parsers)
            events.append(
                UsageEvent(
                    prompt_tokens=usage.prompt_tokens,
                    completion_tokens=usage.completion_tokens,
                    cached_tokens=usage.cached_tokens,
                    reasoning_tokens=min(reasoning, usage.completion_tokens),
                    cache_written_tokens=usage.cache_written_tokens,
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
