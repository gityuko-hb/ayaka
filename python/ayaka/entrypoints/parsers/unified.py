"""Request-scoped output interpretation and transport-neutral event streaming."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

from ayaka.core.types import FinishReason
from ayaka.entrypoints.conversation import (
    ErrorEvent,
    FinishEvent,
    ReasoningDelta,
    TextDelta,
    TokenLogprobs,
    ToolCallEnd,
    UsageEvent,
    merge_deltas,
)

if TYPE_CHECKING:
    from ayaka.core.outputs import RequestOutput
    from ayaka.entrypoints.conversation import OutputEvent
    from ayaka.entrypoints.parsers.markers import Marker
    from ayaka.entrypoints.parsers.reasoning import ReasoningParser
    from ayaka.entrypoints.parsers.tool_calls.base import ToolCallParserLike

__all__ = [
    "ChatResponseProcessor",
    "CombinedParser",
    "OutputInterpretation",
    "UnifiedParser",
]


@dataclass(frozen=True, slots=True)
class OutputInterpretation:
    """Request-scoped controls telling output parsers how to interpret generated tokens.

    Attributes:
        force_reasoning: ``False`` disables reasoning parsing completely for this request;
            ``True`` starts inside a reasoning block; ``None`` uses marker-driven defaults.
        previous_content: Assistant text prefilled in the prompt before generation began.
        stream_reasoning: Whether to emit :class:`~ayaka.entrypoints.conversation.ReasoningDelta`
            events to the client.
        tools: Whether tool definitions were offered on this request.
        parallel_tool_calls: Whether multiple tool calls may appear in one response.
    """

    force_reasoning: bool | None = None
    previous_content: str = ""
    stream_reasoning: bool = True
    tools: bool = False
    parallel_tool_calls: bool = True


class UnifiedParser(Protocol):
    """Transport-neutral incremental parser turning engine outputs into semantic events."""

    def feed(self, output: RequestOutput) -> list[OutputEvent]:
        """Parse the next engine output fragment into semantic events."""
        ...

    @property
    def generated_tokens(self) -> int:
        """Total generated tokens fed so far."""
        ...

    @property
    def reasoning_tokens(self) -> int:
        """Generated tokens attributed to reasoning, excluding marker tokens."""
        ...

    @property
    def done(self) -> bool:
        """Whether the terminal event has been emitted."""
        ...


class CombinedParser:
    """Pairs an optional reasoning parser with an optional tool-call parser."""

    def __init__(
        self,
        reasoning: ReasoningParser | None,
        tool_calls: ToolCallParserLike | None,
        *,
        single_call: bool = False,
        stream_reasoning: bool = True,
        withheld_marker: Marker | None = None,
    ) -> None:
        self._reasoning = reasoning
        self._calls = tool_calls
        self._single_call = single_call
        self._stream_reasoning = stream_reasoning
        self._withheld = withheld_marker
        self._reasoning_tokens = 0
        self._tokens = 0
        self._done = False
        self.ended_early = False

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
        """Tokens attributed to reasoning, excluding marker tokens."""
        return min(self._reasoning_tokens, self._tokens)

    @property
    def in_reasoning(self) -> bool:
        """Whether the output is currently inside a reasoning block."""
        return self._reasoning is not None and self._reasoning.inside

    @property
    def in_call(self) -> bool:
        """Whether the output is currently inside a tool call."""
        return self._calls is not None and self._calls.in_call

    @property
    def opened_calls(self) -> int:
        """Tool-call bodies entered so far."""
        return self._calls.opened_calls if self._calls is not None else 0

    def feed(self, output: RequestOutput) -> list[OutputEvent]:
        """Parse ``output`` into merged semantic events."""
        if self._done:
            return []
        completion = output.outputs[0] if output.outputs else None
        token_ids = completion.token_ids if completion is not None else ()
        text = completion.text if completion is not None else ""
        withheld = self._withheld
        if (
            withheld is not None
            and completion is not None
            and completion.stop_reason == withheld.token_id
            and text.endswith(withheld.text)
        ):
            text = text[: len(text) - len(withheld.text)]
        self._tokens += len(token_ids)
        events: list[OutputEvent] = []
        reasoning = self._reasoning
        if reasoning is not None:
            reasoning.note_tokens(token_ids)
            self._reasoning_tokens += reasoning.last_reasoning_tokens
            pieces = reasoning.feed(text)
        else:
            pieces = [(False, text)] if text else []
        if self._calls is not None:
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
                reasoning_tokens=self.reasoning_tokens,
                cache_written_tokens=usage.cache_written_tokens,
                accepted_prediction_tokens=usage.accepted_prediction_tokens,
                rejected_prediction_tokens=usage.rejected_prediction_tokens,
            )
        )
        events.append(FinishEvent(FinishReason.STOP, tool_calls=True))
        return merge_deltas(events)

    def _route(self, pieces: list[tuple[bool, str]], events: list[OutputEvent]) -> None:
        for is_reasoning, text in pieces:
            if is_reasoning:
                if self._stream_reasoning:
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
            accepted_prediction_tokens=usage.accepted_prediction_tokens,
            rejected_prediction_tokens=usage.rejected_prediction_tokens,
        )

    def _terminal(self, output: RequestOutput) -> FinishEvent | ErrorEvent:
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


class ChatResponseProcessor:
    """Request-scoped output processor driven by :class:`OutputInterpretation`."""

    def __init__(
        self,
        reasoning: ReasoningParser | None,
        tool_calls: ToolCallParserLike | None,
        *,
        interpretation: OutputInterpretation,
        withheld_marker: Marker | None = None,
    ) -> None:
        self._interpretation = interpretation
        effective_reasoning = None if interpretation.force_reasoning is False else reasoning
        self._parser = CombinedParser(
            effective_reasoning,
            tool_calls if interpretation.tools else None,
            single_call=not interpretation.parallel_tool_calls,
            stream_reasoning=interpretation.stream_reasoning,
            withheld_marker=withheld_marker,
        )

    @property
    def interpretation(self) -> OutputInterpretation:
        """The request-scoped interpretation settings in effect."""
        return self._interpretation

    @property
    def done(self) -> bool:
        """Whether the processor has emitted its terminal event."""
        return self._parser.done

    @property
    def generated_tokens(self) -> int:
        """Tokens fed so far."""
        return self._parser.generated_tokens

    @property
    def reasoning_tokens(self) -> int:
        """Tokens attributed to reasoning, excluding marker tokens."""
        return self._parser.reasoning_tokens

    @property
    def ended_early(self) -> bool:
        """Whether generation stopped after the first tool call."""
        return self._parser.ended_early

    @property
    def in_reasoning(self) -> bool:
        """Whether the output is currently inside a reasoning block."""
        return self._parser.in_reasoning

    def feed(self, output: RequestOutput) -> list[OutputEvent]:
        """Process the next output fragment."""
        return self._parser.feed(output)
