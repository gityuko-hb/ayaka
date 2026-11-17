"""The format every protocol speaks inside the server: a conversation in, events out.

The three wire protocols (OpenAI Chat Completions, OpenAI Responses, Anthropic Messages) say the
same things in different shapes. Each has exactly two jobs: turn its request into a
:class:`Conversation`, and turn a sequence of :data:`OutputEvent` values into its own response or
stream. Everything in between (the chat template, tokenizing, the output parsers) works on these
two types only, so a tool call is implemented once.

A conversation becomes a prompt through one serializer, :meth:`Conversation.template_input`, so
the same conversation written in any protocol renders to the same text and the same token ids.
That is what lets a prefix cache hit across clients that speak different protocols.

Events, in the order a stream produces them::

    TextDelta | ReasoningDelta | ToolCallStart ToolCallDelta* ToolCallEnd    (any number, any mix)
    UsageEvent FinishEvent                                                   (a normal end)
    ErrorEvent                                                               (any other end)

Every stream ends with exactly one :class:`FinishEvent` or one :class:`ErrorEvent`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Final, Literal, cast

from ayaka.core.errors import InvalidRequestError
from ayaka.tokenizers.contracts import TemplateInput
from ayaka.utils.parsing import parse_json_strict

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence

    from ayaka.core.types import FinishReason

__all__ = [
    "PART_SEPARATOR",
    "Collected",
    "CollectedToolCall",
    "Conversation",
    "ConversationBuilder",
    "ErrorEvent",
    "FinishEvent",
    "OutputEvent",
    "ReasoningDelta",
    "Role",
    "TextDelta",
    "ToolCall",
    "ToolCallDelta",
    "ToolCallEnd",
    "ToolCallStart",
    "ToolSpec",
    "Turn",
    "UsageEvent",
    "collect",
    "merge_deltas",
    "parse_arguments",
]

#: How the text parts of one message, or the blocks of a system prompt, are joined.
PART_SEPARATOR: Final = "\n"

type Role = Literal["system", "user", "assistant", "tool"]


# -- the request side --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ToolSpec:
    """A function the model may call.

    Attributes:
        name: The function name.
        description: What it does, or ``None`` when the client gave none.
        parameters: Its JSON schema, in the client's key order.
    """

    name: str
    description: str | None
    parameters: Mapping[str, object]


@dataclass(frozen=True, slots=True)
class ToolCall:
    """A call the assistant made earlier in the conversation.

    Attributes:
        id: The id the client knows the call by; tool results point at it.
        name: The function name.
        arguments: The arguments as a JSON object, in the order they were written.
    """

    id: str
    name: str
    arguments: Mapping[str, object]


@dataclass(frozen=True, slots=True)
class Turn:
    """One message of the conversation.

    Attributes:
        role: Who speaks. ``tool`` carries the result of one tool call.
        text: The text; empty for an assistant turn that only calls tools.
        tool_calls: The calls an assistant turn makes.
        tool_call_id: For a ``tool`` turn, the call it answers.
        name: For a ``tool`` turn, the function that was called, when it is known.
    """

    role: Role
    text: str = ""
    tool_calls: tuple[ToolCall, ...] = ()
    tool_call_id: str | None = None
    name: str | None = None


@dataclass(frozen=True, slots=True)
class Conversation:
    """A whole request's prompt, independent of the protocol it came in.

    Attributes:
        system: The system prompt, or ``None``.
        turns: The messages after it, oldest first.
        tools: The functions the model may call.
    """

    system: str | None
    turns: tuple[Turn, ...]
    tools: tuple[ToolSpec, ...] = ()

    def messages(self) -> list[dict[str, object]]:
        """The conversation as chat-template messages (the Hugging Face layout)."""
        messages: list[dict[str, object]] = []
        if self.system is not None:
            messages.append({"role": "system", "content": self.system})
        for turn in self.turns:
            message: dict[str, object] = {"role": turn.role, "content": turn.text}
            if turn.tool_calls:
                message["tool_calls"] = [
                    {
                        "id": call.id,
                        "type": "function",
                        "function": {"name": call.name, "arguments": dict(call.arguments)},
                    }
                    for call in turn.tool_calls
                ]
            if turn.role == "tool":
                message["tool_call_id"] = turn.tool_call_id
                if turn.name is not None:
                    message["name"] = turn.name
            messages.append(message)
        return messages

    def tool_definitions(self) -> list[dict[str, object]]:
        """The tools as chat-template tool definitions."""
        definitions: list[dict[str, object]] = []
        for tool in self.tools:
            function: dict[str, object] = {"name": tool.name}
            if tool.description is not None:
                function["description"] = tool.description
            function["parameters"] = dict(tool.parameters)
            definitions.append({"type": "function", "function": function})
        return definitions

    def template_input(self) -> TemplateInput:
        """The one serialization every protocol's conversation goes through to the template."""
        return TemplateInput(
            messages_json=_dumps(self.messages()), tools_json=_dumps(self.tool_definitions())
        )


def _dumps(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


class ConversationBuilder:
    """Builds a :class:`Conversation` with the same rules for every protocol.

    The rules that keep one conversation one prompt:

    * System text that comes before the first message goes to ``Conversation.system``, parts
      joined by :data:`PART_SEPARATOR`; system text later on becomes a ``system`` turn.
    * A tool call joins the assistant turn just before it, unless a tool result or another speaker
      came in between; otherwise it opens an assistant turn with no text.
    * A tool result names the function of the call it answers when that call is in the history.
    """

    def __init__(self) -> None:
        self._system: list[str] = []
        self._turns: list[Turn] = []
        self._tools: list[ToolSpec] = []
        self._call_names: dict[str, str] = {}

    def system(self, text: str) -> None:
        """Add system text."""
        if self._turns:
            self._turns.append(Turn(role="system", text=text))
        else:
            self._system.append(text)

    def user(self, text: str) -> None:
        """Add a user message."""
        self._turns.append(Turn(role="user", text=text))

    def assistant(self, text: str, tool_calls: Sequence[ToolCall] = ()) -> None:
        """Add an assistant message, with the tool calls it makes."""
        self._turns.append(Turn(role="assistant", text=text))
        for call in tool_calls:
            self.tool_call(call)

    def tool_call(self, call: ToolCall) -> None:
        """Add a tool call to the current assistant turn, or open one for it."""
        if call.id in self._call_names:
            raise InvalidRequestError(f"tool call id {call.id!r} appears twice in the history")
        self._call_names[call.id] = call.name
        last = self._turns[-1] if self._turns else None
        if last is not None and last.role == "assistant":
            self._turns[-1] = Turn(
                role="assistant", text=last.text, tool_calls=(*last.tool_calls, call)
            )
        else:
            self._turns.append(Turn(role="assistant", tool_calls=(call,)))

    def tool_result(self, call_id: str, text: str) -> None:
        """Add the result of a tool call."""
        self._turns.append(
            Turn(role="tool", text=text, tool_call_id=call_id, name=self._call_names.get(call_id))
        )

    def tool(self, spec: ToolSpec) -> None:
        """Offer a function to the model."""
        if any(tool.name == spec.name for tool in self._tools):
            raise InvalidRequestError(f"tool {spec.name!r} is defined twice")
        self._tools.append(spec)

    def build(self) -> Conversation:
        """The conversation.

        Raises:
            InvalidRequestError: If it has no message at all.
        """
        if not self._turns:
            raise InvalidRequestError("the conversation has no message")
        system = PART_SEPARATOR.join(self._system) if self._system else None
        return Conversation(system=system, turns=tuple(self._turns), tools=tuple(self._tools))


def parse_arguments(text: str, *, where: str) -> dict[str, object]:
    """Tool-call arguments sent as JSON text, as an object; ``where`` names the field."""
    try:
        value = parse_json_strict(text)
    except ValueError as exc:
        raise InvalidRequestError(f"{where}: the arguments are not valid JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise InvalidRequestError(f"{where}: the arguments must be a JSON object")
    return cast("dict[str, object]", value)


# -- the output side ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class TextDelta:
    """Answer text."""

    text: str


@dataclass(frozen=True, slots=True)
class ReasoningDelta:
    """Text of the model's reasoning block."""

    text: str


@dataclass(frozen=True, slots=True)
class ToolCallStart:
    """The model started calling a function; its arguments follow.

    Attributes:
        index: Position of this call among the calls of the answer, from 0.
        id: The id the client answers the call with.
        name: The function name.
    """

    index: int
    id: str
    name: str


@dataclass(frozen=True, slots=True)
class ToolCallDelta:
    """A piece of the arguments of call ``index``, as raw JSON text."""

    index: int
    arguments: str


@dataclass(frozen=True, slots=True)
class ToolCallEnd:
    """Call ``index`` is complete.

    Attributes:
        index: The call.
        valid: Whether the arguments the model wrote are one JSON object. They are passed on as
            written either way; nothing repairs them.
    """

    index: int
    valid: bool


@dataclass(frozen=True, slots=True)
class UsageEvent:
    """Token counts of the request, sent once before it finishes.

    ``reasoning_tokens`` is attributed per engine step, so a step whose tokens straddle the end
    of the reasoning block counts by where the step started.
    """

    prompt_tokens: int
    completion_tokens: int
    cached_tokens: int = 0
    reasoning_tokens: int = 0


@dataclass(frozen=True, slots=True)
class FinishEvent:
    """The answer ended normally.

    Attributes:
        reason: ``FinishReason.STOP`` or ``FinishReason.LENGTH``.
        stop_sequence: The stop string that ended it, or ``None`` (end of sequence, stop token).
        tool_calls: Whether the answer called at least one function.
    """

    reason: FinishReason
    stop_sequence: str | None = None
    tool_calls: bool = False


@dataclass(frozen=True, slots=True)
class ErrorEvent:
    """The answer ended with an error: a failure, a timeout or a cancellation.

    Attributes:
        code: The ``ErrorInfo`` code (``cancelled``, ``request_timeout``, ``backend_error``...).
        message: What happened, safe to show the client.
        retryable: Whether the same request may succeed later.
    """

    code: str
    message: str
    retryable: bool = False


type OutputEvent = (
    TextDelta
    | ReasoningDelta
    | ToolCallStart
    | ToolCallDelta
    | ToolCallEnd
    | UsageEvent
    | FinishEvent
    | ErrorEvent
)
"""Anything a request's output stream says."""


def merge_deltas(events: Iterable[OutputEvent]) -> list[OutputEvent]:
    """The events with each run of adjacent text pieces of the same kind joined into one.

    Two streams that cut the same output at different places merge to the same list.
    """
    merged: list[OutputEvent] = []
    for event in events:
        last = merged[-1] if merged else None
        if isinstance(event, TextDelta) and isinstance(last, TextDelta):
            merged[-1] = TextDelta(last.text + event.text)
        elif isinstance(event, ReasoningDelta) and isinstance(last, ReasoningDelta):
            merged[-1] = ReasoningDelta(last.text + event.text)
        elif (
            isinstance(event, ToolCallDelta)
            and isinstance(last, ToolCallDelta)
            and last.index == event.index
        ):
            merged[-1] = ToolCallDelta(event.index, last.arguments + event.arguments)
        elif not _empty(event):
            merged.append(event)
    return merged


def _empty(event: OutputEvent) -> bool:
    match event:
        case TextDelta(text) | ReasoningDelta(text):
            return not text
        case ToolCallDelta(_, arguments):
            return not arguments
        case _:
            return False


@dataclass(slots=True)
class CollectedToolCall:
    """One function call of a whole answer."""

    id: str
    name: str
    arguments: str = ""
    valid: bool = False


@dataclass(slots=True)
class Collected:
    """A whole answer, gathered from its events, for a response that does not stream.

    Attributes:
        text: The answer text.
        reasoning: The reasoning text.
        tool_calls: The function calls, in order.
        usage: The token counts, if the stream reported them.
        finish: How the answer ended normally, or ``None``.
        error: How it ended otherwise, or ``None``.
    """

    text: str = ""
    reasoning: str = ""
    tool_calls: list[CollectedToolCall] = field(default_factory=list[CollectedToolCall])
    usage: UsageEvent | None = None
    finish: FinishEvent | None = None
    error: ErrorEvent | None = None


def collect(events: Iterable[OutputEvent]) -> Collected:
    """Gather a stream of events into one answer."""
    answer = Collected()
    calls: dict[int, CollectedToolCall] = {}
    for event in events:
        match event:
            case TextDelta(text):
                answer.text += text
            case ReasoningDelta(text):
                answer.reasoning += text
            case ToolCallStart(index, call_id, name):
                calls[index] = CollectedToolCall(call_id, name)
                answer.tool_calls.append(calls[index])
            case ToolCallDelta(index, arguments):
                calls[index].arguments += arguments
            case ToolCallEnd(index, valid):
                calls[index].valid = valid
            case UsageEvent():
                answer.usage = event
            case FinishEvent():
                answer.finish = event
            case ErrorEvent():
                answer.error = event
    return answer
