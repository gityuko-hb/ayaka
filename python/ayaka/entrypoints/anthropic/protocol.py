"""The Anthropic Messages wire format: reading requests, and the shapes of answers and errors.

Reading follows the same policy as the OpenAI protocols (``OUT-9``): an unknown optional field is
ignored with a debug log line; a known field that would change the answer and is not available is
refused with 400, naming it.

How the request maps (``OUT-9``):

* ``system``, a string or text blocks, becomes the system prompt; blocks are joined like every
  multi-part text.
* ``cache_control`` marks, on blocks or on the request, are accepted as hints only and change
  nothing in the prompt.
* A ``json_schema`` ``output_format`` (or ``output_config.format``) and a forced ``tool_choice``
  (``any``, ``tool``) become output constraints; the other ``output_config`` fields are hints.
* ``tool_use`` blocks become the assistant's tool calls, ``tool_result`` blocks tool turns;
  ``thinking`` blocks of earlier turns are dropped, as the chat templates of reasoning models do.
* Images and documents are refused until vision lands.
* Tools without an ``input_schema`` (the provider's own tools, such as web search) are left out of
  the prompt with an info log line: only the provider can run them.
* ``thinking.budget_tokens`` is checked as Anthropic checks it but not enforced: the model reasons
  as long as it was trained to, within ``max_tokens``.
"""

from __future__ import annotations

import dataclasses
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final, cast

from ayaka.core.constraints import ConstraintKind
from ayaka.core.errors import InvalidRequestError, UnsupportedFeatureError
from ayaka.core.types import FinishReason
from ayaka.entrypoints.conversation import (
    PART_SEPARATOR,
    ConversationBuilder,
    OutputRequest,
    ToolCall,
    ToolSpec,
)
from ayaka.entrypoints.http import JsonObject, sampling_params

if TYPE_CHECKING:
    from collections.abc import Mapping

    from ayaka.core.params import SamplingParams
    from ayaka.entrypoints.conversation import Conversation, FinishEvent, UsageEvent
    from ayaka.entrypoints.http import ApiFailure

__all__ = [
    "MessagesRequest",
    "anthropic_error",
    "anthropic_usage",
    "parse_count_tokens_request",
    "parse_messages_request",
    "stop_reason",
]

_LOG = logging.getLogger("ayaka.serve")

_ERROR_TYPES: Final = {
    "invalid_request": "invalid_request_error",
    "authentication": "authentication_error",
    "not_found": "not_found_error",
    "too_large": "request_too_large",
    "overloaded": "rate_limit_error",
    "timeout": "api_error",
    "unavailable": "overloaded_error",
    "internal": "api_error",
}


def anthropic_error(failure: ApiFailure) -> dict[str, object]:
    """An error body in Anthropic's shape."""
    return {
        "type": "error",
        "error": {"type": _ERROR_TYPES[failure.kind], "message": failure.message},
    }


def _unsupported(field: str, why: str) -> UnsupportedFeatureError:
    return UnsupportedFeatureError(
        f"{field}: {why}", context={"features": (field,), "param": field}
    )


def _invalid(field: str, why: str) -> InvalidRequestError:
    return InvalidRequestError(f"{field}: {why}", context={"param": field})


@dataclass(frozen=True, slots=True)
class MessagesRequest:
    """A Messages request, read and checked.

    Attributes:
        model: The model the client asked for.
        conversation: The system prompt, messages and tools.
        sampling: The sampling fields; ``max_tokens`` is the client's, not yet capped.
        max_tokens: The client's limit on generated tokens.
        stream: Whether to answer with server-sent events.
        tools: Whether the model may call the tools.
        parallel_tool_calls: Whether it may call several in one answer.
        output: The shape the answer must have (``output_format``, ``tool_choice``).
    """

    model: str | None
    conversation: Conversation
    sampling: SamplingParams
    max_tokens: int
    stream: bool
    tools: bool
    parallel_tool_calls: bool
    output: OutputRequest = dataclasses.field(default_factory=OutputRequest)


_KEYS: Final = frozenset(
    {
        "model",
        "messages",
        "system",
        "max_tokens",
        "stop_sequences",
        "temperature",
        "top_p",
        "top_k",
        "stream",
        "tools",
        "tool_choice",
        "thinking",
        "metadata",
        "service_tier",
        "container",
        "mcp_servers",
        "cache_control",
        "output_format",
        "output_config",
        "n",
    }
)


def _output_format(body: JsonObject) -> tuple[ConstraintKind | None, Mapping[str, object] | None]:
    """A JSON answer from ``output_format`` or ``output_config.format`` (``json_schema``)."""
    output_config = body.child("output_config")
    found = body.child("output_format")
    where = "output_format"
    if found is None and output_config is not None:
        found = output_config.child("format")
        where = "output_config.format"
    if found is None:
        return None, None
    if found.string("type") != "json_schema":
        raise _invalid(where, "the only output format is json_schema")
    schema = found.child("schema")
    if schema is None:
        raise _invalid(found.where("schema"), "a json_schema output needs a schema")
    return ConstraintKind.JSON_SCHEMA, schema.data


def _conversation(body: JsonObject) -> tuple[Conversation, bool, bool]:
    """The conversation, whether the model may call tools, and whether several at once."""
    for field in ("container", "mcp_servers"):
        if body.has(field):
            raise _unsupported(field, "not available on this server")
    if body.has("n") and body.raw("n") != 1:
        raise _unsupported("n", "Messages returns one answer per request")
    builder = ConversationBuilder()
    system = body.raw("system")
    if isinstance(system, str):
        builder.system(system)
    elif system is not None:
        builder.system(_text_blocks(system, "system"))
    messages = body.objects("messages")
    if not messages:
        raise _invalid("messages", "must be a non-empty array")
    for message in messages:
        _message(message, builder)
    if messages[-1].string("role") == "assistant":
        raise _unsupported(
            "messages", "continuing a final assistant message (prefill) is not available"
        )
    tools, parallel, _ = _tools(body, builder)
    return builder.build(), tools, parallel


def parse_messages_request(body: JsonObject) -> MessagesRequest:
    """Read a Messages request.

    Raises:
        InvalidRequestError: For a malformed or missing field, naming it.
        UnsupportedFeatureError: For a field that would change the answer and is not available.
    """
    unknown = body.unknown(_KEYS)
    if unknown:
        _LOG.debug("messages: ignoring unknown fields %s", unknown)
    max_tokens = body.integer("max_tokens", required=True)
    assert max_tokens is not None  # noqa: S101  # required
    if max_tokens < 1:
        raise _invalid("max_tokens", "must be >= 1")
    _thinking(body.child("thinking"), max_tokens)
    conversation, tools, parallel = _conversation(body)
    response, schema = _output_format(body)
    _, _, choice = _tools(body, ConversationBuilder())
    stop = body.array("stop_sequences")
    if stop is not None and not all(isinstance(item, str) for item in stop):
        raise _invalid("stop_sequences", "must be an array of strings")
    sampling = sampling_params(
        temperature=body.number("temperature"),
        top_p=body.number("top_p"),
        top_k=body.integer("top_k"),
        stop=tuple(cast("list[str]", stop)) if stop else None,
        max_tokens=max_tokens,
    )
    return MessagesRequest(
        model=body.string("model"),
        conversation=conversation,
        sampling=sampling,
        max_tokens=max_tokens,
        stream=body.flag("stream"),
        tools=tools,
        parallel_tool_calls=parallel,
        output=OutputRequest(response=response, schema=schema, tool_choice=choice),
    )


def parse_count_tokens_request(body: JsonObject) -> tuple[str | None, Conversation]:
    """Read a ``count_tokens`` request: the model and the conversation to count."""
    unknown = body.unknown(_KEYS)
    if unknown:
        _LOG.debug("count_tokens: ignoring unknown fields %s", unknown)
    conversation, _, _ = _conversation(body)
    return body.string("model"), conversation


def _thinking(thinking: JsonObject | None, max_tokens: int) -> None:
    if thinking is None:
        return
    kind = thinking.string("type")
    if kind == "enabled":
        budget = thinking.integer("budget_tokens", required=True)
        assert budget is not None  # noqa: S101  # required
        if budget < 1024 or budget >= max_tokens:
            raise _invalid(
                thinking.where("budget_tokens"), "must be at least 1024 and less than max_tokens"
            )
        _LOG.debug("thinking budget %d is a hint; the model reasons as trained", budget)
    elif kind not in ("disabled", "adaptive"):
        raise _invalid(thinking.where("type"), "must be enabled, disabled or adaptive")


def _text_blocks(value: object, where: str) -> str:
    """Text from an array of text blocks."""
    if not isinstance(value, list):
        raise _invalid(where, "must be a string or an array of text blocks")
    texts: list[str] = []
    for i, item in enumerate(cast("list[object]", value)):
        block = JsonObject.of(item, f"{where}[{i}]")
        kind = block.string("type")
        if kind == "text":
            texts.append(block.text("text"))
        elif kind in ("image", "document"):
            raise _unsupported(block.where("type"), f"{kind} input needs a vision model")
        else:
            raise _unsupported(block.where("type"), f"{kind!r} blocks here")
    return PART_SEPARATOR.join(texts)


def _message(message: JsonObject, builder: ConversationBuilder) -> None:
    role = message.text("role")
    if role not in ("user", "assistant"):
        raise _invalid(message.where("role"), "must be user or assistant")
    content = message.raw("content")
    if isinstance(content, str):
        if role == "user":
            builder.user(content)
        else:
            builder.assistant(content)
        return
    if not isinstance(content, list):
        raise _invalid(message.where("content"), "must be a string or an array of blocks")
    texts: list[str] = []
    calls: list[ToolCall] = []
    for i, item in enumerate(cast("list[object]", content)):
        block = JsonObject.of(item, f"{message.where('content')}[{i}]")
        kind = block.string("type")
        match kind:
            case "text":
                texts.append(block.text("text"))
            case "tool_use" if role == "assistant":
                arguments = block.child("input") or JsonObject({})
                calls.append(ToolCall(block.text("id"), block.text("name"), arguments.data))
            case "tool_result" if role == "user":
                if texts:
                    builder.user(PART_SEPARATOR.join(texts))
                    texts = []
                result = block.raw("content")
                text = result if isinstance(result, str) else ""
                if result is not None and not isinstance(result, str):
                    text = _text_blocks(result, block.where("content"))
                builder.tool_result(block.text("tool_use_id"), text)
            case "thinking" | "redacted_thinking" if role == "assistant":
                continue
            case "image" | "document":
                raise _unsupported(block.where("type"), f"{kind} input needs a vision model")
            case _:
                raise _unsupported(block.where("type"), f"{kind!r} blocks in {role} messages")
    if role == "assistant":
        builder.assistant(PART_SEPARATOR.join(texts), calls)
    elif texts:
        builder.user(PART_SEPARATOR.join(texts))


def _tools(body: JsonObject, builder: ConversationBuilder) -> tuple[bool, bool, str]:
    """Add the tools; return ``(offered, parallel, choice)``.

    The choice is ``"auto"``, ``"required"`` (``any``) or the one tool's name (``tool``).
    """
    choice = body.child("tool_choice")
    kind = choice.string("type") if choice is not None else "auto"
    parallel = not (choice is not None and choice.flag("disable_parallel_tool_use"))
    if kind not in ("auto", "none", "any", "tool"):
        raise _invalid("tool_choice", "must be auto, any, tool or none")
    constrained = "auto"
    if kind == "any":
        constrained = "required"
    elif kind == "tool":
        assert choice is not None  # noqa: S101  # kind came from it
        constrained = choice.text("name")
    tools = body.objects("tools")
    if kind == "none" or not tools:
        if constrained != "auto":
            raise _invalid("tool_choice", "forces a tool call but the request has no tools")
        return False, parallel, "auto"
    offered = False
    for tool in tools:
        schema = tool.child("input_schema")
        if schema is None:
            _LOG.info("leaving out tool %r: only its provider can run it", tool.string("name"))
            continue
        builder.tool(ToolSpec(tool.text("name"), tool.string("description"), schema.data))
        offered = True
    return offered, parallel, constrained


# -- answers ----------------------------------------------------------------------------------


def stop_reason(finish: FinishEvent) -> str:
    """Anthropic's ``stop_reason`` for a normal end."""
    if finish.tool_calls:
        return "tool_use"
    if finish.reason is FinishReason.LENGTH:
        return "max_tokens"
    return "stop_sequence" if finish.stop_sequence is not None else "end_turn"


def anthropic_usage(usage: UsageEvent | None, *, prompt_tokens: int) -> dict[str, object]:
    """The usage object; the prompt tokens read from the prefix cache, when the engine says."""
    cached = usage.cached_tokens if usage is not None else 0
    return {
        "input_tokens": (usage.prompt_tokens if usage is not None else prompt_tokens) - cached,
        "output_tokens": usage.completion_tokens if usage is not None else 0,
        "cache_creation_input_tokens": 0,
        "cache_read_input_tokens": cached,
    }
