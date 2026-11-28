"""The Anthropic Messages wire format: reading requests, and the shapes of answers and errors.

Reading follows the same policy as the OpenAI protocols: an unknown optional field is
ignored with a debug log line; a known field that would change the answer and is not available is
refused with 400, naming it.

How the request maps:

* ``system``, a string or text blocks, becomes the system prompt; blocks are joined like every
  multi-part text.
* ``cache_control`` marks, on blocks or on the request, are accepted as hints only and change
  nothing in the prompt.
* A ``json_schema`` ``output_format`` (or ``output_config.format``) and a forced ``tool_choice``
  (``any``, ``tool``) become output constraints; the other ``output_config`` fields are hints.
* ``tool_use`` blocks become the assistant's tool calls, ``tool_result`` blocks tool turns;
  ``thinking`` blocks of earlier turns are dropped, as the chat templates of reasoning models do.
* ``image`` blocks on user turns decode base64 PNG payloads; documents and remote image URLs
  are refused.
* Tools without an ``input_schema`` (the provider's own tools, such as web search) are left out of
  the prompt with an info log line: only the provider can run them.
* ``thinking.budget_tokens`` is checked as Anthropic checks it but not enforced: the model reasons
  as long as it was trained to, within ``max_tokens``.
"""

from __future__ import annotations

import dataclasses
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

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
from ayaka.entrypoints.render.reasoning import ThinkingRequest
from ayaka.entrypoints.wire.policy import apply_policy
from ayaka.entrypoints.wire.reader import JsonObject, SamplingFields
from ayaka.multimodal.png import decode_base64_image, decode_data_url_image

if TYPE_CHECKING:
    from collections.abc import Mapping

    from ayaka.core.multimodal import ImageInput
    from ayaka.entrypoints.conversation import Conversation, FinishEvent, UsageEvent
    from ayaka.entrypoints.failures import ApiFailure

__all__ = [
    "MessagesRequest",
    "anthropic_error",
    "anthropic_usage",
    "parse_count_tokens_request",
    "parse_messages_request",
    "stop_reason",
]

_LOG = logging.getLogger("ayaka.serve")


def anthropic_error(failure: ApiFailure) -> dict[str, object]:
    """An error body in Anthropic's shape, with the request id the SDK reports."""
    return {
        "type": "error",
        "error": {"type": failure.row.anthropic_type, "message": failure.message},
        "request_id": failure.request_id,
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
        sampling: The sampling fields, without ``max_tokens``: the server builds them once
            it has capped ``max_tokens`` by the room the prompt leaves.
        max_tokens: The client's limit on generated tokens.
        stream: Whether to answer with server-sent events.
        tools: Whether the model may call the tools.
        parallel_tool_calls: Whether it may call several in one answer.
        output: The shape the answer must have (``output_format``, ``tool_choice``).
    """

    model: str | None
    conversation: Conversation
    sampling: SamplingFields
    max_tokens: int
    stream: bool
    tools: bool
    parallel_tool_calls: bool
    output: OutputRequest = dataclasses.field(default_factory=OutputRequest)
    thinking: ThinkingRequest = dataclasses.field(default_factory=ThinkingRequest)
    continue_final_message: bool = False
    named_tool_choice: bool = False


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


def _conversation(body: JsonObject) -> tuple[Conversation, bool, bool, bool]:
    """The conversation, whether the model may call tools, parallel flag, and prefill flag."""
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
    continue_final = messages[-1].string("role") == "assistant"
    tools, parallel, _, _ = _tools(body, builder)
    return builder.build(), tools, parallel, continue_final


def parse_messages_request(body: JsonObject) -> MessagesRequest:
    """Read a Messages request.

    Raises:
        InvalidRequestError: For a malformed or missing field, naming it.
        UnsupportedFeatureError: For a field that would change the answer and is not available.
    """
    apply_policy(body, "/v1/messages")
    max_tokens = body.integer("max_tokens", required=True)
    assert max_tokens is not None  # noqa: S101  # required
    if max_tokens < 1:
        raise _invalid("max_tokens", "must be >= 1")
    thinking = _thinking(body.child("thinking"), max_tokens)
    conversation, tools, parallel, continue_final = _conversation(body)
    response, schema = _output_format(body)
    _, _, choice, strict = _tools(body, ConversationBuilder())
    stop = body.array("stop_sequences")
    if stop is not None and not all(isinstance(item, str) for item in stop):
        raise _invalid("stop_sequences", "must be an array of strings")
    sampling = SamplingFields(
        {
            "temperature": body.number("temperature"),
            "top_p": body.number("top_p"),
            "top_k": body.integer("top_k"),
            "stop": tuple(cast("list[str]", stop)) if stop else None,
        }
    )
    return MessagesRequest(
        model=body.string("model"),
        conversation=conversation,
        sampling=sampling,
        max_tokens=max_tokens,
        stream=body.flag("stream"),
        tools=tools,
        parallel_tool_calls=parallel,
        output=OutputRequest(response=response, schema=schema, tool_choice=choice, strict=strict),
        thinking=thinking,
        continue_final_message=continue_final,
        named_tool_choice=choice not in ("auto", "required"),
    )


def parse_count_tokens_request(body: JsonObject) -> tuple[str | None, Conversation]:
    """Read a ``count_tokens`` request: the model and the conversation to count."""
    apply_policy(body, "/v1/messages/count_tokens")
    conversation, _, _, _ = _conversation(body)
    return body.string("model"), conversation


def _thinking(thinking: JsonObject | None, max_tokens: int) -> ThinkingRequest:
    if thinking is None:
        return ThinkingRequest()
    kind = thinking.string("type")
    if kind == "enabled":
        budget = thinking.integer("budget_tokens", required=True)
        assert budget is not None  # noqa: S101  # required
        if budget < 1024 or budget >= max_tokens:
            raise _invalid(
                thinking.where("budget_tokens"), "must be at least 1024 and less than max_tokens"
            )
        return ThinkingRequest(enabled=True, budget_tokens=budget)
    if kind == "disabled":
        return ThinkingRequest(enabled=False)
    if kind == "adaptive":
        return ThinkingRequest(enabled=True)
    raise _invalid(thinking.where("type"), "must be enabled, disabled or adaptive")


def _parse_anthropic_image(block: JsonObject) -> ImageInput:
    source = block.child("source")
    if source is None:
        raise _invalid(block.where("source"), "an image block needs a source object")
    kind = source.text("type")
    if kind == "base64":
        return decode_base64_image(
            source.text("data"),
            media_type=source.text("media_type"),
            where=source.where("data"),
        )
    if kind == "url":
        return decode_data_url_image(source.text("url"), where=source.where("url"))
    raise _invalid(source.where("type"), "image source type must be base64 or url")


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
            raise _unsupported(
                block.where("type"), f"{kind} input is only allowed on user messages"
            )
        else:
            raise _unsupported(block.where("type"), f"{kind!r} blocks here")
    return PART_SEPARATOR.join(texts)


def _message(message: JsonObject, builder: ConversationBuilder) -> None:
    role = message.text("role")
    content = message.raw("content")
    if role == "system":
        # A system message inside the conversation (Claude Code 2.1 sends one after the first
        # user turn): text only, kept in place as a system turn.
        builder.system(
            content if isinstance(content, str) else _text_blocks(content, message.where("content"))
        )
        return
    if role not in ("user", "assistant"):
        raise _invalid(message.where("role"), "must be user, assistant or system")
    if isinstance(content, str):
        if role == "user":
            builder.user(content)
        else:
            builder.assistant(content)
        return
    if not isinstance(content, list):
        raise _invalid(message.where("content"), "must be a string or an array of blocks")
    texts: list[str] = []
    thinkings: list[str] = []
    images: list[ImageInput] = []
    calls: list[ToolCall] = []
    for i, item in enumerate(cast("list[object]", content)):
        block = JsonObject.of(item, f"{message.where('content')}[{i}]")
        kind = block.string("type")
        match kind:
            case "text":
                texts.append(block.text("text"))
            case "image" if role == "user":
                images.append(_parse_anthropic_image(block))
            case "tool_use" if role == "assistant":
                arguments = block.child("input") or JsonObject({})
                calls.append(ToolCall(block.text("id"), block.text("name"), arguments.data))
            case "tool_result" if role == "user":
                if texts or images:
                    builder.user(PART_SEPARATOR.join(texts), images=images)
                    texts = []
                    images = []
                result = block.raw("content")
                text = result if isinstance(result, str) else ""
                if result is not None and not isinstance(result, str):
                    text = _text_blocks(result, block.where("content"))
                builder.tool_result(block.text("tool_use_id"), text)
            case "thinking" if role == "assistant":
                thought = block.string("thinking")
                if thought:
                    thinkings.append(thought)
            case "redacted_thinking" if role == "assistant":
                continue
            case "image" | "document":
                raise _unsupported(block.where("type"), f"{kind} input needs a vision model")
            case _:
                raise _unsupported(block.where("type"), f"{kind!r} blocks in {role} messages")
    if role == "assistant":
        builder.assistant(
            PART_SEPARATOR.join(texts),
            calls,
            reasoning=PART_SEPARATOR.join(thinkings) if thinkings else None,
        )
    elif texts or images:
        builder.user(PART_SEPARATOR.join(texts), images=images)


def _tools(body: JsonObject, builder: ConversationBuilder) -> tuple[bool, bool, str, bool]:
    """Add the tools; return ``(offered, parallel, choice, strict)``.

    The choice is ``"auto"``, ``"required"`` (``any``) or the one tool's name (``tool``). With
    ``tool_choice: {type: none}`` the tools are still written into the prompt, so it keeps the
    prefix-cache blocks of the same conversation with ``auto``; the model may not call them.
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
    if not tools and constrained != "auto":
        raise _invalid("tool_choice", "forces a tool call but the request has no tools")
    offered = False
    strict = False
    for tool in tools:
        schema = tool.child("input_schema")
        if schema is None:
            _LOG.info("leaving out tool %r: only its provider can run it", tool.string("name"))
            continue
        builder.tool(ToolSpec(tool.text("name"), tool.string("description"), schema.data))
        offered = True
        strict = strict or tool.flag("strict")
    if kind == "none":
        return False, parallel, "auto", False
    return offered, parallel, constrained, strict


# -- answers ----------------------------------------------------------------------------------


def stop_reason(finish: FinishEvent) -> str:
    """Anthropic's ``stop_reason`` for a normal end."""
    if finish.tool_calls:
        return "tool_use"
    if finish.reason is FinishReason.LENGTH:
        return "max_tokens"
    return "stop_sequence" if finish.stop_sequence is not None else "end_turn"


def anthropic_usage(usage: UsageEvent | None, *, prompt_tokens: int) -> dict[str, int]:
    """The usage object of a whole answer and of a stream's ``message_delta``, the same one.

    The prompt splits three ways, as Anthropic counts it: ``cache_read_input_tokens`` came from
    the prefix cache, ``cache_creation_input_tokens`` were computed and registered in it, and
    ``input_tokens`` are the rest, so the three add up to the prompt. Before the engine reports
    (``usage`` is ``None``, a stream's ``message_start``) the whole prompt is ``input_tokens``.
    """
    if usage is None:
        return {
            "input_tokens": prompt_tokens,
            "output_tokens": 0,
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": 0,
        }
    read, written = usage.cached_tokens, usage.cache_written_tokens
    return {
        "input_tokens": usage.prompt_tokens - read - written,
        "output_tokens": usage.completion_tokens,
        "cache_creation_input_tokens": written,
        "cache_read_input_tokens": read,
    }
