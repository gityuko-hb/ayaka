"""The OpenAI wire formats: reading Chat Completions and Completions requests, and the shapes
of their answers and errors.

Reading follows one policy for every field:

* An optional field this server does not know is ignored and named in a debug log line; clients
  add fields faster than servers learn them.
* A known field that would change the answer and that Ayaka cannot honour is refused with 400,
  naming it. Nothing is accepted and then quietly done differently.
* Sampling fields go to ``SamplingParams`` as they are; the engine refuses the ones it cannot
  serve yet (``prompt_logprobs``) with ``UnsupportedFeatureError``.
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
    parse_arguments,
)
from ayaka.entrypoints.render.reasoning import ThinkingRequest, parse_reasoning_effort
from ayaka.entrypoints.wire.policy import apply_policy
from ayaka.entrypoints.wire.reader import JsonObject, SamplingFields

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from ayaka.core.outputs import TokenLogprob
    from ayaka.entrypoints.conversation import (
        Collected,
        Conversation,
        FinishEvent,
        UsageEvent,
    )
    from ayaka.entrypoints.failures import ApiFailure

__all__ = [
    "ChatRequest",
    "CompletionRequest",
    "chat_finish_reason",
    "chat_message",
    "openai_error",
    "openai_usage",
    "parse_chat_request",
    "parse_completion_request",
]

_LOG = logging.getLogger("ayaka.serve")


def openai_error(failure: ApiFailure) -> dict[str, object]:
    """An error body in OpenAI's shape: the row's type, and the code itself."""
    return {
        "error": {
            "message": failure.message,
            "type": failure.row.openai_type,
            "param": failure.param,
            "code": failure.code,
        }
    }


def _unsupported(field: str, why: str) -> UnsupportedFeatureError:
    return UnsupportedFeatureError(
        f"{field}: {why}", context={"features": (field,), "param": field}
    )


# -- sampling ---------------------------------------------------------------------------------


def _stop(body: JsonObject) -> tuple[str, ...] | None:
    value = body.raw("stop")
    if value is None:
        return None
    if isinstance(value, str):
        return (value,)
    if isinstance(value, list) and all(
        isinstance(item, str) for item in cast("list[object]", value)
    ):
        return tuple(cast("list[str]", value))
    raise InvalidRequestError(
        "stop: must be a string or an array of strings", context={"param": "stop"}
    )


def _logit_bias(body: JsonObject) -> dict[int, float] | None:
    bias = body.child("logit_bias")
    if bias is None:
        return None
    result: dict[int, float] = {}
    for key, value in bias.data.items():
        if not key.isdigit() or isinstance(value, bool) or not isinstance(value, (int, float)):
            raise InvalidRequestError(
                "logit_bias: keys must be token ids and values numbers",
                context={"param": "logit_bias"},
            )
        result[int(key)] = float(value)
    return result


def _sampling(body: JsonObject, *, logprobs: int | None) -> SamplingFields:
    """The sampling fields; the server adds ``max_tokens`` once it knows the prompt's length."""
    fields: dict[str, object] = {
        "n": body.integer("n"),
        "temperature": body.number("temperature"),
        "top_p": body.number("top_p"),
        "top_k": body.integer("top_k"),
        "min_p": body.number("min_p"),
        "presence_penalty": body.number("presence_penalty"),
        "frequency_penalty": body.number("frequency_penalty"),
        "repetition_penalty": body.number("repetition_penalty"),
        "seed": body.integer("seed"),
        "stop": _stop(body),
        "logit_bias": _logit_bias(body),
        "logprobs": logprobs,
        "min_tokens": body.integer("min_tokens"),
        "ignore_eos": body.raw("ignore_eos"),
    }
    return SamplingFields(fields)


def _stream_options(body: JsonObject) -> bool:
    options = body.child("stream_options")
    return options is not None and options.flag("include_usage")


# -- chat completions -------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ChatRequest:
    """A Chat Completions request, read and checked.

    Attributes:
        model: The model the client asked for, or ``None``.
        conversation: The messages and tools.
        sampling: The sampling fields, without ``max_tokens``: the server builds them once
            it has capped ``max_tokens`` by the room the prompt leaves.
        max_tokens: The client's limit on generated tokens, or ``None``.
        stream: Whether to answer with server-sent events.
        include_usage: Whether a streamed answer ends with a usage chunk.
        tools: Whether the model may call the tools.
        parallel_tool_calls: Whether it may call several in one answer.
        output: The shape the answer must have (``response_format``, ``tool_choice``,
            ``strict``).
    """

    model: str | None
    conversation: Conversation
    sampling: SamplingFields
    max_tokens: int | None
    stream: bool
    include_usage: bool
    tools: bool
    parallel_tool_calls: bool
    output: OutputRequest = dataclasses.field(default_factory=OutputRequest)
    thinking: ThinkingRequest = dataclasses.field(default_factory=ThinkingRequest)
    chat_template_kwargs: Mapping[str, object] = dataclasses.field(
        default_factory=dict[str, object]
    )
    continue_final_message: bool = False
    named_tool_choice: bool = False


def parse_chat_request(body: JsonObject) -> ChatRequest:
    """Read a Chat Completions request.

    Raises:
        InvalidRequestError: For a malformed field, naming it.
        UnsupportedFeatureError: For a field that would change the answer and is not available.
    """
    apply_policy(body, "/v1/chat/completions")
    modalities = body.array("modalities")
    if modalities is not None and any(item != "text" for item in modalities):
        raise _unsupported("modalities", "only text output is available")
    response, schema = response_format(body.child("response_format"), "response_format")

    effort_raw = body.string("reasoning_effort")
    effort = None
    if effort_raw is not None:
        try:
            effort = parse_reasoning_effort(effort_raw, where="reasoning_effort")
        except ValueError as exc:
            raise InvalidRequestError(str(exc), context={"param": "reasoning_effort"}) from exc

    template_kwargs_obj = body.child("chat_template_kwargs")
    template_kwargs = dict(template_kwargs_obj.data) if template_kwargs_obj is not None else {}
    continue_final = body.flag("continue_final_message")

    builder = ConversationBuilder()
    messages = body.objects("messages")
    if not messages:
        raise InvalidRequestError(
            "messages: must be a non-empty array", context={"param": "messages"}
        )
    for message in messages:
        _chat_message(message, builder)
    conversation = builder.build()
    if continue_final and conversation.turns[-1].role != "assistant":
        raise InvalidRequestError(
            "continue_final_message: requires a final assistant message",
            context={"param": "continue_final_message"},
        )
    tools, choice, strict = _chat_tools(body, builder)
    conversation = builder.build()

    max_tokens = body.integer("max_completion_tokens")
    if max_tokens is None:
        max_tokens = body.integer("max_tokens")
    logprobs = body.integer("top_logprobs") if body.flag("logprobs") else None
    if body.flag("logprobs") and logprobs is None:
        logprobs = 0
    return ChatRequest(
        model=body.string("model"),
        conversation=conversation,
        sampling=_sampling(body, logprobs=logprobs),
        max_tokens=max_tokens,
        stream=body.flag("stream"),
        include_usage=_stream_options(body),
        tools=tools,
        parallel_tool_calls=body.flag("parallel_tool_calls", default=True),
        output=OutputRequest(response=response, schema=schema, tool_choice=choice, strict=strict),
        thinking=ThinkingRequest(effort=effort),
        chat_template_kwargs=template_kwargs,
        continue_final_message=continue_final,
        named_tool_choice=choice not in ("auto", "required"),
    )


def response_format(
    value: JsonObject | None, field: str
) -> tuple[ConstraintKind | None, Mapping[str, object] | None]:
    """A JSON answer an OpenAI request asks for: ``(kind, schema)``, or ``(None, None)`` for text.

    Reads Chat Completions' ``response_format`` and Responses' ``text.format``:
    ``{"type": "json_object"}``, or ``{"type": "json_schema", "schema": {...}}`` with the schema
    at the top (Responses) or under ``json_schema`` (Chat Completions). A schema is always
    enforced; ``strict`` changes nothing.

    Raises:
        InvalidRequestError: For an unknown type or a missing schema, naming the field.
    """
    if value is None:
        return None, None
    kind = value.string("type")
    if kind in (None, "text"):
        return None, None
    if kind == "json_object":
        return ConstraintKind.JSON_OBJECT, None
    if kind != "json_schema":
        raise InvalidRequestError(
            f"{field}.type: must be text, json_object or json_schema", context={"param": field}
        )
    holder = value.child("json_schema") or value
    schema = holder.child("schema")
    if schema is None:
        raise InvalidRequestError(
            f"{holder.where('schema')}: a json_schema answer needs a schema",
            context={"param": field},
        )
    return ConstraintKind.JSON_SCHEMA, schema.data


def _text_parts(message: JsonObject, key: str, *, allow_null: bool) -> str:
    """A message's content as text: a string, or an array of text parts."""
    content = message.raw(key)
    where = message.where(key)
    if content is None:
        if allow_null:
            return ""
        raise InvalidRequestError(f"{where}: is required", context={"param": where})
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        raise InvalidRequestError(
            f"{where}: must be a string or an array of parts", context={"param": where}
        )
    texts: list[str] = []
    for i, item in enumerate(cast("list[object]", content)):
        part = JsonObject.of(item, f"{where}[{i}]")
        kind = part.string("type")
        if kind in ("text", "input_text", "output_text"):
            texts.append(part.text("text"))
        elif kind == "refusal":
            texts.append(part.text("refusal"))
        elif kind in ("image_url", "input_image"):
            raise _unsupported(part.where("type"), "image input needs a vision model")
        else:
            raise _unsupported(part.where("type"), f"content parts of type {kind!r}")
    return PART_SEPARATOR.join(texts)


def _chat_message(message: JsonObject, builder: ConversationBuilder) -> None:
    role = message.text("role")
    match role:
        case "system" | "developer":
            builder.system(_text_parts(message, "content", allow_null=False))
        case "user":
            builder.user(_text_parts(message, "content", allow_null=False))
        case "assistant":
            calls = [_chat_tool_call(call) for call in message.objects("tool_calls")]
            reasoning = message.string("reasoning_content")
            builder.assistant(
                _text_parts(message, "content", allow_null=True), calls, reasoning=reasoning
            )
        case "tool":
            builder.tool_result(
                message.text("tool_call_id"), _text_parts(message, "content", allow_null=False)
            )
        case "function":
            raise _unsupported(message.where("role"), "function messages are deprecated; use tool")
        case _:
            raise InvalidRequestError(
                f"{message.where('role')}: unknown role {role!r}",
                context={"param": message.where("role")},
            )


def _chat_tool_call(call: JsonObject) -> ToolCall:
    function = call.child("function")
    if function is None:
        raise InvalidRequestError(f"{call.where('function')}: is required")
    arguments = function.string("arguments") or "{}"
    return ToolCall(
        id=call.text("id"),
        name=function.text("name"),
        arguments=parse_arguments(arguments, where=function.where("arguments")),
    )


def _chat_tools(body: JsonObject, builder: ConversationBuilder) -> tuple[bool, str, bool]:
    """Add the request's tools; return whether the model may call them, the choice and strict.

    The choice is ``"auto"``, ``"required"`` or the name of the one function to call.
    """
    tools = body.objects("tools")
    raw = body.raw("tool_choice")
    choice = "auto"
    if isinstance(raw, dict):
        picked = body.child("tool_choice")
        function = picked.child("function") if picked is not None else None
        if function is None or not function.string("name"):
            raise InvalidRequestError(
                "tool_choice: a named choice is {type: function, function: {name}}",
                context={"param": "tool_choice"},
            )
        choice = function.text("name")
    elif raw == "required":
        choice = "required"
    elif raw not in (None, "auto", "none"):
        raise InvalidRequestError("tool_choice: must be auto, none, required or a function")
    if not tools or raw == "none":
        if choice != "auto":
            raise InvalidRequestError(
                "tool_choice: forces a call but the request has no tools",
                context={"param": "tool_choice"},
            )
        return False, "auto", False
    strict = False
    for tool in tools:
        if tool.string("type") != "function":
            raise _unsupported(tool.where("type"), "only function tools are available")
        function = tool.child("function")
        if function is None:
            raise InvalidRequestError(f"{tool.where('function')}: is required")
        strict = strict or function.flag("strict")
        builder.tool(
            ToolSpec(
                name=function.text("name"),
                description=function.string("description"),
                parameters=(function.child("parameters") or JsonObject({})).data,
            )
        )
    return True, choice, strict


# -- completions ------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class CompletionRequest:
    """A Completions request, read and checked."""

    model: str | None
    prompt: str | tuple[int, ...]
    sampling: SamplingFields
    max_tokens: int | None
    stream: bool
    include_usage: bool


def parse_completion_request(body: JsonObject) -> CompletionRequest:
    """Read a Completions request.

    Raises:
        InvalidRequestError: For a malformed field, naming it.
        UnsupportedFeatureError: For ``echo``, ``suffix``, ``best_of > 1`` or several prompts.
    """
    apply_policy(body, "/v1/completions")
    if body.flag("echo"):
        raise _unsupported("echo", "returning the prompt is not available")
    best_of = body.integer("best_of")
    if best_of is not None and best_of > 1:
        raise _unsupported("best_of", "several candidates per request are not available")
    max_tokens = body.integer("max_tokens")
    return CompletionRequest(
        model=body.string("model"),
        prompt=_prompt(body),
        sampling=_sampling(body, logprobs=body.integer("logprobs")),
        max_tokens=16 if max_tokens is None else max_tokens,
        stream=body.flag("stream"),
        include_usage=_stream_options(body),
    )


def _prompt(body: JsonObject) -> str | tuple[int, ...]:
    value = body.raw("prompt")
    if isinstance(value, str) and value:
        return value
    if isinstance(value, list):
        items = cast("list[object]", value)
        if items and all(isinstance(i, int) and not isinstance(i, bool) for i in items):
            return tuple(cast("list[int]", items))
        if len(items) == 1 and isinstance(items[0], (str, list)):
            return _prompt(JsonObject({"prompt": items[0]}))
        if len(items) > 1:
            raise _unsupported("prompt", "several prompts in one request are not available")
    raise InvalidRequestError(
        "prompt: must be a non-empty string or an array of token ids", context={"param": "prompt"}
    )


# -- answers ----------------------------------------------------------------------------------


def chat_finish_reason(finish: FinishEvent) -> str:
    """The OpenAI finish reason of a normal end."""
    if finish.tool_calls:
        return "tool_calls"
    return "length" if finish.reason is FinishReason.LENGTH else "stop"


def chat_logprobs(entries: Sequence[TokenLogprob]) -> dict[str, object]:
    """The ``logprobs`` of a Chat Completions choice: one item per token, with alternatives."""

    def item(text: str, logprob: float) -> dict[str, object]:
        return {"token": text, "logprob": logprob, "bytes": list(text.encode())}

    content: list[dict[str, object]] = []
    for entry in entries:
        names = entry.top_decoded or ("",) * len(entry.top)
        body = item(entry.decoded or "", entry.logprob)
        body["top_logprobs"] = [
            item(name, logprob) for name, (_, logprob) in zip(names, entry.top, strict=True)
        ]
        content.append(body)
    return {"content": content, "refusal": None}


def completion_logprobs(entries: Sequence[TokenLogprob], offset: int = 0) -> dict[str, object]:
    """The legacy ``logprobs`` object of a Completions choice; ``offset`` is the text before."""
    tokens: list[str] = []
    offsets: list[int] = []
    for entry in entries:
        offsets.append(offset)
        text = entry.decoded or ""
        tokens.append(text)
        offset += len(text)
    return {
        "tokens": tokens,
        "token_logprobs": [entry.logprob for entry in entries],
        "top_logprobs": [
            dict(
                zip(
                    entry.top_decoded or ("",) * len(entry.top),
                    (lp for _, lp in entry.top),
                    strict=True,
                )
            )
            for entry in entries
        ],
        "text_offset": offsets,
    }


def openai_usage(usage: UsageEvent | None) -> dict[str, object] | None:
    """The usage object of an answer."""
    if usage is None:
        return None
    return {
        "prompt_tokens": usage.prompt_tokens,
        "completion_tokens": usage.completion_tokens,
        "total_tokens": usage.prompt_tokens + usage.completion_tokens,
        "prompt_tokens_details": {"cached_tokens": usage.cached_tokens},
        "completion_tokens_details": {"reasoning_tokens": usage.reasoning_tokens},
    }


def chat_message(answer: Collected) -> dict[str, object]:
    """The assistant message of a whole answer."""
    message: dict[str, object] = {"role": "assistant", "content": answer.text or None}
    if answer.reasoning:
        message["reasoning_content"] = answer.reasoning
    if answer.tool_calls:
        message["tool_calls"] = [
            {
                "id": call.id,
                "type": "function",
                "function": {"name": call.name, "arguments": call.arguments},
            }
            for call in answer.tool_calls
        ]
    return message
