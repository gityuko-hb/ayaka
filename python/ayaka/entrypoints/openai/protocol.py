"""The OpenAI wire formats: reading Chat Completions and Completions requests, and the shapes
of their answers and errors.

Reading follows one policy for every field (``OUT-9``):

* An optional field this server does not know is ignored and named in a debug log line; clients
  add fields faster than servers learn them.
* A known field that would change the answer and that Ayaka cannot honour is refused with 400,
  naming it. Nothing is accepted and then quietly done differently.
* Sampling fields go to ``SamplingParams`` as they are; the engine refuses the ones it cannot
  serve yet (``prompt_logprobs``) with ``UnsupportedFeatureError``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final, cast

from ayaka.core.errors import InvalidRequestError, UnsupportedFeatureError
from ayaka.core.types import FinishReason
from ayaka.entrypoints.conversation import (
    PART_SEPARATOR,
    ConversationBuilder,
    ToolCall,
    ToolSpec,
    parse_arguments,
)
from ayaka.entrypoints.http import JsonObject, sampling_params

if TYPE_CHECKING:
    from collections.abc import Sequence

    from ayaka.core.outputs import TokenLogprob
    from ayaka.core.params import SamplingParams
    from ayaka.entrypoints.conversation import (
        Collected,
        Conversation,
        FinishEvent,
        UsageEvent,
    )
    from ayaka.entrypoints.http import ApiFailure

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

_ERROR_TYPES: Final = {
    "invalid_request": "invalid_request_error",
    "authentication": "authentication_error",
    "not_found": "not_found_error",
    "too_large": "invalid_request_error",
    "overloaded": "rate_limit_error",
    "timeout": "timeout_error",
    "unavailable": "service_unavailable_error",
    "internal": "server_error",
}


def openai_error(failure: ApiFailure) -> dict[str, object]:
    """An error body in OpenAI's shape."""
    return {
        "error": {
            "message": failure.message,
            "type": _ERROR_TYPES[failure.kind],
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


def _sampling(body: JsonObject, *, max_tokens: int | None, logprobs: int | None) -> SamplingParams:
    return sampling_params(
        n=body.integer("n"),
        temperature=body.number("temperature"),
        top_p=body.number("top_p"),
        top_k=body.integer("top_k"),
        min_p=body.number("min_p"),
        presence_penalty=body.number("presence_penalty"),
        frequency_penalty=body.number("frequency_penalty"),
        repetition_penalty=body.number("repetition_penalty"),
        seed=body.integer("seed"),
        stop=_stop(body),
        logit_bias=_logit_bias(body),
        logprobs=logprobs,
        min_tokens=body.integer("min_tokens"),
        ignore_eos=body.raw("ignore_eos"),
        # The server caps max_tokens by the room the prompt leaves, once it is tokenized.
        max_tokens=max_tokens,
    )


_SAMPLING_KEYS: Final = frozenset(
    {
        "n",
        "temperature",
        "top_p",
        "top_k",
        "min_p",
        "presence_penalty",
        "frequency_penalty",
        "repetition_penalty",
        "seed",
        "stop",
        "logit_bias",
        "min_tokens",
        "ignore_eos",
        "max_tokens",
        "logprobs",
        "stream",
        "stream_options",
        "model",
        "user",
        "metadata",
        "store",
        "service_tier",
    }
)


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
        sampling: The sampling fields; ``max_tokens`` is the client's, not yet capped.
        max_tokens: The client's limit on generated tokens, or ``None``.
        stream: Whether to answer with server-sent events.
        include_usage: Whether a streamed answer ends with a usage chunk.
        tools: Whether the model may call the tools.
        parallel_tool_calls: Whether it may call several in one answer.
    """

    model: str | None
    conversation: Conversation
    sampling: SamplingParams
    max_tokens: int | None
    stream: bool
    include_usage: bool
    tools: bool
    parallel_tool_calls: bool


_CHAT_KEYS: Final = _SAMPLING_KEYS | {
    "messages",
    "tools",
    "tool_choice",
    "parallel_tool_calls",
    "response_format",
    "max_completion_tokens",
    "top_logprobs",
    "functions",
    "function_call",
    "audio",
    "modalities",
    "prediction",
    "web_search_options",
    "reasoning_effort",
}


def parse_chat_request(body: JsonObject) -> ChatRequest:
    """Read a Chat Completions request.

    Raises:
        InvalidRequestError: For a malformed field, naming it.
        UnsupportedFeatureError: For a field that would change the answer and is not available.
    """
    unknown = body.unknown(_CHAT_KEYS)
    if unknown:
        _LOG.debug("chat completions: ignoring unknown fields %s", unknown)
    for field in ("functions", "function_call"):
        if body.has(field):
            raise _unsupported(field, "the deprecated function fields are not served; use tools")
    for field in ("audio", "prediction", "web_search_options"):
        if body.has(field):
            raise _unsupported(field, "not available on this server")
    modalities = body.array("modalities")
    if modalities is not None and any(item != "text" for item in modalities):
        raise _unsupported("modalities", "only text output is available")
    _check_response_format(body.child("response_format"), "response_format")
    if body.has("reasoning_effort"):
        _LOG.debug("chat completions: reasoning_effort is a hint; the model reasons as trained")

    builder = ConversationBuilder()
    messages = body.objects("messages")
    if not messages:
        raise InvalidRequestError(
            "messages: must be a non-empty array", context={"param": "messages"}
        )
    for message in messages:
        _chat_message(message, builder)
    tools = _chat_tools(body, builder)

    max_tokens = body.integer("max_completion_tokens")
    if max_tokens is None:
        max_tokens = body.integer("max_tokens")
    logprobs = body.integer("top_logprobs") if body.flag("logprobs") else None
    if body.flag("logprobs") and logprobs is None:
        logprobs = 0
    return ChatRequest(
        model=body.string("model"),
        conversation=builder.build(),
        sampling=_sampling(body, max_tokens=max_tokens, logprobs=logprobs),
        max_tokens=max_tokens,
        stream=body.flag("stream"),
        include_usage=_stream_options(body),
        tools=tools,
        parallel_tool_calls=body.flag("parallel_tool_calls", default=True),
    )


def _check_response_format(value: JsonObject | None, field: str) -> None:
    if value is None:
        return
    kind = value.string("type")
    if kind not in (None, "text"):
        raise _unsupported(
            field, f"{kind} needs structured output, which this server does not have yet"
        )


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
            builder.assistant(_text_parts(message, "content", allow_null=True), calls)
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


def _chat_tools(body: JsonObject, builder: ConversationBuilder) -> bool:
    """Add the request's tools; return whether the model may call them."""
    tools = body.objects("tools")
    choice = body.raw("tool_choice")
    if isinstance(choice, dict) or choice == "required":
        raise _unsupported("tool_choice", "forcing a tool call needs structured output")
    if choice not in (None, "auto", "none"):
        raise InvalidRequestError("tool_choice: must be auto, none, required or a function")
    if not tools or choice == "none":
        return False
    for tool in tools:
        if tool.string("type") != "function":
            raise _unsupported(tool.where("type"), "only function tools are available")
        function = tool.child("function")
        if function is None:
            raise InvalidRequestError(f"{tool.where('function')}: is required")
        if function.flag("strict"):
            raise _unsupported(function.where("strict"), "strict schemas need structured output")
        builder.tool(
            ToolSpec(
                name=function.text("name"),
                description=function.string("description"),
                parameters=(function.child("parameters") or JsonObject({})).data,
            )
        )
    return True


# -- completions ------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class CompletionRequest:
    """A Completions request, read and checked."""

    model: str | None
    prompt: str | tuple[int, ...]
    sampling: SamplingParams
    max_tokens: int | None
    stream: bool
    include_usage: bool


_COMPLETION_KEYS: Final = _SAMPLING_KEYS | {"prompt", "echo", "suffix", "best_of"}


def parse_completion_request(body: JsonObject) -> CompletionRequest:
    """Read a Completions request.

    Raises:
        InvalidRequestError: For a malformed field, naming it.
        UnsupportedFeatureError: For ``echo``, ``suffix``, ``best_of > 1`` or several prompts.
    """
    unknown = body.unknown(_COMPLETION_KEYS)
    if unknown:
        _LOG.debug("completions: ignoring unknown fields %s", unknown)
    if body.flag("echo"):
        raise _unsupported("echo", "returning the prompt is not available")
    if body.has("suffix"):
        raise _unsupported("suffix", "fill-in-the-middle is not available")
    best_of = body.integer("best_of")
    if best_of is not None and best_of > 1:
        raise _unsupported("best_of", "several candidates per request are not available")
    max_tokens = body.integer("max_tokens")
    return CompletionRequest(
        model=body.string("model"),
        prompt=_prompt(body),
        sampling=_sampling(
            body,
            max_tokens=16 if max_tokens is None else max_tokens,
            logprobs=body.integer("logprobs"),
        ),
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
