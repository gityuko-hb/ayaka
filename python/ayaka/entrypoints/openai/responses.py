"""OpenAI Responses: requests to conversations, events to response objects and stream events.

The server keeps no conversation state at v1.0, so the request must carry the whole conversation
in ``input``; ``previous_response_id``, ``conversation`` and ``store: true`` are
refused with 400 saying so, and a request that omits ``store`` is answered as if it were false.

How the request maps:

* ``instructions`` is the system prompt; ``input`` is a string (one user message) or a list of
  items: messages, ``function_call`` and ``function_call_output``. ``reasoning`` items of earlier
  turns are dropped, as the chat templates of reasoning models do.
* ``tools`` of type ``function`` are offered to the model. The provider's own tools (web search,
  file search, code interpreter...) are left out with an info log line: only the provider can run
  them. ``strict: true`` is refused: it needs structured output.
* ``reasoning.effort`` is a hint: the model reasons as it was trained to.

A streamed answer::

    response.created, response.in_progress
    output_item.added (reasoning)     reasoning_text.delta* reasoning_text.done   output_item.done
    output_item.added (message)       content_part.added output_text.delta* output_text.done
                                      content_part.done                            output_item.done
    output_item.added (function_call) function_call_arguments.delta* .done        output_item.done
    response.completed | response.incomplete (max_output_tokens) | response.failed

Every event carries a ``sequence_number``. A keepalive is an SSE comment line.
"""

from __future__ import annotations

import dataclasses
import logging
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, cast

from aiohttp import web

from ayaka.core.errors import InvalidRequestError, UnsupportedFeatureError
from ayaka.core.types import FinishReason
from ayaka.entrypoints.conversation import (
    PART_SEPARATOR,
    ConversationBuilder,
    ErrorEvent,
    FinishEvent,
    OutputRequest,
    ReasoningDelta,
    TextDelta,
    ToolCall,
    ToolCallDelta,
    ToolCallEnd,
    ToolCallStart,
    ToolSpec,
    UsageEvent,
    parse_arguments,
)
from ayaka.entrypoints.failures import responses_code
from ayaka.entrypoints.http import read_body
from ayaka.entrypoints.openai.protocol import response_format
from ayaka.entrypoints.render.reasoning import ThinkingRequest, parse_reasoning_effort
from ayaka.entrypoints.render.selection import RenderRequest
from ayaka.entrypoints.serving import REQUEST_ID, SERVING
from ayaka.entrypoints.stream import KEEPALIVE_COMMENT, sse_json
from ayaka.entrypoints.wire.policy import apply_policy
from ayaka.entrypoints.wire.reader import JsonObject, SamplingFields

if TYPE_CHECKING:
    from ayaka.entrypoints.conversation import Collected, Conversation, OutputEvent

__all__ = [
    "ResponsesRequest",
    "ResponsesStreamEncoder",
    "create_response",
    "parse_responses_request",
    "response_body",
    "response_output",
]

_LOG = logging.getLogger("ayaka.serve")


def _unsupported(field: str, why: str) -> UnsupportedFeatureError:
    return UnsupportedFeatureError(
        f"{field}: {why}", context={"features": (field,), "param": field}
    )


def _invalid(field: str, why: str) -> InvalidRequestError:
    return InvalidRequestError(f"{field}: {why}", context={"param": field})


# -- reading ----------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ResponsesRequest:
    """A Responses request, read and checked.

    Attributes:
        model: The model the client asked for.
        conversation: The instructions, input items and tools.
        sampling: The sampling fields, without ``max_tokens``: the server builds them once
            it has capped ``max_tokens`` by the room the prompt leaves.
        max_tokens: ``max_output_tokens``, or ``None``.
        stream: Whether to answer with server-sent events.
        tools: Whether the model may call the tools.
        parallel_tool_calls: Whether it may call several in one answer.
        echo: Request fields echoed in the response object.
        output: The shape the answer must have (``text.format``, ``tool_choice``, ``strict``).
    """

    model: str | None
    conversation: Conversation
    sampling: SamplingFields
    max_tokens: int | None
    stream: bool
    tools: bool
    parallel_tool_calls: bool
    echo: dict[str, object]
    output: OutputRequest = dataclasses.field(default_factory=OutputRequest)
    thinking: ThinkingRequest = dataclasses.field(default_factory=ThinkingRequest)
    named_tool_choice: bool = False


def parse_responses_request(body: JsonObject) -> ResponsesRequest:
    """Read a Responses request.

    Raises:
        InvalidRequestError: For a malformed field, or one that needs conversation state.
        UnsupportedFeatureError: For a field that would change the answer and is not available.
    """
    apply_policy(body, "/v1/responses")
    if body.flag("store"):
        raise _invalid("store", "this server stores no responses; send store: false")
    if body.has("n") and body.raw("n") != 1:
        raise _unsupported("n", "the Responses API returns one answer; use Chat Completions")
    if body.string("truncation") not in (None, "disabled"):
        raise _unsupported("truncation", "only disabled is available")
    text = body.child("text")
    text_format = text.child("format") if text is not None else None
    response, schema = response_format(text_format, "text.format")
    reasoning = body.child("reasoning")
    thinking = _parse_responses_thinking(reasoning)

    builder = ConversationBuilder()
    instructions = body.string("instructions")
    if instructions:
        builder.system(instructions)
    _input(body, builder)
    tools, choice, constrained, strict = _tools(body, builder)
    max_tokens = body.integer("max_output_tokens")
    echo: dict[str, object] = {
        "instructions": instructions,
        "max_output_tokens": max_tokens,
        "parallel_tool_calls": body.flag("parallel_tool_calls", default=True),
        "reasoning": {
            "effort": reasoning.string("effort") if reasoning is not None else None,
            "summary": None,
        },
        "temperature": body.number("temperature"),
        "top_p": body.number("top_p"),
        "tool_choice": choice,
        "tools": body.array("tools") or [],
        "metadata": body.raw("metadata") or {},
    }
    return ResponsesRequest(
        model=body.string("model"),
        conversation=builder.build(),
        sampling=SamplingFields(
            {"temperature": body.number("temperature"), "top_p": body.number("top_p")}
        ),
        max_tokens=max_tokens,
        stream=body.flag("stream"),
        tools=tools,
        parallel_tool_calls=body.flag("parallel_tool_calls", default=True),
        echo=echo,
        output=OutputRequest(
            response=response, schema=schema, tool_choice=constrained, strict=strict
        ),
        thinking=thinking,
        named_tool_choice=constrained not in ("auto", "required"),
    )


def _parse_responses_thinking(reasoning: JsonObject | None) -> ThinkingRequest:
    if reasoning is None:
        return ThinkingRequest()
    effort_raw = reasoning.string("effort")
    effort = None
    if effort_raw is not None:
        try:
            effort = parse_reasoning_effort(effort_raw, where=reasoning.where("effort"))
        except ValueError as exc:
            raise _invalid(reasoning.where("effort"), str(exc)) from exc
    enabled = reasoning.flag("enabled") if reasoning.has("enabled") else None
    return ThinkingRequest(enabled=enabled, effort=effort)


def _input(body: JsonObject, builder: ConversationBuilder) -> None:
    value = body.raw("input")
    if isinstance(value, str):
        if not value:
            raise _invalid("input", "must not be empty")
        builder.user(value)
        return
    items = body.objects("input")
    if not items:
        raise _invalid("input", "must be a string or a non-empty array of items")
    for item in items:
        _item(item, builder)


def _item(item: JsonObject, builder: ConversationBuilder) -> None:
    kind = item.string("type") or ("message" if item.has("role") else None)
    match kind:
        case "message":
            role = item.text("role")
            text = _content(item)
            match role:
                case "user":
                    builder.user(text)
                case "system" | "developer":
                    builder.system(text)
                case "assistant":
                    builder.assistant(text)
                case _:
                    raise _invalid(item.where("role"), f"unknown role {role!r}")
        case "function_call":
            arguments = item.string("arguments") or "{}"
            builder.tool_call(
                ToolCall(
                    item.text("call_id"),
                    item.text("name"),
                    parse_arguments(arguments, where=item.where("arguments")),
                )
            )
        case "function_call_output":
            output = item.raw("output")
            text = output if isinstance(output, str) else _parts(output, item.where("output"))
            builder.tool_result(item.text("call_id"), text)
        case "reasoning":
            return
        case "item_reference":
            raise _invalid(item.where("type"), "this server keeps no items; send the item itself")
        case _:
            raise _unsupported(item.where("type"), f"input items of type {kind!r}")


def _content(item: JsonObject) -> str:
    content = item.raw("content")
    if isinstance(content, str):
        return content
    return _parts(content, item.where("content"))


def _parts(value: object, where: str) -> str:
    if not isinstance(value, list):
        raise _invalid(where, "must be a string or an array of content parts")
    texts: list[str] = []
    for i, entry in enumerate(cast("list[object]", value)):
        part = JsonObject.of(entry, f"{where}[{i}]")
        kind = part.string("type")
        if kind in ("input_text", "output_text", "text"):
            texts.append(part.text("text"))
        elif kind == "refusal":
            texts.append(part.text("refusal"))
        elif kind in ("input_image", "input_file", "input_audio"):
            raise _unsupported(part.where("type"), f"{kind} needs a model that reads it")
        else:
            raise _unsupported(part.where("type"), f"content parts of type {kind!r}")
    return PART_SEPARATOR.join(texts)


def _tools(body: JsonObject, builder: ConversationBuilder) -> tuple[bool, object, str, bool]:
    """Add the function tools; return ``(offered, echoed choice, constrained choice, strict)``.

    The constrained choice is ``"auto"``, ``"required"`` or the one function's name. Only an
    explicit ``strict: true`` constrains the arguments to the schema: a tool's schema may use
    something the grammar compiler refuses, and a default must not fail the request.
    """
    choice: object = body.raw("tool_choice") or "auto"
    echoed = choice
    constrained = "auto"
    if isinstance(choice, dict):
        picked = body.child("tool_choice")
        if picked is None or picked.string("type") != "function" or not picked.string("name"):
            raise _invalid("tool_choice", "a named choice is {type: function, name}")
        constrained = picked.text("name")
    elif choice == "required":
        constrained = "required"
    elif choice not in ("auto", "none"):
        raise _invalid("tool_choice", "must be auto, none, required or a tool")
    tools = body.objects("tools")
    if choice == "none" or not tools:
        if constrained != "auto":
            raise _invalid("tool_choice", "forces a call but the request has no function tools")
        return False, echoed, "auto", False
    offered = False
    strict = False
    for tool in tools:
        if tool.string("type") != "function":
            _LOG.info("leaving out tool of type %r: only its provider can run it", tool.raw("type"))
            continue
        strict = strict or tool.flag("strict")
        parameters = tool.child("parameters") or JsonObject({})
        builder.tool(ToolSpec(tool.text("name"), tool.string("description"), parameters.data))
        offered = True
    return offered, echoed, constrained, strict


# -- answers ----------------------------------------------------------------------------------

type _Status = Literal["in_progress", "completed", "incomplete", "failed"]


def _usage(usage: UsageEvent | None) -> dict[str, object] | None:
    if usage is None:
        return None
    return {
        "input_tokens": usage.prompt_tokens,
        # cache_write_tokens: the prompt tokens registered in the prefix cache (the field is
        # required by the current openai client library).
        "input_tokens_details": {
            "cached_tokens": usage.cached_tokens,
            "cache_write_tokens": usage.cache_written_tokens,
        },
        "output_tokens": usage.completion_tokens,
        "output_tokens_details": {"reasoning_tokens": usage.reasoning_tokens},
        "total_tokens": usage.prompt_tokens + usage.completion_tokens,
    }


@dataclass(slots=True)
class _Snapshot:
    """What the response object says, built up as the answer goes."""

    id: str
    created: int
    model: str
    echo: dict[str, object]
    status: _Status = "in_progress"
    output: list[dict[str, object]] | None = None
    usage: UsageEvent | None = None
    error: ErrorEvent | None = None
    incomplete: bool = False

    def render(self) -> dict[str, object]:
        return {
            "id": self.id,
            "object": "response",
            "created_at": self.created,
            "status": self.status,
            "model": self.model,
            "output": list(self.output or []),
            "error": None
            if self.error is None
            else {"code": responses_code(self.error.code), "message": self.error.message},
            "incomplete_details": {"reason": "max_output_tokens"} if self.incomplete else None,
            "previous_response_id": None,
            "store": False,
            "text": {"format": {"type": "text"}},
            "truncation": "disabled",
            "usage": _usage(self.usage),
            **self.echo,
        }


def _finish_status(finish: FinishEvent) -> tuple[_Status, bool]:
    if finish.reason is FinishReason.LENGTH and not finish.tool_calls:
        return "incomplete", True
    return "completed", False


def _reasoning_item(item_id: str, text: str) -> dict[str, object]:
    return {
        "id": item_id,
        "type": "reasoning",
        "summary": [],
        "content": [{"type": "reasoning_text", "text": text}],
    }


def _text_part(text: str) -> dict[str, object]:
    return {"type": "output_text", "text": text, "annotations": [], "logprobs": []}


def _message_item(item_id: str, text: str, status: str) -> dict[str, object]:
    content = [_text_part(text)] if status == "completed" else []
    return {
        "id": item_id,
        "type": "message",
        "status": status,
        "role": "assistant",
        "content": content,
    }


def _call_item(
    item_id: str, call_id: str, name: str, arguments: str, status: str
) -> dict[str, object]:
    return {
        "id": item_id,
        "type": "function_call",
        "status": status,
        "call_id": call_id,
        "name": name,
        "arguments": arguments,
    }


def response_output(answer: Collected, public_id: str) -> list[dict[str, object]]:
    """The output items of a whole answer: reasoning, then the message, then the calls."""
    items: list[dict[str, object]] = []
    if answer.reasoning:
        items.append(_reasoning_item(f"rs_{public_id}_0", answer.reasoning))
    if answer.text:
        items.append(_message_item(f"msg_{public_id}_0", answer.text, "completed"))
    for i, call in enumerate(answer.tool_calls):
        items.append(
            _call_item(f"fc_{public_id}_{i}", call.id, call.name, call.arguments, "completed")
        )
    return items


def response_body(snapshot: _Snapshot, answer: Collected, public_id: str) -> dict[str, object]:
    """The ``response`` object of a whole answer, from the snapshot it started as."""
    finish = answer.finish
    assert finish is not None  # noqa: S101  # collecting raises on an error
    snapshot.status, snapshot.incomplete = _finish_status(finish)
    snapshot.output, snapshot.usage = response_output(answer, public_id), answer.usage
    return snapshot.render()


class ResponsesStreamEncoder:
    """Responses events for one answer.

    Args:
        snapshot: The response object as it starts.
        public_id: The hex id used to name items.
    """

    def __init__(self, snapshot: _Snapshot, public_id: str) -> None:
        self._snapshot = snapshot
        self._public = public_id
        self._sequence = 0
        self._output: list[dict[str, object]] = []
        self._open: Literal["reasoning", "message"] | None = None
        self._texts: list[str] = []  # the open item's text, joined when it closes
        self._item_id = ""
        self._calls: dict[int, tuple[int, dict[str, object]]] = {}  # index -> output index, item
        self._arguments: dict[int, list[str]] = {}  # index -> the open call's argument pieces

    def _event(self, kind: str, **fields: object) -> bytes:
        payload: dict[str, object] = {"type": kind, "sequence_number": self._sequence, **fields}
        self._sequence += 1
        return sse_json(payload, event=kind)

    def _response(self) -> dict[str, object]:
        for index, (_, item) in self._calls.items():  # an open call shows what it has so far
            item["arguments"] = "".join(self._arguments[index])
        self._snapshot.output = self._output
        return self._snapshot.render()

    def start(self) -> bytes:
        return self._event("response.created", response=self._response()) + self._event(
            "response.in_progress", response=self._response()
        )

    def encode(self, event: OutputEvent) -> bytes:
        match event:
            case TextDelta(text):
                return self._delta("message", text)
            case ReasoningDelta(text):
                return self._delta("reasoning", text)
            case ToolCallStart(index, call_id, name):
                frames = self._close()
                position = len(self._output)
                item = _call_item(f"fc_{self._public}_{index}", call_id, name, "", "in_progress")
                self._calls[index] = (position, item)
                self._arguments[index] = []
                self._output.append(item)
                return frames + self._event(
                    "response.output_item.added", output_index=position, item=dict(item)
                )
            case ToolCallDelta(index, arguments):
                position, item = self._calls[index]
                self._arguments[index].append(arguments)
                return self._event(
                    "response.function_call_arguments.delta",
                    item_id=item["id"],
                    output_index=position,
                    delta=arguments,
                )
            case ToolCallEnd(index):
                position, item = self._calls.pop(index)
                item["arguments"] = "".join(self._arguments.pop(index))
                item["status"] = "completed"
                return self._event(
                    "response.function_call_arguments.done",
                    item_id=item["id"],
                    output_index=position,
                    name=item["name"],
                    arguments=item["arguments"],
                ) + self._event("response.output_item.done", output_index=position, item=item)
            case UsageEvent():
                self._snapshot.usage = event
                return b""
            case FinishEvent():
                frames = self._close()
                status, incomplete = _finish_status(event)
                self._snapshot.status, self._snapshot.incomplete = status, incomplete
                kind = "response.completed" if status == "completed" else "response.incomplete"
                return frames + self._event(kind, response=self._response())
            case ErrorEvent():
                self._snapshot.status, self._snapshot.error = "failed", event
                return self._event("response.failed", response=self._response())
            case _:
                return b""

    def keepalive(self) -> bytes:
        return KEEPALIVE_COMMENT

    def end(self) -> bytes:
        return b""

    # -- items -------------------------------------------------------------------------------

    def _delta(self, kind: Literal["reasoning", "message"], text: str) -> bytes:
        frames = b""
        position = len(self._output) - (1 if self._open == kind else 0)
        if self._open != kind:
            frames += self._close()
            position = len(self._output)
            count = sum(1 for item in self._output if item["type"] == kind)
            prefix = "rs" if kind == "reasoning" else "msg"
            self._item_id = f"{prefix}_{self._public}_{count}"
            self._open = kind
            self._texts.clear()
            item: dict[str, object] = (
                {"id": self._item_id, "type": "reasoning", "summary": [], "content": []}
                if kind == "reasoning"
                else _message_item(self._item_id, "", "in_progress")
            )
            self._output.append(item)
            frames += self._event("response.output_item.added", output_index=position, item=item)
            if kind == "message":
                frames += self._event(
                    "response.content_part.added",
                    item_id=self._item_id,
                    output_index=position,
                    content_index=0,
                    part=_text_part(""),
                )
        self._texts.append(text)
        if kind == "reasoning":
            return frames + self._event(
                "response.reasoning_text.delta",
                item_id=self._item_id,
                output_index=position,
                content_index=0,
                delta=text,
            )
        return frames + self._event(
            "response.output_text.delta",
            item_id=self._item_id,
            output_index=position,
            content_index=0,
            delta=text,
            logprobs=[],
        )

    def _close(self) -> bytes:
        if self._open is None:
            return b""
        position = len(self._output) - 1
        text, item_id = "".join(self._texts), self._item_id
        if self._open == "reasoning":
            item = _reasoning_item(item_id, text)
            frames = self._event(
                "response.reasoning_text.done",
                item_id=item_id,
                output_index=position,
                content_index=0,
                text=text,
            )
        else:
            item = _message_item(item_id, text, "completed")
            frames = self._event(
                "response.output_text.done",
                item_id=item_id,
                output_index=position,
                content_index=0,
                text=text,
                logprobs=[],
            ) + self._event(
                "response.content_part.done",
                item_id=item_id,
                output_index=position,
                content_index=0,
                part=_text_part(text),
            )
        self._output[position] = item
        self._open = None
        return frames + self._event("response.output_item.done", output_index=position, item=item)


# -- the handler ------------------------------------------------------------------------------


async def create_response(request: web.Request) -> web.StreamResponse:
    """``POST /v1/responses``."""
    serving = request.app[SERVING]
    responses = parse_responses_request(await read_body(request))
    serving.check_model(responses.model)
    conversation = serving.normalized(
        responses.conversation, request.headers, request_id=request[REQUEST_ID]
    )
    rendered = await serving.render_chat(
        RenderRequest(
            conversation=conversation,
            thinking=responses.thinking,
            named_tool_choice=responses.named_tool_choice,
        ),
        max_tokens=responses.max_tokens,
    )
    prompt = rendered.token_ids
    sampling = responses.sampling.build(serving.output_tokens(len(prompt), responses.max_tokens))
    public_id = request[REQUEST_ID]
    generation = await serving.start(
        prompt,
        sampling,
        public_id=public_id,
        tools=responses.tools,
        parallel_tool_calls=responses.parallel_tool_calls,
        structured=serving.constraint(
            responses.output,
            conversation,
            tools=responses.tools,
            parallel_tool_calls=responses.parallel_tool_calls,
            rendered=rendered,
        ),
        rendered=rendered,
    )
    snapshot = _Snapshot(f"resp_{public_id}", int(time.time()), serving.model_name, responses.echo)
    try:
        if responses.stream:
            encoder = ResponsesStreamEncoder(snapshot, public_id)
            return await serving.stream(request, generation, encoder)
        answer = await serving.collect(generation)
        return web.json_response(response_body(snapshot, answer, public_id))
    finally:
        serving.finish(generation)
