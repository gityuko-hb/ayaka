"""Anthropic Messages and ``count_tokens``: requests to conversations, events to answers.

A streamed answer, per ``OUT-B1``::

    message_start                                   with the prompt's input_tokens
    ping
    content_block_start (thinking) / content_block_delta (thinking_delta)* / content_block_stop
    content_block_start (text)     / content_block_delta (text_delta)*     / content_block_stop
    content_block_start (tool_use) / content_block_delta (input_json_delta) / content_block_stop
    message_delta                                   stop_reason, stop_sequence, output_tokens
    message_stop

A keepalive is a ``ping`` event; an error after the stream started is one ``error`` event, which
ends the stream.

Anthropic's ``tool_use.input`` is a JSON object, so a call is held until it is complete and then
sent whole: a valid call becomes a ``tool_use`` block; a call whose arguments are not one JSON
object cannot be represented and is sent as a text block with what the model wrote (``OUT-11``),
and counted in the ``ayaka_tool_calls{valid="false"}`` metric.
"""

from __future__ import annotations

import json
import time
from dataclasses import replace
from typing import TYPE_CHECKING, Final, Literal, cast

from aiohttp import web

from ayaka.entrypoints.anthropic.protocol import (
    anthropic_error,
    anthropic_usage,
    parse_count_tokens_request,
    parse_messages_request,
    stop_reason,
)
from ayaka.entrypoints.conversation import (
    ErrorEvent,
    FinishEvent,
    ReasoningDelta,
    TextDelta,
    ToolCallDelta,
    ToolCallEnd,
    ToolCallStart,
    UsageEvent,
)
from ayaka.entrypoints.http import REQUEST_ID, SERVING, failure_for_event, read_body
from ayaka.entrypoints.stream import sse_json

if TYPE_CHECKING:
    from ayaka.entrypoints.conversation import Collected, CollectedToolCall, OutputEvent

__all__ = [
    "MessagesStreamEncoder",
    "count_tokens",
    "create_message",
    "message_content",
    "models_listing",
]

_PING: Final = sse_json({"type": "ping"}, event="ping")

type _Block = Literal["text", "thinking"]


def _invalid_call_text(name: str, arguments: str) -> str:
    """What the model wrote for a call whose arguments are not one JSON object."""
    return f'{{"name": {json.dumps(name, ensure_ascii=False)}, "arguments": {arguments}}}'


class MessagesStreamEncoder:
    """Messages events for one answer.

    Args:
        message_id: The ``msg_...`` id.
        model: The model name to report.
        prompt_tokens: The prompt length, reported in ``message_start``.
    """

    def __init__(self, message_id: str, model: str, *, prompt_tokens: int) -> None:
        self._id = message_id
        self._model = model
        self._prompt_tokens = prompt_tokens
        self._index = -1  # index of the last block opened
        self._open: _Block | None = None
        self._calls: dict[int, tuple[str, str, list[str]]] = {}  # index -> id, name, pieces
        self._usage: UsageEvent | None = None

    def start(self) -> bytes:
        message: dict[str, object] = {
            "id": self._id,
            "type": "message",
            "role": "assistant",
            "model": self._model,
            "content": [],
            "stop_reason": None,
            "stop_sequence": None,
            "usage": anthropic_usage(None, prompt_tokens=self._prompt_tokens),
        }
        start = sse_json({"type": "message_start", "message": message}, event="message_start")
        return start + _PING

    def encode(self, event: OutputEvent) -> bytes:
        match event:
            case TextDelta(text):
                return self._delta("text", {"type": "text_delta", "text": text})
            case ReasoningDelta(text):
                return self._delta("thinking", {"type": "thinking_delta", "thinking": text})
            case ToolCallStart(index, call_id, name):
                self._calls[index] = (call_id, name, [])
                return b""
            case ToolCallDelta(index, arguments):
                self._calls[index][2].append(arguments)
                return b""
            case ToolCallEnd(index, valid):
                return self._tool_call(index, valid=valid)
            case UsageEvent():
                self._usage = event
                return b""
            case FinishEvent():
                usage = anthropic_usage(self._usage, prompt_tokens=self._prompt_tokens)
                delta = {"stop_reason": stop_reason(event), "stop_sequence": event.stop_sequence}
                return (
                    self._close()
                    + sse_json(
                        {
                            "type": "message_delta",
                            "delta": delta,
                            "usage": {"output_tokens": usage["output_tokens"]},
                        },
                        event="message_delta",
                    )
                    + sse_json({"type": "message_stop"}, event="message_stop")
                )
            case ErrorEvent():
                failure = failure_for_event(event, self._id)
                return sse_json(anthropic_error(failure), event="error")
            case _:
                return b""

    def keepalive(self) -> bytes:
        return _PING

    def end(self) -> bytes:
        return b""

    # -- blocks ------------------------------------------------------------------------------

    def _delta(self, kind: _Block, delta: dict[str, object]) -> bytes:
        frames = b""
        if self._open != kind:
            frames += self._close()
            frames += self._start(
                {"type": "text", "text": ""}
                if kind == "text"
                else {"type": "thinking", "thinking": "", "signature": ""}
            )
            self._open = kind
        return frames + sse_json(
            {"type": "content_block_delta", "index": self._index, "delta": delta},
            event="content_block_delta",
        )

    def _start(self, block: dict[str, object]) -> bytes:
        self._index += 1
        return sse_json(
            {"type": "content_block_start", "index": self._index, "content_block": block},
            event="content_block_start",
        )

    def _close(self) -> bytes:
        if self._open is None:
            return b""
        self._open = None
        return self._stop()

    def _stop(self) -> bytes:
        return sse_json(
            {"type": "content_block_stop", "index": self._index}, event="content_block_stop"
        )

    def _tool_call(self, index: int, *, valid: bool) -> bytes:
        call_id, name, pieces = self._calls.pop(index)
        arguments = "".join(pieces)
        if not valid:
            return self._delta(
                "text", {"type": "text_delta", "text": _invalid_call_text(name, arguments)}
            )
        frames = self._close()
        frames += self._start({"type": "tool_use", "id": call_id, "name": name, "input": {}})
        frames += sse_json(
            {
                "type": "content_block_delta",
                "index": self._index,
                "delta": {"type": "input_json_delta", "partial_json": arguments},
            },
            event="content_block_delta",
        )
        return frames + self._stop()


def message_content(answer: Collected) -> list[dict[str, object]]:
    """The content blocks of a whole answer: thinking, then text, then the calls."""
    blocks: list[dict[str, object]] = []
    if answer.reasoning:
        blocks.append({"type": "thinking", "thinking": answer.reasoning, "signature": ""})
    texts = [answer.text] if answer.text else []
    calls: list[dict[str, object]] = []
    for call in answer.tool_calls:
        if call.valid:
            calls.append(_tool_use(call))
        else:
            texts.append(_invalid_call_text(call.name, call.arguments))
    if texts:
        blocks.append({"type": "text", "text": "".join(texts)})
    return blocks + calls


def _tool_use(call: CollectedToolCall) -> dict[str, object]:
    arguments = cast("dict[str, object]", json.loads(call.arguments))
    return {"type": "tool_use", "id": call.id, "name": call.name, "input": arguments}


# -- handlers ---------------------------------------------------------------------------------


async def create_message(request: web.Request) -> web.StreamResponse:
    """``POST /v1/messages``."""
    serving = request.app[SERVING]
    message = parse_messages_request(await read_body(request))
    serving.check_model(message.model)
    prompt = await serving.tokenize_chat(message.conversation)
    sampling = replace(
        message.sampling, max_tokens=serving.output_tokens(len(prompt), message.max_tokens)
    )
    public_id = request[REQUEST_ID]
    generation = await serving.start(
        prompt,
        sampling,
        public_id=public_id,
        tools=message.tools,
        parallel_tool_calls=message.parallel_tool_calls,
    )
    message_id = f"msg_{public_id}"
    try:
        if message.stream:
            encoder = MessagesStreamEncoder(
                message_id, serving.model_name, prompt_tokens=len(prompt)
            )
            return await serving.stream(request, generation, encoder)
        answer = await serving.collect(generation)
        assert answer.finish is not None  # noqa: S101  # collect raises on an error
        return web.json_response(
            {
                "id": message_id,
                "type": "message",
                "role": "assistant",
                "model": serving.model_name,
                "content": message_content(answer),
                "stop_reason": stop_reason(answer.finish),
                "stop_sequence": answer.finish.stop_sequence,
                "usage": anthropic_usage(answer.usage, prompt_tokens=len(prompt)),
            }
        )
    finally:
        serving.finish(generation)


async def count_tokens(request: web.Request) -> web.Response:
    """``POST /v1/messages/count_tokens``: the prompt's length, without running the model."""
    serving = request.app[SERVING]
    model, conversation = parse_count_tokens_request(await read_body(request))
    serving.check_model(model)
    prompt = await serving.tokenize_chat(conversation)
    return web.json_response({"input_tokens": len(prompt)})


def models_listing(names: tuple[str, ...], created: int) -> dict[str, object]:
    """``GET /v1/models`` in Anthropic's shape, for clients that send ``anthropic-version``."""
    stamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(created))
    data = [
        {"type": "model", "id": name, "display_name": name, "created_at": stamp} for name in names
    ]
    return {
        "data": data,
        "has_more": False,
        "first_id": names[0] if names else None,
        "last_id": names[-1] if names else None,
    }
