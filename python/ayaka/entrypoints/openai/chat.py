"""OpenAI Chat Completions and Completions: requests to conversations, events to answers.

A streamed chat answer, per ``OUT-B1``::

    {"choices": [{"delta": {"role": "assistant", "content": ""}}]}         first chunk
    {"choices": [{"delta": {"reasoning_content": "..."}}]}                  reasoning
    {"choices": [{"delta": {"content": "..."}}]}                            text
    {"choices": [{"delta": {"tool_calls": [{"index", "id", "function": {"name", ...}}]}}]}
    {"choices": [{"delta": {"tool_calls": [{"index", "function": {"arguments": ...}}]}}]}
    {"choices": [{"delta": {}, "finish_reason": "stop" | "length" | "tool_calls"}]}
    {"choices": [], "usage": {...}}                                         with include_usage
    [DONE]

An error after the stream started is one ``{"error": {...}}`` chunk, then ``[DONE]``. A keepalive
is an SSE comment line.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Final

from aiohttp import web

from ayaka.entrypoints.conversation import (
    ErrorEvent,
    FinishEvent,
    ReasoningDelta,
    TextDelta,
    ToolCallDelta,
    ToolCallStart,
    UsageEvent,
)
from ayaka.entrypoints.http import (
    REQUEST_ID,
    SERVING,
    failure_for_event,
    read_body,
)
from ayaka.entrypoints.openai.protocol import (
    chat_finish_reason,
    chat_message,
    openai_error,
    openai_usage,
    parse_chat_request,
    parse_completion_request,
)
from ayaka.entrypoints.stream import KEEPALIVE_COMMENT, sse_json

if TYPE_CHECKING:
    from ayaka.entrypoints.conversation import OutputEvent

__all__ = [
    "ChatStreamEncoder",
    "CompletionStreamEncoder",
    "create_chat_completion",
    "create_completion",
]

_DONE: Final = b"data: [DONE]\n\n"


@dataclass(frozen=True, slots=True)
class _Meta:
    id: str
    created: int
    model: str


class ChatStreamEncoder:
    """Chat Completions chunks for one answer."""

    def __init__(self, meta: _Meta, *, include_usage: bool) -> None:
        self._meta = meta
        self._include_usage = include_usage
        self._usage: UsageEvent | None = None

    def _chunk(self, delta: dict[str, object], finish: str | None = None) -> bytes:
        return sse_json(
            {
                "id": self._meta.id,
                "object": "chat.completion.chunk",
                "created": self._meta.created,
                "model": self._meta.model,
                "system_fingerprint": None,
                "choices": [
                    {"index": 0, "delta": delta, "logprobs": None, "finish_reason": finish}
                ],
            }
        )

    def start(self) -> bytes:
        return self._chunk({"role": "assistant", "content": ""})

    def encode(self, event: OutputEvent) -> bytes:
        match event:
            case TextDelta(text):
                return self._chunk({"content": text})
            case ReasoningDelta(text):
                return self._chunk({"reasoning_content": text})
            case ToolCallStart(index, call_id, name):
                call = {
                    "index": index,
                    "id": call_id,
                    "type": "function",
                    "function": {"name": name, "arguments": ""},
                }
                return self._chunk({"tool_calls": [call]})
            case ToolCallDelta(index, arguments):
                piece = {"index": index, "function": {"arguments": arguments}}
                return self._chunk({"tool_calls": [piece]})
            case UsageEvent():
                self._usage = event
                return b""
            case FinishEvent():
                frames = self._chunk({}, chat_finish_reason(event))
                if self._include_usage:
                    frames += sse_json(
                        {
                            "id": self._meta.id,
                            "object": "chat.completion.chunk",
                            "created": self._meta.created,
                            "model": self._meta.model,
                            "choices": [],
                            "usage": openai_usage(self._usage),
                        }
                    )
                return frames
            case ErrorEvent():
                return _error_chunk(event, self._meta.id)
            case _:
                return b""

    def keepalive(self) -> bytes:
        return KEEPALIVE_COMMENT

    def end(self) -> bytes:
        return _DONE


class CompletionStreamEncoder:
    """Completions chunks for one answer; reasoning and tool calls do not exist here."""

    def __init__(self, meta: _Meta, *, include_usage: bool) -> None:
        self._meta = meta
        self._include_usage = include_usage
        self._usage: UsageEvent | None = None

    def _chunk(self, text: str, finish: str | None = None, **more: object) -> bytes:
        return sse_json(
            {
                "id": self._meta.id,
                "object": "text_completion",
                "created": self._meta.created,
                "model": self._meta.model,
                "choices": [{"index": 0, "text": text, "logprobs": None, "finish_reason": finish}],
                **more,
            }
        )

    def start(self) -> bytes:
        return b""

    def encode(self, event: OutputEvent) -> bytes:
        match event:
            case TextDelta(text) | ReasoningDelta(text):
                return self._chunk(text)
            case UsageEvent():
                self._usage = event
                return b""
            case FinishEvent():
                more = {"usage": openai_usage(self._usage)} if self._include_usage else {}
                return self._chunk("", chat_finish_reason(event), **more)
            case ErrorEvent():
                return _error_chunk(event, self._meta.id)
            case _:
                return b""

    def keepalive(self) -> bytes:
        return KEEPALIVE_COMMENT

    def end(self) -> bytes:
        return _DONE


def _error_chunk(event: ErrorEvent, request_id: str) -> bytes:
    return sse_json(openai_error(failure_for_event(event, request_id)))


# -- handlers ---------------------------------------------------------------------------------


async def create_chat_completion(request: web.Request) -> web.StreamResponse:
    """``POST /v1/chat/completions``."""
    serving = request.app[SERVING]
    chat = parse_chat_request(await read_body(request))
    serving.check_model(chat.model)
    prompt = await serving.tokenize_chat(chat.conversation)
    sampling = replace(
        chat.sampling, max_tokens=serving.output_tokens(len(prompt), chat.max_tokens)
    )
    public_id = request[REQUEST_ID]
    generation = await serving.start(
        prompt,
        sampling,
        public_id=public_id,
        tools=chat.tools,
        parallel_tool_calls=chat.parallel_tool_calls,
    )
    meta = _Meta(f"chatcmpl-{public_id}", int(time.time()), serving.model_name)
    try:
        if chat.stream:
            encoder = ChatStreamEncoder(meta, include_usage=chat.include_usage)
            return await serving.stream(request, generation, encoder)
        answer = await serving.collect(generation)
        assert answer.finish is not None  # noqa: S101  # collect raises on an error
        return web.json_response(
            {
                "id": meta.id,
                "object": "chat.completion",
                "created": meta.created,
                "model": meta.model,
                "system_fingerprint": None,
                "choices": [
                    {
                        "index": 0,
                        "message": chat_message(answer),
                        "logprobs": None,
                        "finish_reason": chat_finish_reason(answer.finish),
                    }
                ],
                "usage": openai_usage(answer.usage),
            }
        )
    finally:
        serving.finish(generation)


async def create_completion(request: web.Request) -> web.StreamResponse:
    """``POST /v1/completions``."""
    serving = request.app[SERVING]
    completion = parse_completion_request(await read_body(request))
    serving.check_model(completion.model)
    prompt = await serving.tokenize_text(completion.prompt)
    sampling = replace(
        completion.sampling,
        max_tokens=serving.output_tokens(len(prompt), completion.max_tokens),
    )
    public_id = request[REQUEST_ID]
    generation = await serving.start(prompt, sampling, public_id=public_id, tools=False)
    meta = _Meta(f"cmpl-{public_id}", int(time.time()), serving.model_name)
    try:
        if completion.stream:
            encoder = CompletionStreamEncoder(meta, include_usage=completion.include_usage)
            return await serving.stream(request, generation, encoder)
        answer = await serving.collect(generation)
        assert answer.finish is not None  # noqa: S101  # collect raises on an error
        return web.json_response(
            {
                "id": meta.id,
                "object": "text_completion",
                "created": meta.created,
                "model": meta.model,
                "choices": [
                    {
                        "index": 0,
                        "text": answer.reasoning + answer.text,
                        "logprobs": None,
                        "finish_reason": chat_finish_reason(answer.finish),
                    }
                ],
                "usage": openai_usage(answer.usage),
            }
        )
    finally:
        serving.finish(generation)
