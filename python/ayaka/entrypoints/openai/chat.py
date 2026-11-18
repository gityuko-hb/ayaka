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

With ``n > 1`` every chunk names its choice: each choice gets its role chunk, its deltas and its
own finish chunk, and the usage chunk (with ``include_usage``) counts the prompt once and the
tokens of every choice.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Final

from aiohttp import web

from ayaka.entrypoints.conversation import (
    ChoiceEvent,
    ErrorEvent,
    FinishEvent,
    ReasoningDelta,
    TextDelta,
    TokenLogprobs,
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
    chat_logprobs,
    chat_message,
    completion_logprobs,
    openai_error,
    openai_usage,
    parse_chat_request,
    parse_completion_request,
)
from ayaka.entrypoints.stream import KEEPALIVE_COMMENT, sse_json

if TYPE_CHECKING:
    from ayaka.core.outputs import TokenLogprob
    from ayaka.entrypoints.conversation import Collected, OutputEvent
    from ayaka.entrypoints.http import Generation, Serving

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
    """Chat Completions chunks for one answer of ``choices`` choices."""

    def __init__(self, meta: _Meta, *, include_usage: bool, choices: int = 1) -> None:
        self._meta = meta
        self._include_usage = include_usage
        self._choices = choices
        self._usage: UsageEvent | None = None

    def _chunk(
        self,
        delta: dict[str, object],
        finish: str | None = None,
        index: int = 0,
        logprobs: dict[str, object] | None = None,
    ) -> bytes:
        return sse_json(
            {
                "id": self._meta.id,
                "object": "chat.completion.chunk",
                "created": self._meta.created,
                "model": self._meta.model,
                "system_fingerprint": None,
                "choices": [
                    {"index": index, "delta": delta, "logprobs": logprobs, "finish_reason": finish}
                ],
            }
        )

    def _usage_chunk(self) -> bytes:
        return sse_json(
            {
                "id": self._meta.id,
                "object": "chat.completion.chunk",
                "created": self._meta.created,
                "model": self._meta.model,
                "choices": [],
                "usage": openai_usage(self._usage),
            }
        )

    def start(self) -> bytes:
        return b"".join(
            self._chunk({"role": "assistant", "content": ""}, index=index)
            for index in range(self._choices)
        )

    def encode(self, event: OutputEvent) -> bytes:
        match event:
            case ChoiceEvent(index, inner):
                return self._choice(inner, index)
            case UsageEvent():
                self._usage = event
                return b""
            case FinishEvent() if self._choices > 1:
                # Every choice already sent its own finish chunk.
                return self._usage_chunk() if self._include_usage else b""
            case FinishEvent():
                frames = self._chunk({}, chat_finish_reason(event))
                return frames + (self._usage_chunk() if self._include_usage else b"")
            case ErrorEvent():
                return _error_chunk(event, self._meta.id)
            case _:
                return self._choice(event, 0)

    def _choice(self, event: OutputEvent, index: int) -> bytes:
        match event:
            case TextDelta(text):
                return self._chunk({"content": text}, index=index)
            case ReasoningDelta(text):
                return self._chunk({"reasoning_content": text}, index=index)
            case ToolCallStart(call, call_id, name):
                piece = {
                    "index": call,
                    "id": call_id,
                    "type": "function",
                    "function": {"name": name, "arguments": ""},
                }
                return self._chunk({"tool_calls": [piece]}, index=index)
            case ToolCallDelta(call, arguments):
                piece = {"index": call, "function": {"arguments": arguments}}
                return self._chunk({"tool_calls": [piece]}, index=index)
            case FinishEvent():
                return self._chunk({}, chat_finish_reason(event), index=index)
            case TokenLogprobs(entries):
                return self._chunk({}, index=index, logprobs=chat_logprobs(entries))
            case _:
                return b""

    def keepalive(self) -> bytes:
        return KEEPALIVE_COMMENT

    def end(self) -> bytes:
        return _DONE


class CompletionStreamEncoder:
    """Completions chunks for one answer; reasoning and tool calls do not exist here."""

    def __init__(self, meta: _Meta, *, include_usage: bool, choices: int = 1) -> None:
        self._meta = meta
        self._include_usage = include_usage
        self._choices = choices
        self._usage: UsageEvent | None = None
        self._offsets: dict[int, int] = {}

    def _chunk(
        self,
        text: str,
        finish: str | None = None,
        index: int = 0,
        logprobs: dict[str, object] | None = None,
        **more: object,
    ) -> bytes:
        self._offsets[index] = self._offsets.get(index, 0) + len(text)
        return sse_json(
            {
                "id": self._meta.id,
                "object": "text_completion",
                "created": self._meta.created,
                "model": self._meta.model,
                "choices": [
                    {"index": index, "text": text, "logprobs": logprobs, "finish_reason": finish}
                ],
                **more,
            }
        )

    def _logprobs(self, entries: tuple[TokenLogprob, ...], index: int) -> bytes:
        return self._chunk(
            "", index=index, logprobs=completion_logprobs(entries, self._offsets.get(index, 0))
        )

    def start(self) -> bytes:
        return b""

    def encode(self, event: OutputEvent) -> bytes:
        match event:
            case ChoiceEvent(index, TextDelta(text) | ReasoningDelta(text)):
                return self._chunk(text, index=index)
            case ChoiceEvent(index, FinishEvent() as finish):
                return self._chunk("", chat_finish_reason(finish), index=index)
            case ChoiceEvent(index, TokenLogprobs(entries)):
                return self._logprobs(entries, index)
            case TextDelta(text) | ReasoningDelta(text):
                return self._chunk(text)
            case TokenLogprobs(entries):
                return self._logprobs(entries, 0)
            case UsageEvent():
                self._usage = event
                return b""
            case FinishEvent() if self._choices > 1:
                if not self._include_usage:
                    return b""
                return sse_json(
                    {
                        "id": self._meta.id,
                        "object": "text_completion",
                        "created": self._meta.created,
                        "model": self._meta.model,
                        "choices": [],
                        "usage": openai_usage(self._usage),
                    }
                )
            case FinishEvent():
                if self._include_usage:
                    return self._chunk(
                        "", chat_finish_reason(event), 0, usage=openai_usage(self._usage)
                    )
                return self._chunk("", chat_finish_reason(event))
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
            encoder = ChatStreamEncoder(meta, include_usage=chat.include_usage, choices=sampling.n)
            return await serving.stream(request, generation, encoder)
        answers = await _answers(serving, generation, sampling.n)
        return web.json_response(
            {
                "id": meta.id,
                "object": "chat.completion",
                "created": meta.created,
                "model": meta.model,
                "system_fingerprint": None,
                "choices": [
                    {
                        "index": index,
                        "message": chat_message(answer),
                        "logprobs": chat_logprobs(answer.logprobs) if answer.logprobs else None,
                        "finish_reason": _finish(answer),
                    }
                    for index, answer in enumerate(answers)
                ],
                "usage": openai_usage(answers[0].usage),
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
            encoder = CompletionStreamEncoder(
                meta, include_usage=completion.include_usage, choices=sampling.n
            )
            return await serving.stream(request, generation, encoder)
        answers = await _answers(serving, generation, sampling.n)
        return web.json_response(
            {
                "id": meta.id,
                "object": "text_completion",
                "created": meta.created,
                "model": meta.model,
                "choices": [
                    {
                        "index": index,
                        "text": answer.reasoning + answer.text,
                        "logprobs": (
                            completion_logprobs(answer.logprobs) if answer.logprobs else None
                        ),
                        "finish_reason": _finish(answer),
                    }
                    for index, answer in enumerate(answers)
                ],
                "usage": openai_usage(answers[0].usage),
            }
        )
    finally:
        serving.finish(generation)


async def _answers(serving: Serving, generation: Generation, count: int) -> list[Collected]:
    """The whole answer of every choice; raises if the request failed."""
    if count == 1:
        return [await serving.collect(generation)]
    return await serving.collect_choices(generation, count)


def _finish(answer: Collected) -> str:
    finish = answer.finish
    assert finish is not None  # noqa: S101  # collecting raises on an error
    return chat_finish_reason(finish)
