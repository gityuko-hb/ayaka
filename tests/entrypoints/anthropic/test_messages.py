"""Anthropic Messages: reading requests, the exact event sequence, tool use and count_tokens.

Over HTTP, plain text comes from the tiny GPT-2 and is checked against ``greedy_reference``; tool
calls and reasoning come from ``FakeEngine``, since the tiny model does not write them.
"""

from __future__ import annotations

import asyncio
import json
from typing import TYPE_CHECKING, Any

import pytest

from ayaka.core.errors import InvalidRequestError, UnsupportedFeatureError
from ayaka.core.types import FinishReason
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
from tokenizer_fakes import FakeTokenizer

pytest.importorskip("torch")
pytest.importorskip("aiohttp")

from ayaka.entrypoints.anthropic.messages import MessagesStreamEncoder
from ayaka.entrypoints.anthropic.protocol import parse_messages_request
from ayaka.entrypoints.parsers.output import OutputFormat
from ayaka.entrypoints.wire.reader import JsonObject
from serving_support import parse_sse, read_sse, script, serve, write_long_gpt2
from tiny_model import TinyCheckpoint, greedy_reference, write_tiny_gpt2

if TYPE_CHECKING:
    from pathlib import Path

    from fake_engine import FakeEngine

pytestmark = pytest.mark.unit

_MESSAGES = "/v1/messages"
_FAKE = FakeTokenizer()
_TOOLS = OutputFormat.resolve(reasoning="think", tool_calls="hermes")
_WEATHER = {
    "name": "get_weather",
    "description": "Weather for a city",
    "input_schema": {"type": "object", "properties": {"city": {"type": "string"}}},
}


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory: pytest.TempPathFactory) -> TinyCheckpoint:
    return write_tiny_gpt2(tmp_path_factory.mktemp("tiny-gpt2"))


def _parse(**body: Any) -> Any:
    return parse_messages_request(JsonObject({"model": "tiny", "max_tokens": 16, **body}))


class TestReading:
    def test_blocks_tool_turns_and_cache_marks_become_one_conversation(self) -> None:
        message = _parse(
            system=[
                {"type": "text", "text": "be brief", "cache_control": {"type": "ephemeral"}},
                {"type": "text", "text": "and kind"},
            ],
            messages=[
                {"role": "user", "content": "weather?"},
                {
                    "role": "assistant",
                    "content": [
                        {"type": "thinking", "thinking": "hm", "signature": "x"},
                        {"type": "text", "text": "checking"},
                        {
                            "type": "tool_use",
                            "id": "t1",
                            "name": "get_weather",
                            "input": {"city": "Hue"},
                        },
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "t1",
                            "content": [{"type": "text", "text": "rain"}],
                        },
                        {"type": "text", "text": "thanks"},
                    ],
                },
            ],
            tools=[_WEATHER, {"type": "web_search_20250305", "name": "web_search"}],
            tool_choice={"type": "auto", "disable_parallel_tool_use": True},
            thinking={"type": "enabled", "budget_tokens": 1024},
            max_tokens=2048,
        )

        conversation = message.conversation
        assert conversation.system == "be brief\nand kind"
        assert [t.role for t in conversation.turns] == ["user", "assistant", "tool", "user"]
        assert conversation.turns[1].text == "checking"
        assert conversation.turns[1].tool_calls[0].arguments == {"city": "Hue"}
        assert conversation.turns[2].text == "rain"
        assert conversation.turns[2].name == "get_weather"
        assert [tool.name for tool in conversation.tools] == ["get_weather"]
        assert message.tools
        assert not message.parallel_tool_calls

    @pytest.mark.parametrize(
        ("body", "error", "match"),
        [
            ({"max_tokens": None}, InvalidRequestError, "max_tokens: is required"),
            (
                {"messages": [{"role": "user", "content": [{"type": "image", "source": {}}]}]},
                InvalidRequestError,
                r"source\.type",
            ),
            (
                {
                    "messages": [
                        {"role": "user", "content": "x"},
                        {"role": "assistant", "content": "Sure"},
                    ]
                },
                UnsupportedFeatureError,
                "prefill",
            ),
            (
                {"messages": [{"role": "user", "content": "x"}], "tool_choice": {"type": "any"}},
                InvalidRequestError,
                "no tools",
            ),
            (
                {"output_format": {"type": "regex", "pattern": "a+"}},
                InvalidRequestError,
                "json_schema",
            ),
            (
                {
                    "messages": [{"role": "user", "content": "x"}],
                    "thinking": {"type": "enabled", "budget_tokens": 100},
                },
                InvalidRequestError,
                "budget_tokens",
            ),
            ({"messages": [{"role": "tool", "content": "x"}]}, InvalidRequestError, "role"),
        ],
    )
    def test_bad_or_unavailable_fields_name_themselves(
        self, body: dict[str, Any], error: type[Exception], match: str
    ) -> None:
        full = {"messages": [{"role": "user", "content": "x"}], **body}
        with pytest.raises(error, match=match):
            parse_messages_request(JsonObject({"model": "tiny", "max_tokens": 16, **full} | body))


class TestFixedEvents:
    def test_a_fixed_event_sequence_gives_these_exact_events(self) -> None:
        encoder = MessagesStreamEncoder("msg_1", "m", prompt_tokens=11)
        events = [
            ReasoningDelta("hmm"),
            ReasoningDelta(" ok"),
            TextDelta("Hi"),
            ToolCallStart(0, "call_1_0", "f"),
            ToolCallDelta(0, '{"a":'),
            ToolCallDelta(0, "1}"),
            ToolCallEnd(0, valid=True),
            UsageEvent(11, 5),
            FinishEvent(FinishReason.STOP, tool_calls=True),
        ]

        body = encoder.start() + b"".join(encoder.encode(e) for e in events) + encoder.end()

        frames = [(name, json.loads(data)) for name, data in parse_sse(body.decode())]
        assert [name for name, _ in frames] == [
            "message_start",
            "ping",
            "content_block_start",
            "content_block_delta",
            "content_block_delta",
            "content_block_stop",
            "content_block_start",
            "content_block_delta",
            "content_block_stop",
            "content_block_start",
            "content_block_delta",
            "content_block_stop",
            "message_delta",
            "message_stop",
        ]
        assert all(data["type"] == name for name, data in frames)
        start = frames[0][1]["message"]
        assert start["usage"]["input_tokens"] == 11
        assert [f[1]["content_block"]["type"] for f in frames if f[0] == "content_block_start"] == [
            "thinking",
            "text",
            "tool_use",
        ]
        assert frames[3][1]["delta"] == {"type": "thinking_delta", "thinking": "hmm"}
        assert frames[7][1]["delta"] == {"type": "text_delta", "text": "Hi"}
        assert frames[10][1]["delta"] == {"type": "input_json_delta", "partial_json": '{"a":1}'}
        assert [f[1]["index"] for f in frames if "index" in f[1]] == [0, 0, 0, 0, 1, 1, 1, 2, 2, 2]
        assert frames[-2][1] == {
            "type": "message_delta",
            "delta": {"stop_reason": "tool_use", "stop_sequence": None},
            # Cumulative: the whole usage, not only the output tokens.
            "usage": {
                "input_tokens": 11,
                "output_tokens": 5,
                "cache_creation_input_tokens": 0,
                "cache_read_input_tokens": 0,
            },
        }

    def test_an_error_ends_the_stream_with_one_error_event(self) -> None:
        encoder = MessagesStreamEncoder("msg_1", "m", prompt_tokens=1)

        body = encoder.encode(ErrorEvent("cancelled", "the request was cancelled")) + encoder.end()

        [(name, data)] = parse_sse(body.decode())
        assert name == "error"
        assert json.loads(data) == {
            "type": "error",
            "error": {"type": "overloaded_error", "message": "the request was cancelled"},
            "request_id": "1",
        }
        assert parse_sse(encoder.keepalive().decode()) == [("ping", '{"type":"ping"}')]

    def test_a_call_with_broken_arguments_becomes_text(self) -> None:
        encoder = MessagesStreamEncoder("msg_1", "m", prompt_tokens=1)
        events = [
            ToolCallStart(0, "c", "f"),
            ToolCallDelta(0, "{oops"),
            ToolCallEnd(0, valid=False),
        ]

        body = b"".join(encoder.encode(e) for e in events)

        frames = [json.loads(data) for _, data in parse_sse(body.decode())]
        assert frames[0]["content_block"]["type"] == "text"
        assert frames[1]["delta"]["text"] == '{"name": "f", "arguments": {oops}'


def _chat_prompt(text: str) -> list[int]:
    return list(f"<|im_start|>user\n{text}<|im_end|>\n<|im_start|>assistant\n".encode())


def _prompt_total(usage: dict[str, int]) -> int:
    """Anthropic splits the prompt three ways; they add up to the whole prompt."""
    return (
        usage["input_tokens"]
        + usage["cache_read_input_tokens"]
        + usage["cache_creation_input_tokens"]
    )


class TestOverHttp:
    def test_an_answer_and_a_stream_say_the_reference_text(
        self, checkpoint: TinyCheckpoint
    ) -> None:
        tokens = greedy_reference(checkpoint.state, checkpoint.config, _chat_prompt("hi"), 6)
        expected = _FAKE.decode(tokens)
        request = {
            "model": "tiny",
            "max_tokens": 6,
            "temperature": 0,
            "messages": [{"role": "user", "content": "hi"}],
        }

        async def main() -> None:
            async with serve(checkpoint.directory) as (client, _, _):
                answer = await (await client.post(_MESSAGES, json=request)).json()
                assert answer["type"] == "message"
                assert answer["id"].startswith("msg_")
                assert answer["content"] == [{"type": "text", "text": expected}]
                assert answer["stop_reason"] == "max_tokens"
                usage = answer["usage"]
                prompt = len(_chat_prompt("hi"))
                # The first request computes the prompt and registers its whole blocks.
                assert usage["cache_read_input_tokens"] == 0
                assert usage["cache_creation_input_tokens"] > 0
                assert _prompt_total(usage) == prompt
                assert usage["output_tokens"] == 6
                frames = await read_sse(
                    await client.post(_MESSAGES, json={**request, "stream": True})
                )
                text = "".join(
                    data["delta"]["text"] for name, data in frames if name == "content_block_delta"
                )
                assert text == expected
                assert frames[-1] == ("message_stop", {"type": "message_stop"})
                # The same prompt again reads those blocks; message_delta says it, cumulatively.
                [delta] = [data["usage"] for name, data in frames if name == "message_delta"]
                # At most what the first one wrote: the last token is always computed again.
                assert 0 < delta["cache_read_input_tokens"] <= usage["cache_creation_input_tokens"]
                assert delta["cache_creation_input_tokens"] == 0
                assert _prompt_total(delta) == prompt
                assert delta["output_tokens"] == 6
                count = await (
                    await client.post(
                        "/v1/messages/count_tokens",
                        json={"model": "tiny", "messages": request["messages"]},
                    )
                ).json()
                assert count == {"input_tokens": len(_chat_prompt("hi"))}

        asyncio.run(main())

    def test_errors_and_the_model_list_take_anthropic_shapes(
        self, checkpoint: TinyCheckpoint
    ) -> None:
        async def main() -> None:
            async with serve(checkpoint.directory) as (client, _, _):
                missing = await client.post(_MESSAGES, json={"messages": []})
                assert missing.status == 400
                body = await missing.json()
                assert body["type"] == "error"
                assert body["error"]["type"] == "invalid_request_error"
                listing = await (
                    await client.get("/v1/models", headers={"anthropic-version": "2023-06-01"})
                ).json()
                assert listing["data"][0]["type"] == "model"
                assert listing["data"][0]["id"] == "tiny"
                assert listing["has_more"] is False

        asyncio.run(main())

    def test_a_streamed_tool_call_is_one_tool_use_block(self, checkpoint: TinyCheckpoint) -> None:
        answer = script(
            "<think>",
            "need it",
            "</think>",
            "<tool_call>",
            '{"name": "get_weather", "arguments": {"city": "Hue"}}',
            "</tool_call>",
        )

        async def main() -> None:
            async with serve(checkpoint.directory, output_format=_TOOLS, scripts=[answer]) as (
                client,
                _,
                _,
            ):
                request = {
                    "model": "tiny",
                    "max_tokens": 64,
                    "stream": True,
                    "messages": [{"role": "user", "content": "weather?"}],
                    "tools": [_WEATHER],
                }
                frames = await read_sse(await client.post(_MESSAGES, json=request))
                starts = [d["content_block"] for n, d in frames if n == "content_block_start"]
                assert [block["type"] for block in starts] == ["thinking", "tool_use"]
                assert starts[1]["name"] == "get_weather"
                partial = "".join(
                    d["delta"].get("partial_json", "")
                    for n, d in frames
                    if n == "content_block_delta"
                )
                assert json.loads(partial) == {"city": "Hue"}
                delta = next(d for n, d in frames if n == "message_delta")
                assert delta["delta"]["stop_reason"] == "tool_use"

        asyncio.run(main())


_HISTORY_CHAT = {
    "model": "tiny",
    "max_tokens": 4,
    "messages": [
        {"role": "system", "content": "be brief"},
        {"role": "user", "content": "weather in Hue?"},
        {
            "role": "assistant",
            "content": "checking",
            "tool_calls": [
                {
                    "id": "t1",
                    "type": "function",
                    "function": {"name": "get_weather", "arguments": '{"city": "Hue"}'},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "t1", "content": "rain"},
        {"role": "user", "content": "thanks"},
    ],
    "tools": [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Weather for a city",
                "parameters": _WEATHER["input_schema"],
            },
        }
    ],
}

_HISTORY_MESSAGES = {
    "model": "tiny",
    "max_tokens": 4,
    "system": "be brief",
    "messages": [
        {"role": "user", "content": "weather in Hue?"},
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "checking"},
                {"type": "tool_use", "id": "t1", "name": "get_weather", "input": {"city": "Hue"}},
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": "t1", "content": "rain"},
                {"type": "text", "text": "thanks"},
            ],
        },
    ],
    "tools": [_WEATHER],
}


def test_one_conversation_in_two_protocols_gives_one_prompt(tmp_path: Path) -> None:
    directory = write_long_gpt2(tmp_path)

    async def main() -> None:
        async with serve(directory, output_format=_TOOLS, scripts=[], num_blocks=None) as (
            client,
            _,
            llm,
        ):
            assert (await client.post("/v1/chat/completions", json=_HISTORY_CHAT)).status == 200
            assert (await client.post(_MESSAGES, json=_HISTORY_MESSAGES)).status == 200
            count = await (
                await client.post("/v1/messages/count_tokens", json=_HISTORY_MESSAGES)
            ).json()
            scripted: FakeEngine = llm
            chat, messages = (request.prompt for request in scripted.requests)
            assert isinstance(chat, tuple)
            assert chat == messages
            assert count["input_tokens"] == len(chat)
            rendered = bytes(chat)
            assert b"tools\nget_weather" in rendered  # the tools reached the template
            assert b'<tool_call>get_weather{"city":"Hue"}</tool_call>' in rendered
            assert b"[t1:get_weather]" in rendered

    asyncio.run(main())
