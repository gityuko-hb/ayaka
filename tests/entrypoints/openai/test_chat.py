"""OpenAI Chat Completions: reading requests, the exact chunk sequence, tool calls and reasoning.

Tool calls and reasoning need a model that writes them, which the tiny GPT-2 does not, so these
tests answer with ``FakeEngine``: real HTTP, real tokenizing and parsing, scripted engine output.
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

pytest.importorskip("torch")
pytest.importorskip("aiohttp")

from ayaka.entrypoints.openai.chat import ChatStreamEncoder, _Meta  # pyright: ignore[reportPrivateUsage]
from ayaka.entrypoints.openai.protocol import parse_chat_request
from ayaka.entrypoints.parsers.output import OutputFormat
from ayaka.entrypoints.wire.reader import JsonObject
from serving_support import parse_sse, read_sse, script, serve
from tiny_model import TinyCheckpoint, write_tiny_gpt2

if TYPE_CHECKING:
    from ayaka.core.outputs import RequestOutput
    from fake_engine import FakeEngine

pytestmark = pytest.mark.unit

_CHAT = "/v1/chat/completions"
_TOOLS = OutputFormat.resolve(reasoning="think", tool_calls="hermes")
_WEATHER = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Weather for a city",
        "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
    },
}


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory: pytest.TempPathFactory) -> TinyCheckpoint:
    return write_tiny_gpt2(tmp_path_factory.mktemp("tiny-gpt2"))


def _parse(body: dict[str, Any]) -> Any:
    return parse_chat_request(JsonObject(body))


class TestReading:
    def test_roles_tool_turns_and_arguments_become_one_conversation(self) -> None:
        chat = _parse(
            {
                "messages": [
                    {"role": "developer", "content": "be brief"},
                    {
                        "role": "user",
                        "content": [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}],
                    },
                    {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "c1",
                                "type": "function",
                                "function": {"name": "get_weather", "arguments": '{"city":"Hue"}'},
                            }
                        ],
                    },
                    {"role": "tool", "tool_call_id": "c1", "content": "rain"},
                ],
                "tools": [_WEATHER],
                "max_completion_tokens": 9,
                "unknown_field": 1,
            }
        )

        conversation = chat.conversation
        assert conversation.system == "be brief"
        assert [t.role for t in conversation.turns] == ["user", "assistant", "tool"]
        assert conversation.turns[0].text == "a\nb"
        assert conversation.turns[1].tool_calls[0].arguments == {"city": "Hue"}
        assert conversation.turns[2].name == "get_weather"
        assert conversation.tools[0].description == "Weather for a city"
        assert chat.tools
        assert chat.max_tokens == 9
        assert chat.parallel_tool_calls

    def test_tool_choice_none_keeps_the_tools_in_the_prompt_and_offers_no_call(self) -> None:
        chat = _parse(
            {
                "messages": [{"role": "user", "content": "x"}],
                "tools": [_WEATHER],
                "tool_choice": "none",
            }
        )

        assert not chat.tools
        assert [tool.name for tool in chat.conversation.tools] == ["get_weather"]

    @pytest.mark.parametrize(
        ("body", "error", "match"),
        [
            (
                {"messages": [{"role": "wizard", "content": "x"}]},
                InvalidRequestError,
                "unknown role",
            ),
            (
                {
                    "messages": [
                        {
                            "role": "assistant",
                            "tool_calls": [
                                {"id": "c", "function": {"name": "f", "arguments": "{"}}
                            ],
                        }
                    ]
                },
                InvalidRequestError,
                r"messages\[0\]\.tool_calls\[0\]\.function\.arguments",
            ),
            (
                {"messages": [{"role": "user", "content": "x"}], "functions": [{}]},
                UnsupportedFeatureError,
                "deprecated",
            ),
            (
                {"messages": [{"role": "user", "content": "x"}], "tool_choice": "required"},
                InvalidRequestError,
                "no tools",
            ),
            (
                {
                    "messages": [{"role": "user", "content": "x"}],
                    "response_format": {"type": "json_schema", "json_schema": {"name": "a"}},
                },
                InvalidRequestError,
                "needs a schema",
            ),
            (
                {"messages": [{"role": "user", "content": "x"}], "stop": [1]},
                InvalidRequestError,
                "stop",
            ),
        ],
    )
    def test_bad_or_unavailable_fields_name_themselves(
        self, body: dict[str, Any], error: type[Exception], match: str
    ) -> None:
        with pytest.raises(error, match=match):
            _parse(body)


class TestFixedEvents:
    def test_a_fixed_event_sequence_gives_these_exact_chunks(self) -> None:
        encoder = ChatStreamEncoder(_Meta("chatcmpl-1", 7, "m"), include_usage=True)
        events = [
            ReasoningDelta("hmm"),
            TextDelta("Hi"),
            ToolCallStart(0, "call_1_0", "f"),
            ToolCallDelta(0, '{"a":1}'),
            ToolCallEnd(0, valid=True),
            UsageEvent(3, 5, reasoning_tokens=1),
            FinishEvent(FinishReason.STOP, tool_calls=True),
        ]

        body = encoder.start() + b"".join(encoder.encode(e) for e in events) + encoder.end()

        frames = parse_sse(body.decode())
        assert frames[-1] == (None, "[DONE]")
        chunks = [json.loads(data) for _, data in frames[:-1]]
        deltas = [c["choices"][0]["delta"] if c["choices"] else None for c in chunks]
        assert deltas == [
            {"role": "assistant", "content": ""},
            {"reasoning_content": "hmm"},
            {"content": "Hi"},
            {
                "tool_calls": [
                    {
                        "index": 0,
                        "id": "call_1_0",
                        "type": "function",
                        "function": {"name": "f", "arguments": ""},
                    }
                ]
            },
            {"tool_calls": [{"index": 0, "function": {"arguments": '{"a":1}'}}]},
            {},
            None,
        ]
        assert chunks[-2]["choices"][0]["finish_reason"] == "tool_calls"
        assert chunks[-1]["usage"] == {
            "prompt_tokens": 3,
            "completion_tokens": 5,
            "total_tokens": 8,
            "prompt_tokens_details": {"cached_tokens": 0},
            "completion_tokens_details": {"reasoning_tokens": 1},
        }
        assert all(
            c["object"] == "chat.completion.chunk" and c["id"] == "chatcmpl-1" for c in chunks
        )

    def test_an_error_is_one_error_chunk_then_done(self) -> None:
        encoder = ChatStreamEncoder(_Meta("chatcmpl-1", 7, "m"), include_usage=False)

        body = encoder.encode(ErrorEvent("cancelled", "the request was cancelled")) + encoder.end()

        frames = parse_sse(body.decode())
        assert json.loads(frames[0][1])["error"]["code"] == "cancelled"
        assert frames[1] == (None, "[DONE]")
        assert encoder.keepalive().startswith(b":")


def _tool_script() -> list[RequestOutput]:
    return script(
        "<think>",
        "weather needed",
        "</think>",
        "\n\n",
        "<tool_call>",
        '\n{"name": "get_weather", ',
        '"arguments": {"city": "Hue"}}',
        "\n</tool_call>",
    )


class TestToolCallsOverHttp:
    def test_a_streamed_call_reaches_the_client_in_pieces(self, checkpoint: TinyCheckpoint) -> None:
        async def main() -> None:
            async with serve(
                checkpoint.directory, output_format=_TOOLS, scripts=[_tool_script()]
            ) as (client, _, _):
                request = {
                    "messages": [{"role": "user", "content": "weather in Hue?"}],
                    "tools": [_WEATHER],
                    "stream": True,
                }
                frames = await read_sse(await client.post(_CHAT, json=request))
                chunks = [data for _, data in frames[:-1]]
                deltas = [c["choices"][0]["delta"] for c in chunks]
                assert {"reasoning_content": "weather needed"} in deltas
                calls = [d["tool_calls"][0] for d in deltas if "tool_calls" in d]
                assert calls[0]["function"]["name"] == "get_weather"
                assert calls[0]["id"].startswith("call_")
                arguments = "".join(c["function"]["arguments"] for c in calls)
                assert json.loads(arguments) == {"city": "Hue"}
                assert chunks[-1]["choices"][0]["finish_reason"] == "tool_calls"
                assert not any("content" in d and d["content"] for d in deltas)

        asyncio.run(main())

    def test_a_whole_answer_carries_the_call_and_the_reasoning(
        self, checkpoint: TinyCheckpoint
    ) -> None:
        async def main() -> None:
            async with serve(
                checkpoint.directory, output_format=_TOOLS, scripts=[_tool_script()]
            ) as (client, _, _):
                request = {"messages": [{"role": "user", "content": "x"}], "tools": [_WEATHER]}
                answer = await (await client.post(_CHAT, json=request)).json()
                message = answer["choices"][0]["message"]
                assert message["content"] is None
                assert message["reasoning_content"] == "weather needed"
                [call] = message["tool_calls"]
                assert call["function"] == {"name": "get_weather", "arguments": '{"city": "Hue"}'}
                assert answer["choices"][0]["finish_reason"] == "tool_calls"

        asyncio.run(main())

    def test_without_parallel_calls_the_answer_ends_after_the_first(
        self, checkpoint: TinyCheckpoint
    ) -> None:
        two = script(
            "<tool_call>",
            '{"name": "a", "arguments": {}}',
            "</tool_call>",
            "<tool_call>",
            '{"name": "b", "arguments": {}}',
            "</tool_call>",
        )

        async def main() -> None:
            async with serve(checkpoint.directory, output_format=_TOOLS, scripts=[two]) as (
                client,
                _,
                llm,
            ):
                request = {
                    "messages": [{"role": "user", "content": "x"}],
                    "tools": [_WEATHER],
                    "parallel_tool_calls": False,
                }
                answer = await (await client.post(_CHAT, json=request)).json()
                calls = answer["choices"][0]["message"]["tool_calls"]
                assert [c["function"]["name"] for c in calls] == ["a"]
                assert answer["choices"][0]["finish_reason"] == "tool_calls"
                scripted: FakeEngine = llm
                assert scripted.aborted == [scripted.requests[0].request_id]

        asyncio.run(main())
