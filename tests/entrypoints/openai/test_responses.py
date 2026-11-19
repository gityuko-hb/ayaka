"""OpenAI Responses: reading requests, the exact event sequence, function calls, and one prompt
for one conversation across all three protocols."""

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

from ayaka.entrypoints.http import JsonObject
from ayaka.entrypoints.openai.responses import (
    ResponsesStreamEncoder,
    _Snapshot,  # pyright: ignore[reportPrivateUsage]
    parse_responses_request,
)
from ayaka.entrypoints.parsers.output import OutputFormat
from serving_support import parse_sse, read_sse, script, serve, write_long_gpt2
from tiny_model import TinyCheckpoint, greedy_reference, write_tiny_gpt2

if TYPE_CHECKING:
    from pathlib import Path

    from fake_engine import FakeEngine

pytestmark = pytest.mark.unit

_RESPONSES = "/v1/responses"
_FAKE = FakeTokenizer()
_TOOLS = OutputFormat.resolve(reasoning="think", tool_calls="hermes")
_SCHEMA = {"type": "object", "properties": {"city": {"type": "string"}}}
_WEATHER = {
    "type": "function",
    "name": "get_weather",
    "description": "Weather for a city",
    "parameters": _SCHEMA,
}


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory: pytest.TempPathFactory) -> TinyCheckpoint:
    return write_tiny_gpt2(tmp_path_factory.mktemp("tiny-gpt2"))


def _parse(**body: Any) -> Any:
    return parse_responses_request(JsonObject({"model": "tiny", **body}))


class TestReading:
    def test_items_become_one_conversation(self) -> None:
        request = _parse(
            instructions="be brief",
            input=[
                {"role": "user", "content": "weather?"},
                {"type": "reasoning", "id": "rs_1", "summary": []},
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "checking"}],
                },
                {
                    "type": "function_call",
                    "call_id": "c1",
                    "name": "get_weather",
                    "arguments": '{"city":"Hue"}',
                },
                {"type": "function_call_output", "call_id": "c1", "output": "rain"},
            ],
            tools=[_WEATHER, {"type": "web_search"}],
            max_output_tokens=7,
            store=False,
            include=["reasoning.encrypted_content"],
            reasoning={"effort": "low"},
        )

        conversation = request.conversation
        assert conversation.system == "be brief"
        assert [t.role for t in conversation.turns] == ["user", "assistant", "tool"]
        assert conversation.turns[1].text == "checking"
        assert conversation.turns[1].tool_calls[0].id == "c1"
        assert conversation.turns[2].name == "get_weather"
        assert [tool.name for tool in conversation.tools] == ["get_weather"]
        assert request.max_tokens == 7
        assert request.echo["reasoning"] == {"effort": "low", "summary": None}

    def test_a_string_input_is_one_user_message(self) -> None:
        assert _parse(input="hi").conversation.turns[0].text == "hi"

    @pytest.mark.parametrize(
        ("body", "error", "match"),
        [
            ({"previous_response_id": "resp_1"}, InvalidRequestError, "whole conversation"),
            ({"store": True}, InvalidRequestError, "store: false"),
            (
                {"input": [{"type": "item_reference", "id": "x"}]},
                InvalidRequestError,
                "keeps no items",
            ),
            (
                {"input": [{"role": "user", "content": [{"type": "input_image"}]}]},
                UnsupportedFeatureError,
                "input_image",
            ),
            (
                {"text": {"format": {"type": "json_schema"}}},
                InvalidRequestError,
                "needs a schema",
            ),
            ({"text": {"format": {"type": "xml"}}}, InvalidRequestError, "text.format"),
            ({"tool_choice": "required"}, InvalidRequestError, "no function tools"),
            ({"input": []}, InvalidRequestError, "input"),
        ],
    )
    def test_bad_or_unavailable_fields_name_themselves(
        self, body: dict[str, Any], error: type[Exception], match: str
    ) -> None:
        with pytest.raises(error, match=match):
            _parse(**{"input": "x", **body})


def _encoder() -> ResponsesStreamEncoder:
    return ResponsesStreamEncoder(_Snapshot("resp_1", 7, "m", {}), "1")


class TestFixedEvents:
    def test_a_fixed_event_sequence_gives_these_exact_events(self) -> None:
        encoder = _encoder()
        events = [
            ReasoningDelta("hmm"),
            TextDelta("Hi"),
            TextDelta("!"),
            ToolCallStart(0, "call_1_0", "f"),
            ToolCallDelta(0, '{"a":'),
            ToolCallDelta(0, "1}"),
            ToolCallEnd(0, valid=True),
            UsageEvent(3, 5, reasoning_tokens=1),
            FinishEvent(FinishReason.STOP, tool_calls=True),
        ]

        body = encoder.start() + b"".join(encoder.encode(e) for e in events) + encoder.end()

        frames = [(name, json.loads(data)) for name, data in parse_sse(body.decode())]
        assert [name for name, _ in frames] == [
            "response.created",
            "response.in_progress",
            "response.output_item.added",
            "response.reasoning_text.delta",
            "response.reasoning_text.done",
            "response.output_item.done",
            "response.output_item.added",
            "response.content_part.added",
            "response.output_text.delta",
            "response.output_text.delta",
            "response.output_text.done",
            "response.content_part.done",
            "response.output_item.done",
            "response.output_item.added",
            "response.function_call_arguments.delta",
            "response.function_call_arguments.delta",
            "response.function_call_arguments.done",
            "response.output_item.done",
            "response.completed",
        ]
        assert [data["sequence_number"] for _, data in frames] == list(range(len(frames)))
        assert all(data["type"] == name for name, data in frames)
        final = frames[-1][1]["response"]
        assert final["status"] == "completed"
        assert [item["type"] for item in final["output"]] == [
            "reasoning",
            "message",
            "function_call",
        ]
        assert final["output"][1]["content"][0]["text"] == "Hi!"
        assert final["output"][2]["arguments"] == '{"a":1}'
        assert final["output"][2]["status"] == "completed"
        assert final["usage"]["output_tokens_details"] == {"reasoning_tokens": 1}
        assert [data.get("output_index") for _, data in frames[2:-1]] == [
            0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2,
        ]  # fmt: skip

    def test_length_is_incomplete_and_an_error_is_failed(self) -> None:
        length = _encoder()
        body = length.encode(TextDelta("a")) + length.encode(FinishEvent(FinishReason.LENGTH))
        name, data = parse_sse(body.decode())[-1]
        assert name == "response.incomplete"
        response = json.loads(data)["response"]
        assert response["incomplete_details"] == {"reason": "max_output_tokens"}

        failed = _encoder()
        [(name, data)] = parse_sse(failed.encode(ErrorEvent("cancelled", "gone")).decode())
        assert name == "response.failed"
        assert json.loads(data)["response"]["error"] == {"code": "cancelled", "message": "gone"}


def _chat_prompt(text: str) -> list[int]:
    return list(f"<|im_start|>user\n{text}<|im_end|>\n<|im_start|>assistant\n".encode())


class TestOverHttp:
    def test_an_answer_and_a_stream_say_the_reference_text(
        self, checkpoint: TinyCheckpoint
    ) -> None:
        tokens = greedy_reference(checkpoint.state, checkpoint.config, _chat_prompt("hi"), 6)
        expected = _FAKE.decode(tokens)
        request = {"model": "tiny", "input": "hi", "max_output_tokens": 6, "temperature": 0}

        async def main() -> None:
            async with serve(checkpoint.directory) as (client, _, _):
                answer = await (await client.post(_RESPONSES, json=request)).json()
                assert answer["object"] == "response"
                assert answer["id"].startswith("resp_")
                assert answer["status"] == "incomplete"
                [message] = answer["output"]
                assert message["content"][0]["text"] == expected
                assert answer["usage"]["output_tokens"] == 6
                frames = await read_sse(
                    await client.post(_RESPONSES, json={**request, "stream": True})
                )
                text = "".join(
                    data["delta"] for name, data in frames if name == "response.output_text.delta"
                )
                assert text == expected
                assert frames[-1][0] == "response.incomplete"
                refused = await client.post(
                    _RESPONSES, json={**request, "previous_response_id": "resp_x"}
                )
                assert refused.status == 400
                assert (await refused.json())["error"]["param"] == "previous_response_id"

        asyncio.run(main())

    def test_a_streamed_function_call(self, checkpoint: TinyCheckpoint) -> None:
        answer = script(
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
                    "input": "weather?",
                    "tools": [_WEATHER],
                    "stream": True,
                }
                frames = await read_sse(await client.post(_RESPONSES, json=request))
                done = next(d for n, d in frames if n == "response.function_call_arguments.done")
                assert json.loads(done["arguments"]) == {"city": "Hue"}
                final = frames[-1][1]["response"]
                assert frames[-1][0] == "response.completed"
                [call] = final["output"]
                assert call["type"] == "function_call"
                assert call["name"] == "get_weather"
                assert call["call_id"].startswith("call_")

        asyncio.run(main())


def test_one_conversation_in_three_protocols_gives_one_prompt(tmp_path: Path) -> None:
    directory = write_long_gpt2(tmp_path)
    chat = {
        "model": "tiny",
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
                    "parameters": _SCHEMA,
                },
            }
        ],
    }
    messages = {
        "model": "tiny",
        "max_tokens": 4,
        "system": "be brief",
        "messages": [
            {"role": "user", "content": "weather in Hue?"},
            {
                "role": "assistant",
                "content": [
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
                    {"type": "tool_result", "tool_use_id": "t1", "content": "rain"},
                    {"type": "text", "text": "thanks"},
                ],
            },
        ],
        "tools": [
            {"name": "get_weather", "description": "Weather for a city", "input_schema": _SCHEMA}
        ],
    }
    responses = {
        "model": "tiny",
        "instructions": "be brief",
        "input": [
            {"role": "user", "content": "weather in Hue?"},
            {"role": "assistant", "content": "checking"},
            {
                "type": "function_call",
                "call_id": "t1",
                "name": "get_weather",
                "arguments": '{"city": "Hue"}',
            },
            {"type": "function_call_output", "call_id": "t1", "output": "rain"},
            {"role": "user", "content": "thanks"},
        ],
        "tools": [_WEATHER],
    }

    async def main() -> None:
        async with serve(directory, output_format=_TOOLS, scripts=[], num_blocks=None) as (
            client,
            _,
            llm,
        ):
            for path, body in (
                ("/v1/chat/completions", chat),
                ("/v1/messages", messages),
                (_RESPONSES, responses),
            ):
                response = await client.post(path, json=body)
                assert response.status == 200, await response.text()
            scripted: FakeEngine = llm
            prompts = [request.prompt for request in scripted.requests]
            assert prompts[0] == prompts[1] == prompts[2]

    asyncio.run(main())
