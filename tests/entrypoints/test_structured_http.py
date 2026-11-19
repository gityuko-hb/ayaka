"""Structured output over HTTP on the tiny GPT-2, in all three protocols.

The server runs the real engine with a byte tokenizer that has the Hermes tool-call markers as
tokens, so a forced or lazy tool call goes through the grammar, the parser and the encoder.
"""

from __future__ import annotations

import asyncio
import json
from typing import TYPE_CHECKING, Any

import pytest

pytest.importorskip("torch")
pytest.importorskip("aiohttp")
pytest.importorskip("llguidance")

from ayaka.entrypoints.parsers.output import OutputFormat
from serving_support import serve, write_long_gpt2
from tokenizer_fakes import ToolTokenizer, loaded_for

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable
    from pathlib import Path

    from aiohttp.test_utils import TestClient

pytestmark = pytest.mark.unit

_SCHEMA = {
    "type": "object",
    "properties": {"answer": {"type": "string", "maxLength": 5}},
    "required": ["answer"],
    "additionalProperties": False,
}
_PARAMETERS = {
    "type": "object",
    "properties": {"city": {"type": "string", "maxLength": 5}},
    "required": ["city"],
    "additionalProperties": False,
}
_OPENAI_TOOL = {
    "type": "function",
    "function": {"name": "get_weather", "parameters": _PARAMETERS, "strict": True},
}


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory: pytest.TempPathFactory) -> Path:
    # 1024 positions: a prompt with tool definitions leaves room for a whole call.
    return write_long_gpt2(tmp_path_factory.mktemp("long-gpt2"))


def _run(
    checkpoint: Path,
    body: Callable[[TestClient[Any, Any]], Awaitable[None]],
    *,
    tool_calls: str = "hermes",
) -> None:
    async def main() -> None:
        tokenizer = ToolTokenizer()
        async with serve(
            checkpoint,
            tokenizer=loaded_for(tokenizer, stream=False),
            output_format=OutputFormat.resolve(tool_calls=tool_calls, tokenizer=tokenizer),
            max_tokens_per_step=64,
            num_blocks=600,  # a whole 1024-token context and then some
        ) as (client, _, _):
            await body(client)

    asyncio.run(main())


def _valid_answer(text: str) -> None:
    value = json.loads(text)
    assert set(value) == {"answer"}
    assert len(value["answer"]) <= 5


class TestChat:
    def test_a_json_schema_answer_validates(self, checkpoint: Path) -> None:
        async def body(client: TestClient[Any, Any]) -> None:
            request = {
                "messages": [{"role": "user", "content": "hi"}],
                "temperature": 1.0,
                "seed": 5,
                "max_tokens": 40,
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {"name": "a", "schema": _SCHEMA, "strict": True},
                },
            }
            answer = await (await client.post("/v1/chat/completions", json=request)).json()

            _valid_answer(answer["choices"][0]["message"]["content"])
            assert answer["choices"][0]["finish_reason"] == "stop"

        _run(checkpoint, body)

    def test_a_named_tool_choice_gives_a_valid_call(self, checkpoint: Path) -> None:
        async def body(client: TestClient[Any, Any]) -> None:
            request = {
                "messages": [{"role": "user", "content": "weather?"}],
                "tools": [_OPENAI_TOOL],
                "tool_choice": {"type": "function", "function": {"name": "get_weather"}},
                "parallel_tool_calls": False,
                "temperature": 1.0,
                "seed": 2,
                "max_tokens": 90,
            }
            answer = await (await client.post("/v1/chat/completions", json=request)).json()

            choice = answer["choices"][0]
            assert choice["finish_reason"] == "tool_calls"
            [call] = choice["message"]["tool_calls"]
            assert call["function"]["name"] == "get_weather"
            assert set(json.loads(call["function"]["arguments"])) == {"city"}

        _run(checkpoint, body)

    def test_auto_tools_are_accepted_and_free_text_is_not_forced(self, checkpoint: Path) -> None:
        async def body(client: TestClient[Any, Any]) -> None:
            request = {
                "messages": [{"role": "user", "content": "hi"}],
                "tools": [_OPENAI_TOOL],
                "temperature": 0,
                "max_tokens": 6,
            }
            response = await client.post("/v1/chat/completions", json=request)

            assert response.status == 200
            answer = await response.json()
            assert answer["choices"][0]["message"]["content"]  # greedy text, never a call

        _run(checkpoint, body)


class TestFormatWithoutMarkers:
    """``llama3_json`` has no marker to start a lazy grammar on: free text, a forced call written
    from the first token, or a clear refusal for strict arguments of an optional call."""

    def test_auto_tools_are_accepted_and_free_text_is_not_forced(self, checkpoint: Path) -> None:
        async def body(client: TestClient[Any, Any]) -> None:
            plain = {"type": "function", "function": {"name": "f", "parameters": _PARAMETERS}}
            request = {
                "messages": [{"role": "user", "content": "hi"}],
                "tools": [plain],
                "temperature": 0,
                "max_tokens": 6,
            }
            response = await client.post("/v1/chat/completions", json=request)

            assert response.status == 200
            answer = await response.json()
            assert answer["choices"][0]["message"]["content"]

        _run(checkpoint, body, tool_calls="llama3_json")

    @pytest.mark.parametrize(
        "choice",
        ["required", {"type": "function", "function": {"name": "get_weather"}}],
        ids=["required", "named"],
    )
    @pytest.mark.parametrize("parallel", [False, True])
    def test_a_forced_call_is_written_from_the_first_token(
        self, checkpoint: Path, choice: object, parallel: bool
    ) -> None:
        """No marker to wait for: the grammar is the whole answer, ``{"name", "parameters"}``."""

        async def body(client: TestClient[Any, Any]) -> None:
            request = {
                "messages": [{"role": "user", "content": "weather?"}],
                "tools": [_OPENAI_TOOL],
                "tool_choice": choice,
                "parallel_tool_calls": parallel,
                "temperature": 1.0,
                "seed": 2,
                "max_tokens": 90,
            }
            response = await client.post("/v1/chat/completions", json=request)

            assert response.status == 200, await response.text()
            choice_ = (await response.json())["choices"][0]
            assert choice_["finish_reason"] == "tool_calls"
            # The random model may go on with "; {" and be cut by max_tokens in a later call.
            first, *rest = choice_["message"]["tool_calls"]
            assert first["function"]["name"] == "get_weather"
            assert set(json.loads(first["function"]["arguments"])) == {"city"}
            assert parallel or not rest

        _run(checkpoint, body, tool_calls="llama3_json")

    def test_strict_arguments_of_an_optional_call_are_refused(self, checkpoint: Path) -> None:
        async def body(client: TestClient[Any, Any]) -> None:
            request = {
                "messages": [{"role": "user", "content": "weather?"}],
                "tools": [_OPENAI_TOOL],  # strict, and the call is optional
                "max_tokens": 20,
            }
            response = await client.post("/v1/chat/completions", json=request)

            assert response.status == 400
            error = (await response.json())["error"]
            assert "llama3_json" in error["message"]

        _run(checkpoint, body, tool_calls="llama3_json")


class TestResponses:
    def test_a_text_format_schema_answer_validates(self, checkpoint: Path) -> None:
        async def body(client: TestClient[Any, Any]) -> None:
            request = {
                "input": "hi",
                "temperature": 1.0,
                "max_output_tokens": 40,
                "text": {"format": {"type": "json_schema", "name": "a", "schema": _SCHEMA}},
            }
            answer = await (await client.post("/v1/responses", json=request)).json()

            [message] = [item for item in answer["output"] if item["type"] == "message"]
            _valid_answer(message["content"][0]["text"])

        _run(checkpoint, body)


class TestMessages:
    def test_an_output_format_answer_validates(self, checkpoint: Path) -> None:
        async def body(client: TestClient[Any, Any]) -> None:
            request = {
                "max_tokens": 40,
                "messages": [{"role": "user", "content": "hi"}],
                "temperature": 1.0,
                "output_format": {"type": "json_schema", "schema": _SCHEMA},
            }
            answer = await (await client.post("/v1/messages", json=request)).json()

            _valid_answer(answer["content"][0]["text"])

        _run(checkpoint, body)

    def test_a_forced_tool_gives_a_tool_use_block(
        self, checkpoint: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A forced call at temperature 1 on the random tiny model, with a fixed request id.

        Messages has no ``seed``: the seed comes from the request id, which is random. For about
        one id in twenty the random model, free inside a non-strict ``arguments`` object, writes
        past ``max_tokens`` and the call ends cut (a text block). The id is fixed so the test
        checks the forced call, not the luck of the draw.
        """
        import types

        from ayaka.entrypoints import http as http_module

        fixed = types.SimpleNamespace(hex="0" * 32)
        monkeypatch.setattr(http_module, "uuid", types.SimpleNamespace(uuid4=lambda: fixed))

        async def body(client: TestClient[Any, Any]) -> None:
            request = {
                "max_tokens": 90,
                "messages": [{"role": "user", "content": "weather?"}],
                "temperature": 1.0,
                "tools": [{"name": "get_weather", "input_schema": _PARAMETERS}],
                # One call, then only an end: the random tiny model writes undecodable ids freely.
                "tool_choice": {
                    "type": "tool",
                    "name": "get_weather",
                    "disable_parallel_tool_use": True,
                },
            }
            answer = await (await client.post("/v1/messages", json=request)).json()

            assert answer["stop_reason"] == "tool_use"
            [block] = [b for b in answer["content"] if b["type"] == "tool_use"]
            assert block["name"] == "get_weather"
            assert isinstance(block["input"], dict)

        _run(checkpoint, body)
