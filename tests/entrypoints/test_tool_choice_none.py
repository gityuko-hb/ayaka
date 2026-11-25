"""``tool_choice: none`` keeps the tools in the prompt and forbids the call.

A coding agent switches ``tool_choice`` between turns of one session (a summary turn sends
``none``). When ``none`` dropped the tool definitions, the prompt lost its bytes from the tool
block on, and the prefix cache with them: the whole context was computed again. These tests pin
the three protocols to one prompt for ``auto`` and ``none``, check that the cache is read across
the switch on the real engine, and that a model with a one-token call marker is stopped there.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

import pytest

pytest.importorskip("torch")
pytest.importorskip("aiohttp")

from ayaka.entrypoints.anthropic.protocol import parse_messages_request
from ayaka.entrypoints.openai.protocol import parse_chat_request
from ayaka.entrypoints.openai.responses import parse_responses_request
from ayaka.entrypoints.parsers.output import OutputFormat
from ayaka.entrypoints.wire.reader import JsonObject
from serving_support import serve, write_long_gpt2
from tokenizer_fakes import FakeTokenizer, ToolTokenizer, loaded_for

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable
    from pathlib import Path

    from aiohttp.test_utils import TestClient

    from fake_engine import FakeEngine

pytestmark = pytest.mark.unit

_PARAMETERS = {"type": "object", "properties": {"path": {"type": "string"}}}
# Long enough that the prompt fills several cache blocks before the user's question.
_DESCRIPTION = "Read a file of the repository and return its text. " * 4
_QUESTION = "Summarize what we changed in this session in two sentences, without calling a tool."
_OPENER = 260  # ToolTokenizer's <tool_call>

_CHAT_TOOL = {
    "type": "function",
    "function": {"name": "read_file", "description": _DESCRIPTION, "parameters": _PARAMETERS},
}
_MESSAGES_TOOL = {"name": "read_file", "description": _DESCRIPTION, "input_schema": _PARAMETERS}
_RESPONSES_TOOL = {
    "type": "function",
    "name": "read_file",
    "description": _DESCRIPTION,
    "parameters": _PARAMETERS,
}


def _chat(choice: object) -> dict[str, Any]:
    return {
        "messages": [{"role": "user", "content": _QUESTION}],
        "tools": [_CHAT_TOOL],
        "tool_choice": choice,
        "max_tokens": 1,
    }


def _messages(choice: object) -> dict[str, Any]:
    return {
        "model": "tiny",
        "max_tokens": 1,
        "messages": [{"role": "user", "content": _QUESTION}],
        "tools": [_MESSAGES_TOOL],
        "tool_choice": choice,
    }


def _responses(choice: object) -> dict[str, Any]:
    return {
        "input": _QUESTION,
        "tools": [_RESPONSES_TOOL],
        "tool_choice": choice,
        "max_output_tokens": 1,
    }


# (route, body for auto, body for none, path of cached tokens in the usage)
_PROTOCOLS = [
    pytest.param(
        "/v1/chat/completions",
        _chat("auto"),
        _chat("none"),
        ("usage", "prompt_tokens_details", "cached_tokens"),
        id="chat",
    ),
    pytest.param(
        "/v1/messages",
        _messages({"type": "auto"}),
        _messages({"type": "none"}),
        ("usage", "cache_read_input_tokens"),
        id="messages",
    ),
    pytest.param(
        "/v1/responses",
        _responses("auto"),
        _responses("none"),
        ("usage", "input_tokens_details", "cached_tokens"),
        id="responses",
    ),
]


class TestReading:
    def test_chat_keeps_the_tools_and_offers_no_call(self) -> None:
        chat = parse_chat_request(JsonObject(_chat("none")))

        assert not chat.tools
        assert [tool.name for tool in chat.conversation.tools] == ["read_file"]
        assert chat.output.tool_choice == "auto"

    def test_messages_keeps_the_tools_and_offers_no_call(self) -> None:
        message = parse_messages_request(JsonObject(_messages({"type": "none"})))

        assert not message.tools
        assert [tool.name for tool in message.conversation.tools] == ["read_file"]
        assert message.output.tool_choice == "auto"

    def test_responses_keeps_the_tools_and_offers_no_call(self) -> None:
        responses = parse_responses_request(JsonObject(_responses("none")))

        assert not responses.tools
        assert [tool.name for tool in responses.conversation.tools] == ["read_file"]
        assert responses.output.tool_choice == "auto"

    def test_none_without_tools_is_still_valid(self) -> None:
        chat = parse_chat_request(
            JsonObject({"messages": [{"role": "user", "content": "x"}], "tool_choice": "none"})
        )

        assert not chat.tools
        assert chat.conversation.tools == ()


def _with_fake_engine(
    body: Callable[[TestClient[Any, Any], FakeEngine], Awaitable[None]],
    checkpoint: Path,
    *,
    tool_calls: str,
) -> None:
    async def main() -> None:
        tokenizer = ToolTokenizer()
        async with serve(
            checkpoint,
            tokenizer=loaded_for(tokenizer, stream=False),
            output_format=OutputFormat.resolve(tool_calls=tool_calls, tokenizer=tokenizer),
            scripts=[],
            num_blocks=600,  # the long GPT-2 reserves its whole 1024-token context
        ) as (client, _, engine):
            await body(client, engine)

    asyncio.run(main())


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory: pytest.TempPathFactory) -> Path:
    # 1024 positions: the tool block and the question fit with room to spare.
    return write_long_gpt2(tmp_path_factory.mktemp("long-gpt2"))


class TestPrompt:
    @pytest.mark.parametrize(("route", "auto", "none", "cached"), _PROTOCOLS)
    def test_auto_and_none_send_the_same_prompt_ids(
        self,
        checkpoint: Path,
        route: str,
        auto: dict[str, Any],
        none: dict[str, Any],
        cached: tuple[str, ...],
    ) -> None:
        del cached

        async def body(client: TestClient[Any, Any], engine: FakeEngine) -> None:
            for request in (auto, none):
                response = await client.post(route, json=request)
                assert response.status == 200, await response.text()
            first, second = engine.requests

            assert first.prompt == second.prompt
            prompt = ToolTokenizer().decode(list(first.prompt))  # type: ignore[arg-type]
            assert "read_file" in prompt  # the tool block is in both prompts

        _with_fake_engine(body, checkpoint, tool_calls="hermes")

    @pytest.mark.parametrize(("route", "auto", "none", "cached"), _PROTOCOLS)
    def test_none_stops_at_the_call_marker_and_auto_does_not(
        self,
        checkpoint: Path,
        route: str,
        auto: dict[str, Any],
        none: dict[str, Any],
        cached: tuple[str, ...],
    ) -> None:
        del cached

        async def body(client: TestClient[Any, Any], engine: FakeEngine) -> None:
            for request in (auto, none):
                response = await client.post(route, json=request)
                assert response.status == 200, await response.text()
            with_auto, with_none = engine.requests

            assert _OPENER not in with_auto.sampling.stop_token_ids
            assert _OPENER in with_none.sampling.stop_token_ids

        _with_fake_engine(body, checkpoint, tool_calls="hermes")

    def test_a_format_without_a_marker_token_adds_no_stop(self, checkpoint: Path) -> None:
        # llama3_json opens a call with a bare JSON object: no one token marks it.
        async def body(client: TestClient[Any, Any], engine: FakeEngine) -> None:
            response = await client.post("/v1/chat/completions", json=_chat("none"))
            assert response.status == 200, await response.text()

            assert engine.requests[0].sampling.stop_token_ids == frozenset()

        _with_fake_engine(body, checkpoint, tool_calls="llama3_json")


class TestPrefixCache:
    @pytest.mark.parametrize(("route", "auto", "none", "cached"), _PROTOCOLS)
    def test_switching_auto_to_none_reads_the_prompt_from_the_cache(
        self,
        checkpoint: Path,
        route: str,
        auto: dict[str, Any],
        none: dict[str, Any],
        cached: tuple[str, ...],
    ) -> None:
        async def main() -> None:
            # The byte tokenizer decodes all 300 ids of the tiny model; ToolTokenizer knows 262
            # and the random model may draw one past them. The cache needs no marker token.
            async with serve(
                checkpoint,
                output_format=OutputFormat.resolve(tool_calls="hermes", tokenizer=FakeTokenizer()),
                num_blocks=600,
            ) as (client, _, _):
                answers = []
                # auto, then none, then auto again: none must read as much of the prompt from
                # the cache as a repeat of the same auto request does.
                for request in (auto, none, auto):
                    response = await client.post(route, json=request)
                    assert response.status == 200, await response.text()
                    answers.append(await response.json())

            def read(answer: dict[str, Any]) -> int:
                value: Any = answer
                for key in cached:
                    value = value[key]
                return int(value)

            first, switched, repeated = (read(answer) for answer in answers)
            assert first == 0
            assert repeated > 0
            assert switched == repeated

        asyncio.run(main())
