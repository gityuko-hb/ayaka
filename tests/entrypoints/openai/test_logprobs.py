"""Logprobs and penalties over HTTP on the tiny GPT-2: Chat Completions and Completions."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

import pytest

pytest.importorskip("torch")
pytest.importorskip("aiohttp")

from serving_support import read_sse, serve
from tiny_model import TinyCheckpoint, write_tiny_gpt2

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from aiohttp.test_utils import TestClient

pytestmark = pytest.mark.unit


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory: pytest.TempPathFactory) -> TinyCheckpoint:
    return write_tiny_gpt2(tmp_path_factory.mktemp("tiny-gpt2"))


def _run(
    checkpoint: TinyCheckpoint, body: Callable[[TestClient[Any, Any]], Awaitable[None]]
) -> None:
    async def main() -> None:
        async with serve(checkpoint.directory) as (client, _, _):
            await body(client)

    asyncio.run(main())


_CHAT = {
    "messages": [{"role": "user", "content": "hello"}],
    "temperature": 0,
    "max_tokens": 5,
    "logprobs": True,
    "top_logprobs": 2,
}


class TestChat:
    def test_every_token_has_its_logprob_and_two_alternatives(
        self, checkpoint: TinyCheckpoint
    ) -> None:
        async def body(client: TestClient[Any, Any]) -> None:
            answer = await (await client.post("/v1/chat/completions", json=_CHAT)).json()

            choice = answer["choices"][0]
            content = choice["logprobs"]["content"]
            assert len(content) == answer["usage"]["completion_tokens"] == 5
            assert "".join(item["token"] for item in content) == choice["message"]["content"]
            for item in content:
                assert item["logprob"] <= 0
                assert item["bytes"] == list(item["token"].encode())
                assert len(item["top_logprobs"]) == 2
                assert item["top_logprobs"][0]["token"] == item["token"]  # greedy: the best

        _run(checkpoint, body)

    def test_streamed_logprobs_equal_the_whole_answer(self, checkpoint: TinyCheckpoint) -> None:
        async def body(client: TestClient[Any, Any]) -> None:
            whole = await (await client.post("/v1/chat/completions", json=_CHAT)).json()
            frames = await read_sse(
                await client.post("/v1/chat/completions", json={**_CHAT, "stream": True})
            )

            streamed = [
                item
                for _, chunk in frames[:-1]
                for choice in chunk["choices"]
                if choice["logprobs"]
                for item in choice["logprobs"]["content"]
            ]
            expected = whole["choices"][0]["logprobs"]["content"]
            # Two requests: the second attaches the first's prompt from the prefix cache, which
            # may move the last float bits, never the tokens.
            assert [item["token"] for item in streamed] == [item["token"] for item in expected]
            for got, want in zip(streamed, expected, strict=True):
                assert got["logprob"] == pytest.approx(want["logprob"], abs=1e-5)

        _run(checkpoint, body)

    def test_without_logprobs_the_field_is_null(self, checkpoint: TinyCheckpoint) -> None:
        async def body(client: TestClient[Any, Any]) -> None:
            request = {key: value for key, value in _CHAT.items() if "logprobs" not in key}
            answer = await (await client.post("/v1/chat/completions", json=request)).json()

            assert answer["choices"][0]["logprobs"] is None

        _run(checkpoint, body)

    def test_too_many_alternatives_are_refused(self, checkpoint: TinyCheckpoint) -> None:
        async def body(client: TestClient[Any, Any]) -> None:
            response = await client.post("/v1/chat/completions", json={**_CHAT, "top_logprobs": 21})

            assert response.status == 400

        _run(checkpoint, body)

    def test_penalties_and_a_bias_are_accepted(self, checkpoint: TinyCheckpoint) -> None:
        async def body(client: TestClient[Any, Any]) -> None:
            request = {
                "messages": [{"role": "user", "content": "hello"}],
                "temperature": 0,
                "max_tokens": 4,
                "presence_penalty": 0.5,
                "frequency_penalty": 0.5,
                "logit_bias": {str(ord("q")): 100},
            }
            answer = await (await client.post("/v1/chat/completions", json=request)).json()

            assert answer["choices"][0]["message"]["content"] == "qqqq"

        _run(checkpoint, body)


class TestCompletions:
    def test_the_legacy_logprobs_object(self, checkpoint: TinyCheckpoint) -> None:
        async def body(client: TestClient[Any, Any]) -> None:
            request = {"prompt": "once upon", "temperature": 0, "max_tokens": 4, "logprobs": 2}
            answer = await (await client.post("/v1/completions", json=request)).json()
            frames = await read_sse(
                await client.post("/v1/completions", json={**request, "stream": True})
            )

            choice = answer["choices"][0]
            logprobs = choice["logprobs"]
            assert "".join(logprobs["tokens"]) == choice["text"]
            assert len(logprobs["token_logprobs"]) == 4
            assert all(len(top) <= 2 for top in logprobs["top_logprobs"])
            offsets = logprobs["text_offset"]
            assert offsets == sorted(offsets)
            assert offsets[0] == 0
            streamed = [
                token
                for _, chunk in frames[:-1]
                for c in chunk["choices"]
                if c["logprobs"]
                for token in c["logprobs"]["tokens"]
            ]
            assert streamed == logprobs["tokens"]

        _run(checkpoint, body)
