"""Several choices (``n > 1``) over HTTP on the tiny GPT-2: Chat Completions and Completions.

The answer of every choice must be what the engine itself returns for the same request, the prompt
must be counted once in the usage, and Responses and Messages, which have no ``n``, must refuse it.
"""

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

# top_k keeps the draws among the printable tokens the fixture favours, which the tokenizer decodes.
_SAMPLING = {"n": 2, "temperature": 1.0, "top_k": 4, "seed": 11, "max_tokens": 6, "min_tokens": 6}


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


class TestChat:
    def test_two_choices_in_one_answer_with_one_prompt_in_the_usage(
        self, checkpoint: TinyCheckpoint
    ) -> None:
        async def body(client: TestClient[Any, Any]) -> None:
            request = {"messages": [{"role": "user", "content": "hello"}], **_SAMPLING}
            answer = await (await client.post("/v1/chat/completions", json=request)).json()
            single = await (
                await client.post("/v1/chat/completions", json={**request, "n": 1})
            ).json()

            choices = answer["choices"]
            assert [c["index"] for c in choices] == [0, 1]
            assert all(c["finish_reason"] == "length" for c in choices)
            assert choices[0]["message"]["content"] != choices[1]["message"]["content"]
            assert choices[0]["message"]["content"] == single["choices"][0]["message"]["content"]
            usage = answer["usage"]
            assert usage["prompt_tokens"] == single["usage"]["prompt_tokens"]
            assert usage["completion_tokens"] == 12
            assert usage["prompt_tokens_details"]["cached_tokens"] > 0

        _run(checkpoint, body)

    def test_a_stream_of_two_choices_matches_the_whole_answer(
        self, checkpoint: TinyCheckpoint
    ) -> None:
        async def body(client: TestClient[Any, Any]) -> None:
            request = {"messages": [{"role": "user", "content": "hello"}], **_SAMPLING}
            whole = await (await client.post("/v1/chat/completions", json=request)).json()
            frames = await read_sse(
                await client.post(
                    "/v1/chat/completions",
                    json={**request, "stream": True, "stream_options": {"include_usage": True}},
                )
            )

            assert frames[-1][1] == "[DONE]"
            chunks = [data for _, data in frames[:-1]]
            text = {0: "", 1: ""}
            roles: list[int] = []
            finishes: dict[int, str] = {}
            for chunk in chunks:
                for choice in chunk["choices"]:
                    delta = choice["delta"]
                    if "role" in delta:
                        roles.append(choice["index"])
                    text[choice["index"]] += delta.get("content") or ""
                    if choice["finish_reason"] is not None:
                        finishes[choice["index"]] = choice["finish_reason"]
            assert sorted(roles) == [0, 1]
            assert finishes == {0: "length", 1: "length"}
            assert text[0] == whole["choices"][0]["message"]["content"]
            assert text[1] == whole["choices"][1]["message"]["content"]
            usage_chunks = [c for c in chunks if c.get("usage")]
            assert len(usage_chunks) == 1
            assert usage_chunks[0]["choices"] == []
            assert usage_chunks[0]["usage"]["completion_tokens"] == 12

        _run(checkpoint, body)


class TestCompletions:
    def test_two_choices_streamed_and_whole(self, checkpoint: TinyCheckpoint) -> None:
        async def body(client: TestClient[Any, Any]) -> None:
            request = {"prompt": "once upon a time", **_SAMPLING}
            whole = await (await client.post("/v1/completions", json=request)).json()
            frames = await read_sse(
                await client.post("/v1/completions", json={**request, "stream": True})
            )

            assert [c["index"] for c in whole["choices"]] == [0, 1]
            text = {0: "", 1: ""}
            for _, chunk in frames[:-1]:
                for choice in chunk["choices"]:
                    text[choice["index"]] += choice["text"]
            assert text[0] == whole["choices"][0]["text"]
            assert text[1] == whole["choices"][1]["text"]
            assert whole["usage"]["completion_tokens"] == 12

        _run(checkpoint, body)


class TestRefusals:
    @pytest.mark.parametrize(
        ("path", "request_body"),
        [
            ("/v1/responses", {"input": "hi", "n": 2}),
            (
                "/v1/messages",
                {"max_tokens": 4, "messages": [{"role": "user", "content": "hi"}], "n": 2},
            ),
        ],
        ids=["responses", "messages"],
    )
    def test_protocols_without_n_refuse_it(
        self, checkpoint: TinyCheckpoint, path: str, request_body: dict[str, Any]
    ) -> None:
        async def body(client: TestClient[Any, Any]) -> None:
            response = await client.post(path, json=request_body)
            payload = await response.json()

            assert response.status == 400
            assert "n:" in str(payload)

        _run(checkpoint, body)
