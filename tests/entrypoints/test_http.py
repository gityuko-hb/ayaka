"""The HTTP server on the tiny GPT-2: chat and completions, streaming, errors and life cycle.

Requests go over real HTTP on localhost to ``create_app``. The oracle for generated text is
``greedy_reference`` decoded by the byte tokenizer, as in the engine tests.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import socket
import time
from typing import TYPE_CHECKING, Any

import pytest

from ayaka.configs.server import ServerConfig
from ayaka.core.errors import ConfigError, OverloadedError
from ayaka.core.params import SamplingParams
from ayaka.core.request import GenerateRequest
from ayaka.core.types import RequestId
from tokenizer_fakes import FakeTokenizer, loaded_for

torch: Any = pytest.importorskip("torch")
pytest.importorskip("aiohttp")

from ayaka.engine.async_llm import AsyncLLM  # noqa: E402
from ayaka.engine.build import build_engine  # noqa: E402
from ayaka.entrypoints.conversation import ErrorEvent, FinishEvent  # noqa: E402
from ayaka.entrypoints.parsers.output import OutputFormat  # noqa: E402
from ayaka.entrypoints.stream import SlowClientError, pump  # noqa: E402
from serving_support import parse_sse, read_sse, script, serve  # noqa: E402
from tiny_model import (  # noqa: E402  # needs the importorskip above
    TinyCheckpoint,
    assert_cache_released,
    greedy_reference,
    tiny_engine_config,
    tiny_tokenizer,
    write_tiny_gpt2,
)

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from aiohttp.test_utils import TestClient

    from ayaka.entrypoints.serving import Serving
    from fake_engine import FakeEngine

pytestmark = pytest.mark.unit

_FAKE = FakeTokenizer()
_CHAT = "/v1/chat/completions"


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory: pytest.TempPathFactory) -> TinyCheckpoint:
    return write_tiny_gpt2(tmp_path_factory.mktemp("tiny-gpt2"))


type Client = TestClient[Any, Any]
type Body = Callable[[Client, Serving, Any], Awaitable[None]]  # the engine client: real or fake


def _run(checkpoint: TinyCheckpoint, body: Body, **options: Any) -> None:
    async def main() -> None:
        async with serve(checkpoint.directory, **options) as (client, serving, llm):
            await body(client, serving, llm)

    asyncio.run(main())


def _chat_prompt(text: str) -> list[int]:
    """The ids the fake template and tokenizer make of one user message."""
    rendered = f"<|im_start|>user\n{text}<|im_end|>\n<|im_start|>assistant\n"
    return list(rendered.encode())


def _expected_chat(checkpoint: TinyCheckpoint, text: str, count: int) -> str:
    tokens = greedy_reference(checkpoint.state, checkpoint.config, _chat_prompt(text), count)
    return _FAKE.decode(tokens, skip_special_tokens=True)


def _chat(text: str, max_tokens: int, **more: Any) -> dict[str, Any]:
    return {
        "model": "tiny",
        "messages": [{"role": "user", "content": text}],
        "max_tokens": max_tokens,
        "temperature": 0,
        **more,
    }


class TestChat:
    def test_an_answer_is_the_reference_text_with_usage(self, checkpoint: TinyCheckpoint) -> None:
        expected = _expected_chat(checkpoint, "hi", 6)

        async def body(client: Client, serving: Serving, llm: AsyncLLM) -> None:
            response = await client.post(_CHAT, json=_chat("hi", 6))
            assert response.status == 200
            assert response.headers["x-request-id"]
            answer = await response.json()
            assert answer["object"] == "chat.completion"
            assert answer["id"].startswith("chatcmpl-")
            assert answer["model"] == "tiny"
            [choice] = answer["choices"]
            assert choice["message"] == {"role": "assistant", "content": expected}
            assert choice["finish_reason"] == "length"
            usage = answer["usage"]
            assert usage["prompt_tokens"] == len(_chat_prompt("hi"))
            assert usage["completion_tokens"] == 6
            assert usage["total_tokens"] == usage["prompt_tokens"] + 6

        _run(checkpoint, body)

    def test_a_stream_says_the_same_text_and_ends_with_usage_and_done(
        self, checkpoint: TinyCheckpoint
    ) -> None:
        expected = _expected_chat(checkpoint, "hi", 8)

        async def body(client: Client, serving: Serving, llm: AsyncLLM) -> None:
            request = _chat("hi", 8, stream=True, stream_options={"include_usage": True})
            response = await client.post(_CHAT, json=request)
            assert response.status == 200
            assert response.headers["Content-Type"].startswith("text/event-stream")
            frames = await read_sse(response)
            chunks = [data for _, data in frames[:-1]]
            assert frames[-1] == (None, "[DONE]")
            assert chunks[0]["choices"][0]["delta"] == {"role": "assistant", "content": ""}
            text = "".join(c["choices"][0]["delta"].get("content", "") for c in chunks[1:-2])
            assert text == expected
            assert chunks[-2]["choices"][0]["finish_reason"] == "length"
            assert chunks[-1]["choices"] == []
            assert chunks[-1]["usage"]["completion_tokens"] == 8
            assert {c["id"] for c in chunks} == {chunks[0]["id"]}

        _run(checkpoint, body)

    def test_a_stop_string_over_two_tokens_and_two_frames_is_cut(
        self, checkpoint: TinyCheckpoint
    ) -> None:
        text = _expected_chat(checkpoint, "hi", 12)
        at = next(i for i in range(3, 10) if text.find(text[i : i + 2]) == i)
        stop = text[at : at + 2]

        async def body(client: Client, serving: Serving, llm: AsyncLLM) -> None:
            request = _chat(
                "hi", 12, stop=[stop], stream=True, stream_options={"include_usage": True}
            )
            frames = await read_sse(await client.post(_CHAT, json=request))
            chunks = [data for _, data in frames[:-1]]
            pieces = [c["choices"][0]["delta"].get("content", "") for c in chunks[1:-2]]
            assert "".join(pieces) == text[: text.index(stop)]
            assert chunks[-2]["choices"][0]["finish_reason"] == "stop"
            # Every token up to the one that completed the stop string was generated.
            assert chunks[-1]["usage"]["completion_tokens"] == text.index(stop) + 2

        _run(checkpoint, body)

    def test_two_requests_at_once_both_finish(self, checkpoint: TinyCheckpoint) -> None:
        first, second = _expected_chat(checkpoint, "one", 7), _expected_chat(checkpoint, "two", 9)

        async def body(client: Client, serving: Serving, llm: AsyncLLM) -> None:
            a, b = await asyncio.gather(
                client.post(_CHAT, json=_chat("one", 7)), client.post(_CHAT, json=_chat("two", 9))
            )
            assert (await a.json())["choices"][0]["message"]["content"] == first
            assert (await b.json())["choices"][0]["message"]["content"] == second

        _run(checkpoint, body)


class TestCompletions:
    def test_a_text_prompt_and_a_token_prompt(self, checkpoint: TinyCheckpoint) -> None:
        prompt = [256, *b"once"]
        expected = _FAKE.decode(greedy_reference(checkpoint.state, checkpoint.config, prompt, 5))

        async def body(client: Client, serving: Serving, llm: AsyncLLM) -> None:
            by_text = await client.post(
                "/v1/completions", json={"prompt": "once", "max_tokens": 5, "temperature": 0}
            )
            answer = await by_text.json()
            assert answer["object"] == "text_completion"
            assert answer["choices"][0]["text"] == expected
            by_ids = await client.post(
                "/v1/completions",
                json={"prompt": prompt, "max_tokens": 5, "temperature": 0, "stream": True},
            )
            frames = await read_sse(by_ids)
            text = "".join(data["choices"][0]["text"] for _, data in frames[:-1])
            assert text == expected
            assert frames[-2][1]["choices"][0]["finish_reason"] == "length"

        _run(checkpoint, body)


class TestErrors:
    @pytest.mark.parametrize(
        ("request_body", "status", "code", "param"),
        [
            ({"messages": []}, 400, "invalid_request", "messages"),
            ({"messages": [{"role": "user", "content": 3}]}, 400, "invalid_request", None),
            (_chat("x", 2, top_logprobs=50, logprobs=True), 400, "invalid_request", None),
            (
                _chat("x", 2, response_format={"type": "yaml"}),
                400,
                "invalid_request",
                "response_format",
            ),
            (_chat("x", 2, tool_choice="required"), 400, "invalid_request", "tool_choice"),
            (
                _chat("x", 2, tools=[{"type": "function", "function": {"name": "f"}}]),
                400,
                "unsupported_feature",
                "tools",
            ),
            (
                {"messages": [{"role": "user", "content": [{"type": "image_url"}]}]},
                400,
                "unsupported_feature",
                None,
            ),
            (_chat("x", 2, model="gpt-4o"), 404, "not_found", None),
            (_chat("x", 2, temperature="hot"), 400, "invalid_request", "temperature"),
        ],
    )
    def test_a_refused_request_gets_an_openai_error(
        self,
        checkpoint: TinyCheckpoint,
        request_body: dict[str, Any],
        status: int,
        code: str,
        param: str | None,
    ) -> None:
        async def body(client: Client, serving: Serving, llm: AsyncLLM) -> None:
            response = await client.post(_CHAT, json=request_body)
            assert response.status == status
            error = (await response.json())["error"]
            assert error["code"] == code
            assert error["message"]
            if param is not None:
                assert error["param"] == param

        _run(checkpoint, body)

    def test_a_body_that_is_not_json_or_too_large(self, checkpoint: TinyCheckpoint) -> None:
        async def body(client: Client, serving: Serving, llm: AsyncLLM) -> None:
            bad = await client.post(_CHAT, data=b"{not json")
            assert bad.status == 400
            assert "not valid JSON" in (await bad.json())["error"]["message"]
            big = await client.post(_CHAT, data=b"x" * 4096)
            assert big.status == 413
            assert (await big.json())["error"]["type"] == "invalid_request_error"

        _run(checkpoint, body, config=ServerConfig(max_body_bytes=2048))

    def test_a_prompt_longer_than_the_context_says_the_limit(
        self, checkpoint: TinyCheckpoint
    ) -> None:
        async def body(client: Client, serving: Serving, llm: AsyncLLM) -> None:
            response = await client.post(_CHAT, json=_chat("x" * 400, 2))
            assert response.status == 400
            assert "128 tokens" in (await response.json())["error"]["message"]

        _run(checkpoint, body)

    def test_a_large_max_tokens_is_capped_by_the_context(self, checkpoint: TinyCheckpoint) -> None:
        async def body(client: Client, serving: Serving, llm: AsyncLLM) -> None:
            response = await client.post(_CHAT, json=_chat("hi", 100_000))
            assert response.status == 200
            answer = await response.json()
            assert answer["choices"][0]["finish_reason"] == "length"
            used = answer["usage"]
            assert used["prompt_tokens"] + used["completion_tokens"] == 128

        _run(checkpoint, body)

    def test_an_overloaded_engine_answers_429_with_retry_after(
        self, checkpoint: TinyCheckpoint
    ) -> None:
        async def body(client: Client, serving: Serving, llm: AsyncLLM) -> None:
            response = await client.post(_CHAT, json=_chat("x", 2))
            assert response.status == 429
            assert response.headers["Retry-After"] == "1"
            assert (await response.json())["error"]["type"] == "rate_limit_error"

        _run(checkpoint, body, scripts=[OverloadedError("the waiting queue is full")])

    def test_an_engine_that_dies_mid_stream_ends_it_with_an_error_and_done(
        self, checkpoint: TinyCheckpoint
    ) -> None:
        async def body(client: Client, serving: Serving, llm: FakeEngine) -> None:
            # The script has no final output: the request hangs until the engine dies.
            sent = asyncio.ensure_future(client.post(_CHAT, json=_chat("x", 50, stream=True)))
            while not llm.requests:  # noqa: ASYNC110
                await asyncio.sleep(0.01)
            llm.die()
            frames = await read_sse(await sent)
            assert frames[-1] == (None, "[DONE]")
            error = frames[-2][1]["error"]
            assert error["code"] == "engine_dead"
            assert error["type"] == "service_unavailable_error"
            health = await client.get("/health")
            assert health.status == 503
            assert (await health.json()) == {"status": "dead"}
            refused = await client.post(_CHAT, json=_chat("x", 2))
            assert refused.status == 503

        _run(checkpoint, body, scripts=[script("one", "two")[:-1]])


class TestKeysAndCors:
    def test_a_key_is_needed_everywhere_but_health(self, checkpoint: TinyCheckpoint) -> None:
        async def body(client: Client, serving: Serving, llm: AsyncLLM) -> None:
            assert (await client.get("/health")).status == 200
            missing = await client.post(_CHAT, json=_chat("x", 1))
            assert missing.status == 401
            assert (await missing.json())["error"]["type"] == "authentication_error"
            wrong = await client.get("/v1/models", headers={"Authorization": "Bearer nope"})
            assert wrong.status == 401
            bearer = await client.get("/v1/models", headers={"Authorization": "Bearer s3cret"})
            assert bearer.status == 200
            header = await client.get("/metrics", headers={"x-api-key": "s3cret"})
            assert header.status == 200

        _run(checkpoint, body, config=ServerConfig(api_key="s3cret"))

    def test_cors_answers_preflight_and_tags_allowed_origins(
        self, checkpoint: TinyCheckpoint
    ) -> None:
        async def body(client: Client, serving: Serving, llm: AsyncLLM) -> None:
            origin = {"Origin": "http://app.local"}
            preflight = await client.options(
                _CHAT, headers={**origin, "Access-Control-Request-Method": "POST"}
            )
            assert preflight.status == 204
            assert preflight.headers["Access-Control-Allow-Origin"] == "http://app.local"
            other = await client.get("/health", headers={"Origin": "http://evil.local"})
            assert "Access-Control-Allow-Origin" not in other.headers

        _run(checkpoint, body, config=ServerConfig(cors_origins=("http://app.local",)))


class TestStateEndpoints:
    def test_health_is_ready_only_once_an_engine_is_attached(
        self, checkpoint: TinyCheckpoint
    ) -> None:
        async def body(client: Client, serving: Serving, llm: AsyncLLM) -> None:
            starting = await client.get("/health")
            assert starting.status == 503
            assert await starting.json() == {"status": "starting"}
            refused = await client.post(_CHAT, json=_chat("x", 1))
            assert refused.status == 503
            assert refused.headers["Retry-After"] == "1"
            serving.attach(llm, llm.engine.tokenizer)
            assert (await client.get("/health")).status == 200
            serving.begin_draining()
            draining = await client.get("/health")
            assert await draining.json() == {"status": "draining"}
            assert (await client.post(_CHAT, json=_chat("x", 1))).status == 503

        _run(checkpoint, body, attach=False)

    def test_models_lists_every_served_name(self, checkpoint: TinyCheckpoint) -> None:
        async def body(client: Client, serving: Serving, llm: AsyncLLM) -> None:
            listing = await (await client.get("/v1/models")).json()
            assert [model["id"] for model in listing["data"]] == ["tiny-a", "tiny-b"]
            assert listing["data"][0]["max_model_len"] == 128
            assert "suggested_tool_call_parser" not in listing["data"][0]
            ok = await client.post(_CHAT, json=_chat("x", 1, model="tiny-b"))
            assert ok.status == 200

        _run(checkpoint, body, config=ServerConfig(served_model_names=("tiny-a", "tiny-b")))

    def test_models_suggests_a_tool_call_parser_from_the_chat_template(
        self, checkpoint: TinyCheckpoint
    ) -> None:
        class _LlamaTokenizer(FakeTokenizer):
            chat_template = (
                "<|start_header_id|>user<|end_header_id|>\n"
                'Respond in the format {"name": function name, "parameters": dict}.\n'
                "<|eot_id|>"
            )

        async def body(client: Client, serving: Serving, llm: AsyncLLM) -> None:
            listing = await (await client.get("/v1/models")).json()
            assert listing["data"][0]["suggested_tool_call_parser"] == "llama3_json"

        _run(checkpoint, body, tokenizer=loaded_for(_LlamaTokenizer()))

    def test_metrics_count_requests_without_per_request_labels(
        self, checkpoint: TinyCheckpoint
    ) -> None:
        async def body(client: Client, serving: Serving, llm: AsyncLLM) -> None:
            await client.post(_CHAT, json=_chat("x", 3))
            response = await client.get("/metrics")
            assert response.headers["Content-Type"].startswith("text/plain")
            text = await response.text()
            assert (
                'ayaka_http_requests_total{route="/v1/chat/completions",status="200"} 1.0' in text
            )
            assert "ayaka_generation_tokens_total 3.0" in text
            assert 'ayaka_requests_finished_total{outcome="length"} 1.0' in text
            assert "ayaka-" not in text  # no request id in any label

        _run(checkpoint, body)


class TestStalls:
    def test_a_stalled_engine_turns_health_to_503_and_back(
        self, checkpoint: TinyCheckpoint
    ) -> None:
        async def body(client: Client, serving: Serving, llm: FakeEngine) -> None:
            assert (await client.get("/health")).status == 200
            llm.stall()
            await asyncio.sleep(0.2)
            stalled = await client.get("/health")
            assert stalled.status == 503
            assert await stalled.json() == {"status": "stalled"}
            age = _gauge(await (await client.get("/metrics")).text())
            assert age > 0.1
            llm.resume()
            ready = await client.get("/health")
            assert ready.status == 200
            assert await ready.json() == {"status": "ready"}
            assert _gauge(await (await client.get("/metrics")).text()) == 0.0

        _run(checkpoint, body, scripts=[], config=ServerConfig(stall_seconds=0.1))

    def test_an_idle_engine_is_never_stalled(self, checkpoint: TinyCheckpoint) -> None:
        async def body(client: Client, serving: Serving, llm: AsyncLLM) -> None:
            assert (await client.post(_CHAT, json=_chat("x", 2))).status == 200
            assert await llm.drain(5.0)
            await asyncio.sleep(0.2)  # idle for longer than the limit
            health = await client.get("/health")
            assert health.status == 200
            assert _gauge(await (await client.get("/metrics")).text()) == 0.0

        _run(checkpoint, body, config=ServerConfig(stall_seconds=0.1))


def _gauge(metrics: str) -> float:
    [line] = [
        line
        for line in metrics.splitlines()
        if line.startswith("ayaka_engine_progress_age_seconds ")
    ]
    return float(line.split()[1])


class TestShutdown:
    def test_a_client_that_stopped_reading_does_not_hold_shutdown(
        self, checkpoint: TinyCheckpoint
    ) -> None:
        # 64 fragments of 256 KiB: far more than the socket buffers of both ends hold, so the
        # write really blocks once the client stops reading. The script has no final output.
        hanging = script(*[(120, "x" * 262_144)] * 64)[:-1]

        async def body(client: Client, serving: Serving, llm: FakeEngine) -> None:
            server = client.server
            reader, writer = await _stuck_client(str(server.host), int(server.port or 0))
            try:
                payload = json.dumps(_chat("x", 50, stream=True)).encode()
                writer.write(
                    b"POST /v1/chat/completions HTTP/1.1\r\nHost: x\r\n"
                    b"Content-Type: application/json\r\n"
                    + f"Content-Length: {len(payload)}\r\n\r\n".encode()
                    + payload
                )
                await writer.drain()
                while not llm.requests:  # noqa: ASYNC110
                    await asyncio.sleep(0.01)
                await asyncio.sleep(1.0)  # the fragments fill the buffers; nothing is read
                started = time.monotonic()
                cut = await serving.shutdown(0.0, flush=0.5)
                waited = time.monotonic() - started
                assert cut == 1  # the stream was still blocked once the engine had ended it
                assert waited < 0.5 + 2.0
                assert (await client.get("/health")).status == 503
            finally:
                writer.close()
            del reader

        _run(checkpoint, body, scripts=[hanging], config=ServerConfig(slow_client_seconds=60))


async def _stuck_client(host: str, port: int) -> tuple[asyncio.StreamReader, asyncio.StreamWriter]:
    """A connection with a small receive buffer whose owner never reads."""
    raw = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    raw.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 4096)
    raw.setblocking(False)
    await asyncio.get_running_loop().sock_connect(raw, (host, port))
    return await asyncio.open_connection(sock=raw, limit=4096)


class TestDisconnectsAndSlowClients:
    def test_a_client_that_leaves_mid_stream_frees_its_request(
        self, checkpoint: TinyCheckpoint
    ) -> None:
        async def body(client: Client, serving: Serving, llm: AsyncLLM) -> None:
            response = await client.post(_CHAT, json=_chat("x", 100, stream=True))
            await response.content.readuntil(b"\n\n")
            response.close()
            deadline = time.monotonic() + 10
            # The server notices the disconnect on its own task; poll until it has.
            while llm.num_streams and time.monotonic() < deadline:  # noqa: ASYNC110
                await asyncio.sleep(0.02)
            assert llm.num_streams == 0
            assert await llm.drain(5.0)
            assert_cache_released(llm.engine)

        _run(checkpoint, body)

    def test_a_request_that_ends_early_logs_the_tokens_it_generated(
        self, checkpoint: TinyCheckpoint, caplog: pytest.LogCaptureFixture
    ) -> None:
        async def body(client: Client, serving: Serving, llm: AsyncLLM) -> None:
            response = await client.post(_CHAT, json=_chat("x", 100, stream=True))
            frames = 0
            while frames < 4:  # a few text frames: the engine has generated tokens by then
                if b'"content"' in await response.content.readuntil(b"\n\n"):
                    frames += 1
            response.close()
            deadline = time.monotonic() + 10
            while llm.num_streams and time.monotonic() < deadline:  # noqa: ASYNC110
                await asyncio.sleep(0.02)

        with caplog.at_level(logging.INFO, logger="ayaka"):
            _run(checkpoint, body)

        lines = [r.getMessage() for r in caplog.records if "generated tokens" in r.getMessage()]
        assert len(lines) == 1
        match = re.search(r": client_gone, \d+ prompt and (\d+) generated tokens", lines[0])
        assert match is not None, lines[0]
        # Not 0: no usage event came, so the parser's count is logged. The first of the four
        # frames carries only the role, so at least three tokens reached the client.
        assert int(match.group(1)) >= 3

    def test_a_slow_reader_is_cancelled_and_others_keep_going(
        self, checkpoint: TinyCheckpoint
    ) -> None:
        engine = build_engine(tiny_engine_config(checkpoint.directory), tokenizer=tiny_tokenizer())
        llm = AsyncLLM(engine)
        prompt = (256, *b"fast")
        expected = greedy_reference(checkpoint.state, checkpoint.config, list(prompt), 30)

        class Stuck:
            def __init__(self) -> None:
                self.writes = 0

            async def write(self, data: bytes) -> None:
                self.writes += 1
                if self.writes > 1:
                    await asyncio.Event().wait()  # the socket never drains

        class Sink:
            def __init__(self) -> None:
                self.data = b""

            async def write(self, data: bytes) -> None:
                self.data += data

        from ayaka.entrypoints.openai.chat import CompletionStreamEncoder, _Meta  # pyright: ignore[reportPrivateUsage]

        def request(name: str, ids: tuple[int, ...]) -> GenerateRequest:
            sampling = SamplingParams(temperature=0, max_tokens=30)
            return GenerateRequest(request_id=RequestId(name), prompt=ids, sampling=sampling)

        async def main() -> None:
            llm.start()
            try:
                fmt = OutputFormat()
                slow = await llm.add(request("slow", (256, *b"slow")))
                fast = await llm.add(request("fast", prompt))
                meta = _Meta("x", 0, "tiny")

                async def slow_side() -> float:
                    started = time.monotonic()
                    with pytest.raises(SlowClientError):
                        await pump(
                            slow,
                            fmt.parser(call_id_prefix="c", tools=False),
                            CompletionStreamEncoder(meta, include_usage=False),
                            Stuck(),
                            keepalive=5,
                            write_deadline=0.3,
                        )
                    llm.abort(RequestId("slow"))
                    return time.monotonic() - started

                sink = Sink()

                async def fast_side() -> FinishEvent | ErrorEvent:
                    return await pump(
                        fast,
                        fmt.parser(call_id_prefix="c", tools=False),
                        CompletionStreamEncoder(meta, include_usage=False),
                        sink,
                        keepalive=5,
                        write_deadline=5,
                    )

                waited, terminal = await asyncio.gather(slow_side(), fast_side())
                assert waited < 3
                assert isinstance(terminal, FinishEvent)
                assert _text_of(sink.data) == _FAKE.decode(expected)
                assert await llm.drain(5.0)
                assert_cache_released(llm.engine)
            finally:
                await llm.shutdown()

        asyncio.run(main())


def _text_of(body: bytes) -> str:
    return "".join(
        json.loads(data)["choices"][0]["text"]
        for _, data in parse_sse(body.decode())
        if data != "[DONE]"
    )


def test_a_server_config_refuses_bad_values() -> None:
    with pytest.raises(ConfigError, match=r"server\.port"):
        ServerConfig(port=70000)
    with pytest.raises(ConfigError, match="UNKNOWN_KEY"):
        ServerConfig.from_mapping({"prot": 1})
    with pytest.raises(ConfigError, match=r"server\.stall_seconds"):
        ServerConfig(stall_seconds=0)
    config = ServerConfig.from_mapping({"cors_origins": ["*"], "api_key": "s3cret-key"})
    assert config.cors_origins == ("*",)
    assert "s3cret-key" not in config.canonical_json()
    assert "s3cret-key" not in repr(config)
    assert not ServerConfig(host="0.0.0.0").is_loopback  # noqa: S104  # the check under test
    assert ServerConfig().is_loopback
