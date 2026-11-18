"""The server, end to end, with the official OpenAI and Anthropic SDKs and real signals.

The SDK tests run the server in this process on a localhost port, on the tiny GPT-2 (text) or on
scripted engine output (tool calls and reasoning, which the tiny model does not write), and talk
to it with ``AsyncOpenAI`` and ``AsyncAnthropic``. Whatever the SDKs send and parse is what a coding
tool built on them sends and parses.

The signal tests start ``ayaka serve`` in a child process and send it SIGINT and SIGTERM; they need
POSIX signals and skip on Windows.

Needs the ``serve`` extra and the ``sdk`` dependency group (CI job ``serving``)::

    uv sync --extra cpu --extra serve --group sdk
    uv run pytest -m integration tests/e2e/test_serving.py
"""

from __future__ import annotations

import asyncio
import json
import os
import signal
import socket
import subprocess
import sys
import textwrap
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest

from ayaka.configs.server import ServerConfig
from tokenizer_fakes import FakeTokenizer

pytest.importorskip("torch")
pytest.importorskip("aiohttp")
openai = pytest.importorskip("openai")
anthropic = pytest.importorskip("anthropic")

from ayaka.entrypoints.parsers.output import OutputFormat  # noqa: E402
from serving_support import script, serve  # noqa: E402
from tiny_model import TinyCheckpoint, greedy_reference, write_tiny_gpt2  # noqa: E402

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from aiohttp.test_utils import TestClient

pytestmark = pytest.mark.integration

_FAKE = FakeTokenizer()
_TOOLS = OutputFormat.resolve(reasoning="think", tool_calls="hermes")
_KEY = "test-key"
_SCHEMA = {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}
_CALL = script(
    "<think>",
    "need the weather",
    "</think>",
    "<tool_call>",
    '{"name": "get_weather", ',
    '"arguments": {"city": "Hue"}}',
    "</tool_call>",
)


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory: pytest.TempPathFactory) -> TinyCheckpoint:
    return write_tiny_gpt2(tmp_path_factory.mktemp("tiny-gpt2"))


def _expected(checkpoint: TinyCheckpoint, text: str, count: int) -> str:
    prompt = list(f"<|im_start|>user\n{text}<|im_end|>\n<|im_start|>assistant\n".encode())
    return _FAKE.decode(greedy_reference(checkpoint.state, checkpoint.config, prompt, count))


type Body = Callable[[str], Awaitable[None]]


_CLIENTS: list[Any] = []


def _sdk(kind: Any, **options: Any) -> Any:
    """An SDK client that ``_run`` closes when the test body ends."""
    client = kind(**options)
    _CLIENTS.append(client)
    return client


def _run(checkpoint: TinyCheckpoint, body: Body, **options: Any) -> None:
    async def main() -> None:
        config = ServerConfig(api_key=_KEY, served_model_names=("tiny",))
        async with serve(checkpoint.directory, config=config, **options) as (client, _, _):
            try:
                await body(_base(client))
            finally:
                while _CLIENTS:
                    await _CLIENTS.pop().close()

    asyncio.run(main())


def _base(client: TestClient[Any, Any]) -> str:
    return str(client.make_url("")).rstrip("/")


class TestOpenAISdk:
    def test_two_choices_whole_and_streamed(self, checkpoint: TinyCheckpoint) -> None:
        async def body(base: str) -> None:
            client = _sdk(openai.AsyncOpenAI, base_url=f"{base}/v1", api_key=_KEY)
            options: dict[str, Any] = {
                "model": "tiny",
                "messages": [{"role": "user", "content": "hi"}],
                "n": 2,
                "temperature": 1.0,
                "seed": 5,
                "max_tokens": 5,
                # top_k keeps the draws among tokens the fixture's tokenizer decodes.
                "extra_body": {"top_k": 4, "min_tokens": 5},
            }
            whole = await client.chat.completions.create(**options)
            assert [choice.index for choice in whole.choices] == [0, 1]
            assert whole.usage is not None
            assert whole.usage.completion_tokens == 10

            stream = await client.chat.completions.create(
                **options, stream=True, stream_options={"include_usage": True}
            )
            text = {0: "", 1: ""}
            usage = None
            async for chunk in stream:
                if chunk.usage is not None:
                    usage = chunk.usage
                for choice in chunk.choices:
                    text[choice.index] += choice.delta.content or ""
            assert text[0] == whole.choices[0].message.content
            assert text[1] == whole.choices[1].message.content
            assert usage is not None
            assert usage.prompt_tokens == whole.usage.prompt_tokens

        _run(checkpoint, body)

    def test_a_repeated_prompt_reports_cached_tokens(self, checkpoint: TinyCheckpoint) -> None:
        """Two requests at once both finish; a third with the same prefix reports a cache hit."""
        text = "tell me about the printing press and the history of books"
        expected = _expected(checkpoint, text, 4)

        async def body(base: str) -> None:
            client = _sdk(openai.AsyncOpenAI, base_url=f"{base}/v1", api_key=_KEY)

            async def ask() -> Any:
                return await client.chat.completions.create(
                    model="tiny",
                    messages=[{"role": "user", "content": text}],
                    max_tokens=4,
                    temperature=0,
                )

            together = await asyncio.gather(ask(), ask())
            again = await ask()
            for answer in (*together, again):
                assert answer.choices[0].message.content == expected
            assert again.usage.prompt_tokens_details.cached_tokens > 0
            assert again.usage.prompt_tokens_details.cached_tokens < again.usage.prompt_tokens

        _run(checkpoint, body)

    def test_chat_completions_stream_text(self, checkpoint: TinyCheckpoint) -> None:
        expected = _expected(checkpoint, "hi", 8)

        async def body(base: str) -> None:
            client = _sdk(openai.AsyncOpenAI, base_url=f"{base}/v1", api_key=_KEY)
            stream = await client.chat.completions.create(
                model="tiny",
                messages=[{"role": "user", "content": "hi"}],
                max_tokens=8,
                temperature=0,
                stream=True,
                stream_options={"include_usage": True},
            )
            text, finish, usage = "", None, None
            async for chunk in stream:
                if chunk.usage is not None:
                    usage = chunk.usage
                for choice in chunk.choices:
                    text += choice.delta.content or ""
                    finish = choice.finish_reason or finish
            assert text == expected
            assert finish == "length"
            assert usage is not None
            assert usage.completion_tokens == 8
            whole = await client.chat.completions.create(
                model="tiny",
                messages=[{"role": "user", "content": "hi"}],
                max_tokens=8,
                temperature=0,
            )
            assert whole.choices[0].message.content == expected

        _run(checkpoint, body)

    def test_chat_completions_stream_a_tool_call(self, checkpoint: TinyCheckpoint) -> None:
        async def body(base: str) -> None:
            client = _sdk(openai.AsyncOpenAI, base_url=f"{base}/v1", api_key=_KEY)
            stream = await client.chat.completions.create(
                model="tiny",
                messages=[{"role": "user", "content": "weather in Hue?"}],
                tools=[
                    {
                        "type": "function",
                        "function": {"name": "get_weather", "parameters": _SCHEMA},
                    }
                ],
                stream=True,
            )
            name, arguments, call_id, finish = "", "", "", None
            async for chunk in stream:
                for choice in chunk.choices:
                    finish = choice.finish_reason or finish
                    for call in choice.delta.tool_calls or ():
                        call_id = call.id or call_id
                        if call.function is not None:
                            name += call.function.name or ""
                            arguments += call.function.arguments or ""
            assert (name, json.loads(arguments), finish) == (
                "get_weather",
                {"city": "Hue"},
                "tool_calls",
            )
            assert call_id.startswith("call_")

        _run(checkpoint, body, output_format=_TOOLS, scripts=[_CALL])

    def test_responses_stream_text_and_a_function_call(self, checkpoint: TinyCheckpoint) -> None:
        expected = _expected(checkpoint, "hi", 6)

        async def text_body(base: str) -> None:
            client = _sdk(openai.AsyncOpenAI, base_url=f"{base}/v1", api_key=_KEY)
            stream = await client.responses.create(
                model="tiny", input="hi", max_output_tokens=6, temperature=0, stream=True
            )
            deltas: list[str] = []
            final: Any = None
            async for event in stream:
                if event.type == "response.output_text.delta":
                    deltas.append(event.delta)
                if event.type in ("response.completed", "response.incomplete"):
                    final = event.response
            assert "".join(deltas) == expected
            assert final is not None
            assert final.output_text == expected
            assert final.usage.output_tokens == 6

        _run(checkpoint, text_body)

        async def call_body(base: str) -> None:
            client = _sdk(openai.AsyncOpenAI, base_url=f"{base}/v1", api_key=_KEY)
            stream = await client.responses.create(
                model="tiny",
                input=[{"role": "user", "content": "weather in Hue?"}],
                tools=[
                    {
                        "type": "function",
                        "name": "get_weather",
                        "parameters": _SCHEMA,
                        "strict": False,
                    }
                ],
                store=False,
                stream=True,
            )
            final: Any = None
            async for event in stream:
                if event.type == "response.completed":
                    final = event.response
            assert final is not None
            calls = [item for item in final.output if item.type == "function_call"]
            assert [(c.name, json.loads(c.arguments)) for c in calls] == [
                ("get_weather", {"city": "Hue"})
            ]

        _run(checkpoint, call_body, output_format=_TOOLS, scripts=[_CALL])

    def test_a_refused_request_and_a_wrong_key_raise_the_sdk_errors(
        self, checkpoint: TinyCheckpoint
    ) -> None:
        async def body(base: str) -> None:
            client = _sdk(openai.AsyncOpenAI, base_url=f"{base}/v1", api_key=_KEY)
            with pytest.raises(openai.BadRequestError) as caught:
                await client.chat.completions.create(
                    model="tiny",
                    messages=[{"role": "user", "content": "x"}],
                    response_format={"type": "json_object"},
                )
            assert caught.value.code == "unsupported_feature"
            wrong = _sdk(openai.AsyncOpenAI, base_url=f"{base}/v1", api_key="nope", max_retries=0)
            with pytest.raises(openai.AuthenticationError):
                await wrong.models.list()
            models = await client.models.list()
            assert [model.id for model in models.data] == ["tiny"]

        _run(checkpoint, body)


class TestAnthropicSdk:
    def test_messages_stream_text_and_count_tokens(self, checkpoint: TinyCheckpoint) -> None:
        expected = _expected(checkpoint, "hi", 6)

        async def body(base: str) -> None:
            client = _sdk(anthropic.AsyncAnthropic, base_url=base, api_key=_KEY)
            messages = [{"role": "user", "content": "hi"}]
            async with client.messages.stream(
                model="tiny", max_tokens=6, messages=messages, extra_body={"temperature": 0}
            ) as stream:
                text = "".join([piece async for piece in stream.text_stream])
                final = await stream.get_final_message()
            assert text == expected
            assert final.stop_reason == "max_tokens"
            assert final.usage.output_tokens == 6
            count = await client.messages.count_tokens(model="tiny", messages=messages)
            assert count.input_tokens == final.usage.input_tokens

        _run(checkpoint, body)

    def test_messages_stream_a_tool_use_with_thinking(self, checkpoint: TinyCheckpoint) -> None:
        async def body(base: str) -> None:
            client = _sdk(anthropic.AsyncAnthropic, base_url=base, api_key=_KEY)
            async with client.messages.stream(
                model="tiny",
                max_tokens=2048,
                messages=[{"role": "user", "content": "weather in Hue?"}],
                tools=[{"name": "get_weather", "input_schema": _SCHEMA}],
                thinking={"type": "enabled", "budget_tokens": 1024},
            ) as stream:
                final = await stream.get_final_message()
            assert [block.type for block in final.content] == ["thinking", "tool_use"]
            assert final.content[0].thinking == "need the weather"
            tool = final.content[1]
            assert (tool.name, tool.input) == ("get_weather", {"city": "Hue"})
            assert final.stop_reason == "tool_use"

        _run(checkpoint, body, output_format=_TOOLS, scripts=[_CALL])

    def test_a_refused_request_raises_the_sdk_error(self, checkpoint: TinyCheckpoint) -> None:
        async def body(base: str) -> None:
            client = _sdk(anthropic.AsyncAnthropic, base_url=base, api_key=_KEY, max_retries=0)
            with pytest.raises(anthropic.BadRequestError) as caught:
                await client.messages.create(
                    model="tiny",
                    max_tokens=8,
                    messages=[{"role": "user", "content": "x"}],
                    tool_choice={"type": "any"},
                    tools=[{"name": "f", "input_schema": {"type": "object"}}],
                )
            assert "forcing a tool call" in str(caught.value)

        _run(checkpoint, body)


# -- real signals, in a child process ---------------------------------------------------------

_CHILD = textwrap.dedent(
    """
    import sys, time
    sys.path.insert(0, {support!r})
    from ayaka.execution.runner import ModelRunner
    from ayaka.tokenizers.factory import DefaultTokenizerFactory
    from tiny_model import tiny_tokenizer

    DefaultTokenizerFactory.from_config = classmethod(lambda cls, config: tiny_tokenizer())
    real = ModelRunner.execute

    def slowed(self, step, drafts=None):
        time.sleep(0.02)  # long enough for a signal to land while a request runs
        return real(self, step, drafts)

    ModelRunner.execute = slowed
    from ayaka.entrypoints.cli import main
    sys.exit(main(sys.argv[1:]))
    """
)

_posix = pytest.mark.skipif(sys.platform == "win32", reason="needs POSIX signals")


def _free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


class _Child:
    def __init__(self, checkpoint: TinyCheckpoint, tmp_path: Path) -> None:
        support = str(Path(__file__).resolve().parents[1] / "_support")
        script_path = tmp_path / "child.py"
        script_path.write_text(_CHILD.format(support=support))
        self.port = _free_port()
        self.url = f"http://127.0.0.1:{self.port}"
        self.process = subprocess.Popen(  # noqa: S603
            [
                sys.executable,
                str(script_path),
                "serve",
                str(checkpoint.directory),
                "--port",
                str(self.port),
                "--max-running",
                "2",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        deadline = time.monotonic() + 120
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                raise AssertionError(f"the server exited: {self.process.communicate()}")
            if _status(f"{self.url}/health") == 200:
                return
            time.sleep(0.1)
        self.process.kill()
        raise AssertionError("the server did not get ready")

    def signal(self, number: int) -> None:
        os.kill(self.process.pid, number)

    def wait(self) -> tuple[int, str]:
        _, err = self.process.communicate(timeout=60)
        return self.process.returncode, err


def _status(url: str, body: dict[str, Any] | None = None) -> int:
    data = json.dumps(body).encode() if body is not None else None
    request = urllib.request.Request(  # noqa: S310  # a local test server
        url, data=data, headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(request, timeout=5) as response:  # noqa: S310
            return int(response.status)
    except urllib.error.HTTPError as exc:
        return exc.code
    except OSError:
        return 0


def _stream_in_background(url: str) -> tuple[threading.Thread, list[str]]:
    body = {
        "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 60,
        "temperature": 0,
        "stream": True,
    }
    received: list[str] = []

    def read() -> None:
        request = urllib.request.Request(  # noqa: S310  # a local test server
            f"{url}/v1/chat/completions",
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(request, timeout=60) as response:  # noqa: S310
                received.extend(line.decode() for line in response)
        except OSError as exc:
            received.append(f"failed: {exc}")

    thread = threading.Thread(target=read)
    thread.start()
    deadline = time.monotonic() + 30
    while not received and time.monotonic() < deadline:
        time.sleep(0.01)
    return thread, received


@_posix
@pytest.mark.parametrize("number", [signal.SIGINT, signal.SIGTERM], ids=["SIGINT", "SIGTERM"])
def test_a_signal_drains_requests_in_flight_then_exits_cleanly(
    checkpoint: TinyCheckpoint, tmp_path: Path, number: int
) -> None:
    child = _Child(checkpoint, tmp_path)
    thread, received = _stream_in_background(child.url)

    child.signal(number)
    time.sleep(0.3)
    refused = _status(
        f"{child.url}/v1/chat/completions",
        {"messages": [{"role": "user", "content": "x"}], "max_tokens": 1},
    )
    thread.join(60)
    code, err = child.wait()

    assert refused in (503, 0)  # 503 while draining; 0 if it already closed
    assert code == 0, err
    assert "stopping" in err
    assert "stopped" in err
    assert any('"finish_reason":"length"' in line for line in received), received
    assert received[-2].strip() == "data: [DONE]" or received[-1].strip() == "data: [DONE]"


@_posix
def test_a_second_signal_exits_at_once(checkpoint: TinyCheckpoint, tmp_path: Path) -> None:
    child = _Child(checkpoint, tmp_path)
    thread, _ = _stream_in_background(child.url)

    started = time.monotonic()
    child.signal(signal.SIGINT)
    time.sleep(0.1)
    child.signal(signal.SIGINT)
    code, _ = child.wait()
    thread.join(10)

    assert code == 130
    assert time.monotonic() - started < 1.0
