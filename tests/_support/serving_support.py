"""A server on the tiny GPT-2, a scripted engine, and an SSE reader, for the serving tests.

* ``serve(...)``: an async context manager that builds the tiny engine, starts an ``AsyncLLM``,
  attaches it to a ``Serving`` and yields an aiohttp test client for ``create_app``. Requests go
  through real HTTP on localhost.
* ``ScriptedLLM``: an ``AsyncLLM`` whose requests are answered with outputs a test wrote, while
  tokenizing goes through the real tokenizer service. It records every admitted request, so a
  test can compare prompt ids, and every abort. Use it where the tiny model cannot produce what
  the test needs (a tool call, a reasoning block).
* ``read_sse(body)``: the frames of an SSE body as ``(event, data)`` pairs.
* ``write_long_gpt2(directory)``: the tiny GPT-2 with 1024 positions, for prompts with a tool
  history that the 128-position fixture cannot hold (only tokenized, never run, in these tests).

Import this module only after ``pytest.importorskip("torch")`` and ``("aiohttp")``.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
from typing import TYPE_CHECKING, Any

from aiohttp.test_utils import TestClient, TestServer

from ayaka.configs.server import ServerConfig
from ayaka.core.outputs import CompletionOutput, RequestOutput, Usage
from ayaka.core.types import FinishReason
from ayaka.engine.async_llm import AsyncLLM, RequestStream
from ayaka.engine.build import build_engine
from ayaka.entrypoints.http import create_app
from ayaka.entrypoints.serving import Serving
from gpt2_support import hf_state, tiny_config, write_checkpoint
from tiny_model import tiny_engine_config, tiny_tokenizer
from tokenizer_fakes import MODEL_VOCAB

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Sequence
    from pathlib import Path

    from ayaka.core.request import GenerateRequest
    from ayaka.core.types import RequestId
    from ayaka.engine.core import EngineCore
    from ayaka.entrypoints.parsers.output import OutputFormat
    from ayaka.tokenizers.contracts import LoadedTokenizer

__all__ = ["ScriptedLLM", "parse_sse", "read_sse", "script", "serve", "write_long_gpt2"]


def write_long_gpt2(directory: Path, *, positions: int = 1024) -> Path:
    """Write a tiny GPT-2 with a longer context to ``directory`` and return it."""
    config = tiny_config(vocab_size=MODEL_VOCAB, max_position_embeddings=positions)
    write_checkpoint(directory, hf_state(config), config)
    return directory


def script(
    *pieces: str | tuple[int, str],
    reason: FinishReason = FinishReason.STOP,
    stop: str | int | None = None,
) -> list[RequestOutput]:
    """Outputs for a scripted request: one fragment per piece, then the final one.

    A piece is text (its token ids are its bytes) or ``(token id, text)``.
    """
    fragments: list[tuple[tuple[int, ...], str]] = []
    for piece in pieces:
        if isinstance(piece, tuple):
            fragments.append(((piece[0],), piece[1]))
        else:
            fragments.append((tuple(piece.encode()), piece))
    count = sum(len(ids) for ids, _ in fragments)
    outputs: list[RequestOutput] = []
    for i, (ids, text) in enumerate(fragments):
        last = i == len(fragments) - 1
        outputs.append(
            RequestOutput(
                request_id="scripted",  # type: ignore[arg-type]  # replaced per request
                outputs=(
                    CompletionOutput(
                        index=0,
                        token_ids=ids,
                        text=text,
                        finish_reason=reason if last else None,
                        stop_reason=stop if last else None,
                    ),
                ),
                finished=last,
                usage=Usage(prompt_tokens=0, completion_tokens=count if last else 0),
            )
        )
    return outputs


class ScriptedLLM(AsyncLLM):
    """Answers each request with the next script; tokenizing uses the real engine's tokenizer.

    Args:
        engine: A built engine (only its tokenizer and configuration are used).
        scripts: One list of outputs per expected request, in order.
    """

    def __init__(self, engine: EngineCore, scripts: Sequence[list[RequestOutput]] = ()) -> None:
        super().__init__(engine)
        self.scripts = list(scripts)
        self.requests: list[GenerateRequest] = []
        self.aborted: list[RequestId] = []

    async def add_request(self, request: GenerateRequest) -> RequestStream:
        self.requests.append(request)
        outputs = self.scripts.pop(0) if self.scripts else script("ok")
        stream = RequestStream(request.request_id)
        prompt = len(request.prompt) if isinstance(request.prompt, tuple) else 0

        async def feed() -> None:
            for output in outputs:
                await asyncio.sleep(0.005)
                stream.put(
                    RequestOutput(
                        request_id=request.request_id,
                        outputs=output.outputs,
                        finished=output.finished,
                        usage=Usage(
                            prompt_tokens=prompt, completion_tokens=output.usage.completion_tokens
                        ),
                        error=output.error,
                    )
                )

        task = asyncio.get_running_loop().create_task(feed())
        stream.feeder = task  # type: ignore[attr-defined]  # keep a reference to the task
        return stream

    def abort(self, request_id: RequestId) -> None:
        self.aborted.append(request_id)


@contextlib.asynccontextmanager
async def serve(
    directory: Path,
    *,
    config: ServerConfig | None = None,
    output_format: OutputFormat | None = None,
    tokenizer: LoadedTokenizer | None = None,
    scripts: Sequence[list[RequestOutput]] | None = None,
    attach: bool = True,
    **engine: Any,
) -> AsyncGenerator[tuple[TestClient[Any, Any], Serving, AsyncLLM]]:
    """A test client for a server on the tiny model (or on scripts, when given)."""
    core = build_engine(
        tiny_engine_config(directory, **engine), tokenizer=tokenizer or tiny_tokenizer()
    )
    llm: AsyncLLM = ScriptedLLM(core, scripts) if scripts is not None else AsyncLLM(core)
    llm.start()
    serving = Serving(config or ServerConfig(), model_name="tiny")
    if attach:
        serving.attach(llm, output_format=output_format)
    server = TestServer(create_app(serving), handler_cancellation=True)
    try:
        async with TestClient(server) as client:
            yield client, serving, llm
    finally:
        await llm.shutdown()


def parse_sse(body: str) -> list[tuple[str | None, str]]:
    """``(event, data)`` for each frame; comment-only frames give ``(":", comment)``."""
    frames: list[tuple[str | None, str]] = []
    for block in body.split("\n\n"):
        if not block.strip():
            continue
        event: str | None = None
        data: list[str] = []
        comment: str | None = None
        for line in block.split("\n"):
            if line.startswith(":"):
                comment = line[1:].strip()
            elif line.startswith("event:"):
                event = line[len("event:") :].strip()
            elif line.startswith("data:"):
                data.append(line[len("data:") :].strip())
        if data:
            frames.append((event, "\n".join(data)))
        elif comment is not None:
            frames.append((":", comment))
    return frames


async def read_sse(response: Any) -> list[tuple[str | None, Any]]:
    """The frames of a streamed response, with JSON data decoded (``[DONE]`` stays a string)."""
    body = await response.text()
    frames: list[tuple[str | None, Any]] = []
    for event, data in parse_sse(body):
        if event == ":" or data == "[DONE]":
            frames.append((event, data))
        else:
            frames.append((event, json.loads(data)))
    return frames
