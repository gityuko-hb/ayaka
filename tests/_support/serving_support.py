"""A server on the tiny GPT-2, a fake engine, and an SSE reader, for the serving tests.

* ``serve(...)``: an async context manager that builds the tiny engine, attaches it to a ``Serving``
  and yields an aiohttp test client for ``create_app``, the ``Serving`` and the engine client.
  Requests go through real HTTP on localhost. With ``scripts`` the engine client is a
  ``FakeEngine`` (``fake_engine.py``) that answers with outputs a test wrote, while tokenizing
  goes through the real tokenizer service; without, it is a real ``AsyncLLM`` on the tiny model.
  Use scripts where the tiny model cannot produce what the test needs (a tool call, a reasoning
  block) or where the test needs the engine to fail on cue.
* ``script(...)``: the outputs of one scripted request, from ``fake_engine``.
* ``read_sse(body)``: the frames of an SSE body as ``(event, data)`` pairs.
* ``write_long_gpt2(directory)``: the tiny GPT-2 with 1024 positions, for prompts with a tool
  history that the 128-position fixture cannot hold (only tokenized, never run, in these tests).

Import this module only after ``pytest.importorskip("torch")`` and ``("aiohttp")``.
"""

from __future__ import annotations

import contextlib
import json
from typing import TYPE_CHECKING, Any

from aiohttp.test_utils import TestClient, TestServer

from ayaka.configs.server import ServerConfig
from ayaka.engine.async_llm import AsyncLLM
from ayaka.engine.build import build_engine
from ayaka.entrypoints.http import create_app
from ayaka.entrypoints.serving import Serving
from fake_engine import FakeEngine, script
from gpt2_support import hf_state, tiny_config, write_checkpoint
from tiny_model import tiny_engine_config, tiny_tokenizer
from tokenizer_fakes import MODEL_VOCAB

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Sequence
    from pathlib import Path

    from ayaka.entrypoints.parsers.output import OutputFormat
    from ayaka.tokenizers.contracts import LoadedTokenizer
    from fake_engine import Script

__all__ = ["parse_sse", "read_sse", "script", "serve", "write_long_gpt2"]


def write_long_gpt2(directory: Path, *, positions: int = 1024) -> Path:
    """Write a tiny GPT-2 with a longer context to ``directory`` and return it."""
    config = tiny_config(vocab_size=MODEL_VOCAB, max_position_embeddings=positions)
    write_checkpoint(directory, hf_state(config), config)
    return directory


@contextlib.asynccontextmanager
async def serve(
    directory: Path,
    *,
    config: ServerConfig | None = None,
    output_format: OutputFormat | None = None,
    tokenizer: LoadedTokenizer | None = None,
    scripts: Sequence[Script] | None = None,
    attach: bool = True,
    **engine: Any,
) -> AsyncGenerator[tuple[TestClient[Any, Any], Serving, Any]]:
    """A test client for a server on the tiny model (or on a fake engine, when scripts are given).

    The third value is the engine client: an ``AsyncLLM``, or a ``FakeEngine`` with ``scripts``.
    """
    core = build_engine(
        tiny_engine_config(directory, **engine), tokenizer=tokenizer or tiny_tokenizer()
    )
    llm: AsyncLLM | FakeEngine
    if scripts is None:
        llm = AsyncLLM(core)
        llm.start()
    else:
        llm = FakeEngine(core.info, scripts)  # only the tokenizer and the limits are the core's
    serving = Serving(config or ServerConfig(), model_name="tiny")
    if attach:
        serving.attach(llm, core.tokenizer, output_format=output_format)
    server = TestServer(create_app(serving), handler_cancellation=True)
    try:
        async with TestClient(server) as client:
            yield client, serving, llm
    finally:
        await llm.shutdown()
        core.shutdown()  # idempotent; a fake engine does not own the core


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
