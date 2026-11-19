"""Replay the recorded client requests against the server.

The fixtures under ``tests/entrypoints/fixtures/`` are sanitized recordings of real coding
tools' requests; that directory's README says what was removed and how to redo a recording.
Each replay sends a recording to the server on the fixture GPT-2 with a scripted answer, and
checks what #50 asks for: the request is accepted, fields the server does not know are ignored
with a debug line that names them, a refused field names itself, and the streamed answer ends
with the protocol's terminal event. ``tests/e2e/test_serving.py`` parses the same recordings'
answers with the tools' own client libraries.

The replays need no GPU and run on the CPU. The fixture prompts are far longer than the tiny
GPT-2's 128 positions, so the checkpoint is written with room for them (it is only tokenized:
the engine that answers is scripted).
"""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import pytest

pytest.importorskip("torch")
pytest.importorskip("aiohttp")

from serving_support import read_sse, script, serve, write_long_gpt2  # needs the importorskip above

if TYPE_CHECKING:
    from collections.abc import Iterator

pytestmark = pytest.mark.unit

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture(autouse=True)
def _let_the_adapters_propagate() -> Iterator[None]:
    """``main(...)`` installs its own handler on the ``ayaka`` logger, at INFO, without propagation.

    The replay tests read the adapters' debug lines through pytest's root capture, so propagation
    comes back and the level opens for the test; the CLI's logger goes away afterwards.
    """
    logger = logging.getLogger("ayaka")
    handlers, level, propagate = list(logger.handlers), logger.level, logger.propagate
    logger.handlers, logger.level, logger.propagate = [], logging.DEBUG, True
    yield
    logger.handlers, logger.level, logger.propagate = handlers, level, propagate


_TERMINAL = {
    "anthropic": "message_stop",
    "openai-responses": "response.completed",
    "openai-chat": "[DONE]",
}


def _cases() -> Iterator[tuple[str, Path]]:
    """The recorded-request fixtures: one directory per tool, not the normalizer's shapes or the
    golden wire bytes."""
    for directory in sorted(FIXTURES.iterdir()):
        if not directory.is_dir() or directory.name in ("normalize", "golden"):
            continue
        for file in sorted(directory.glob("*.json")):
            yield f"{directory.name}/{file.name}", file


_CASES = list(_cases())


def _protocol(recording: dict[str, Any]) -> str:
    """The protocol a recording was sent in, from the path the tool called."""
    path = str(recording["path"])
    if "/v1/messages" in path:
        return "anthropic"
    if "/v1/responses" in path:
        return "openai-responses"
    if "/v1/chat/completions" in path:
        return "openai-chat"
    raise AssertionError(f"no protocol for {path}")


def _load(fixture_path: Path, *, stream: bool | None = None) -> dict[str, Any]:
    """A recording, retargeted to the test server's name (and streaming, when asked).

    The fixtures name the model of the machine they were recorded on; the test server serves
    ``tiny``.
    """
    recording = json.loads(fixture_path.read_text(encoding="utf-8"))
    recording["body"]["model"] = "tiny"
    if stream is not None:
        recording["body"]["stream"] = stream
    return recording


@pytest.fixture(scope="module")
def long_checkpoint(tmp_path_factory: pytest.TempPathFactory) -> Any:
    pytest.importorskip("torch")
    pytest.importorskip("aiohttp")
    return write_long_gpt2(tmp_path_factory.mktemp("replay-gpt2"), positions=65536)


def _server(tokenizer: Any, **scripts: Any) -> Any:
    """The arguments of ``serve`` for a replay: the hermes output format, a scripted engine."""
    pytest.importorskip("llguidance")
    from ayaka.entrypoints.parsers.output import OutputFormat
    from tokenizer_fakes import loaded_for

    return dict(
        tokenizer=loaded_for(tokenizer, stream=False),
        output_format=OutputFormat.resolve(tool_calls="hermes", tokenizer=tokenizer),
        max_tokens_per_step=128,
        num_blocks=17000,  # one request may hold the whole fixture prompt (65536 positions)
        **scripts,
    )


async def _post(client: Any, recording: dict[str, Any]) -> Any:
    """The recorded request as the tool sent it: its path, its headers, its body."""
    return await client.post(
        recording["path"].split("?")[0],
        data=json.dumps(recording["body"]).encode(),
        headers=recording["headers"],
    )


class TestReplay:
    @pytest.mark.parametrize(("case", "fixture_path"), _CASES, ids=[case for case, _ in _CASES])
    def test_the_recording_is_accepted_and_streamed(
        self, long_checkpoint: Any, fixture_path: Path, case: str
    ) -> None:
        del case  # the id only
        recording = _load(fixture_path)
        protocol = _protocol(recording)
        from tokenizer_fakes import ToolTokenizer

        async def run() -> Any:
            async with serve(
                long_checkpoint, scripts=[script("Hello!")], **_server(ToolTokenizer())
            ) as (client, _, _):
                response = await _post(client, recording)
                frames = await read_sse(response)
                return response.status, frames

        status, frames = asyncio.run(run())

        assert status == 200, frames
        terminal = _TERMINAL[protocol]
        if terminal == "[DONE]":
            assert frames[-1][1] == "[DONE]"
        else:
            assert terminal in {event for event, _ in frames}

    @pytest.mark.parametrize(("case", "fixture_path"), _CASES, ids=[case for case, _ in _CASES])
    def test_the_streamed_answer_is_the_golden_bytes(
        self, long_checkpoint: Any, fixture_path: Path, case: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The whole wire of a recorded request's streamed answer, with the ids and clock fixed.

        The request id is the only random part of a stream and ``created`` the only clock one;
        with both fixed, the bytes depend on the recording, the scripted answer and the encoders.
        """
        import types

        from ayaka.entrypoints import http as http_module
        from ayaka.entrypoints.openai import chat as chat_module
        from ayaka.entrypoints.openai import responses as responses_module
        from golden import check_golden
        from tokenizer_fakes import ToolTokenizer

        fixed_id = types.SimpleNamespace(hex="0123456789abcdef0123456789abcdef")
        monkeypatch.setattr(http_module, "uuid", types.SimpleNamespace(uuid4=lambda: fixed_id))
        clock = types.SimpleNamespace(time=lambda: 1700000000.0)
        monkeypatch.setattr(chat_module, "time", clock)
        monkeypatch.setattr(responses_module, "time", clock)
        recording = _load(fixture_path, stream=True)

        async def run() -> tuple[int, bytes]:
            async with serve(
                long_checkpoint,
                scripts=[script("Hel", "lo!")],
                **_server(ToolTokenizer()),
            ) as (client, _, _):
                response = await _post(client, recording)
                return response.status, await response.read()

        status, body = asyncio.run(run())

        assert status == 200, body
        check_golden(f"replay/{case.removesuffix('.json')}.sse", body)

    def test_a_refused_field_names_itself(self, long_checkpoint: Any) -> None:
        """Claude Code can turn richer input on; a block the server cannot take names itself."""
        recording = _load(FIXTURES / "claude-code-2.1.327" / "001-anthropic.json")
        recording["body"]["messages"][0]["content"].append(
            {
                "type": "image",
                "source": {"type": "base64", "media_type": "image/png", "data": "aGVsbG8="},
            }
        )
        from tokenizer_fakes import ToolTokenizer

        async def run() -> Any:
            async with serve(
                long_checkpoint,
                scripts=[script("Hello!")],
                **_server(ToolTokenizer()),
            ) as (client, _, _):
                response = await _post(client, recording)
                return response.status, await response.text()

        status, text = asyncio.run(run())

        assert status == 400
        assert "image" in text

    @pytest.mark.parametrize(
        "fixture_path",
        [
            FIXTURES / "claude-code-2.1.327" / "001-anthropic.json",
            FIXTURES / "codex-0.163.0" / "001-openai-responses.json",
            FIXTURES / "opencode-2.3.2" / "002-openai-chat.json",
        ],
        ids=["claude-code", "codex", "opencode"],
    )
    def test_an_unknown_field_is_ignored_with_a_debug_line(
        self, long_checkpoint: Any, fixture_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        recording = _load(fixture_path)
        field = "a_field_the_tool_invented"
        recording["body"][field] = {"invented": True}
        from tokenizer_fakes import ToolTokenizer

        async def run() -> Any:
            async with serve(
                long_checkpoint,
                scripts=[script("Hello!")],
                **_server(ToolTokenizer()),
            ) as (client, _, _):
                response = await _post(client, recording)
                return response.status

        with caplog.at_level(logging.DEBUG):
            status = asyncio.run(run())

        assert status == 200
        assert "ignoring unknown fields" in caplog.text
        assert field in caplog.text


class TestFixtureHygiene:
    """A fixture may be committed only when no private marker survives in it."""

    @pytest.mark.parametrize(("case", "fixture_path"), _CASES, ids=[case for case, _ in _CASES])
    def test_no_private_content_survives_in_the_fixture(
        self, fixture_path: Path, case: str
    ) -> None:
        del case  # the id only
        from sanitize_recording import assert_clean

        recording = json.loads(fixture_path.read_text(encoding="utf-8"))

        assert assert_clean(recording) == []

    def test_the_sanitizer_keeps_the_shape_and_drops_the_text(self) -> None:
        from sanitize_recording import REDACTED_TEXT, sanitize

        recording = {
            "path": "/v1/chat/completions",
            "headers": {"Authorization": "Bearer sk-secret", "user-agent": "tool/1"},
            "body": {
                "model": "m",
                "stream": True,
                "messages": [
                    {"role": "system", "content": "the project's private instructions"},
                    {"role": "user", "content": "a private prompt"},
                ],
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "read",
                            "description": "the tool's product description",
                            "parameters": {
                                "type": "object",
                                "properties": {"path": {"type": "string"}},
                            },
                        },
                    }
                ],
            },
        }

        clean = cast("dict[str, Any]", sanitize(recording))

        assert clean["headers"] == {"user-agent": "tool/1"}
        assert clean["body"]["model"] == "m"
        assert clean["body"]["stream"] is True
        assert clean["body"]["messages"][0]["content"] == REDACTED_TEXT  # a string stays a string
        assert clean["body"]["messages"][1]["content"] == REDACTED_TEXT
        tool = clean["body"]["tools"][0]
        assert tool["function"]["name"] == "read"
        assert tool["function"]["description"].startswith("<redacted")
        assert tool["function"]["parameters"]["properties"] == {"path": {"type": "string"}}
