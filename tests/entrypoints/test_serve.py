"""``ayaka serve``: settings from flags and files, and the server's life in this process.

The server runs on the tiny GPT-2 on a free port; ``stop`` stands in for SIGINT (the real signals
are tested in a child process by ``tests/e2e/test_serving.py``).
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import TYPE_CHECKING, Any

import pytest

from ayaka.configs.server import ServerConfig
from ayaka.core.errors import ConfigError, InvalidRequestError
from ayaka.entrypoints.cli import _parser, describe_server, main, server_config  # pyright: ignore[reportPrivateUsage]

torch: Any = pytest.importorskip("torch")
aiohttp = pytest.importorskip("aiohttp")

from ayaka.engine.build import build_engine  # noqa: E402
from ayaka.entrypoints.serve import serve  # noqa: E402
from serving_support import parse_sse  # noqa: E402
from tiny_model import (  # noqa: E402
    TinyCheckpoint,
    patch_tokenizer_loading,
    tiny_engine_config,
    tiny_tokenizer,
    write_tiny_gpt2,
)

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

    from ayaka.engine.core import EngineCore
    from ayaka.scheduler.step import StepOutput

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def _restore_logging() -> Iterator[None]:
    """``main(["serve", ...])`` installs a log handler on stderr; take it away afterwards."""
    logger = logging.getLogger("ayaka")
    handlers, level, propagate = list(logger.handlers), logger.level, logger.propagate
    yield
    for handler in list(logger.handlers):
        if handler not in handlers:
            logger.removeHandler(handler)
    logger.setLevel(level)
    logger.propagate = propagate


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory: pytest.TempPathFactory) -> TinyCheckpoint:
    return write_tiny_gpt2(tmp_path_factory.mktemp("tiny-gpt2"))


def _args(*argv: str) -> Any:
    return _parser().parse_args(["serve", "model", *argv])


class TestSettings:
    def test_flags_win_over_the_file_which_wins_over_defaults(self, tmp_path: Path) -> None:
        file = tmp_path / "ayaka.toml"
        file.write_text('[server]\nport = 9000\nhost = "0.0.0.0"\ncors_origins = ["*"]\n')

        config = server_config(_args("--config", str(file), "--port", "9100"))

        assert config.port == 9100
        assert config.host == "0.0.0.0"  # noqa: S104  # read from the file
        assert config.cors_origins == ("*",)
        assert config.shutdown_timeout == ServerConfig().shutdown_timeout

    def test_repeated_flags_become_tuples(self) -> None:
        config = server_config(
            _args("--served-model-name", "a", "--served-model-name", "b", "--cors-origin", "x")
        )

        assert config.served_model_names == ("a", "b")
        assert config.cors_origins == ("x",)

    def test_normalization_is_on_by_default_and_the_file_and_the_flag_agree(
        self, tmp_path: Path
    ) -> None:
        file = tmp_path / "ayaka.toml"
        file.write_text("[server]\nnormalize_prompts = false\n")

        assert server_config(_args()).normalize_prompts is True
        assert server_config(_args("--config", str(file))).normalize_prompts is False
        assert server_config(_args("--no-normalize")).normalize_prompts is False

    def test_the_key_comes_from_a_flag_or_a_file_but_not_both(self, tmp_path: Path) -> None:
        key = tmp_path / "key"
        key.write_text("s3cret\n")

        assert server_config(_args("--api-key-file", str(key))).api_key == "s3cret"
        assert server_config(_args("--api-key", "k")).api_key == "k"
        with pytest.raises(InvalidRequestError, match="not both"):
            server_config(_args("--api-key", "k", "--api-key-file", str(key)))
        (tmp_path / "empty").write_text("")
        with pytest.raises(InvalidRequestError, match="empty"):
            server_config(_args("--api-key-file", str(tmp_path / "empty")))

    def test_a_bad_file_is_a_config_error(self, tmp_path: Path) -> None:
        bad = tmp_path / "bad.toml"
        bad.write_text("[server]\nprot = 1\n")
        other = tmp_path / "other.toml"
        other.write_text("[engine]\n")

        with pytest.raises(ConfigError, match="UNKNOWN_KEY"):
            server_config(_args("--config", str(bad)))
        with pytest.raises(ConfigError, match="unknown tables"):
            server_config(_args("--config", str(other)))

    def test_the_settings_lines_warn_off_loopback_without_a_key_and_hide_the_key(self) -> None:
        engine = tiny_engine_config("model")  # type: ignore[arg-type]  # only the device is read
        open_config = ServerConfig(host="0.0.0.0")  # noqa: S104  # the case under test
        keyed = ServerConfig(host="0.0.0.0", api_key="s3cret")  # noqa: S104

        open_lines = describe_server(open_config, engine, "model")
        keyed_lines = describe_server(keyed, engine, "model")

        assert any(line.startswith("warning:") for line in open_lines)
        assert not any(line.startswith("warning:") for line in keyed_lines)
        assert "s3cret" not in "\n".join(keyed_lines)
        assert "API key: required" in keyed_lines[1]

    def test_help_exits_0_without_loading_anything(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        with pytest.raises(SystemExit) as caught:
            main(["serve", "--help"])

        assert caught.value.code == 0
        assert "--tool-call-parser" in capsys.readouterr().out


def _factory(checkpoint: TinyCheckpoint, *, slow: float = 0.0) -> Any:
    def build() -> EngineCore:
        core = build_engine(tiny_engine_config(checkpoint.directory), tokenizer=tiny_tokenizer())
        if slow:
            real = core.runner.execute

            def slowed(step: Any, drafts: Any = None) -> StepOutput:
                time.sleep(slow)
                return real(step, drafts)

            core.runner.execute = slowed  # type: ignore[method-assign]  # slow the steps down
        return core

    return build


_STREAM = {
    "messages": [{"role": "user", "content": "hi"}],
    "max_tokens": 60,
    "temperature": 0,
    "stream": True,
}


async def _run_server(
    checkpoint: TinyCheckpoint,
    config: ServerConfig,
    *,
    slow: float = 0.0,
    record: Path | None = None,
) -> tuple[asyncio.Task[None], asyncio.Event, str, list[str]]:
    stop = asyncio.Event()
    ready: asyncio.Future[str] = asyncio.get_running_loop().create_future()
    said: list[str] = []
    task = asyncio.create_task(
        serve(
            config,
            _factory(checkpoint, slow=slow),
            model_name="tiny",
            say=said.append,
            stop=stop,
            handle_signals=False,
            on_ready=lambda host, port: ready.set_result(f"http://{host}:{port}"),
            record=record,
        )
    )
    url = await asyncio.wait_for(ready, 60)
    return task, stop, url, said


class TestLifeCycle:
    def test_record_writes_each_request_but_not_the_warm_up(
        self, checkpoint: TinyCheckpoint, tmp_path: Path
    ) -> None:
        from ayaka.engine.recorder import read_session

        async def main_() -> None:
            config = ServerConfig(port=0)
            task, stop, url, said = await _run_server(checkpoint, config, record=tmp_path)
            async with aiohttp.ClientSession() as session:
                body = {**_STREAM, "stream": False, "max_tokens": 5}
                for _ in range(2):
                    assert (await session.post(f"{url}/v1/chat/completions", json=body)).ok
            stop.set()
            await asyncio.wait_for(task, 30)
            assert any(line.startswith("recording every request to") for line in said)

        asyncio.run(main_())
        [path] = tmp_path.glob("session-*.jsonl")
        records = read_session(path)
        assert len(records) == 2
        assert [record.completion_tokens for record in records] == [5, 5]

    def test_stopping_drains_requests_in_flight_and_refuses_new_ones(
        self, checkpoint: TinyCheckpoint
    ) -> None:
        async def main_() -> None:
            config = ServerConfig(port=0, shutdown_timeout=30)
            task, stop, url, said = await _run_server(checkpoint, config, slow=0.01)
            async with aiohttp.ClientSession() as session:
                assert (await session.get(f"{url}/health")).status == 200
                response = await session.post(f"{url}/v1/chat/completions", json=_STREAM)
                await response.content.readuntil(b"\n\n")
                stop.set()
                await asyncio.sleep(0.1)
                refused = await session.post(f"{url}/v1/chat/completions", json=_STREAM)
                assert refused.status == 503
                health = await session.get(f"{url}/health")
                assert await health.json() == {"status": "draining"}
                frames = parse_sse(await response.text())
                assert frames[-1] == (None, "[DONE]")
                last = json.loads(frames[-2][1])
                assert last["choices"][0]["finish_reason"] == "length"
            await asyncio.wait_for(task, 30)
            assert said[0].startswith("listening on http://127.0.0.1:")
            assert said[1].startswith("ready: tiny")
            assert said[-1] == "stopped"

        asyncio.run(main_())

    def test_with_no_grace_requests_in_flight_end_cancelled(
        self, checkpoint: TinyCheckpoint
    ) -> None:
        async def main_() -> None:
            config = ServerConfig(port=0, shutdown_timeout=0)
            task, stop, url, _ = await _run_server(checkpoint, config, slow=0.02)
            async with aiohttp.ClientSession() as session:
                response = await session.post(f"{url}/v1/chat/completions", json=_STREAM)
                await response.content.readuntil(b"\n\n")
                stop.set()
                frames = parse_sse(await response.text())
                assert frames[-1] == (None, "[DONE]")
                assert json.loads(frames[-2][1])["error"]["code"] == "cancelled"
            await asyncio.wait_for(task, 30)

        asyncio.run(main_())

    def test_a_model_that_cannot_load_stops_the_server_and_exits_3(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        patch_tokenizer_loading(monkeypatch)

        code = main(["serve", str(tmp_path / "nowhere"), "--port", "0", "--offline"])

        err = capsys.readouterr().err
        assert code == 3
        assert "listening on http://127.0.0.1:" in err
        assert "ayaka: error:" in err

    def test_a_tool_call_parser_that_does_not_exist_is_refused(
        self, checkpoint: TinyCheckpoint, capsys: pytest.CaptureFixture[str]
    ) -> None:
        code = main(["serve", str(checkpoint.directory), "--tool-call-parser", "nope"])

        assert code == 2
        assert "unknown tool call parser" in capsys.readouterr().err


class TestLogContent:
    def test_no_prompt_or_answer_text_reaches_the_default_log_or_metrics(
        self, checkpoint: TinyCheckpoint, caplog: pytest.LogCaptureFixture
    ) -> None:
        marker = "PROMPT-MARKER-ZX81"
        answers: list[str] = []
        ids: list[str] = []
        metrics: list[str] = []

        async def main_() -> None:
            config = ServerConfig(port=0)
            task, stop, url, _ = await _run_server(checkpoint, config)
            async with aiohttp.ClientSession() as session:
                for stream in (False, True):
                    body = {
                        "messages": [{"role": "user", "content": f"{marker} please answer"}],
                        "max_tokens": 24,
                        "temperature": 0,
                        "stream": stream,
                    }
                    response = await session.post(f"{url}/v1/chat/completions", json=body)
                    ids.append(response.headers["x-request-id"])
                    if stream:
                        frames = parse_sse(await response.text())
                        answers.append(
                            "".join(
                                json.loads(data)["choices"][0]["delta"].get("content") or ""
                                for _, data in frames
                                if data != "[DONE]" and json.loads(data)["choices"]
                            )
                        )
                    else:
                        answer = await response.json()
                        answers.append(answer["choices"][0]["message"]["content"] or "")
                metrics.append(await (await session.get(f"{url}/metrics")).text())
            stop.set()
            await asyncio.wait_for(task, 30)

        with caplog.at_level(logging.INFO, logger="ayaka"):
            asyncio.run(main_())

        log = "\n".join(record.getMessage() for record in caplog.records)
        texts = [answer for answer in answers if len(answer.strip()) >= 4]
        assert texts, "the model answered nothing to look for"
        for leaked in (marker, *texts):
            assert leaked not in log
            assert leaked not in metrics[0]
        for request_id in ids:
            assert request_id in log  # the line is there, with its id
        assert "prompt and" in log
        assert "generated tokens" in log
