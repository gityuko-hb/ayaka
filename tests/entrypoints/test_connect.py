"""``ayaka connect``: records, rendering, the diffed write, the seven steps and the proxy."""

from __future__ import annotations

import asyncio
import http.server
import json
import threading
import time
import tomllib
import urllib.error
import urllib.request
from typing import TYPE_CHECKING, Any

import pytest

from ayaka.entrypoints.cli import main
from ayaka.entrypoints.connect.check import (
    _cache_share,  # pyright: ignore[reportPrivateUsage]
    check_lines,
    run_check,
)
from ayaka.entrypoints.connect.observe import (
    agent_version,
    count_parts,
    largest,
    observer,
    recorder,
    redact_headers,
    save_observation,
    serve_proxy,
)
from ayaka.entrypoints.connect.tools import TOOLS, ServerInfo, render_config
from ayaka.entrypoints.connect.write import write_config
from ayaka.entrypoints.console import EXIT_OK, EXIT_USAGE

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

pytestmark = pytest.mark.unit

_SERVER = ServerInfo("http://127.0.0.1:8000", "sk-secret-123", "tiny", context_tokens=1024)


class TestRender:
    def test_claude_code_merges_its_variables_into_the_settings(self) -> None:
        current = json.dumps({"model": "opus", "env": {"KEEP": "1"}, "hooks": {"a": 1}})

        settings = json.loads(render_config(TOOLS["claude-code"], _SERVER, current))

        assert settings["model"] == "opus"
        assert settings["hooks"] == {"a": 1}
        env = settings["env"]
        assert env["KEEP"] == "1"
        assert env["ANTHROPIC_BASE_URL"] == "http://127.0.0.1:8000"
        assert env["ANTHROPIC_MODEL"] == env["ANTHROPIC_DEFAULT_HAIKU_MODEL"] == "tiny"
        assert env["CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC"] == "1"
        assert env["CLAUDE_CODE_MAX_CONTEXT_TOKENS"] == "1024"

    def test_codex_gets_a_profile_file_that_parses(self) -> None:
        profile = tomllib.loads(render_config(TOOLS["codex"], _SERVER))

        assert profile["model"] == "tiny"
        assert profile["model_provider"] == "ayaka"
        assert profile["model_context_window"] == 1024
        provider = profile["model_providers"]["ayaka"]
        assert provider["base_url"] == "http://127.0.0.1:8000/v1"
        assert provider["wire_api"] == "responses"
        assert "secret" not in json.dumps(profile)

    def test_opencode_adds_a_provider_and_keeps_the_others(self) -> None:
        current = json.dumps({"providers": {"other": {"name": "x"}}, "theme": "dark"})

        config = json.loads(render_config(TOOLS["opencode"], _SERVER, current, "2.3.2"))

        assert config["theme"] == "dark"
        assert config["providers"]["other"] == {"name": "x"}
        ayaka = config["providers"]["ayaka"]
        assert ayaka["settings"]["baseURL"] == "http://127.0.0.1:8000/v1"
        assert ayaka["env"] == ["AYAKA_API_KEY"]
        assert ayaka["models"]["tiny"]["limit"] == {"context": 1024}
        assert config["model"] == "ayaka/tiny"

    def test_opencode_1_gets_the_provider_layout_it_reads(self) -> None:
        current = json.dumps({"provider": {"other": {"npm": "x"}}, "theme": "dark"})

        config = json.loads(render_config(TOOLS["opencode"], _SERVER, current, "1.19.6"))

        assert config["theme"] == "dark"
        assert config["provider"]["other"] == {"npm": "x"}
        assert "providers" not in config  # opencode 1.19.6 drops this table
        ayaka = config["provider"]["ayaka"]
        assert ayaka["npm"] == "@ai-sdk/openai-compatible"
        assert ayaka["options"] == {
            "baseURL": "http://127.0.0.1:8000/v1",
            "apiKey": "{env:AYAKA_API_KEY}",
        }
        # opencode 1.19.6 refuses a limit without its output.
        assert ayaka["models"]["tiny"] == {
            "name": "tiny",
            "tool_call": True,
            "limit": {"context": 1024, "output": 1024},
        }
        assert config["model"] == "ayaka/tiny"
        assert "secret" not in json.dumps(config)

    @pytest.mark.parametrize("version", [None, "opencode 1.2.0", "unknown"])
    def test_opencode_without_a_2_version_gets_the_1_layout(self, version: str | None) -> None:
        config = json.loads(render_config(TOOLS["opencode"], _SERVER, "", version))

        assert "provider" in config
        assert "providers" not in config

    def test_opencode_1_names_the_small_model(self) -> None:
        server = ServerInfo("http://127.0.0.1:8000", None, "big", small_model="small")

        config = json.loads(render_config(TOOLS["opencode"], server, "", "1.19.6"))

        assert set(config["provider"]["ayaka"]["models"]) == {"big", "small"}
        assert config["small_model"] == "ayaka/small"

    def test_every_record_has_a_home_path(self) -> None:
        for record in TOOLS.values():
            path = record.config_path({})
            assert path.name.endswith((".json", ".toml"))
            assert "~" not in str(path)

    def test_the_tools_own_variable_moves_the_file(self, tmp_path: Path) -> None:
        env = {"CODEX_HOME": str(tmp_path), "CLAUDE_CONFIG_DIR": str(tmp_path / "c")}

        assert TOOLS["codex"].config_path(env) == tmp_path / "ayaka.config.toml"
        assert TOOLS["claude-code"].config_path(env) == tmp_path / "c" / "settings.json"

    def test_every_measured_floor_says_where_it_comes_from(self) -> None:
        for record in TOOLS.values():
            if record.min_context_tokens is not None:
                assert record.measured is not None
                assert "tokenizer" in record.measured

    def test_claude_code_maps_every_model_alias_to_the_served_model(self) -> None:
        env = json.loads(render_config(TOOLS["claude-code"], _SERVER))["env"]

        assert {env[f"ANTHROPIC_DEFAULT_{size}_MODEL"] for size in ("OPUS", "SONNET", "HAIKU")} == {
            "tiny"
        }


class TestWrite:
    def test_a_diff_a_backup_and_an_atomic_write(self, tmp_path: Path) -> None:
        path = tmp_path / "settings.json"
        path.write_text('{"a": 1}\n', encoding="utf-8")
        shown: list[str] = []

        result = write_config(
            path,
            '{"a": 2, "key": "sk-secret-123"}\n',
            confirm=lambda question: True,
            show=shown.append,
            redactor=lambda text: text.replace("sk-secret-123", "<api key>"),
        )

        assert result.written
        assert result.backup is not None
        assert result.backup.read_text(encoding="utf-8") == '{"a": 1}\n'
        assert ".bak" in result.backup.name
        assert path.read_text(encoding="utf-8") == '{"a": 2, "key": "sk-secret-123"}\n'
        assert any('+{"a": 2' in line for line in shown)
        assert not any("sk-secret-123" in line for line in shown)
        assert [p.name for p in tmp_path.iterdir() if p.name.startswith(".")] == []

    def test_nothing_is_written_without_a_yes(self, tmp_path: Path) -> None:
        path = tmp_path / "settings.json"
        path.write_text("old\n", encoding="utf-8")

        result = write_config(path, "new\n", confirm=lambda question: False, show=lambda _: None)

        assert not result.written
        assert path.read_text(encoding="utf-8") == "old\n"
        assert len(list(tmp_path.iterdir())) == 1

    def test_a_missing_folder_needs_create_dirs(self, tmp_path: Path) -> None:
        path = tmp_path / "nested" / "config.toml"

        with pytest.raises(FileNotFoundError, match="--create-dirs"):
            write_config(path, "x\n", confirm=lambda question: True, show=lambda _: None)
        result = write_config(
            path, "x\n", confirm=lambda question: True, create_dirs=True, show=lambda _: None
        )
        assert result.written
        assert result.backup is None

    def test_the_command_refuses_to_write_without_a_terminal_or_yes(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("CODEX_HOME", str(tmp_path))
        monkeypatch.setattr("sys.stdin", None)

        code = main(["connect", "codex", "--model", "tiny", "--write", "--url", "http://x"])

        assert code == EXIT_USAGE

    def test_printing_shows_only_what_ayaka_adds(self, capsys: pytest.CaptureFixture[str]) -> None:
        code = main(["connect", "claude-code", "--model", "tiny", "--url", "http://127.0.0.1:1"])

        out = capsys.readouterr().out
        assert code == EXIT_OK
        assert "merge into" in out
        body = out.split("\n", 1)[1].rsplit("# then:", 1)[0]
        assert set(json.loads(body)) == {"env"}

    def test_an_unknown_tool_gets_the_generic_setup(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        code = main(["connect", "some-editor", "--model", "tiny", "--url", "http://127.0.0.1:1"])

        out = capsys.readouterr().out
        assert code == EXIT_OK
        assert "OpenAI-compatible" in out
        assert "Anthropic-compatible" in out


class TestCountsAndRecordings:
    @staticmethod
    def _count(text: str) -> int:
        return len(text.encode())

    def test_anthropic_parts(self) -> None:
        body = {
            "system": [{"type": "text", "text": "abc"}],
            "tools": [{"name": "t"}],
            "messages": [{"role": "user", "content": "hello"}],
        }

        assert count_parts("anthropic", body, self._count) == (3, 15, 5, 1)

    def test_openai_chat_and_responses_parts(self) -> None:
        chat = {
            "messages": [
                {"role": "system", "content": "sys"},
                {"role": "user", "content": [{"type": "text", "text": "hi"}]},
            ]
        }
        responses = {"instructions": "be brief", "input": "hello"}

        assert count_parts("openai-chat", chat, self._count) == (4, 0, 2, 0)
        assert count_parts("openai-responses", responses, self._count) == (8, 0, 5, 0)

    def test_the_observer_keeps_counts_only(self) -> None:
        found: list[Any] = []
        observe = observer(TOOLS["claude-code"], "tok", self._count, found)
        body = json.dumps({"system": "secret system text", "messages": [], "max_tokens": 1})

        observe("/v1/messages?beta=true", {"authorization": "Bearer x"}, body.encode())
        observe("/v1/messages/count_tokens", {}, body.encode())

        assert len(found) == 1
        assert found[0].system_tokens == len("secret system text")
        assert "secret" not in repr(found[0])

    def test_a_recording_has_no_credentials(self, tmp_path: Path) -> None:
        keep = recorder(tmp_path / "rec", limit=1)

        done = keep(
            "/v1/chat/completions",
            {"Authorization": "Bearer sk-x", "x-api-key": "k", "user-agent": "tool/1"},
            b'{"model": "m"}',
        )

        [file] = (tmp_path / "rec").iterdir()
        saved = json.loads(file.read_text(encoding="utf-8"))
        assert done
        assert saved["headers"] == {"user-agent": "tool/1"}
        assert saved["body"] == {"model": "m"}
        assert redact_headers({"Cookie": "c", "a": "b"}) == {"a": "b"}

    def test_the_proxy_forwards_and_records(self, tmp_path: Path) -> None:
        class Upstream(http.server.BaseHTTPRequestHandler):
            def log_message(self, format: str, *args: object) -> None:  # noqa: A002
                del format, args

            def do_POST(self) -> None:
                self.rfile.read(int(self.headers["content-length"]))
                self.send_response(200)
                self.send_header("content-type", "application/json")
                self.end_headers()
                self.wfile.write(b'{"ok": true}')

        upstream = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Upstream)
        threading.Thread(target=upstream.serve_forever, daemon=True).start()
        stop = threading.Event()
        keep = recorder(tmp_path / "rec")

        def on_request(path: str, headers: dict[str, str], body: bytes) -> None:
            keep(path, headers, body)

        port = serve_proxy(
            0,
            f"http://127.0.0.1:{upstream.server_address[1]}",
            on_request,
            stop=stop,
        )
        try:
            request = urllib.request.Request(
                f"http://127.0.0.1:{port}/v1/messages",
                data=b'{"messages": []}',
                headers={"authorization": "Bearer sk-x", "content-type": "application/json"},
            )
            with urllib.request.urlopen(request, timeout=5) as response:  # noqa: S310  # local
                assert json.loads(response.read()) == {"ok": True}
        finally:
            stop.set()
            upstream.shutdown()
            upstream.server_close()
        [file] = (tmp_path / "rec").iterdir()
        assert "sk-x" not in file.read_text(encoding="utf-8")


class TestServerLookup:
    """``_connect_server`` asks a running server for its model and context size."""

    @staticmethod
    def _models_server(data: list[dict[str, Any]]) -> tuple[str, Callable[[], None]]:
        class Models(http.server.BaseHTTPRequestHandler):
            def log_message(self, format: str, *args: object) -> None:  # noqa: A002
                del format, args

            def do_GET(self) -> None:
                if self.path.startswith("/v1/models"):
                    self.send_response(200)
                    self.send_header("content-type", "application/json")
                    self.end_headers()
                    self.wfile.write(json.dumps({"data": data}).encode())
                else:
                    self.send_response(404)
                    self.end_headers()

        server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Models)
        server.daemon_threads = True

        def close() -> None:
            server.shutdown()
            server.server_close()

        threading.Thread(target=server.serve_forever, daemon=True).start()
        return f"http://127.0.0.1:{server.server_address[1]}", close

    def test_the_config_carries_the_server_context(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        base, stop = self._models_server(
            [
                {"id": "big", "max_model_len": 4096, "max_context_tokens": 2048},
                {"id": "small", "max_model_len": 1024},
            ]
        )
        try:
            code = main(["connect", "codex", "--url", base])
        finally:
            stop()
        out = capsys.readouterr().out
        profile = tomllib.loads(out.split("\n", 1)[1].rsplit("# then:", 1)[0])

        assert code == EXIT_OK
        assert profile["model"] == "big"
        assert profile["model_context_window"] == 2048

    def test_the_named_model_selects_its_own_entry(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        base, stop = self._models_server(
            [{"id": "big", "max_context_tokens": 2048}, {"id": "small", "max_model_len": 1024}]
        )
        try:
            code = main(["connect", "codex", "--url", base, "--model", "small"])
        finally:
            stop()
        out = capsys.readouterr().out
        profile = tomllib.loads(out.split("\n", 1)[1].rsplit("# then:", 1)[0])

        assert code == EXIT_OK
        assert profile["model_context_window"] == 1024

    def test_a_silent_server_leaves_the_window_out(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        code = main(["connect", "codex", "--url", "http://127.0.0.1:9", "--model", "tiny"])
        out = capsys.readouterr().out
        profile = tomllib.loads(out.split("\n", 1)[1].rsplit("# then:", 1)[0])

        assert code == EXIT_OK
        assert profile["model"] == "tiny"
        assert "model_context_window" not in profile


# -- the seven steps against a real server on the tiny GPT-2 ----------------------------------

pytest.importorskip("torch")


@pytest.fixture(scope="module")
def long_checkpoint(tmp_path_factory: pytest.TempPathFactory) -> Path:
    pytest.importorskip("aiohttp")
    from gpt2_support import tiny_config, write_checkpoint
    from tiny_model import tiny_gpt2_state
    from tokenizer_fakes import MODEL_VOCAB

    # The fixture model, quiet outside printable bytes, with room for a prompt with tools.
    config = tiny_config(vocab_size=MODEL_VOCAB, max_position_embeddings=1024)
    directory = tmp_path_factory.mktemp("connect-gpt2")
    write_checkpoint(directory, tiny_gpt2_state(config), config)
    return directory


def _check(checkpoint: Path, record: str, **server: Any) -> Any:
    pytest.importorskip("aiohttp")
    pytest.importorskip("llguidance")
    from ayaka.entrypoints.parsers.output import OutputFormat
    from serving_support import serve
    from tokenizer_fakes import ToolTokenizer, loaded_for

    async def main() -> Any:
        tokenizer = ToolTokenizer()
        async with serve(
            checkpoint,
            tokenizer=loaded_for(tokenizer, stream=False),
            output_format=OutputFormat.resolve(tool_calls="hermes", tokenizer=tokenizer),
            max_tokens_per_step=128,
            num_blocks=600,
            **server,
        ) as (client, _, _):
            base = str(client.make_url("")).rstrip("/")
            info = ServerInfo(base, None, "tiny")
            return await asyncio.to_thread(run_check, TOOLS[record], info, timeout=30.0)

    return asyncio.run(main())


class TestCheck:
    @pytest.mark.parametrize("record", ["claude-code", "codex", "opencode"])
    def test_every_step_passes_on_a_working_server(
        self, long_checkpoint: Path, record: str
    ) -> None:
        report = _check(long_checkpoint, record)

        results = {step.id: step.ok for step in report.steps}
        details = [f"{step.id} {step.ok} {step.detail}" for step in report.steps]
        # The tiny model gives a request 1024 tokens: less than any tool's measured floor.
        assert results == {1: True, 2: True, 3: True, 4: True, 5: True, 6: True, 7: False}, details
        assert not report.ok
        payload = report.as_json()
        assert payload["protocol"] == TOOLS[record].protocol
        assert [step["id"] for step in payload["steps"]] == [1, 2, 3, 4, 5, 6, 7]

    def test_a_server_that_is_down_fails_the_first_step_with_a_fix(self) -> None:
        report = run_check(
            TOOLS["opencode"], ServerInfo("http://127.0.0.1:9", None, "x"), timeout=1
        )

        [step] = report.steps
        assert step.ok is False
        assert step.fix is not None
        assert not report.ok


# -- each step's failure against a scripted server --------------------------------------------


class _Script:
    """What the fake server answers: a working OpenAI-compatible server unless changed."""

    def __init__(self) -> None:
        self.health: tuple[int, dict[str, Any]] = (200, {"status": "ready"})
        self.models: list[dict[str, Any]] = [
            {"id": "tiny", "max_context_tokens": 65536, "prefix_cache": True}
        ]
        self.text = "Hello there."
        self.stream_end: list[str] = ["data: [DONE]"]
        self.arguments = '{"path": "README.md"}'
        self.cached = 80
        self.prompt = 90
        self.posts = 0


def _fake_server(script: _Script) -> tuple[str, Callable[[], None]]:
    class Handler(http.server.BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: object) -> None:  # noqa: A002
            del format, args

        def _json(self, status: int, body: object) -> None:
            data = json.dumps(body).encode()
            self.send_response(status)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self) -> None:
            if self.path == "/health":
                self._json(*script.health)
            else:
                self._json(200, {"object": "list", "data": script.models})

        def do_POST(self) -> None:
            request = json.loads(self.rfile.read(int(self.headers["content-length"])))
            script.posts += 1
            if request.get("stream"):
                chunk = {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}
                body = f"data: {json.dumps(chunk)}\n\n" + "".join(
                    f"{line}\n\n" for line in script.stream_end
                )
                self.send_response(200)
                self.send_header("content-type", "text/event-stream")
                self.end_headers()
                self.wfile.write(body.encode())
                return
            usage = {
                "prompt_tokens": script.prompt,
                "prompt_tokens_details": {"cached_tokens": script.cached},
            }
            if request.get("tools"):
                call = {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "read_file", "arguments": script.arguments},
                }
                message: dict[str, Any] = {"role": "assistant", "tool_calls": [call]}
            else:
                message = {"role": "assistant", "content": script.text}
                usage = {"prompt_tokens": 9}
            self._json(200, {"choices": [{"index": 0, "message": message}], "usage": usage})

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, daemon=True).start()

    def close() -> None:
        server.shutdown()
        server.server_close()

    return f"http://127.0.0.1:{server.server_address[1]}", close


def _scripted(script: _Script, *, key: str | None = None) -> Any:
    base, close = _fake_server(script)
    try:
        return run_check(TOOLS["opencode"], ServerInfo(base, key, "tiny"), timeout=1.0)
    finally:
        close()


def _step(report: Any, number: int) -> Any:
    return next(step for step in report.steps if step.id == number)


class TestCheckSteps:
    def test_a_working_server_passes_every_step(self) -> None:
        report = _scripted(_Script())

        assert [step.ok for step in report.steps] == [True] * 7, check_lines(report)

    def test_a_server_still_loading_fails_ready_after_the_timeout(self) -> None:
        script = _Script()
        script.health = (503, {"status": "starting"})

        report = _scripted(script)

        assert [step.id for step in report.steps] == [1]
        assert report.steps[0].ok is False
        assert "--timeout" in (report.steps[0].fix or "")

    def test_a_missing_alias_names_the_flag_that_adds_it(self) -> None:
        script = _Script()
        script.models = [{"id": "other", "max_context_tokens": 65536}]

        step = _step(_scripted(script), 2)

        assert step.ok is False
        assert step.fix == "serve every name the tool sends: --served-model-name tiny"

    def test_empty_text_fails_the_text_step(self) -> None:
        script = _Script()
        script.text = ""

        assert _step(_scripted(script), 3).ok is False

    def test_a_stream_without_its_terminal_event_fails(self) -> None:
        script = _Script()
        script.stream_end = []

        step = _step(_scripted(script), 4)

        assert step.ok is False
        assert "does not end with" in step.detail
        assert "proxy" in (step.fix or "")

    def test_tool_arguments_that_are_not_schema_json_fail(self) -> None:
        for arguments in ('{"path": ', '{"file": "a"}', '{"path": "' + "x" * 41 + '"}'):
            script = _Script()
            script.arguments = arguments

            assert _step(_scripted(script), 5).ok is False, arguments

    def test_a_failed_tool_call_names_the_suggested_parser_when_reported(self) -> None:
        script = _Script()
        script.arguments = "{broken"
        script.models[0]["suggested_tool_call_parser"] = "llama3_json"

        step = _step(_scripted(script), 5)

        assert step.ok is False
        assert "--tool-call-parser llama3_json" in (step.fix or "")

    def test_the_anthropic_prompt_is_the_three_parts_of_its_usage(self) -> None:
        usage = {
            "input_tokens": 4,
            "cache_read_input_tokens": 40,
            "cache_creation_input_tokens": 8,
        }

        assert _cache_share("anthropic", {"usage": usage}) == (40, 52)

    def test_a_cache_miss_says_where_the_prompts_diverge(self) -> None:
        script = _Script()
        script.cached = 0

        step = _step(_scripted(script), 6)

        assert step.ok is False
        assert "diverge in the cache at token 0" in step.detail

    def test_a_server_without_a_prefix_cache_is_not_measured(self) -> None:
        script = _Script()
        script.models[0]["prefix_cache"] = False

        report = _scripted(script)

        assert _step(report, 6).ok is None
        assert script.posts == 3  # no repeat was sent
        assert report.ok

    def test_too_short_a_context_fails_with_the_plan_command(self) -> None:
        script = _Script()
        script.models[0]["max_context_tokens"] = 4096

        step = _step(_scripted(script), 7)

        assert step.ok is False
        assert "ayaka plan <model> --for opencode" in (step.fix or "")

    def test_a_server_that_breaks_mid_check_fails_steps_instead_of_raising(self) -> None:
        script = _Script()
        base, close = _fake_server(script)
        close()  # health is never answered, nor anything after

        report = run_check(TOOLS["opencode"], ServerInfo(base, None, "tiny"), timeout=0.5)

        assert report.steps[0].ok is False
        assert "nothing" in report.steps[0].detail


class TestCheckCommand:
    def test_json_output_has_the_stable_shape_and_exit_codes(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        key = tmp_path / "key"
        key.write_text("sk-secret-123\n", encoding="utf-8")
        script = _Script()
        base, close = _fake_server(script)
        try:
            argv = ["connect", "opencode", "--check", "--url", base, "--model", "tiny"]
            argv += ["--api-key-file", str(key), "--output", "json", "--timeout", "2"]
            passed = main(argv)
            first = capsys.readouterr().out
            script.text = ""
            failed = main(argv)
            second = capsys.readouterr().out
        finally:
            close()

        report = json.loads(first)
        assert passed == EXIT_OK
        assert failed == 1
        assert set(report) == {"tool", "version", "protocol", "ok", "ayaka", "steps"}
        assert [set(step) for step in report["steps"]] == [
            {"id", "name", "ok", "detail", "fix"}
        ] * 7
        assert json.loads(second)["ok"] is False
        assert "sk-secret-123" not in first + second

    def test_a_check_of_an_unknown_tool_cannot_run(self) -> None:
        assert main(["connect", "some-editor", "--check", "--url", "http://127.0.0.1:9"]) == 2


class TestNoKeyLeaks:
    def test_the_key_never_appears_in_printed_configurations(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        key = tmp_path / "key"
        key.write_text("sk-secret-123\n", encoding="utf-8")

        for tool in TOOLS:
            assert (
                main(
                    [
                        "connect",
                        tool,
                        "--url",
                        "http://127.0.0.1:9",
                        "--model",
                        "m",
                        "--api-key-file",
                        str(key),
                    ]
                )
                == EXIT_OK
            )

        assert "sk-secret-123" not in capsys.readouterr().out

    def test_the_key_never_appears_in_the_write_diff(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        key = tmp_path / "key"
        key.write_text("sk-secret-123\n", encoding="utf-8")
        monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(tmp_path / "claude"))
        (tmp_path / "claude").mkdir()
        (tmp_path / "claude" / "settings.json").write_text('{"model": "opus"}\n', encoding="utf-8")

        code = main(
            [
                "connect",
                "claude-code",
                "--write",
                "--yes",
                "--url",
                "http://127.0.0.1:9",
                "--model",
                "m",
                "--api-key-file",
                str(key),
            ]
        )

        out = capsys.readouterr().out
        written = json.loads((tmp_path / "claude" / "settings.json").read_text(encoding="utf-8"))
        assert code == EXIT_OK
        assert "<api key>" in out
        assert "sk-secret-123" not in out
        assert written["env"]["ANTHROPIC_AUTH_TOKEN"] == "sk-secret-123"  # the tool needs it
        assert written["model"] == "opus"
        backups = list((tmp_path / "claude").glob("settings.json.*.bak"))
        assert [b.read_text(encoding="utf-8") for b in backups] == ['{"model": "opus"}\n']


class TestWriteRefusals:
    def test_a_broken_file_is_not_overwritten(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(tmp_path))
        (tmp_path / "settings.json").write_text("{not json", encoding="utf-8")

        code = main(
            [
                "connect",
                "claude-code",
                "--write",
                "--yes",
                "--url",
                "http://127.0.0.1:9",
                "--model",
                "m",
            ]
        )

        assert code == EXIT_USAGE
        assert (tmp_path / "settings.json").read_text(encoding="utf-8") == "{not json"

    def test_two_backups_in_one_second_are_both_kept(self, tmp_path: Path) -> None:
        path = tmp_path / "f.json"
        for content in ("1\n", "2\n", "3\n"):
            write_config(path, content, confirm=lambda question: True, show=lambda _: None)

        backups = sorted(p.read_text(encoding="utf-8") for p in tmp_path.glob("f.json.*.bak"))
        assert backups == ["1\n", "2\n"]

    def test_codex_names_a_key_variable_only_when_the_server_has_a_key(self) -> None:
        keyless = tomllib.loads(render_config(TOOLS["codex"], ServerInfo("http://h", None, "m")))
        keyed = tomllib.loads(render_config(TOOLS["codex"], _SERVER))

        assert "env_key" not in keyless["model_providers"]["ayaka"]
        assert keyed["model_providers"]["ayaka"]["env_key"] == "AYAKA_API_KEY"


class TestObserve:
    @staticmethod
    def _count(text: str) -> int:
        return len(text.encode())

    def test_the_largest_request_is_the_floor_and_the_version_comes_from_the_agent(self) -> None:
        found: list[Any] = []
        observe = observer(TOOLS["opencode"], "tok", self._count, found, installed="9.9")
        title = {"messages": [{"role": "system", "content": "title please"}]}
        agent = {
            "messages": [{"role": "system", "content": "you are an agent"}],
            "tools": [{"type": "function", "function": {"name": "read"}}],
        }
        headers = {"User-Agent": "opencode/latest/2.3.2/cli"}

        observe("/v1/chat/completions", headers, json.dumps(title).encode())
        observe("/v1/chat/completions", headers, json.dumps(agent).encode())
        observe("/v1/chat/completions", {}, b"not json")

        floor = largest(found)
        assert len(found) == 2
        assert floor is not None
        assert floor.tool_count == 1
        assert floor.version == "2.3.2"
        assert floor.tokenizer == "tok"
        assert found[0].version == "2.3.2"

    def test_another_tools_agent_is_not_taken_for_this_ones_version(self) -> None:
        assert agent_version({"user-agent": "claude-cli/2.1.327 (external)"}, "claude-cli") == (
            "2.1.327"
        )
        assert agent_version({"user-agent": "codex_exec/0.163.0 (Windows)"}, "codex_exec") == (
            "0.163.0"
        )
        assert agent_version({"user-agent": "curl/8"}, "opencode") is None

    def test_an_observation_file_holds_counts_and_a_new_row_per_run(self, tmp_path: Path) -> None:
        found: list[Any] = []
        observe = observer(TOOLS["claude-code"], "tok", self._count, found)
        body = {"system": "secret system text", "messages": [{"role": "user", "content": "hi"}]}
        observe("/v1/messages", {"user-agent": "claude-cli/2.1.327"}, json.dumps(body).encode())

        save_observation(found[0], tmp_path)
        path = save_observation(found[0], tmp_path)

        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        assert len(rows) == 2
        assert rows[0]["fixed_tokens"] == len("secret system text")
        assert "secret" not in path.read_text(encoding="utf-8")

    def test_the_proxy_answers_502_when_the_server_is_gone(self) -> None:
        stop = threading.Event()
        port = serve_proxy(0, "http://127.0.0.1:9", lambda path, headers, body: None, stop=stop)
        try:
            request = urllib.request.Request(
                f"http://127.0.0.1:{port}/v1/chat/completions", data=b"{}", method="POST"
            )
            with pytest.raises(urllib.error.HTTPError) as caught:
                urllib.request.urlopen(request, timeout=5)  # noqa: S310  # local
        finally:
            stop.set()
        caught.value.close()
        assert caught.value.code == 502

    def test_the_request_that_stops_the_proxy_still_gets_its_answer(self) -> None:
        class Slow(http.server.BaseHTTPRequestHandler):
            def log_message(self, format: str, *args: object) -> None:  # noqa: A002
                del format, args

            def do_POST(self) -> None:
                self.rfile.read(int(self.headers["content-length"]))
                time.sleep(0.5)  # the proxy is asked to stop while this answer is on its way
                self.send_response(200)
                self.send_header("content-type", "application/json")
                self.end_headers()
                self.wfile.write(b'{"late": true}')

        upstream = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Slow)
        threading.Thread(target=upstream.serve_forever, daemon=True).start()
        stop, closed = threading.Event(), threading.Event()
        port = serve_proxy(
            0,
            f"http://127.0.0.1:{upstream.server_address[1]}",
            lambda path, headers, body: stop.set(),
            stop=stop,
            closed=closed,
        )
        try:
            request = urllib.request.Request(
                f"http://127.0.0.1:{port}/v1/messages", data=b"{}", method="POST"
            )
            with urllib.request.urlopen(request, timeout=10) as response:  # noqa: S310  # local
                answer = json.loads(response.read())
        finally:
            upstream.shutdown()
            upstream.server_close()

        assert answer == {"late": True}
        assert closed.wait(10)
