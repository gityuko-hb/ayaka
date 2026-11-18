"""``ayaka connect``: records, rendering, the diffed write, the seven steps and the proxy."""

from __future__ import annotations

import asyncio
import http.server
import json
import threading
import tomllib
import urllib.request
from typing import TYPE_CHECKING, Any

import pytest

from ayaka.entrypoints.cli import EXIT_OK, EXIT_USAGE, main
from ayaka.entrypoints.connect import (
    TOOLS,
    ServerInfo,
    count_parts,
    observer,
    recorder,
    redact_headers,
    render_config,
    run_check,
    serve_proxy,
    write_config,
)

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

        config = json.loads(render_config(TOOLS["opencode"], _SERVER, current))

        assert config["theme"] == "dark"
        assert config["providers"]["other"] == {"name": "x"}
        ayaka = config["providers"]["ayaka"]
        assert ayaka["settings"]["baseURL"] == "http://127.0.0.1:8000/v1"
        assert ayaka["env"] == ["AYAKA_API_KEY"]
        assert ayaka["models"]["tiny"]["limit"] == {"context": 1024}
        assert config["model"] == "ayaka/tiny"

    def test_every_record_has_a_home_path(self) -> None:
        for record in TOOLS.values():
            path = record.config_path({})
            assert path.name.endswith((".json", ".toml"))
            assert "~" not in str(path)

    def test_the_tools_own_variable_moves_the_file(self, tmp_path: Path) -> None:
        env = {"CODEX_HOME": str(tmp_path), "CLAUDE_CONFIG_DIR": str(tmp_path / "c")}

        assert TOOLS["codex"].config_path(env) == tmp_path / "ayaka.config.toml"
        assert TOOLS["claude-code"].config_path(env) == tmp_path / "c" / "settings.json"

    def test_no_record_claims_a_measured_floor_it_does_not_have(self) -> None:
        assert all(record.min_context_tokens is None for record in TOOLS.values())


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
        observe = observer("claude-code", "2.1", self._count, found)
        body = json.dumps({"system": "secret system text", "messages": [], "max_tokens": 1})

        observe("/v1/messages?beta=true", {"authorization": "Bearer x"}, body.encode())
        observe("/v1/messages", {}, body.encode())

        assert len(found) == 1
        assert found[0].system_tokens == len("secret system text")
        assert "secret" not in json.dumps(found[0].__repr__())

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
        assert results == {1: True, 2: True, 3: True, 4: True, 5: True, 6: True, 7: None}, details
        assert report.ok
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
