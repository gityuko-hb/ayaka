"""``ayaka info`` and ``ayaka config show``, and where the user's configuration file lives."""

from __future__ import annotations

import json
import sys
from typing import TYPE_CHECKING

import pytest

from ayaka.configs.server import ServerConfig, user_config_path
from ayaka.core.errors import ConfigError
from ayaka.entrypoints.cli import EXIT_OK, EXIT_USAGE, main
from ayaka.entrypoints.info import config_rows, info_lines, info_report

if TYPE_CHECKING:
    from pathlib import Path

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def _isolated(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """No real user file or cache: the test controls every path."""
    monkeypatch.setenv("AYAKA_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    monkeypatch.setenv("APPDATA", str(tmp_path / "appdata"))
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path / "hub"))


class TestInfo:
    def test_the_report_names_the_version_extras_and_paths(self) -> None:
        report = info_report()

        assert report.version
        assert "serve" in report.extras
        names = [entry.name for entry in report.paths]
        assert names == ["startup profiles", "Hugging Face cache", "configuration file"]
        assert any(backend.name == "torch" for backend in report.backends)

    def test_the_lines_say_missing_rather_than_zero(self) -> None:
        lines = info_lines(info_report())

        assert any(line.startswith("ayaka      ") for line in lines)
        assert any("(missing)" in line for line in lines if line.startswith("path"))

    def test_json_output_parses(self, capsys: pytest.CaptureFixture[str]) -> None:
        code = main(["info", "--output", "json"])

        report = json.loads(capsys.readouterr().out)
        assert code == EXIT_OK
        assert report["version"]
        assert isinstance(report["devices"], list)


class TestConfigShow:
    def test_each_setting_names_its_source(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        file = tmp_path / "ayaka.toml"
        file.write_text('[server]\nport = 9001\nhost = "127.0.0.1"\n', encoding="utf-8")

        code = main(["config", "show", "--config", str(file), "--port", "9002"])

        out = capsys.readouterr().out
        rows = {line.split()[0]: line for line in out.splitlines()[1:]}
        assert code == EXIT_OK
        assert "9002" in rows["port"]
        assert rows["port"].endswith("[flag]")
        assert rows["host"].endswith(f"[file {file}]")
        assert rows["keepalive_seconds"].endswith("[default]")

    def test_the_api_key_is_never_printed(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        key = tmp_path / "key"
        key.write_text("s3cret-value\n", encoding="utf-8")

        main(["config", "show", "--api-key-file", str(key)])

        out = capsys.readouterr().out
        assert "s3cret-value" not in out
        assert "api_key" in out
        assert "set" in out

    def test_the_user_file_is_read_without_a_flag(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        path = user_config_path()
        path.parent.mkdir(parents=True)
        path.write_text("[server]\nport = 9100\n", encoding="utf-8")

        main(["config", "show"])

        out = capsys.readouterr().out
        assert f"configuration file: {path}" in out
        assert "9100" in out

    def test_a_typo_suggests_the_nearest_key(self) -> None:
        with pytest.raises(ConfigError, match="did you mean port"):
            ServerConfig.from_mapping({"prot": 1})

    def test_config_without_an_action_is_a_usage_error(self) -> None:
        assert main(["config"]) == EXIT_USAGE

    def test_rows_follow_flag_then_file_then_default(self) -> None:
        rows = {
            name: (value, source)
            for name, value, source in config_rows({"port": 1}, {"port": 2, "host": "::1"}, None)
        }

        assert rows["port"] == ("1", "flag")
        assert rows["host"] == ("::1", "file None")


class TestUserConfigPath:
    @pytest.mark.parametrize(
        ("platform", "expected"),
        [("win32", "appdata"), ("linux", "xdg")],
    )
    def test_the_os_convention_is_followed(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, platform: str, expected: str
    ) -> None:
        monkeypatch.setattr(sys, "platform", platform)  # read by user_config_path at call time

        path = user_config_path()

        assert path.parts[-2:] == ("ayaka", "config.toml")
        assert str(tmp_path / expected) in str(path)
