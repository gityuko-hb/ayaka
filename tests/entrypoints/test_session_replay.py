"""Unit tests for ``ayaka replay`` session recording replay and host-KV CLI flags."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import pytest

from ayaka.configs.model_source import ModelSourceConfig
from ayaka.core.params import SamplingParams
from ayaka.engine.recorder import SessionRecorder
from ayaka.entrypoints.cli import _engine_config, _parser, main
from ayaka.entrypoints.replay import replay_lines, replay_session

torch: Any = pytest.importorskip("torch")

from ayaka.engine.llm import LLM  # noqa: E402
from tiny_model import (  # noqa: E402
    TinyCheckpoint,
    greedy_reference,
    prompt_ids,
    tiny_engine_config,
    tiny_tokenizer,
    write_tiny_gpt2,
)

if TYPE_CHECKING:
    from pathlib import Path

pytestmark = pytest.mark.unit


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory: pytest.TempPathFactory) -> TinyCheckpoint:
    return write_tiny_gpt2(tmp_path_factory.mktemp("session-replay"))


def _write_recorded_session(tmp_path: Path, checkpoint: TinyCheckpoint) -> Path:
    rec_dir = tmp_path / "rec"
    recorder = SessionRecorder(rec_dir, name="test-session.jsonl")
    cfg = tiny_engine_config(checkpoint.directory)
    with LLM.from_config(cfg, tokenizer=tiny_tokenizer(), recorder=recorder) as llm:
        tokens = tuple(prompt_ids("shared system header for replay verification"))
        llm.generate(tokens, SamplingParams(temperature=0.0, max_tokens=4))
        llm.generate(tokens, SamplingParams(temperature=0.0, max_tokens=4))
    return recorder.path


def test_host_kv_cli_flags_and_replay_parser_wire_config() -> None:
    parser = _parser()
    args = parser.parse_args(
        [
            "serve",
            "org/model",
            "--host-kv",
            "--ram-keep",
            "0.30",
            "--host-kv-max-mb",
            "512",
        ]
    )
    cfg = _engine_config(args)
    assert cfg.host_memory.enabled is True
    assert cfg.host_memory.ram_keep_fraction == pytest.approx(0.30)
    assert cfg.host_memory.max_host_kv_mb == 512

    default_args = parser.parse_args(["serve", "org/model"])
    default_cfg = _engine_config(default_args)
    assert default_cfg.host_memory.enabled is False

    replay_args = parser.parse_args(
        ["replay", "trace.jsonl", "--model", "org/model", "--host-kv", "--output", "json"]
    )
    assert replay_args.command == "replay"
    assert replay_args.output == "json"
    assert replay_args.host_kv is True


def test_replay_session_verifies_exact_token_parity_and_cache_hits(
    tmp_path: Path,
    checkpoint: TinyCheckpoint,
) -> None:
    session_path = _write_recorded_session(tmp_path, checkpoint)
    cfg = tiny_engine_config(checkpoint.directory)

    report = replay_session(
        session_path,
        cfg,
        engine_factory=lambda c: LLM.from_config(c, tokenizer=tiny_tokenizer()),
    )
    assert report.ok
    assert report.total_requests == 2
    assert report.parity_matched == 2
    assert report.parity_mismatched == 0
    assert report.items[1].cached_tokens > 0
    assert report.cache_hit_ratio > 0.0

    # Verify expected greedy oracle matches replayed tokens too.
    expected = greedy_reference(
        checkpoint.state,
        checkpoint.config,
        prompt_ids("shared system header for replay verification"),
        4,
    )
    assert list(report.items[0].actual_tokens) == expected

    lines = replay_lines(report)
    assert "parity: 2/2" in lines[1]
    assert "[MATCH]" in lines[2]


def test_replay_detects_parity_mismatch_and_returns_engine_exit_code(
    tmp_path: Path,
    checkpoint: TinyCheckpoint,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session_path = _write_recorded_session(tmp_path, checkpoint)
    lines = session_path.read_text(encoding="utf-8").splitlines()
    second = json.loads(lines[2])
    second["request"]["outputs"][0]["token_ids"] = [0, 0, 0, 0]
    lines[2] = json.dumps(second)
    corrupted = tmp_path / "mismatch.jsonl"
    corrupted.write_text("\n".join(lines) + "\n", encoding="utf-8")

    # Stub LLM.from_config so CLI command uses the test tiny tokenizer cleanly.
    orig = LLM.from_config
    monkeypatch.setattr(
        LLM,
        "from_config",
        classmethod(lambda cls, c, **kw: orig(c, tokenizer=tiny_tokenizer())),
    )
    code = main(
        [
            "replay",
            str(corrupted),
            "--model",
            str(checkpoint.directory),
            "--output",
            "json",
        ]
    )
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert code != 0
    assert payload["ok"] is False
    assert payload["parity_matched"] == 1
    assert payload["parity_mismatched"] == 1
    assert isinstance(ModelSourceConfig("org/model").model, str)
