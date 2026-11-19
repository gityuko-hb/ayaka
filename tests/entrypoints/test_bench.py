"""``ayaka bench``: the prompt set, results.json, the power source and a run on the tiny model."""

from __future__ import annotations

import copy
import json
import statistics
from typing import TYPE_CHECKING, Any

import pytest

from ayaka.entrypoints import bench
from ayaka.entrypoints.bench import SCHEMA, BenchOptions
from ayaka.entrypoints.bench_machine import machine_slug, power_source
from ayaka.entrypoints.bench_prompts import GROUPS, build_prompt, material_text, prompt_digest
from ayaka.entrypoints.bench_results import (
    summary_line,
    summary_lines,
    validate_results,
    write_results,
)

if TYPE_CHECKING:
    from pathlib import Path

pytestmark = pytest.mark.unit


def _bytes(text: str) -> list[int]:
    return list(text.encode())


class TestPromptSet:
    @pytest.mark.parametrize("group", GROUPS)
    @pytest.mark.parametrize("length", [256, 1024, 4096])
    def test_a_prompt_has_exactly_its_length(self, group: str, length: int) -> None:
        prompt = build_prompt(group, length, _bytes)

        assert len(prompt.token_ids) == length
        assert bytes(prompt.token_ids[-40:]).decode(errors="replace")  # the instruction ends it

    def test_the_set_is_deterministic(self) -> None:
        first = [prompt_digest(build_prompt(g, 2048, _bytes)) for g in GROUPS]
        second = [prompt_digest(build_prompt(g, 2048, _bytes)) for g in GROUPS]

        assert first == second
        assert len(set(first)) == 3
        assert material_text("edit", 3) == material_text("edit", 3)

    def test_the_edit_material_has_the_function_its_instruction_names(self) -> None:
        assert "def compute_total(" in material_text("edit", 64)

    def test_an_unknown_group_or_a_tiny_length_is_refused(self) -> None:
        with pytest.raises(ValueError, match="unknown prompt group"):
            build_prompt("poetry", 1024, _bytes)
        with pytest.raises(ValueError, match="cannot hold"):
            build_prompt("prose", 10, _bytes)


def _results(**changes: Any) -> dict[str, Any]:
    raw = [100.0, 120.0, 110.0]
    data: dict[str, Any] = {
        "schema": SCHEMA,
        "valid": True,
        "model": {"name": "tiny"},
        "engine": {},
        "backends": {},
        "plan": {},
        "machine": {"gpu": "NVIDIA GeForce RTX 3050 Laptop GPU", "ayaka": {"commit": "abc"}},
        "power": "ac",
        "thermal": {},
        "prompt_set": {},
        "prompt_read": [
            {
                "group": "prose",
                "length": 1024,
                "tokens_per_s": raw,
                "median_tokens_per_s": statistics.median(raw),
            }
        ],
        "generation": [
            {"context": 1024, "tokens_per_s": raw, "median_tokens_per_s": statistics.median(raw)}
        ],
        "ttft": {
            "system_tokens": 100,
            "cold_s": [0.2, 0.3],
            "warm_s": [0.1, 0.1],
            "cold_median_s": 0.25,
            "warm_median_s": 0.1,
        },
        "agent": {"turns": [], "cached_share": 0.9, "total_s": 1.0},
        "sanity": [{"name": "tool call", "passed": True, "detail": ""}],
        "memory": {"peak_vram_bytes": 2**30, "peak_rss_bytes": 2**29},
        "notes": [],
        "started": "2026-11-19T10:00:00+00:00",
        "finished": "2026-11-19T10:05:00+00:00",
    }
    data.update(changes)
    return data


class TestResults:
    def test_a_sound_result_validates(self) -> None:
        assert validate_results(_results()) == []

    def test_a_median_that_disagrees_with_its_runs_is_caught(self) -> None:
        data = _results()
        data["prompt_read"][0]["median_tokens_per_s"] = 999.0

        assert any("median" in problem for problem in validate_results(data))

    def test_valid_must_equal_every_sanity_item(self) -> None:
        failed = _results(sanity=[{"name": "tool call", "passed": False, "detail": "x"}])

        assert validate_results(failed) == ["valid does not equal all(sanity passed)"]
        assert validate_results({**failed, "valid": False}) == []

    def test_a_missing_part_or_another_schema_is_caught(self) -> None:
        data = _results()
        del data["ttft"]

        assert "missing 'ttft'" in validate_results(data)
        assert validate_results(_results(schema="other/1"))

    def test_the_summary_reports_parts_separately_and_the_history_line(
        self, tmp_path: Path
    ) -> None:
        data = _results()

        lines = summary_lines(data)
        path, line_path = write_results(data, tmp_path / "out", tmp_path / "history")

        assert any(line.startswith("read") for line in lines)
        assert any(line.startswith("generate") for line in lines)
        assert any(line.startswith("first tok") for line in lines)
        assert not any("overall" in line or "score" in line for line in lines)
        assert json.loads(path.read_text(encoding="utf-8")) == data
        assert line_path is not None
        [row] = [json.loads(x) for x in line_path.read_text(encoding="utf-8").splitlines()]
        assert row == summary_line(data)
        assert row["machine"] == "rtx-3050-laptop-gpu"
        assert row["read_tok_s"] == {"prose/1024": 110.0}


class TestMachine:
    def test_a_hub_cache_path_is_named_by_its_repo_id(self) -> None:
        from ayaka.entrypoints.bench import model_label

        path = "/home/someone/.cache/huggingface/hub/models--unsloth--Llama-3.2-1B/snapshots/5a8"

        assert model_label(path) == "unsloth/Llama-3.2-1B"
        assert model_label("org/model") == "org/model"

    def test_the_folder_name_is_the_gpu_not_the_host(self) -> None:
        assert machine_slug("NVIDIA GeForce RTX 3050 Laptop GPU") == "rtx-3050-laptop-gpu"
        assert machine_slug(None) == "cpu"

    @staticmethod
    def _supply(root: Path, name: str, kind: str, online: str | None = None) -> None:
        entry = root / name
        entry.mkdir(parents=True)
        (entry / "type").write_text(kind + "\n", encoding="utf-8")
        if online is not None:
            (entry / "online").write_text(online + "\n", encoding="utf-8")

    def test_linux_reads_the_power_supply(self, tmp_path: Path) -> None:
        plugged, unplugged, desktop = tmp_path / "a", tmp_path / "b", tmp_path / "c"
        self._supply(plugged, "AC", "Mains", "1")
        self._supply(plugged, "BAT0", "Battery")
        self._supply(unplugged, "AC", "Mains", "0")
        self._supply(unplugged, "BAT0", "Battery")
        desktop.mkdir()

        def source(root: Path) -> str:
            return power_source(platform_name="linux", sysfs=root, wsl=False)

        assert source(plugged) == "ac"
        assert source(unplugged) == "battery"
        assert source(desktop) == "ac"
        assert source(tmp_path / "missing") == "unknown"

    def test_windows_and_wsl_ask_windows(self) -> None:
        assert power_source(platform_name="win32", windows_status=lambda: True) == "ac"
        assert power_source(platform_name="linux", wsl=True, windows_status=lambda: False) == (
            "battery"
        )
        assert power_source(platform_name="win32", windows_status=lambda: None) == "unknown"


# -- one run on the tiny model, on the CPU -----------------------------------------------------

torch: Any = pytest.importorskip("torch")


@pytest.fixture(scope="module")
def long_checkpoint(tmp_path_factory: pytest.TempPathFactory) -> Path:
    from gpt2_support import tiny_config, write_checkpoint
    from tiny_model import tiny_gpt2_state
    from tokenizer_fakes import MODEL_VOCAB

    config = tiny_config(vocab_size=MODEL_VOCAB, max_position_embeddings=1024)
    directory = tmp_path_factory.mktemp("bench-gpt2")
    write_checkpoint(directory, tiny_gpt2_state(config), config)
    return directory


def _run(
    checkpoint: Path,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    run_bench: Any = bench.run_bench,  # bound here: a test may replace the module's name
) -> dict[str, Any]:
    from ayaka.engine.llm import LLM
    from tiny_model import tiny_engine_config, tiny_tokenizer

    monkeypatch.setattr(bench, "QUICK_LENGTHS", (256, 384))
    monkeypatch.setattr(bench, "GENERATION_TOKENS", 24)
    config = tiny_engine_config(checkpoint, num_blocks=600, max_tokens_per_step=128)
    options = BenchOptions(model="tiny", quick=True, record=tmp_path / "rec")
    return run_bench(
        config,
        options,
        say=lambda line: None,
        load=lambda c: LLM.from_config(c, tokenizer=tiny_tokenizer()),
    )


class TestRun:
    def test_a_quick_run_writes_a_result_that_validates(
        self, long_checkpoint: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        results = _run(long_checkpoint, monkeypatch, tmp_path)

        assert validate_results(results) == [], validate_results(results)
        assert [(row["group"], row["length"]) for row in results["prompt_read"]] == [
            (group, length) for group in GROUPS for length in (256, 384)
        ]
        for row in results["prompt_read"]:
            assert len(row["tokens_per_s"]) == 2
            assert all(rate > 0 for rate in row["tokens_per_s"])
        assert [row["context"] for row in results["generation"]] == [256, 384]
        assert results["ttft"]["cold_s"]
        assert results["ttft"]["warm_s"]
        turns = results["agent"]["turns"]
        assert len(turns) == 3
        assert turns[1]["cached_tokens"] > 0  # the session's start is read from the cache
        assert {item["name"] for item in results["sanity"]} == {"tool call", "structured output"}
        assert results["valid"] is all(item["passed"] for item in results["sanity"])
        assert results["memory"]["samples"] > 0
        plan = results["plan"]
        assert plan["engine_context_tokens"] <= plan["max_context_tokens"]
        assert results["prompt_set"]["digests"]
        sessions = list((tmp_path / "rec").glob("session-*.jsonl"))
        assert len(sessions) == 1

    def test_a_failed_sanity_check_marks_the_result_and_exits_2(
        self,
        long_checkpoint: Path,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        from ayaka.entrypoints.bench import SanityItem
        from ayaka.entrypoints.cli import main

        def failing(self: Any) -> list[SanityItem]:
            del self
            return [SanityItem("tool call", passed=False, detail="scripted failure")]

        results_box: list[dict[str, Any]] = []

        def run(config: Any, options: Any, **kwargs: Any) -> dict[str, Any]:
            del config, kwargs
            monkeypatch.setattr(bench._Runner, "sanity", failing)  # pyright: ignore[reportPrivateUsage]
            results = _run(long_checkpoint, monkeypatch, tmp_path)
            results_box.append(results)
            return results

        monkeypatch.setattr(bench, "run_bench", run)
        code = main(
            [
                "bench",
                str(long_checkpoint),
                "--out",
                str(tmp_path / "out"),
                "--history",
                str(tmp_path / "history"),  # never the repository's own history file
                "--quick",
            ]
        )
        written = json.loads((tmp_path / "out" / "results.json").read_text(encoding="utf-8"))

        assert code == 2
        assert (tmp_path / "history" / "bench.jsonl").is_file()
        assert written["valid"] is False
        assert any("sanity check failed" in note for note in written["notes"])
        assert "FAIL  tool call" in capsys.readouterr().out
        assert copy.deepcopy(results_box[0]) == written
