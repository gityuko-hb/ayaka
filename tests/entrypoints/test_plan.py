"""``ayaka plan``: the plan of a start-up, printed without loading a weight.

The test of record is that the plan equals reality: the numbers ``plan_engine`` reports are the
ones ``build_engine`` then builds the engine with, on the same machine.
"""

from __future__ import annotations

import dataclasses
import json
from typing import TYPE_CHECKING, Any

import pytest

from ayaka.configs.cache import CacheConfig
from ayaka.core.errors import CapacityError
from ayaka.entrypoints.cli import main
from ayaka.entrypoints.console import EXIT_LOAD, EXIT_MEMORY, EXIT_OK
from ayaka.entrypoints.plan import shortfall_lines, shortfall_of

torch: Any = pytest.importorskip("torch")

from ayaka.engine.build import build_engine, plan_engine  # noqa: E402
from ayaka.model_loader import loader  # noqa: E402
from tiny_model import (  # noqa: E402  # needs the importorskip above
    TinyCheckpoint,
    patch_tokenizer_loading,
    tiny_engine_config,
    tiny_tokenizer,
    write_tiny_gpt2,
)

if TYPE_CHECKING:
    from pathlib import Path

pytestmark = pytest.mark.unit


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory: pytest.TempPathFactory) -> TinyCheckpoint:
    return write_tiny_gpt2(tmp_path_factory.mktemp("plan-gpt2"))


@pytest.fixture(autouse=True)
def _profiles_in_tmp(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AYAKA_CACHE_DIR", str(tmp_path / "cache"))
    patch_tokenizer_loading(monkeypatch)


class TestPlanEqualsReality:
    @pytest.mark.parametrize("num_blocks", [None, 40], ids=["derived", "fixed"])
    def test_the_engine_is_built_with_the_planned_numbers(
        self, checkpoint: TinyCheckpoint, num_blocks: int | None
    ) -> None:
        config = tiny_engine_config(checkpoint.directory, num_blocks=num_blocks)

        plan = plan_engine(config)
        engine = build_engine(config, tokenizer=tiny_tokenizer())

        try:
            stats = engine.kv.stats()
            assert stats.usable_blocks == (plan.kv_blocks - 1,)
            assert engine.kv.table.max_requests == plan.max_running
            assert engine.kv.max_tokens_per_request >= plan.max_context_tokens
            assert engine.scheduler.token_budget == plan.chunk_tokens
            weights = sum(p.numel() * p.element_size() for p in engine.runner.model.parameters())
            assert weights == plan.weights_bytes
            # What could fit follows the free memory of the moment, which moves between the two
            # readings; the blocks allocated above do not.
            [planned], [built] = plan.budget.kv_blocks, engine.plan.kv_blocks
            assert abs(planned - built) <= 0.01 * planned
        finally:
            engine.shutdown()

    def test_planning_reads_no_weight(
        self, checkpoint: TinyCheckpoint, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def no_weights(*args: object, **kwargs: object) -> None:
            raise AssertionError("a plan read a weight")

        monkeypatch.setattr(loader, "_stream", no_weights)

        plan = plan_engine(tiny_engine_config(checkpoint.directory))

        assert plan.weights_bytes > 0
        assert plan.directory == checkpoint.directory


class TestCommand:
    def test_a_plan_that_fits_prints_the_table_and_exits_0(
        self, checkpoint: TinyCheckpoint, capsys: pytest.CaptureFixture[str]
    ) -> None:
        code = main(["plan", str(checkpoint.directory)])

        out = capsys.readouterr().out
        assert code == EXIT_OK
        assert "verdict    FITS." in out
        assert "not measured on this machine" in out
        assert "no weight read, nothing downloaded" in out

    def test_json_output_has_the_same_numbers(
        self, checkpoint: TinyCheckpoint, capsys: pytest.CaptureFixture[str]
    ) -> None:
        code = main(["plan", str(checkpoint.directory), "--output", "json"])

        report = json.loads(capsys.readouterr().out)
        plan = plan_engine(dataclasses.replace(tiny_engine_config(checkpoint.directory)))
        assert code == EXIT_OK
        assert report["fits"] is True
        assert report["memory"]["weights_bytes"] == plan.weights_bytes
        assert report["kv"]["bytes_per_token"] == plan.group.spec.bytes_per_token()
        assert report["speed"] is None

    def test_a_plan_that_does_not_fit_exits_5_with_three_ways_out(
        self, checkpoint: TinyCheckpoint, capsys: pytest.CaptureFixture[str]
    ) -> None:
        code = main(["plan", str(checkpoint.directory), "--memory-ratio", "0.0000001"])

        out = capsys.readouterr().out
        assert code == EXIT_MEMORY
        assert "DOES NOT FIT" in out
        assert out.count("  fix      ") >= 2

    def test_for_a_tool_says_why_the_tiny_context_does_not_hold_it(
        self, checkpoint: TinyCheckpoint, capsys: pytest.CaptureFixture[str]
    ) -> None:
        code = main(["plan", str(checkpoint.directory), "--for", "opencode"])
        text = capsys.readouterr().out
        json_code = main(
            ["plan", str(checkpoint.directory), "--for", "opencode", "--output", "json"]
        )
        report = json.loads(capsys.readouterr().out)

        assert code == json_code == EXIT_MEMORY
        assert "for        opencode: DOES NOT FIT" in text
        fit = report["for"]
        assert fit["fits"] is False
        assert fit["needed_tokens"] == fit["floor_tokens"] + fit["margin_tokens"]
        assert fit["context_tokens"] == report["context"]["max_tokens_per_request"]

    def test_a_tool_fit_reads_the_plans_context(self) -> None:
        from ayaka.entrypoints.connect.tools import TOOLS
        from ayaka.entrypoints.plan import ToolFit, tool_fit_lines

        record = TOOLS["codex"]
        assert record.min_context_tokens is not None
        roomy = ToolFit("codex", 10**6, record.min_context_tokens, 8192, record.measured)
        unmeasured = ToolFit("codex", 10**6, None, 8192, None)

        assert roomy.fits is True
        assert "FITS" in tool_fit_lines(roomy)[0]
        assert unmeasured.fits is None
        assert "not measured" in tool_fit_lines(unmeasured)[0]

    def test_a_repo_that_is_not_cached_is_never_downloaded(
        self, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        pytest.importorskip("huggingface_hub", reason="the hf extra resolves repo ids")
        monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path / "empty-hub"))

        code = main(["plan", "nobody/not-a-cached-model"])

        assert code == EXIT_LOAD
        assert "error" in capsys.readouterr().err


class TestShortfall:
    def test_a_shorter_context_is_offered_with_the_number_that_fits(self) -> None:
        error = CapacityError(
            "KV floor: need 4000 bytes, have 1000, short 3000",
            context={
                "what": "KV floor",
                "need": 4000,
                "have": 1000,
                "short": 3000,
                "bytes_per_token": 100,
                "weights_bytes": 5000,
                "kv_reserve_tokens": 128,
                "model_context": 128,
            },
        )

        lines = shortfall_lines(shortfall_of(error))

        assert any("--kv-reserve-tokens 98" in line for line in lines)
        assert any("smaller model" in line for line in lines)

    def test_a_kv_reserve_beyond_the_context_is_refused(self, checkpoint: TinyCheckpoint) -> None:
        config = tiny_engine_config(checkpoint.directory, num_blocks=None)
        config = dataclasses.replace(config, cache=CacheConfig(kv_reserve_tokens=10_000))

        with pytest.raises(Exception, match="exceeds the model context"):
            plan_engine(config)
