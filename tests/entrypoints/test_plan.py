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
from ayaka.configs.device import DeviceConfig
from ayaka.core.errors import CapacityError
from ayaka.device.info import DeviceInfo
from ayaka.engine.profile import StartupProfile, save_profile
from ayaka.entrypoints.cli import main
from ayaka.entrypoints.console import EXIT_LOAD, EXIT_MEMORY, EXIT_OK
from ayaka.entrypoints.plan import shortfall_lines, shortfall_of
from ayaka.memory.budget import BudgetInputs, KVCost, plan_budget
from ayaka.utils.dtypes import DType

torch: Any = pytest.importorskip("torch")

import ayaka.device.context as device_context  # noqa: E402
import ayaka.device.info as device_info  # noqa: E402
from ayaka.device.context import MemoryReading  # noqa: E402
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
        assert report["memory"]["state_bytes"] == 0
        assert report["memory"]["latent_kv_bytes"] == 0
        assert report["memory"]["vision_weights_bytes"] == 0
        assert report["kv"]["groups"] == 1
        assert report["kv"]["bytes_per_token"] == plan.group.spec.bytes_per_token()
        assert report["speed"] is None

    def test_hybrid_mla_and_vision_fields_surface_in_plan_json_and_lines(
        self, checkpoint: TinyCheckpoint
    ) -> None:
        from ayaka.entrypoints.plan import plan_lines, plan_report

        base = plan_engine(dataclasses.replace(tiny_engine_config(checkpoint.directory)))
        enriched = dataclasses.replace(
            base,
            kv_groups=2,
            state_bytes=8 * 2**20,
            latent_kv_bytes=4 * 2**20,
            vision_weights_bytes=32 * 2**20,
            vision_activation_bytes=6 * 2**20,
            vision_load="lazy",
        )
        rep: dict[str, Any] = plan_report(enriched, "hybrid-vision")
        assert rep["kv"]["groups"] == 2
        assert rep["memory"]["state_bytes"] == 8 * 2**20
        assert rep["memory"]["latent_kv_bytes"] == 4 * 2**20
        assert rep["memory"]["vision_weights_bytes"] == 32 * 2**20
        assert rep["memory"]["vision_activation_bytes"] == 6 * 2**20
        assert rep["memory"]["vision_load"] == "lazy"

        text = "\n".join(plan_lines(enriched, "hybrid-vision"))
        assert "2 groups" in text
        assert "  state    " in text
        assert "  latent   " in text
        assert "  vision   " in text
        assert "lazy" in text

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

    def test_a_sliding_window_shorter_than_the_context_caps_the_plan(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        windowed = write_tiny_gpt2(tmp_path / "windowed")
        config_path = windowed.directory / "config.json"
        raw = json.loads(config_path.read_text(encoding="utf-8"))
        config_path.write_text(json.dumps({**raw, "sliding_window": 16}), encoding="utf-8")

        with caplog.at_level("WARNING", logger="ayaka.engine.build"):
            plan = plan_engine(tiny_engine_config(windowed.directory, num_blocks=None))

        assert plan.max_context_tokens == 16
        assert "16-token sliding window" in caplog.text


class TestColdCudaRuntimeGap:
    _MIB = 2**20

    @staticmethod
    def _mock_cuda(
        monkeypatch: pytest.MonkeyPatch,
        *,
        nvml_free: int,
        cuda_free: int,
        total: int = 16 * 2**20,
        baseline_reserved: int = 0,
    ) -> None:
        fake_info = DeviceInfo(
            kind="cuda",
            index=0,
            name="Mock GPU",
            total_bytes=total,
            free_bytes=nvml_free,
            compute_capability=(8, 6),
            num_sms=20,
        )
        monkeypatch.setattr(device_info, "probe_devices", lambda: (fake_info,))

        class _FakeDevice:
            def __init__(self, spec: object) -> None:
                self.spec = spec

            def memory(self) -> MemoryReading:
                return MemoryReading(
                    free=cuda_free,
                    total=total,
                    allocated=baseline_reserved,
                    reserved=baseline_reserved,
                    peak_allocated=baseline_reserved,
                )

        monkeypatch.setattr(device_context, "Device", _FakeDevice)
        if baseline_reserved:
            monkeypatch.setattr(torch.cuda, "is_initialized", lambda: True)
            monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
            monkeypatch.setattr(torch.cuda, "memory_reserved", lambda _index=0: baseline_reserved)
        else:
            monkeypatch.setattr(torch.cuda, "is_initialized", lambda: False)

    @staticmethod
    def _cuda_config(checkpoint: TinyCheckpoint) -> Any:
        base = tiny_engine_config(
            checkpoint.directory, block_size=4, num_blocks=None, max_running=None
        )
        return dataclasses.replace(
            base,
            device=DeviceConfig(
                device="cuda", dtype=DType.FP16, backends=("torch",), cuda_graph=True
            ),
            cache=CacheConfig(
                block_size=4, num_blocks=None, kv_reserve_tokens=32, memory_ratio=0.9
            ),
        )

    @staticmethod
    def _expected_budget(plan: Any, *, baseline: int, late_alloc_bytes: int) -> Any:
        return plan_budget(
            BudgetInputs(
                baseline_bytes=baseline,
                memory_ratio=0.9,
                weights_bytes=plan.weights_bytes,
                kv=(
                    KVCost(
                        bytes_per_block=plan.group.spec.bytes_per_block(),
                        fixed_bytes=plan.group.spec.fixed_bytes(),
                        block_size=plan.block_size,
                    ),
                ),
                kv_reserve_tokens=32,
                late_alloc_bytes=late_alloc_bytes,
            )
        )

    def test_cold_cuda_plan_accounts_for_nvml_to_cuda_gap(
        self, checkpoint: TinyCheckpoint, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        config = self._cuda_config(checkpoint)
        nvml_free = 10 * self._MIB
        cuda_free = 6 * self._MIB

        self._mock_cuda(monkeypatch, nvml_free=nvml_free, cuda_free=nvml_free)
        no_gap_plan = plan_engine(config)

        self._mock_cuda(monkeypatch, nvml_free=nvml_free, cuda_free=cuda_free)
        cold_plan = plan_engine(config)

        expected = self._expected_budget(
            cold_plan, baseline=nvml_free, late_alloc_bytes=nvml_free - cuda_free
        )
        assert not cold_plan.profiled
        assert cold_plan.budget == expected
        assert cold_plan.budget.kv_blocks[0] < no_gap_plan.budget.kv_blocks[0]

    def test_cold_cuda_plan_does_not_double_count_pre_existing_reservations(
        self, checkpoint: TinyCheckpoint, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        config = self._cuda_config(checkpoint)
        nvml_free = 10 * self._MIB
        cuda_free = 6 * self._MIB

        self._mock_cuda(
            monkeypatch, nvml_free=nvml_free, cuda_free=cuda_free, baseline_reserved=3 * self._MIB
        )
        shared_plan = plan_engine(config)

        expected = self._expected_budget(
            shared_plan, baseline=nvml_free, late_alloc_bytes=nvml_free - cuda_free
        )
        assert shared_plan.budget == expected

    def test_cold_cuda_plan_rejects_when_gap_exceeds_usable_memory(
        self, checkpoint: TinyCheckpoint, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        config = self._cuda_config(checkpoint)
        self._mock_cuda(monkeypatch, nvml_free=10 * self._MIB, cuda_free=512 * 1024)

        with pytest.raises(CapacityError):
            plan_engine(config)

    def test_warm_cuda_plan_never_drops_below_observed_nvml_to_cuda_gap(
        self, checkpoint: TinyCheckpoint, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        config = self._cuda_config(checkpoint)
        nvml_free = 10 * self._MIB
        cuda_free = 6 * self._MIB
        self._mock_cuda(monkeypatch, nvml_free=nvml_free, cuda_free=cuda_free)
        cold_plan = plan_engine(config)

        stale = StartupProfile(
            key=cold_plan.profile_key,
            weights_bytes=cold_plan.weights_bytes,
            kv_bytes=cold_plan.kv_bytes,
            runtime_bytes=1 * self._MIB,
            activation_tokens=16,
            activation_peak_bytes=0,
        )
        save_profile(stale)
        stale_plan = plan_engine(config)
        assert stale_plan.profiled
        assert stale_plan.budget == self._expected_budget(
            cold_plan, baseline=nvml_free, late_alloc_bytes=nvml_free - cuda_free
        )

        honest = dataclasses.replace(stale, runtime_bytes=5 * self._MIB)
        save_profile(honest)
        honest_plan = plan_engine(config)
        assert honest_plan.profiled
        assert honest_plan.budget == self._expected_budget(
            cold_plan, baseline=nvml_free, late_alloc_bytes=5 * self._MIB
        )


class TestQuantizedPlan:
    @staticmethod
    def _fp8_llama(directory: Path) -> Path:
        from fp8_support import HF_FP8_BLOCK, fp8_data, quantize_state
        from gpt2_support import tensor_data
        from llama_support import llama_config, llama_config_dict, llama_state
        from safetensors_files import write_safetensors

        config = llama_config()
        stored, _ = quantize_state(llama_state(config), "block", scale_name="weight_scale_inv")
        directory.mkdir(parents=True)
        raw = {**llama_config_dict(config), "quantization_config": HF_FP8_BLOCK}
        (directory / "config.json").write_text(json.dumps(raw), encoding="utf-8")
        write_safetensors(
            directory / "model.safetensors",
            {
                name: fp8_data(t) if t.dtype == torch.float8_e4m3fn else tensor_data(t)
                for name, t in stored.items()
            },
        )
        return directory

    def test_the_plan_says_in_one_line_that_the_weights_are_sliced(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        directory = self._fp8_llama(tmp_path / "fp8")

        code = main(["plan", str(directory)])

        out = capsys.readouterr().out
        assert code == EXIT_OK
        assert "fp8_e4m3/block-16x16 projections" in out
        quantized = [line for line in out.splitlines() if line.startswith("quantized  ")]
        assert len(quantized) == 1
        assert "dequantized in slices" in quantized[0]

    def test_the_planned_weights_are_the_loaded_eight_bit_weights(self, tmp_path: Path) -> None:
        from ayaka.configs.model_source import ModelSourceConfig

        directory = self._fp8_llama(tmp_path / "fp8")
        config = tiny_engine_config(directory, num_blocks=None)

        plan = plan_engine(config)
        model = loader.load_model(ModelSourceConfig(str(directory)), config.device)

        assert plan.quantization == "fp8_e4m3/block-16x16"
        assert plan.weights_bytes == sum(p.numel() * p.element_size() for p in model.parameters())
        assert "+fp8_e4m3/block-16x16" in plan.profile_key.model


class TestGGUFPlan:
    @staticmethod
    def _gguf(directory: Path) -> Path:
        from ayaka.weights.gguf import GGMLType
        from gguf_files import llama_metadata, write_gguf

        entries = [
            ("token_embd.weight", [256, 256], GGMLType.Q6_K),
            ("output.weight", [256, 256], GGMLType.Q4_K),
            ("output_norm.weight", [32], GGMLType.F32),
            ("blk.0.attn_q.weight", [32, 32], GGMLType.Q8_0),
            ("blk.0.ffn_down.weight", [64, 32], GGMLType.Q4_0),
        ]
        table, offset = [], 0
        for name, ne, ggml in entries:
            table.append((name, ne, ggml.type_id, offset))
            offset += -(-ggml.nbytes(tuple(reversed(ne))) // 32) * 32
        directory.mkdir(parents=True)
        return write_gguf(
            directory / "tiny-Q4_K_M.gguf", llama_metadata(), table, data_bytes=offset
        )

    def test_the_weights_are_the_sum_of_the_header_sizes(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        from ayaka.weights.gguf import read_gguf

        path = self._gguf(tmp_path / "gguf")

        code = main(["plan", str(path), "--output", "json"])

        report = json.loads(capsys.readouterr().out)
        assert code == EXIT_OK
        assert report["memory"]["weights_bytes"] == read_gguf(path).weights_bytes
        assert report["gguf_types"] == {"F32": 1, "Q4_0": 1, "Q4_K": 1, "Q6_K": 1, "Q8_0": 1}
        assert report["architecture"] == "LlamaForCausalLM"

    def test_the_table_names_the_types_and_says_it_does_not_run(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        path = self._gguf(tmp_path / "gguf")

        code = main(["plan", str(path)])

        out = capsys.readouterr().out
        assert code == EXIT_OK
        assert "GGUF header read; no tensor read" in out
        assert "GGUF tensors: 1 F32, 1 Q4_0, 1 Q4_K, 1 Q6_K, 1 Q8_0" in out
        assert "running GGUF tensors is not implemented" in out

    def test_loading_a_gguf_file_is_refused_with_the_reason(self, tmp_path: Path) -> None:
        from ayaka.configs.model_source import ModelSourceConfig
        from ayaka.core.errors import ModelLoadError

        path = self._gguf(tmp_path / "gguf")

        with pytest.raises(ModelLoadError, match="running GGUF tensors is not implemented"):
            loader.load_model(ModelSourceConfig(str(path)), DeviceConfig())
