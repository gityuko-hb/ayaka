"""The start-up profile: its key, saving and reading it back, the ledger check, the dummy run."""

from __future__ import annotations

import logging
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import pytest

from ayaka.configs.device import DeviceConfig
from ayaka.engine.profile import (
    ProfileKey,
    StartupProfile,
    load_profile,
    profile_directory,
    profile_save_issues,
    reconcile,
    save_profile,
    should_save_profile,
)
from ayaka.memory.budget import BudgetInputs, KVCost, MemoryBudget, plan_budget
from ayaka.utils.dtypes import DType

torch: Any = pytest.importorskip("torch")

from ayaka.device.context import Device, MemoryReading  # noqa: E402
from ayaka.engine import build as engine_build  # noqa: E402
from ayaka.engine.build import _startup_run  # noqa: E402  # pyright: ignore[reportPrivateUsage]
from ayaka.engine.profile import measure_startup  # noqa: E402
from tiny_model import build_tiny_stack, tiny_engine_config  # noqa: E402  # needs the importorskip above

if TYPE_CHECKING:
    from pathlib import Path

pytestmark = pytest.mark.unit

MIB = 2**20


def _key(**overrides: str) -> ProfileKey:
    fields = {
        "device": "Test GPU sm_86",
        "model": "abc",
        "dtype": "fp16",
        "providers": {"attention": "flashinfer", "norm": "torch"}.items(),
        "block_size": 16,
    }
    fields.update(overrides)
    return ProfileKey.build(**fields)  # type: ignore[arg-type]  # the fields, by design


def _profile(key: ProfileKey | None = None) -> StartupProfile:
    return StartupProfile(
        key=key or _key(),
        weights_bytes=100 * MIB,
        kv_bytes=200 * MIB,
        runtime_bytes=300 * MIB,
        activation_tokens=512,
        activation_peak_bytes=51 * MIB,
        graph_bytes=7 * MIB,
        prefill_seconds=0.25,
        decode_seconds={1: 0.01, 4: 0.02},
    )


class TestKey:
    def test_the_providers_are_sorted_and_the_version_is_included(self) -> None:
        key = _key()

        assert key.providers == "attention=flashinfer,norm=torch"
        assert key.versions.startswith("ayaka=")

    def test_any_part_changes_the_digest(self) -> None:
        assert _key().digest() == _key().digest()
        assert _key().digest() != _key(dtype="bf16").digest()
        assert _key().digest() != _key(model="other").digest()


class TestFile:
    def test_a_saved_profile_is_read_back_equal(self, tmp_path: Path) -> None:
        profile = _profile()

        path = save_profile(profile, tmp_path)

        assert path is not None
        assert path.name == f"{profile.key.digest()}.json"
        assert load_profile(profile.key, tmp_path) == profile

    def test_nothing_saved_reads_as_none(self, tmp_path: Path) -> None:
        assert load_profile(_key(), tmp_path) is None

    def test_a_damaged_file_is_ignored(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        key = _key()
        (tmp_path / f"{key.digest()}.json").write_text("{not json", encoding="utf-8")

        assert load_profile(key, tmp_path) is None
        assert "ignoring the startup profile" in caplog.text

    def test_a_profile_of_another_version_is_ignored(self, tmp_path: Path) -> None:
        profile = _profile()
        path = save_profile(profile, tmp_path)
        assert path is not None
        path.write_text(path.read_text(encoding="utf-8").replace('"version": 1', '"version": 0'))

        assert load_profile(profile.key, tmp_path) is None

    def test_activation_per_token_rounds_up(self) -> None:
        assert _profile().activation_bytes_per_token == -(-51 * MIB // 512)

    def test_the_directory_follows_the_environment(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setenv("AYAKA_CACHE_DIR", str(tmp_path))

        assert profile_directory() == tmp_path / "profiles"

        monkeypatch.delenv("AYAKA_CACHE_DIR")
        monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg"))
        assert profile_directory() == tmp_path / "xdg" / "ayaka" / "profiles"


class TestReconcile:
    def _reading(self, free: int, *, reserved: int = 0) -> MemoryReading:
        return MemoryReading(
            free=free,
            total=4096 * MIB,
            allocated=0,
            reserved=reserved,
            peak_allocated=0,
        )

    def test_a_ledger_that_explains_the_allocator_matches(self) -> None:
        budget = MemoryBudget(3000 * MIB)
        budget.reserve("weights", 1000 * MIB)
        budget.reserve("kv", 500 * MIB)

        result = reconcile(
            budget,
            baseline_bytes=3000 * MIB,
            reading=self._reading(1480 * MIB, reserved=1520 * MIB),
        )

        assert result.measured_bytes == 1520 * MIB
        assert result.device_bytes == 1520 * MIB
        assert result.difference_bytes == 20 * MIB
        assert result.tolerance_bytes == 76 * MIB  # 5 % of 1520 MiB
        assert result.matches

    def test_memory_the_ledger_does_not_know_is_a_mismatch(self) -> None:
        budget = MemoryBudget(3000 * MIB)
        budget.reserve("weights", 1000 * MIB)

        result = reconcile(
            budget,
            baseline_bytes=3000 * MIB,
            reading=self._reading(1500 * MIB, reserved=1500 * MIB),
        )

        assert not result.matches
        assert result.difference_bytes == 500 * MIB

    def test_another_process_returning_memory_does_not_break_the_match(self) -> None:
        budget = MemoryBudget(3000 * MIB)
        budget.reserve("weights", 1000 * MIB)

        result = reconcile(
            budget,
            baseline_bytes=3000 * MIB,
            reading=self._reading(3500 * MIB, reserved=1000 * MIB),
        )

        assert result.measured_bytes == 1000 * MIB
        assert result.device_bytes == 0
        assert result.device_difference_bytes == -1000 * MIB
        assert result.matches

    def test_another_process_taking_memory_does_not_break_the_match(self) -> None:
        budget = MemoryBudget(3000 * MIB)
        budget.reserve("weights", 1000 * MIB)

        result = reconcile(
            budget,
            baseline_bytes=3000 * MIB,
            reading=self._reading(2500 * MIB, reserved=1000 * MIB),
        )

        assert result.measured_bytes == 1000 * MIB
        assert result.device_bytes == 500 * MIB
        assert result.matches

    def test_weights_in_pre_baseline_memory_are_counted(self) -> None:
        budget = MemoryBudget(3000 * MIB)
        budget.reserve("weights", 1000 * MIB)

        result = reconcile(
            budget,
            baseline_bytes=3000 * MIB,
            baseline_reserved_bytes=500 * MIB,
            reading=self._reading(2500 * MIB, reserved=1000 * MIB),
            reused_bytes=500 * MIB,
        )

        assert result.measured_bytes == 1000 * MIB  # (1000 - 500) + 500
        assert result.matches

    def test_the_runtime_owner_is_added_back(self) -> None:
        budget = MemoryBudget(3000 * MIB)
        budget.reserve("weights", 800 * MIB)
        budget.reserve("runtime", 200 * MIB)

        result = reconcile(
            budget,
            baseline_bytes=3000 * MIB,
            reading=self._reading(2200 * MIB, reserved=800 * MIB),
        )

        assert result.measured_bytes == 1000 * MIB
        assert result.matches


def test_the_dummy_run_measures_on_the_cpu() -> None:
    stack = build_tiny_stack(num_blocks=40)
    before = {layer: (b.k.clone(), b.v.clone()) for layer, b in stack.runner.kv_buffers.items()}

    profile = measure_startup(
        stack.model,
        stack.runner.kv_buffers,
        key=_key(),
        device=Device("cpu"),
        baseline_bytes=0,
        kv_bytes=stack.pool.nbytes,
        chunk_tokens=16,
        buckets=(1, 2),
        block_size=4,
        usable_blocks=stack.pool.usable_blocks,
    )

    assert profile.activation_tokens == 16
    assert profile.kv_bytes == stack.pool.nbytes
    assert profile.weights_bytes == sum(p.numel() * 4 for p in stack.model.parameters())
    assert set(profile.decode_seconds) == {1, 2}
    # The dummy steps wrote only blocks 1..4; block 0 and the rest are as they were.
    for layer, buffers in stack.runner.kv_buffers.items():
        assert torch.equal(buffers.k[5:], before[layer][0][5:])
        assert torch.equal(buffers.k[0], before[layer][0][0])


class TestShouldSaveProfile:
    def _reading(self, *, free: int = 2000 * MIB, reserved: int = 600 * MIB) -> MemoryReading:
        return MemoryReading(
            free=free,
            total=4096 * MIB,
            allocated=reserved,
            reserved=reserved,
            peak_allocated=reserved,
        )

    def _reconcile_for(
        self,
        profile: StartupProfile,
        reading: MemoryReading,
        *,
        baseline: int = 3000 * MIB,
        baseline_reserved: int = 0,
    ) -> Any:
        budget = MemoryBudget(baseline)
        budget.reserve("weights", profile.weights_bytes)
        budget.reserve("kv", profile.kv_bytes)
        budget.reserve("runtime", profile.runtime_bytes)
        return reconcile(
            budget,
            baseline_bytes=baseline,
            baseline_reserved_bytes=baseline_reserved,
            reading=reading,
            reused_bytes=profile.reused_bytes,
        )

    def test_a_clean_startup_is_saved(self) -> None:
        profile = _profile()
        reading = self._reading(free=2400 * MIB, reserved=300 * MIB)
        result = self._reconcile_for(profile, reading)

        assert result.matches
        assert profile_save_issues(result, reading) == ()
        assert should_save_profile(result, reading)

    def test_a_reconciliation_mismatch_is_rejected(self) -> None:
        profile = _profile()
        reading = self._reading(free=1000 * MIB, reserved=1500 * MIB)
        result = self._reconcile_for(profile, reading)

        assert not result.matches
        assert not should_save_profile(result, reading)
        assert any("mismatch" in issue for issue in profile_save_issues(result, reading))

    def test_zero_free_memory_is_rejected_even_when_reconciliation_matches(self) -> None:
        profile = _profile()
        reading = self._reading(free=0, reserved=300 * MIB)
        result = self._reconcile_for(profile, reading)

        assert result.matches
        assert not should_save_profile(result, reading)
        assert any("0 free" in issue for issue in profile_save_issues(result, reading))

    def test_reused_pre_baseline_memory_is_allowed_when_reconciliation_matches(self) -> None:
        profile = StartupProfile(
            key=_key(),
            weights_bytes=100 * MIB,
            kv_bytes=200 * MIB,
            runtime_bytes=300 * MIB,
            activation_tokens=512,
            activation_peak_bytes=51 * MIB,
            reused_bytes=100 * MIB,
        )
        reading = self._reading(free=2400 * MIB, reserved=700 * MIB)
        result = self._reconcile_for(profile, reading, baseline_reserved=500 * MIB)

        assert result.matches
        assert profile_save_issues(result, reading) == ()
        assert should_save_profile(result, reading)


class TestStartupRunSaveGate:
    def _run_startup(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        *,
        profile: StartupProfile,
        reading: MemoryReading,
        baseline: int = 3000 * MIB,
        baseline_reserved: int = 0,
    ) -> None:
        monkeypatch.setenv("AYAKA_CACHE_DIR", str(tmp_path / "cache"))

        class _FakeDevice:
            def __init__(self, spec: object) -> None:
                self.spec = spec

            def synchronize(self) -> None:
                pass

            def memory(self) -> MemoryReading:
                return reading

        import ayaka.device.context as device_context

        monkeypatch.setattr(device_context, "Device", _FakeDevice)
        monkeypatch.setattr(engine_build, "measure_startup", lambda *args, **kwargs: profile)

        base_config = tiny_engine_config(tmp_path)
        config = type(base_config)(
            source=base_config.source,
            device=DeviceConfig(device="cuda", dtype=DType.FP16, cuda_graph=False),
            cache=base_config.cache,
            scheduler=base_config.scheduler,
        )
        plan = plan_budget(
            BudgetInputs(
                baseline_bytes=baseline,
                memory_ratio=0.9,
                weights_bytes=profile.weights_bytes,
                kv=(KVCost(bytes_per_block=MIB, block_size=16),),
                kv_reserve_tokens=64,
            )
        )
        runner: Any = SimpleNamespace(model=object(), kv_buffers={})
        pool: Any = SimpleNamespace(
            nbytes=profile.kv_bytes,
            usable_blocks=16,
            group=SimpleNamespace(spec=SimpleNamespace(block_size=16)),
        )
        _startup_run(
            config,
            runner,
            pool,
            key=profile.key,
            plan=plan,
            baseline=baseline,
            baseline_reserved=baseline_reserved,
            chunk_tokens=16,
            running=4,
        )

    def test_clean_startup_saves_the_profile(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        profile = _profile()
        reading = MemoryReading(
            free=2400 * MIB,
            total=4096 * MIB,
            allocated=300 * MIB,
            reserved=300 * MIB,
            peak_allocated=300 * MIB,
        )

        self._run_startup(tmp_path, monkeypatch, profile=profile, reading=reading)

        assert load_profile(profile.key) == profile

    def test_clean_startup_with_reused_pre_baseline_memory_saves_the_profile(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        profile = StartupProfile(
            key=_key(),
            weights_bytes=100 * MIB,
            kv_bytes=200 * MIB,
            runtime_bytes=300 * MIB,
            activation_tokens=512,
            activation_peak_bytes=51 * MIB,
            reused_bytes=100 * MIB,
        )
        reading = MemoryReading(
            free=2400 * MIB,
            total=4096 * MIB,
            allocated=700 * MIB,
            reserved=700 * MIB,
            peak_allocated=700 * MIB,
        )

        self._run_startup(
            tmp_path,
            monkeypatch,
            profile=profile,
            reading=reading,
            baseline_reserved=500 * MIB,
        )

        assert load_profile(profile.key) == profile

    @pytest.mark.parametrize(
        ("case", "reserved", "free", "expected_log"),
        [
            ("mismatch", 1500 * MIB, 1000 * MIB, "mismatch"),
            ("zero_free", 300 * MIB, 0, "0 free"),
        ],
    )
    def test_untrustworthy_startup_skips_saving_and_keeps_prior_profile(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
        case: str,
        reserved: int,
        free: int,
        expected_log: str,
    ) -> None:
        key = _key()
        prior = _profile(key)
        monkeypatch.setenv("AYAKA_CACHE_DIR", str(tmp_path / "cache"))
        save_profile(prior)

        bad = StartupProfile(
            key=key,
            weights_bytes=100 * MIB,
            kv_bytes=200 * MIB,
            runtime_bytes=123 * MIB,
            activation_tokens=512,
            activation_peak_bytes=51 * MIB,
        )
        reading = MemoryReading(
            free=free,
            total=4096 * MIB,
            allocated=reserved,
            reserved=reserved,
            peak_allocated=reserved,
        )

        with caplog.at_level(logging.WARNING, logger="ayaka.engine.build"):
            self._run_startup(tmp_path, monkeypatch, profile=bad, reading=reading)

        assert load_profile(key) == prior
        assert any(
            record.levelno == logging.WARNING
            and "skipping saving startup profile" in record.message
            and expected_log in record.message
            for record in caplog.records
        ), (case, caplog.text)
