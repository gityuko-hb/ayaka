"""Unit tests for ``ayaka doctor`` diagnostic checks and CLI output."""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import replace
from typing import TYPE_CHECKING

import pytest

from ayaka.device.info import DeviceInfo
from ayaka.engine.profile import ProfileKey, StartupProfile
from ayaka.entrypoints.cli import main
from ayaka.entrypoints.doctor import (
    DoctorCheck,
    DoctorProbes,
    doctor_lines,
    run_doctor,
)
from ayaka.utils.host_info import CgroupMemory

if TYPE_CHECKING:
    from pathlib import Path

pytestmark = pytest.mark.unit

_GIB = 2**30
_MIB = 2**20


def _gpu_device(*, free_mib: int = 3500, total_mib: int = 4096) -> DeviceInfo:
    return DeviceInfo(
        kind="cuda",
        index=0,
        name="NVIDIA GeForce RTX 3050",
        total_bytes=total_mib * _MIB,
        free_bytes=free_mib * _MIB,
        compute_capability=(8, 6),
        num_sms=16,
    )


def _cpu_device() -> DeviceInfo:
    return DeviceInfo(kind="cpu", index=0, name="cpu", total_bytes=16 * _GIB, free_bytes=8 * _GIB)


def _healthy_probes(tmp_path: Path, *, gpu: bool = True, wsl2: bool = False) -> DoctorProbes:
    profiles = tmp_path / "profiles"
    profiles.mkdir(parents=True, exist_ok=True)
    hf_cache = tmp_path / "hf_cache"
    hf_cache.mkdir(parents=True, exist_ok=True)
    devices = (_gpu_device(), _cpu_device()) if gpu else (_cpu_device(),)
    torch_cuda = (
        ("2.14.1+cu130", "13.0", True, ((0, 3400 * _MIB, 4096 * _MIB),))
        if gpu
        else ("2.14.1+cpu", None, False, ())
    )
    return DoctorProbes(
        devices=lambda: devices,
        torch_cuda=lambda: torch_cuda,
        wsl2=lambda: wsl2,
        platform="linux",
        host_ram=lambda: (16 * _GIB, 8 * _GIB),
        cgroup=lambda: CgroupMemory(),
        psi_memory=lambda: (0.0, 0.0),
        memlock_limit=lambda: None,
        package_installed=lambda _name: False,
        which=lambda _cmd: None,
        profile_dir=lambda: profiles,
        hub_cache_dir=lambda: hf_cache,
        disk_free=lambda _path: 50 * _GIB,
    )


def test_doctor_check_requires_fix_for_warn_and_fail() -> None:
    with pytest.raises(ValueError, match="needs a fix"):
        DoctorCheck("cuda_driver", "fail", "missing GPU")
    with pytest.raises(ValueError, match="needs a fix"):
        DoctorCheck("host_pinned_ram", "warn", "low RAM", fix="   ")


def test_healthy_gpu_machine_passes_all_checks(tmp_path: Path) -> None:
    report = run_doctor(probes=_healthy_probes(tmp_path, gpu=True))

    assert report.ok
    assert not report.has_warnings
    assert all(check.status == "pass" for check in report.checks)


def test_cpu_only_machine_warns_on_cuda_and_stays_ok(tmp_path: Path) -> None:
    report = run_doctor(probes=_healthy_probes(tmp_path, gpu=False))
    by_name = {c.name: c for c in report.checks}

    assert report.ok
    assert report.has_warnings
    assert by_name["cuda_driver"].status == "warn"
    assert by_name["cuda_driver"].fix is not None
    assert by_name["nvml_cuda_gap"].status == "pass"


def test_gpu_present_with_cpu_torch_fails_with_cu130_fix(tmp_path: Path) -> None:
    base = _healthy_probes(tmp_path, gpu=True)
    probes = replace(base, torch_cuda=lambda: ("2.14.1+cpu", None, False, ()))

    report = run_doctor(probes=probes)
    cuda = {c.name: c for c in report.checks}["cuda_driver"]

    assert not report.ok
    assert cuda.status == "fail"
    assert cuda.fix is not None
    assert "cu130" in cuda.fix


def test_nvml_cuda_gap_detects_wsl2_oversubscription_and_zero_free(tmp_path: Path) -> None:
    base = _healthy_probes(tmp_path, gpu=True, wsl2=True)
    warn_probes = replace(
        base,
        devices=lambda: (_gpu_device(free_mib=3900, total_mib=4096), _cpu_device()),
        torch_cuda=lambda: ("2.14.1+cu130", "13.0", True, ((0, 2900 * _MIB, 4096 * _MIB),)),
    )
    warn_report = run_doctor(probes=warn_probes)
    gap = {c.name: c for c in warn_report.checks}["nvml_cuda_gap"]
    assert gap.status == "warn"
    assert "WSL2/WDDM" in gap.summary
    assert gap.fix is not None
    assert "--memory-ratio" in gap.fix

    fail_probes = replace(
        base,
        torch_cuda=lambda: ("2.14.1+cu130", "13.0", True, ((0, 0, 4096 * _MIB),)),
    )
    fail_report = run_doctor(probes=fail_probes)
    assert not fail_report.ok
    assert {c.name: c for c in fail_report.checks}["nvml_cuda_gap"].status == "fail"


def test_host_ram_checks_low_ram_psi_and_memlock(tmp_path: Path) -> None:
    base = _healthy_probes(tmp_path, gpu=True)

    low_ram = run_doctor(probes=replace(base, host_ram=lambda: (8 * _GIB, 256 * _MIB)))
    assert not low_ram.ok
    assert {c.name: c for c in low_ram.checks}["host_pinned_ram"].status == "fail"

    memlock = run_doctor(probes=replace(base, memlock_limit=lambda: 64 * 1024))
    assert memlock.ok
    check = {c.name: c for c in memlock.checks}["host_pinned_ram"]
    assert check.status == "warn"
    assert check.fix is not None
    assert "ulimit" in check.fix

    psi = run_doctor(probes=replace(base, psi_memory=lambda: (15.0, 8.5)))
    assert {c.name: c for c in psi.checks}["host_pinned_ram"].status == "warn"


def test_flashinfer_requires_nvcc_on_linux(tmp_path: Path) -> None:
    base = _healthy_probes(tmp_path, gpu=True)
    missing_nvcc = replace(
        base,
        package_installed=lambda name: name == "flashinfer-python",
        which=lambda _cmd: None,
    )
    report = run_doctor(probes=missing_nvcc)
    check = {c.name: c for c in report.checks}["flashinfer_nvcc"]
    assert not report.ok
    assert check.status == "fail"
    assert check.fix is not None
    assert "PATH" in check.fix

    with_nvcc = replace(
        base,
        package_installed=lambda name: name == "flashinfer-python",
        which=lambda cmd: "/usr/local/cuda/bin/nvcc" if cmd == "nvcc" else None,
    )
    assert run_doctor(probes=with_nvcc).ok


def test_disk_space_and_startup_profile_validation(tmp_path: Path) -> None:
    base = _healthy_probes(tmp_path, gpu=True)
    bad_profile = tmp_path / "profiles" / "corrupt.json"
    bad_profile.write_text("{not valid json", encoding="utf-8")

    report = run_doctor(
        min_disk_gib=5.0,
        probes=replace(base, disk_free=lambda _p: 512 * _MIB),
    )
    by_name = {c.name: c for c in report.checks}
    assert not report.ok
    assert by_name["cache_disk_space"].status == "fail"
    assert by_name["startup_profiles"].status == "fail"
    assert "corrupt.json" in (by_name["startup_profiles"].fix or "")

    bad_profile.unlink()
    valid = StartupProfile(
        key=ProfileKey.build(
            device="RTX 3050 sm_86",
            model="tiny",
            dtype="fp16",
            providers=(("attention", "torch"),),
            block_size=16,
        ),
        weights_bytes=1024,
        kv_bytes=2048,
        runtime_bytes=512,
        activation_tokens=16,
        activation_peak_bytes=256,
    )
    (tmp_path / "profiles" / f"{valid.key.digest()}.json").write_text(
        valid.to_json(), encoding="utf-8"
    )
    ok_report = run_doctor(probes=base)
    assert ok_report.ok
    assert {c.name: c for c in ok_report.checks}["startup_profiles"].status == "pass"


def test_cli_doctor_json_and_text_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from ayaka.entrypoints import doctor as doctor_module

    probes = _healthy_probes(tmp_path, gpu=True)
    monkeypatch.setattr(doctor_module, "DoctorProbes", lambda: probes)

    code = main(["doctor", "--output", "json"])
    payload = json.loads(capsys.readouterr().out)
    assert code == 0
    assert payload["ok"] is True
    assert len(payload["checks"]) == 6

    code_text = main(["doctor"])
    out = capsys.readouterr().out
    assert code_text == 0
    assert "[PASS]" in out
    assert doctor_lines(run_doctor(probes=probes))


def test_doctor_help_does_not_load_torch() -> None:
    code = (
        "import sys\n"
        "from ayaka.entrypoints.cli import main\n"
        "try:\n"
        "    main(['doctor', '--help'])\n"
        "except SystemExit:\n"
        "    pass\n"
        "assert 'torch' not in sys.modules\n"
    )
    result = subprocess.run(  # noqa: S603
        [sys.executable, "-I", "-c", code], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr
