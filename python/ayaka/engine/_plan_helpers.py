"""Internal device probing, KV codec validation, and budget helpers for engine construction."""

from __future__ import annotations

import os
import sys
from typing import TYPE_CHECKING, Final

from ayaka.core.errors import (
    AyakaError,
    CapabilityError,
    CapacityError,
    ConfigError,
    ModelLoadError,
)
from ayaka.memory.budget import BudgetInputs, KVCost, StateCost, plan_budget
from ayaka.ops.kvcache import FP8_E4M3_CODEC, RAW_CODEC
from ayaka.tokenizers.service import TokenizerService
from ayaka.utils.dtypes import DType
from ayaka.utils.math_utils import div_ceil

if TYPE_CHECKING:
    from collections.abc import Iterable

    from ayaka.configs.device import DeviceConfig
    from ayaka.configs.engine import EngineConfig
    from ayaka.device.info import DeviceInfo
    from ayaka.engine.profile import StartupProfile
    from ayaka.kvcache.storage.spec import KVGroup, LatentKVGroup, RecurrentStateGroup
    from ayaka.memory.budget import BudgetPlan
    from ayaka.models.base import CausalLM
    from ayaka.tokenizers.contracts import LoadedTokenizer

__all__ = [
    "compute_budget",
    "free_device_bytes",
    "library_versions",
    "max_running",
    "release_cached_device_memory",
    "resolve_kv_codec",
    "tokenizer_service",
    "weights_bytes",
]

#: Requests that run at once on CPU when max_running is unset.
_HOST_MAX_RUNNING: Final = 8


def resolve_kv_codec(config: EngineConfig) -> tuple[str, float]:
    """Validate and resolve the KV cache storage codec and static FP8 scale."""
    if config.cache.kv_dtype == "auto":
        return RAW_CODEC, 1.0
    if config.device.device_type != "cuda" or config.device.dtype not in (DType.BF16, DType.FP16):
        raise ConfigError(
            "cache.kv_dtype",
            "FP8_KV_UNSUPPORTED",
            f"--kv-dtype fp8 requires CUDA in fp16 or bf16 (got {config.device.device} / "
            f"{config.device.dtype.label})",
        )
    return FP8_E4M3_CODEC, config.cache.kv_fp8_scale


def _state_cost(
    config: EngineConfig,
    state_group: RecurrentStateGroup | None,
    kv_costs: tuple[KVCost, ...],
    usable_pool_estimate: int,
    reserve_tokens: int,
    block_size: int,
) -> StateCost | None:
    if state_group is None:
        return None
    per_slot = state_group.spec.bytes_per_slot()
    if config.scheduler.max_running is not None:
        slots = config.scheduler.max_running + 1
    elif config.device.device_type == "cpu":
        slots = _HOST_MAX_RUNNING + 1
    else:
        kv_per_req = div_ceil(reserve_tokens, block_size) * sum(c.bytes_per_block for c in kv_costs)
        fit = max(1, usable_pool_estimate // max(kv_per_req + per_slot, 1))
        slots = min(fit, 32) + 1
    return StateCost(bytes_per_slot=per_slot, max_slots=slots)


def compute_budget(
    config: EngineConfig,
    groups: tuple[KVGroup | LatentKVGroup, ...],
    state_group: RecurrentStateGroup | None,
    weights: int,
    baseline: int,
    reserve_tokens: int,
    previous: StartupProfile | None,
    *,
    initial_runtime_bytes: int = 0,
) -> BudgetPlan:
    """Compute the engine's memory budget from free memory, weights, and KV/state geometry."""
    late_alloc_bytes = (
        max(previous.runtime_bytes, initial_runtime_bytes)
        if previous is not None
        else initial_runtime_bytes
    )
    kv_costs = tuple(
        KVCost(
            bytes_per_block=group.spec.bytes_per_block(),
            fixed_bytes=group.spec.fixed_bytes(),
            block_size=group.spec.block_size,
        )
        for group in groups
    )
    approx_usable = max(0, int(baseline * config.cache.memory_ratio) - weights - late_alloc_bytes)
    state_cost = _state_cost(
        config, state_group, kv_costs, approx_usable, reserve_tokens, groups[0].spec.block_size
    )
    return plan_budget(
        BudgetInputs(
            baseline_bytes=baseline,
            memory_ratio=config.cache.memory_ratio,
            weights_bytes=weights,
            kv=kv_costs,
            state=state_cost,
            kv_reserve_tokens=reserve_tokens,
            kv_blocks_override=config.cache.num_blocks,
            late_alloc_bytes=late_alloc_bytes,
            activation_bytes_per_token=previous.activation_bytes_per_token if previous else 0,
            activation_fixed_bytes=(
                max(previous.workspace_bytes - previous.activation_peak_bytes, 0) if previous else 0
            ),
            graph_bytes_estimate=(
                previous.graph_bytes if previous and config.device.cuda_graph else 0
            ),
        )
    )


def release_cached_device_memory(device: DeviceConfig) -> int:
    """Release cached torch allocator segments back to the driver before reading free memory."""
    torch = sys.modules.get("torch")
    if device.device_type != "cuda" or torch is None or not torch.cuda.is_initialized():
        return 0
    import gc

    gc.collect()
    torch.cuda.empty_cache()
    return int(torch.cuda.memory_reserved(device.device_index or 0))


def library_versions(backends: Iterable[str]) -> dict[str, str]:
    """Versions of torch and kernel libraries the ops come from, for the profile key."""
    from importlib.metadata import PackageNotFoundError, version

    names = {"torch", *(f"{name}-python" for name in backends if name == "flashinfer")}
    found: dict[str, str] = {}
    for name in sorted(names):
        try:
            found[name] = version(name)
        except PackageNotFoundError:
            continue
    return found


def free_device_bytes(device: DeviceConfig) -> tuple[DeviceInfo, int]:
    """Return the configured device and its free memory before this engine takes any."""
    from ayaka.device.info import probe_devices

    wanted = device.device_index or 0
    for found in probe_devices():
        if found.kind == device.device_type and (found.kind == "cpu" or found.index == wanted):
            free = found.free_bytes
            if free <= 0 and found.kind == "cpu":
                free = _host_free_bytes_without_proc()
            if free <= 0:
                raise CapacityError(
                    f"cannot read the free memory of {device.device}; "
                    "without it the memory plan has nothing to divide",
                    context={"device": device.device},
                )
            return found, free
    raise CapabilityError(
        f"device.{device.device}",
        detail=f"no {device.device} device was found",
        remedy="check the NVIDIA driver, or run on the CPU with device 'cpu'",
    )


def _host_free_bytes_without_proc() -> int:
    """Available RAM on hosts without ``/proc/meminfo``; 0 if unknown."""
    if sys.platform == "win32":
        import ctypes

        class _MemoryStatus(ctypes.Structure):
            _fields_ = (
                ("length", ctypes.c_ulong),
                ("memory_load", ctypes.c_ulong),
                ("total_phys", ctypes.c_ulonglong),
                ("avail_phys", ctypes.c_ulonglong),
                ("total_page_file", ctypes.c_ulonglong),
                ("avail_page_file", ctypes.c_ulonglong),
                ("total_virtual", ctypes.c_ulonglong),
                ("avail_virtual", ctypes.c_ulonglong),
                ("avail_extended_virtual", ctypes.c_ulonglong),
            )

        status = _MemoryStatus()
        status.length = ctypes.sizeof(_MemoryStatus)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            return int(status.avail_phys)
        return 0
    try:
        return os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
    except (AttributeError, ValueError, OSError):
        return 0


def weights_bytes(model: CausalLM) -> int:
    """Bytes of the model's parameters, a shared (tied) tensor counted once."""
    return sum(parameter.numel() * parameter.element_size() for parameter in model.parameters())


def max_running(config: EngineConfig, planned: int) -> int:
    """Requests the cache is built for: config limit or planner capacity."""
    if config.scheduler.max_running is not None:
        return config.scheduler.max_running
    limit = max(1, planned)
    return min(limit, _HOST_MAX_RUNNING) if config.device.device_type == "cpu" else limit


def tokenizer_service(
    config: EngineConfig, vocab_size: int, context: int, loaded: LoadedTokenizer | None
) -> TokenizerService:
    """Build or validate the tokenizer service against the model's vocab/context."""
    tokenizer_config = config.tokenizer_config
    try:
        return TokenizerService(
            tokenizer_config, max_model_len=context, model_vocab_size=vocab_size, loaded=loaded
        )
    except AyakaError:
        raise
    except (OSError, ValueError) as exc:
        raise ModelLoadError(
            f"cannot use the tokenizer {str(tokenizer_config.tokenizer)!r}: {exc}",
            context={"tokenizer": str(tokenizer_config.tokenizer)},
        ) from exc
