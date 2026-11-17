"""The one place that wires the packages of the engine together.

:func:`build_engine` takes an :class:`~ayaka.configs.engine.EngineConfig` and returns a running
:class:`~ayaka.engine.core.EngineCore`. The order is the order of the dependencies::

    free memory of the device         measured first, before anything is allocated
    device capabilities               the dtype the config asks for exists here
    ops backend                       from the config's backend list, for this device and dtype
    model                             resolve the source, check and load the weights
    memory plan                       plan_budget over the weights, the KV cost of the model and,
                                      on a GPU, the last start-up profile measured for this key
    KV cache                          tensors (KVPool) and bookkeeping (KVCacheManager)
    tokenizer                         checked against the model's vocabulary and context
    scheduler, runner, processors     and the engine that runs them
    start-up run (GPU only)           measure activations and runtime memory, reserve it all in
                                      the ledger, compare the ledger with the device, save the
                                      profile for the next start-up

Nothing here is a decision of its own except one: the memory plan hands the whole pool to the KV
cache, which is right for a GPU that holds only this engine and wrong for the host's RAM, so on the
CPU the cache is sized for the requests that can run at once, not for all the free memory.

This module imports torch inside :func:`build_engine`, not at import, so importing the engine
package stays cheap and a command line that only prints help never loads it.
"""

from __future__ import annotations

import logging
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
from ayaka.device.capabilities import device_capabilities
from ayaka.engine.core import EngineCore
from ayaka.engine.input import InputProcessor
from ayaka.engine.output import OutputProcessor
from ayaka.engine.profile import (
    ProfileKey,
    StartupReport,
    load_profile,
    measure_startup,
    reconcile,
    save_profile,
)
from ayaka.kvcache.manager import KVCacheManager
from ayaka.memory.budget import BudgetInputs, KVCost, MemoryBudget, plan_budget
from ayaka.scheduler.scheduler import Scheduler
from ayaka.tokenizers.service import TokenizerService
from ayaka.utils.math_utils import div_ceil

if TYPE_CHECKING:
    from collections.abc import Iterable

    from ayaka.configs.device import DeviceConfig
    from ayaka.configs.engine import EngineConfig
    from ayaka.device.info import DeviceInfo
    from ayaka.execution.runner import ModelRunner
    from ayaka.kvcache.storage.pool import KVPool
    from ayaka.memory.budget import BudgetPlan
    from ayaka.models.base import CausalLM
    from ayaka.tokenizers.contracts import LoadedTokenizer

__all__ = ["build_engine"]

logger = logging.getLogger(__name__)

#: Requests that run at once on the host CPU when the config does not say. The CPU shares its RAM
#: with everything else, so the KV cache is sized for this many requests of full context.
_HOST_MAX_RUNNING: Final = 8


def build_engine(config: EngineConfig, *, tokenizer: LoadedTokenizer | None = None) -> EngineCore:
    """Build an engine: load the model, size and allocate the cache, and wire the loop.

    Args:
        config: Everything that describes the run.
        tokenizer: A tokenizer to use instead of loading the one the configuration names; for
            tests and for callers that already hold one.

    Returns:
        An engine ready to take requests. Call :meth:`~ayaka.engine.core.EngineCore.shutdown`
        when done.

    Raises:
        ConfigError: If CUDA graphs are requested (not implemented yet), the device cannot run
            the dtype, the backend list cannot serve the device and dtype, the KV reserve exceeds
            the model context, or the tokenizer settings cannot work.
        CapabilityError: If torch, the requested GPU, the Hub client or a tokenizer package is
            missing.
        ModelLoadError: If the source cannot be resolved, the checkpoint does not match the model,
            the model needs more than one KV group, or the tokenizer cannot be loaded or does not
            match the model.
        CapacityError: If the weights and a context of KV do not fit the memory the config allows.
    """
    from ayaka.execution.runner import ModelRunner
    from ayaka.kvcache.storage.pool import KVPool
    from ayaka.model_loader.loader import load_model
    from ayaka.ops.backend import load_backend

    if config.device.cuda_graph:
        raise ConfigError(
            "device.cuda_graph",
            "CUDA_GRAPH_UNSUPPORTED",
            "CUDA graph capture is not implemented yet; set cuda_graph=False",
        )

    device = config.device
    baseline_reserved = _release_cached_device_memory(device)
    info, baseline = _free_bytes(device)
    capabilities = device_capabilities(info)
    capabilities.check(device)
    logger.info("device: %s", capabilities.describe())
    ops = load_backend(device.backends, device=device.device_type, dtype=device.dtype)
    model = load_model(config.source, device, ops=ops)
    model_config = model.config

    groups = model.kv_specs(block_size=config.cache.block_size, layout=ops.kvcache.layout())
    if len(groups) != 1:
        raise ModelLoadError(
            f"{model_config.architecture} needs {len(groups)} KV cache groups; "
            "this engine runs models with one",
            context={"groups": len(groups)},
        )
    [group] = groups
    block_size = group.spec.block_size
    context = model_config.max_position_embeddings
    reserve_tokens = config.cache.kv_reserve_tokens or context
    if reserve_tokens > context:
        raise ConfigError(
            "cache.kv_reserve_tokens",
            "RESERVE_EXCEEDS_CONTEXT",
            f"kv_reserve_tokens={reserve_tokens} exceeds the model context of {context} tokens",
        )
    key = ProfileKey.build(
        device=f"{info.name} {capabilities.arch}",
        model=model_config.fingerprint,
        dtype=device.dtype.label,
        providers=ops.providers.items(),
        block_size=block_size,
        libraries=_library_versions(ops.providers.values()),
    )
    previous = load_profile(key) if device.device_type == "cuda" else None
    if previous is not None:
        logger.info(
            "planning with the startup profile measured before: %d bytes of runtime memory, "
            "%d activation bytes per token, %d bytes of graphs",
            previous.runtime_bytes,
            previous.activation_bytes_per_token,
            previous.graph_bytes,
        )
    plan = plan_budget(
        BudgetInputs(
            baseline_bytes=baseline,
            memory_ratio=config.cache.memory_ratio,
            weights_bytes=_weights_bytes(model),
            kv=(
                KVCost(
                    bytes_per_block=group.spec.bytes_per_block(),
                    fixed_bytes=group.spec.fixed_bytes(),
                    block_size=block_size,
                ),
            ),
            kv_reserve_tokens=reserve_tokens,
            kv_blocks_override=config.cache.num_blocks,
            late_alloc_bytes=previous.runtime_bytes if previous else 0,
            activation_bytes_per_token=previous.activation_bytes_per_token if previous else 0,
            activation_fixed_bytes=(
                max(previous.workspace_bytes - previous.activation_peak_bytes, 0) if previous else 0
            ),
        )
    )

    max_blocks = div_ceil(context, block_size)
    running = _max_running(config, plan.max_running)
    blocks = plan.kv_blocks[0]
    if config.cache.num_blocks is None:
        floor = 1 + div_ceil(reserve_tokens, block_size)
        blocks = min(blocks, max(1 + running * max_blocks, floor))
    logger.info(
        "memory plan: %d KV blocks fit, %d allocated for up to %d running requests "
        "(%d tokens each at most)",
        plan.kv_blocks[0],
        blocks,
        running,
        min(context, (blocks - 1) * block_size),
    )

    kv = KVCacheManager([group], [blocks], max_requests=running, max_blocks_per_request=max_blocks)
    pool = KVPool(group, blocks, device=device.device)

    service = _tokenizer_service(config, model_config.vocab_size, context, tokenizer)
    scheduler = Scheduler(
        config.scheduler,
        kv,
        max_model_len=context,
        max_chunk_tokens=plan.max_chunk_tokens,
        eos_token_ids=service.eos_token_ids,
    )
    runner = ModelRunner(
        model,
        [pool],
        kv.table,
        # No draft proposer exists yet, so lookahead is 0. A proposer must pass its k here;
        # required_rows then reserves one row per request plus k draft rows.
        max_rows=ModelRunner.required_rows(
            token_budget=scheduler.token_budget,
            max_running=scheduler.max_running,
            lookahead=0,
        ),
        device=device.device,
    )
    startup = None
    if device.device_type == "cuda":
        startup = _startup_run(
            config,
            runner,
            pool,
            key=key,
            plan=plan,
            baseline=baseline,
            baseline_reserved=baseline_reserved,
            chunk_tokens=scheduler.token_budget,
            running=scheduler.max_running,
        )
    return EngineCore(
        scheduler=scheduler,
        runner=runner,
        kv=kv,
        input_processor=InputProcessor(
            service, max_model_len=context, vocab_size=model_config.vocab_size
        ),
        output_processor=OutputProcessor(service, abort=scheduler.abort),
        tokenizer=service,
        model_config=model_config,
        plan=plan,
        startup=startup,
    )


def _startup_run(
    config: EngineConfig,
    runner: ModelRunner,
    pool: KVPool,
    *,
    key: ProfileKey,
    plan: BudgetPlan,
    baseline: int,
    baseline_reserved: int,
    chunk_tokens: int,
    running: int,
) -> StartupReport:
    """Measure, fill the ledger and check it against the device (GPU only)."""
    from ayaka.device.context import Device

    target = Device(config.device.device)
    buckets = tuple(bucket for bucket in config.device.graph_buckets if bucket <= running) or (1,)
    profile = measure_startup(
        runner.model,
        runner.kv_buffers,
        key=key,
        device=target,
        baseline_bytes=baseline,
        baseline_reserved_bytes=baseline_reserved,
        kv_bytes=pool.nbytes,
        chunk_tokens=chunk_tokens,
        buckets=buckets,
        block_size=pool.group.spec.block_size,
        usable_blocks=pool.usable_blocks,
    )
    budget = MemoryBudget(baseline)
    budget.reserve("weights", profile.weights_bytes)
    budget.reserve("kv", profile.kv_bytes)
    budget.reserve("runtime", min(profile.runtime_bytes, budget.free_bytes))
    budget.reserve("workspace", min(profile.workspace_bytes, budget.free_bytes))
    target.synchronize()
    reconciliation = reconcile(
        budget, baseline_bytes=baseline, reading=target.memory(), reused_bytes=profile.reused_bytes
    )
    log = logger.info if reconciliation.matches else logger.warning
    log(
        "memory ledger %d bytes, device lost %d bytes since the baseline (difference %d, "
        "tolerance %d); plan: %d usable, %d headroom",
        reconciliation.ledger_bytes,
        reconciliation.measured_bytes,
        reconciliation.difference_bytes,
        reconciliation.tolerance_bytes,
        plan.usable_bytes,
        plan.headroom_bytes,
    )
    save_profile(profile)
    return StartupReport(profile=profile, budget=budget, reconciliation=reconciliation)


def _release_cached_device_memory(device: DeviceConfig) -> int:
    """Give memory torch caches but no tensor uses back to the driver, before the baseline is read.

    Only matters when this process already used the GPU (an earlier engine, a test): the baseline
    must be what is free to this engine, not what an earlier one left in torch's cache. In a fresh
    process torch is not imported yet and nothing happens.

    Returns:
        What torch's allocator still holds afterwards (tensors that are still alive), so the
        start-up run counts only what this engine reserves on top; 0 in a fresh process.
    """
    torch = sys.modules.get("torch")
    if device.device_type != "cuda" or torch is None or not torch.cuda.is_initialized():
        return 0
    import gc

    gc.collect()
    torch.cuda.empty_cache()
    return int(torch.cuda.memory_reserved(device.device_index or 0))


def _library_versions(backends: Iterable[str]) -> dict[str, str]:
    """Versions of torch and of every kernel library the ops come from, for the profile key."""
    from importlib.metadata import PackageNotFoundError, version

    names = {"torch", *(f"{name}-python" for name in backends if name == "flashinfer")}
    found: dict[str, str] = {}
    for name in sorted(names):
        try:
            found[name] = version(name)
        except PackageNotFoundError:
            continue
    return found


def _free_bytes(device: DeviceConfig) -> tuple[DeviceInfo, int]:
    """The configured device and its free memory before this engine takes any (the baseline)."""
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
    """Available RAM on a host with no ``/proc/meminfo`` (Windows, macOS); 0 if unknown.

    The device probe reads ``/proc`` and so reports nothing there. This is the engine's own
    fallback, not a second probe: it runs only when the probe found no free memory.
    """
    if sys.platform == "win32":
        import ctypes

        class _MemoryStatus(ctypes.Structure):
            _fields_ = (  # the layout of MEMORYSTATUSEX
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


def _weights_bytes(model: CausalLM) -> int:
    """Bytes of the model's parameters, a shared (tied) one counted once."""
    return sum(parameter.numel() * parameter.element_size() for parameter in model.parameters())


def _max_running(config: EngineConfig, planned: int) -> int:
    """Requests the cache is built for: the config's limit, else what the plan allows."""
    if config.scheduler.max_running is not None:
        return config.scheduler.max_running
    limit = max(1, planned)
    return min(limit, _HOST_MAX_RUNNING) if config.device.device_type == "cpu" else limit


def _tokenizer_service(
    config: EngineConfig, vocab_size: int, context: int, loaded: LoadedTokenizer | None
) -> TokenizerService:
    tokenizer_config = config.tokenizer_config
    try:
        return TokenizerService(
            tokenizer_config, max_model_len=context, model_vocab_size=vocab_size, loaded=loaded
        )
    except AyakaError:
        raise  # typed errors (a bad config, a missing package) already say what to do
    except (OSError, ValueError) as exc:
        # The adapters raise these for missing files, a bad tokenizer or a vocabulary the model
        # cannot embed.
        raise ModelLoadError(
            f"cannot use the tokenizer {str(tokenizer_config.tokenizer)!r}: {exc}",
            context={"tokenizer": str(tokenizer_config.tokenizer)},
        ) from exc
