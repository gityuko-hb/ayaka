"""The one place that wires the packages of the engine together.

:func:`build_engine` takes an :class:`~ayaka.configs.engine.EngineConfig` and returns a running
:class:`~ayaka.engine.core.EngineCore`. The order is the order of the dependencies::

    free memory of the device         measured first, before anything is allocated
    device capabilities               the dtype and CUDA graphs the config asks for exist here
    ops backend                       from the config's backend list, for this device and dtype
                                      (graph-safe families only when CUDA graphs are on)
    memory plan                       plan_budget over the weights (the model built on the meta
                                      device: nothing read yet), the KV cost of the model and, on
                                      a GPU, the last start-up profile measured for this key;
                                      plan_engine stops here, which is what ``ayaka plan`` prints
    model                             load the weights
    KV cache                          tensors (KVPool), bookkeeping (KVCacheManager) and, unless
                                      the config turns it off, the prefix cache
    tokenizer                         checked against the model's vocabulary and context
    scheduler, runner, processors     and the engine that runs them, with the stat loggers
    start-up run (GPU only)           measure activations and runtime memory, capture the decode
                                      graphs, reserve it all in the ledger, compare the ledger with
                                      the device, save the profile for the next start-up

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
from dataclasses import dataclass, replace
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
    StartupProfile,
    StartupReport,
    load_profile,
    measure_startup,
    profile_save_issues,
    reconcile,
    save_profile,
    should_save_profile,
)
from ayaka.kvcache.manager import KVCacheManager
from ayaka.kvcache.prefix.hash_cache import HashPrefixCache
from ayaka.kvcache.prefix.hashing import ChainRoot
from ayaka.memory.budget import BudgetInputs, KVCost, MemoryBudget, plan_budget
from ayaka.metrics.loggers import LoggingStatLogger
from ayaka.metrics.manager import StatLoggerManager
from ayaka.scheduler.scheduler import Scheduler
from ayaka.tokenizers.service import TokenizerService
from ayaka.utils.math_utils import div_ceil

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence
    from pathlib import Path

    from ayaka.configs.device import DeviceConfig
    from ayaka.configs.engine import EngineConfig
    from ayaka.configs.model import ModelConfig
    from ayaka.device.capabilities import DeviceCapabilities
    from ayaka.device.info import DeviceInfo
    from ayaka.engine.recorder import SessionRecorder
    from ayaka.execution.runner import ModelRunner
    from ayaka.kvcache.storage.pool import KVPool
    from ayaka.kvcache.storage.spec import KVGroup
    from ayaka.memory.budget import BudgetPlan
    from ayaka.metrics.loggers import StatLogger
    from ayaka.models.base import CausalLM
    from ayaka.ops.backend import OpsBackend
    from ayaka.tokenizers.contracts import LoadedTokenizer

__all__ = ["STATS_LOG_INTERVAL", "EnginePlan", "build_engine", "plan_engine"]

logger = logging.getLogger(__name__)

#: Seconds between two lines of the logging stat logger.
STATS_LOG_INTERVAL: Final = 10.0

#: Requests that run at once on the host CPU when the config does not say. The CPU shares its RAM
#: with everything else, so the KV cache is sized for this many requests of full context.
_HOST_MAX_RUNNING: Final = 8


@dataclass(frozen=True, slots=True)
class EnginePlan:
    """What an engine would get on this machine, worked out without reading a weight.

    :func:`plan_engine` makes it and :func:`build_engine` builds from it, so ``ayaka plan`` and
    a real start-up print the same numbers from the same code.

    Attributes:
        directory: The checkpoint directory.
        device: The device.
        baseline_bytes: Its free memory, read before anything was allocated.
        capabilities: What the device can run.
        providers: Which backend serves each op family.
        model_config: The model's architecture.
        group: The KV cache group of the model.
        weights_bytes: Bytes of the weights as they sit on the device (8-bit weights and their
            scales for a quantized checkpoint), a shared tensor counted once.
        budget: The memory plan (the same function the engine uses, with the last start-up
            profile measured for this machine and model when there is one).
        kv_blocks: KV blocks the engine allocates (on the CPU, sized for the requests that run).
        max_running: Requests that run at once.
        max_blocks_per_request: Blocks of one request's longest context.
        reserve_tokens: Tokens of KV one request must be able to hold.
        chunk_tokens: Most tokens one prefill step carries: the scheduler's budget, capped by
            what the memory headroom supports.
        profile_key: The key of the start-up profile.
        profiled: Whether a start-up profile measured before fed the plan.
        quantization: The label of the checkpoint's 8-bit recipe
            (:attr:`~ayaka.weights.quant.QuantFormat.name`), or ``None`` for 16- or 32-bit weights.
        stored_types: For a GGUF file, the tensors of each GGML type (its ``weights_bytes`` are
            the header's tensor sizes); ``None`` for a safetensors checkpoint.
    """

    directory: Path
    device: DeviceInfo
    baseline_bytes: int
    capabilities: DeviceCapabilities
    providers: Mapping[str, str]
    model_config: ModelConfig
    group: KVGroup
    weights_bytes: int
    budget: BudgetPlan
    kv_blocks: int
    max_running: int
    max_blocks_per_request: int
    reserve_tokens: int
    chunk_tokens: int
    profile_key: ProfileKey
    profiled: bool
    quantization: str | None = None
    stored_types: Mapping[str, int] | None = None

    @property
    def block_size(self) -> int:
        """Tokens per KV block."""
        return self.group.spec.block_size

    @property
    def kv_bytes(self) -> int:
        """Bytes of the KV pool the engine allocates."""
        return self.kv_blocks * self.group.spec.bytes_per_block()

    @property
    def max_context_tokens(self) -> int:
        """The longest context one request can reach here: the model's or the cache's limit."""
        return min(self.model_config.context_length, (self.kv_blocks - 1) * self.block_size)


def plan_engine(config: EngineConfig) -> EnginePlan:
    """Work out the memory plan of an engine without loading or allocating the model.

    Reads ``config.json`` and builds the model on torch's meta device, so the weights' size and
    the KV geometry come from the model's own code, then runs the planner the engine runs.

    Raises:
        As :func:`build_engine`, except that nothing is loaded; ``CapacityError`` when the plan
        does not fit.
    """
    plan, _, _ = _plan(config)
    return plan


def _plan(config: EngineConfig) -> tuple[EnginePlan, OpsBackend, int]:
    """The plan, the ops backend it was made for, and the bytes torch held before it."""
    from ayaka.model_loader.loader import describe_model, read_model_config, stored_weights
    from ayaka.ops.backend import load_backend

    device = config.device
    baseline_reserved = _release_cached_device_memory(device)
    info, baseline = _free_bytes(device)
    capabilities = device_capabilities(info)
    capabilities.check(device)
    logger.info("device: %s", capabilities.describe())
    ops = load_backend(
        device.backends,
        device=device.device_type,
        dtype=device.dtype,
        cuda_graph=device.cuda_graph,
        head_dim=read_model_config(config.source).head_dim,
    )
    initial_runtime_bytes = 0
    if device.device_type == "cuda":
        # Cross-check the pre-start-up NVML baseline against CUDA's first reading. The difference
        # is the CUDA context, backend allocations and, under WSL2, the gap between the two
        # reporters; carry it into the cold plan so the first start does not over-commit VRAM.
        from ayaka.device.context import Device

        cuda_free = Device(device.device).memory().free
        initial_runtime_bytes = max(baseline - cuda_free, 0)
        logger.info(
            "device free cross-check: NVML %d bytes before the backend, CUDA %d bytes after it "
            "(%d bytes runtime gap)",
            baseline,
            cuda_free,
            initial_runtime_bytes,
        )
    directory, model = describe_model(config.source, device, ops=ops)
    model_config = model.config
    quant = model.quant_format

    groups = model.kv_specs(block_size=config.cache.block_size, layout=ops.kvcache.layout())
    if len(groups) != 1:
        raise ModelLoadError(
            f"{model_config.architecture} needs {len(groups)} KV cache groups; "
            "this engine runs models with one",
            context={"groups": len(groups)},
        )
    [group] = groups
    block_size = group.spec.block_size
    context = model_config.context_length
    if context < model_config.max_position_embeddings:
        logger.warning(
            "%s was trained on %d tokens but uses a %d-token sliding window, which this "
            "engine does not implement: serving at most %d tokens per request",
            model_config.architecture,
            model_config.max_position_embeddings,
            context,
            context,
        )
    reserve_tokens = config.cache.kv_reserve_tokens or context
    if reserve_tokens > context:
        raise ConfigError(
            "cache.kv_reserve_tokens",
            "RESERVE_EXCEEDS_CONTEXT",
            f"kv_reserve_tokens={reserve_tokens} exceeds the model context of {context} tokens",
        )
    key = ProfileKey.build(
        device=f"{info.name} {capabilities.arch}",
        model=model_config.fingerprint
        if quant is None
        else f"{model_config.fingerprint}+{quant.name}",
        dtype=device.dtype.label,
        providers=ops.providers.items(),
        block_size=block_size,
        libraries=_library_versions(ops.providers.values()),
    )
    previous = load_profile(key) if device.device_type == "cuda" else None
    if previous is not None:
        if initial_runtime_bytes > previous.runtime_bytes:
            logger.warning(
                "saved startup profile measured %d bytes of runtime memory, below the %d-byte "
                "NVML/CUDA gap observed now; planning with %d bytes",
                previous.runtime_bytes,
                initial_runtime_bytes,
                initial_runtime_bytes,
            )
        logger.info(
            "planning with the startup profile measured before: %d bytes of runtime memory, "
            "%d activation bytes per token, %d bytes of graphs",
            previous.runtime_bytes,
            previous.activation_bytes_per_token,
            previous.graph_bytes,
        )
    stored = stored_weights(config.source)
    weights = _weights_bytes(model) if stored is None else stored[0]
    try:
        budget = _budget(
            config,
            group,
            weights,
            baseline,
            reserve_tokens,
            previous,
            initial_runtime_bytes=initial_runtime_bytes,
        )
    except CapacityError as exc:
        # The facts a reader needs to choose a fix: what one token of KV and the weights cost.
        raise CapacityError(
            exc.message,
            context={
                **exc.context,
                "bytes_per_token": group.spec.bytes_per_token(),
                "weights_bytes": weights,
                "kv_reserve_tokens": reserve_tokens,
                "model_context": context,
            },
        ) from exc

    max_blocks = div_ceil(context, block_size)
    running = _max_running(config, budget.max_running)
    blocks = budget.kv_blocks[0]
    if config.cache.num_blocks is None:
        floor = 1 + div_ceil(reserve_tokens, block_size)
        blocks = min(blocks, max(1 + running * max_blocks, floor))
    plan = EnginePlan(
        directory=directory,
        device=info,
        baseline_bytes=baseline,
        capabilities=capabilities,
        providers={str(family): name for family, name in ops.providers.items()},
        model_config=model_config,
        group=group,
        weights_bytes=weights,
        budget=budget,
        kv_blocks=blocks,
        max_running=running,
        max_blocks_per_request=max_blocks,
        reserve_tokens=reserve_tokens,
        chunk_tokens=min(config.scheduler.max_tokens_per_step, budget.max_chunk_tokens),
        profile_key=key,
        profiled=previous is not None,
        quantization=None if quant is None else quant.name,
        stored_types=None if stored is None else stored[1],
    )
    return plan, ops, baseline_reserved


def _budget(
    config: EngineConfig,
    group: KVGroup,
    weights: int,
    baseline: int,
    reserve_tokens: int,
    previous: StartupProfile | None,
    *,
    initial_runtime_bytes: int = 0,
) -> BudgetPlan:
    """The memory plan of the engine, from the free memory, the weights and the KV geometry."""
    spec = group.spec
    late_alloc_bytes = (
        max(previous.runtime_bytes, initial_runtime_bytes)
        if previous is not None
        else initial_runtime_bytes
    )
    return plan_budget(
        BudgetInputs(
            baseline_bytes=baseline,
            memory_ratio=config.cache.memory_ratio,
            weights_bytes=weights,
            kv=(
                KVCost(
                    bytes_per_block=spec.bytes_per_block(),
                    fixed_bytes=spec.fixed_bytes(),
                    block_size=spec.block_size,
                ),
            ),
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


def build_engine(
    config: EngineConfig,
    *,
    tokenizer: LoadedTokenizer | None = None,
    stat_loggers: Sequence[StatLogger] = (),
    recorder: SessionRecorder | None = None,
) -> EngineCore:
    """Build an engine: plan the memory, load the model, allocate the cache, and wire the loop.

    Args:
        config: Everything that describes the run.
        tokenizer: A tokenizer to use instead of loading the one the configuration names; for
            tests and for callers that already hold one.
        stat_loggers: Loggers that get every step besides the logging one (which writes a line
            every :data:`STATS_LOG_INTERVAL` seconds at INFO); the server passes its Prometheus
            logger.
        recorder: Writes every request the engine serves as token ids (``--record``); the engine
            owns it from now on and closes it when it shuts down.

    Returns:
        An engine ready to take requests. Call :meth:`~ayaka.engine.core.EngineCore.shutdown`
        when done.

    Raises:
        ConfigError: If the device cannot run the dtype or CUDA graphs the config asks for, the
            backend list cannot serve the device and dtype (graph-safe families included, with
            graphs on), the KV reserve exceeds the model context, or the tokenizer settings cannot
            work.
        CapabilityError: If torch, the requested GPU, the Hub client or a tokenizer package is
            missing.
        ModelLoadError: If the source cannot be resolved, the checkpoint does not match the model,
            the model needs more than one KV group, or the tokenizer cannot be loaded or does not
            match the model.
        CapacityError: If the weights and a context of KV do not fit the memory the config allows;
            raised by the plan, before a weight is read.
    """
    from ayaka.execution.runner import ModelRunner
    from ayaka.kvcache.storage.pool import KVPool
    from ayaka.model_loader.loader import load_model

    device = config.device
    engine_plan, ops, baseline_reserved = _plan(config)
    model = load_model(config.source, device, ops=ops)
    model_config = engine_plan.model_config
    group = engine_plan.group
    block_size = engine_plan.block_size
    context = model_config.context_length
    plan = engine_plan.budget
    key = engine_plan.profile_key
    baseline = engine_plan.baseline_bytes
    max_blocks = engine_plan.max_blocks_per_request
    running = engine_plan.max_running
    blocks = engine_plan.kv_blocks
    logger.info(
        "memory plan: %d KV blocks fit, %d allocated for up to %d running requests "
        "(%d tokens each at most)",
        plan.kv_blocks[0],
        blocks,
        running,
        engine_plan.max_context_tokens,
    )

    prefix = None
    if config.cache.prefix_cache:
        prefix = HashPrefixCache(
            ChainRoot(
                model_id=f"{config.source.fingerprint}/{model_config.fingerprint}",
                dtype=group.spec.dtype.label,
                codec=group.spec.codec,
                block_size=block_size,
            )
        )
    kv = KVCacheManager(
        [group], [blocks], max_requests=running, max_blocks_per_request=max_blocks, prefix=prefix
    )
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
        eos_token_ids=service.eos_token_ids,
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
        stats=StatLoggerManager(
            loggers=(LoggingStatLogger(), *stat_loggers), log_interval=STATS_LOG_INTERVAL
        ),
        recorder=recorder,
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
    """Measure, capture the graphs, fill the ledger and check it against the device (GPU only)."""
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
    captured = ()
    if config.device.cuda_graph:
        captured = tuple(runner.capture_graphs(buckets))
        graph_bytes = sum(item.bytes for item in captured)
        budget.reserve("graph", min(graph_bytes, budget.free_bytes))
        profile = replace(profile, graph_bytes=graph_bytes)
    target.synchronize()
    reading = target.memory()
    reconciliation = reconcile(
        budget,
        baseline_bytes=baseline,
        baseline_reserved_bytes=baseline_reserved,
        reading=reading,
        reused_bytes=profile.reused_bytes,
    )
    log = logger.info if reconciliation.matches else logger.warning
    log(
        "memory ledger %d bytes, allocator plus runtime %d bytes (difference %d, tolerance %d); "
        "the driver reports %d bytes lost since the baseline (diagnostic difference %d); "
        "plan: %d usable, %d headroom",
        reconciliation.ledger_bytes,
        reconciliation.measured_bytes,
        reconciliation.difference_bytes,
        reconciliation.tolerance_bytes,
        reconciliation.device_bytes,
        reconciliation.device_difference_bytes,
        plan.usable_bytes,
        plan.headroom_bytes,
    )
    if should_save_profile(reconciliation, reading):
        save_profile(profile)
    else:
        issues = "; ".join(profile_save_issues(reconciliation, reading))
        logger.warning("skipping saving startup profile %s: %s", key.digest(), issues)
    return StartupReport(
        profile=profile, budget=budget, reconciliation=reconciliation, graphs=captured
    )


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
