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
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Final, Literal

from ayaka.core.errors import CapacityError, ConfigError, ModelLoadError
from ayaka.device.capabilities import device_capabilities
from ayaka.engine._plan_helpers import (
    compute_budget,
    free_device_bytes,
    library_versions,
    max_running,
    release_cached_device_memory,
    resolve_kv_codec,
    tokenizer_service,
    weights_bytes,
)
from ayaka.engine.core import EngineCore
from ayaka.engine.input import InputProcessor
from ayaka.engine.output import OutputProcessor
from ayaka.engine.profile import (
    ProfileKey,
    load_profile,
)
from ayaka.engine.startup import startup_run
from ayaka.kvcache.manager import KVCacheManager
from ayaka.kvcache.prefix.hash_cache import HashPrefixCache
from ayaka.kvcache.prefix.hashing import ChainRoot
from ayaka.kvcache.state_slots import StateCheckpointStore, StateSlotPool
from ayaka.kvcache.storage.spec import KVGroup, KVSpec, LatentKVGroup, RecurrentStateGroup
from ayaka.memory.budget import DraftCost
from ayaka.metrics.loggers import LoggingStatLogger
from ayaka.metrics.manager import StatLoggerManager
from ayaka.ops.kvcache import RAW_CODEC
from ayaka.scheduler.scheduler import Scheduler
from ayaka.utils.math_utils import div_ceil

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from pathlib import Path

    from ayaka.configs.engine import EngineConfig
    from ayaka.configs.model import ModelConfig
    from ayaka.device.capabilities import DeviceCapabilities
    from ayaka.device.info import DeviceInfo
    from ayaka.engine.recorder import SessionRecorder
    from ayaka.kvcache.storage.pool import KVPool, LatentKVPool, RecurrentStatePool
    from ayaka.memory.budget import BudgetPlan
    from ayaka.metrics.loggers import StatLogger
    from ayaka.ops.backend import OpsBackend
    from ayaka.tokenizers.contracts import LoadedTokenizer

__all__ = ["STATS_LOG_INTERVAL", "EnginePlan", "build_engine", "plan_engine"]

logger = logging.getLogger(__name__)

#: Seconds between two lines of the logging stat logger.
STATS_LOG_INTERVAL: Final = 10.0


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
    group: KVGroup | LatentKVGroup
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
    groups: tuple[KVGroup | LatentKVGroup, ...] = ()
    state_group: RecurrentStateGroup | None = None
    kv_groups: int = 1
    state_bytes: int = 0
    latent_kv_bytes: int = 0
    vision_weights_bytes: int = 0
    vision_activation_bytes: int = 0
    vision_load: Literal["eager", "lazy"] = "eager"

    def __post_init__(self) -> None:
        if not self.groups:
            object.__setattr__(self, "groups", (self.group,))

    displaced_kv_blocks: int = 0

    @property
    def block_size(self) -> int:
        """Tokens per KV block."""
        return self.group.spec.block_size

    @property
    def kv_bytes(self) -> int:
        """Bytes of the KV/latent-KV pools the engine allocates."""
        return sum(
            self.kv_blocks * group.spec.bytes_per_block() + group.spec.fixed_bytes()
            for group in (self.groups or (self.group,))
        )

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
    baseline_reserved = release_cached_device_memory(device)
    info, baseline = free_device_bytes(device)
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
    kv_codec, kv_scale = resolve_kv_codec(config)

    groups = model.kv_specs(
        block_size=config.cache.block_size,
        layout=ops.kvcache.layout(),
        codec=kv_codec,
        kv_scale=kv_scale,
    )
    if not groups:
        raise ModelLoadError(
            f"{model_config.architecture} declared zero KV cache groups",
            context={"groups": 0},
        )
    group = groups[0]
    plain_groups = groups
    state_group = model.state_spec()
    draft_costs: tuple[DraftCost, ...] = ()
    if (
        config.speculative.enabled
        and config.speculative.method == "mtp"
        and model_config.num_nextn_predict_layers >= 1
        and len(groups) == 1
        and isinstance(group, KVGroup)
    ):
        from ayaka.models.mimo import describe_mimo_mtp_bytes

        # The prediction layer keeps its KV as one more layer of the model's single KV group.
        mtp_bytes = describe_mimo_mtp_bytes(model_config, dtype=device.dtype)
        draft_costs = (DraftCost(name="mtp", fixed_bytes=mtp_bytes, bytes_per_kv_token=0),)
        group = KVGroup(
            layer_ids=(*group.layer_ids, model_config.num_layers),
            spec=replace(group.spec, num_layers=group.spec.num_layers + 1),
        )
        groups = (group,)
    block_size = group.spec.block_size
    context = model_config.context_length
    if (
        len(groups) == 1
        and isinstance(group.spec, KVSpec)
        and group.spec.window is None
        and context < model_config.max_position_embeddings
    ):
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
    base_fp = (
        model_config.fingerprint if quant is None else f"{model_config.fingerprint}+{quant.name}"
    )
    model_fp = base_fp if kv_codec == RAW_CODEC else f"{base_fp}+kv:{kv_codec}"
    if model_config.has_vision:
        model_fp = f"{model_fp}+vision:{device.vision_load}"
    if draft_costs:
        model_fp += "+" + "+".join(cost.name for cost in draft_costs)
    key = ProfileKey.build(
        device=f"{info.name} {capabilities.arch}",
        model=model_fp,
        dtype=device.dtype.label,
        providers=ops.providers.items(),
        block_size=block_size,
        libraries=library_versions(ops.providers.values()),
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
    vision_weights = model.vision_weight_bytes()
    base_weights = weights_bytes(model) if stored is None else stored[0]
    weights = base_weights + (vision_weights if device.vision_load == "eager" else 0)
    try:
        budget = compute_budget(
            config,
            groups,
            state_group,
            weights,
            baseline,
            reserve_tokens,
            previous,
            initial_runtime_bytes=initial_runtime_bytes,
            drafts=draft_costs,
        )
    except CapacityError as exc:
        # The facts a reader needs to choose a fix: what one token of KV and the weights cost.
        bytes_per_token = sum(g.spec.bytes_per_token() for g in groups)
        raise CapacityError(
            exc.message,
            context={
                **exc.context,
                "bytes_per_token": bytes_per_token,
                "weights_bytes": weights,
                "kv_reserve_tokens": reserve_tokens,
                "model_context": context,
            },
        ) from exc

    if any(name == "mtp" for name, _ in budget.skipped) and groups is not plain_groups:
        groups = plain_groups
        group = groups[0]
        budget = compute_budget(
            config,
            groups,
            state_group,
            weights,
            baseline,
            reserve_tokens,
            previous,
            initial_runtime_bytes=initial_runtime_bytes,
            drafts=(),
        )
    displaced_blocks = 0
    if budget.loaded_drafts:
        try:
            plain_budget = compute_budget(
                config,
                plain_groups,
                state_group,
                weights,
                baseline,
                reserve_tokens,
                previous,
                initial_runtime_bytes=initial_runtime_bytes,
                drafts=(),
            )
            displaced_blocks = max(plain_budget.kv_blocks[0] - budget.kv_blocks[0], 0)
        except CapacityError:
            displaced_blocks = 0

    max_blocks = div_ceil(context, block_size)
    running = max_running(config, budget.max_running)
    blocks = budget.kv_blocks[0]
    if config.cache.num_blocks is None:
        floor = 1 + div_ceil(reserve_tokens, block_size)
        blocks = min(blocks, max(1 + running * max_blocks, floor))
    latent_bytes = sum(
        blocks * g.spec.bytes_per_block() + g.spec.fixed_bytes()
        for g in groups
        if isinstance(g, LatentKVGroup)
    )
    plan = EnginePlan(
        directory=directory,
        device=info,
        baseline_bytes=baseline,
        capabilities=capabilities,
        providers={str(family): name for family, name in ops.providers.items()},
        model_config=model_config,
        group=group,
        groups=groups,
        state_group=state_group,
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
        kv_groups=len(groups),
        state_bytes=budget.state_bytes,
        latent_kv_bytes=latent_bytes,
        vision_weights_bytes=vision_weights,
        vision_activation_bytes=model.vision_activation_bytes(),
        vision_load=device.vision_load,
        displaced_kv_blocks=displaced_blocks,
    )
    return plan, ops, baseline_reserved


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
    from ayaka.engine.speculation import build_speculation
    from ayaka.execution.runner import ModelRunner
    from ayaka.kvcache.storage.pool import KVPool, LatentKVPool, RecurrentStatePool
    from ayaka.model_loader.loader import load_model

    device = config.device
    engine_plan, ops, baseline_reserved = _plan(config)
    model = load_model(config.source, device, ops=ops)
    model_config = engine_plan.model_config
    group = engine_plan.group
    groups = engine_plan.groups
    state_group = engine_plan.state_group
    block_size = engine_plan.block_size
    context = model_config.context_length
    plan = engine_plan.budget
    key = engine_plan.profile_key
    baseline = engine_plan.baseline_bytes
    max_blocks = engine_plan.max_blocks_per_request
    running = engine_plan.max_running
    blocks = engine_plan.kv_blocks
    logger.info(
        "memory plan: %d KV blocks fit across %d group(s), %d allocated for up to %d running "
        "requests (%d tokens each at most)",
        plan.kv_blocks[0],
        len(groups),
        blocks,
        running,
        engine_plan.max_context_tokens,
    )

    spec_cfg = config.speculative
    mtp_active = (
        spec_cfg.enabled
        and spec_cfg.method == "mtp"
        and model_config.num_nextn_predict_layers >= 1
        and "mtp" in plan.loaded_drafts
    )
    prefix = None
    if config.cache.prefix_cache:
        prefix = HashPrefixCache(
            ChainRoot(
                model_id=f"{config.source.fingerprint}/{model_config.fingerprint}",
                dtype=group.spec.dtype.label,
                codec=group.spec.codec,
                block_size=block_size,
                lookahead=1 if mtp_active else 0,
            )
        )
    state_slots: StateSlotPool | None = None
    state_checkpoints: StateCheckpointStore | None = None
    state_pool: RecurrentStatePool | None = None
    if state_group is not None:
        state_slots = StateSlotPool(running + 1)
        state_pool = RecurrentStatePool(state_group, running + 1, device=device.device)
        if prefix is not None:
            state_checkpoints = StateCheckpointStore(
                config.cache.state_checkpoint_max_bytes,
                stride_blocks=config.cache.state_checkpoint_stride,
            )

    block_counts = [blocks] * len(groups)
    kv = KVCacheManager(
        groups,
        block_counts,
        max_requests=running,
        max_blocks_per_request=max_blocks,
        prefix=prefix,
        state_slots=state_slots,
        state_checkpoints=state_checkpoints,
    )
    pools: list[KVPool | LatentKVPool] = [
        LatentKVPool(grp, blocks, device=device.device)
        if isinstance(grp, LatentKVGroup)
        else KVPool(grp, blocks, device=device.device)
        for grp in groups
    ]

    service = tokenizer_service(config, model_config.vocab_size, context, tokenizer)
    speculation = build_speculation(
        spec_cfg,
        device=device,
        model=model,
        ops=ops,
        engine_plan=engine_plan,
        pool=pools[0],
        group=group,
        mtp_active=mtp_active,
        has_recurrent_state=state_group is not None,
    )
    proposer, controller, lookahead = (
        speculation.proposer,
        speculation.controller,
        speculation.lookahead,
    )
    scheduler = Scheduler(
        config.scheduler,
        kv,
        max_model_len=context,
        max_chunk_tokens=plan.max_chunk_tokens,
        eos_token_ids=service.eos_token_ids,
    )
    runner = ModelRunner(
        model,
        pools,
        kv.table,
        max_rows=ModelRunner.required_rows(
            token_budget=scheduler.token_budget,
            max_running=scheduler.max_running,
            lookahead=lookahead,
        ),
        device=device.device,
        eos_token_ids=service.eos_token_ids,
        state_pool=state_pool,
        state_checkpoints=state_checkpoints,
        speculative=spec_cfg,
        proposer=proposer,
        controller=controller,
    )
    startup = None
    if device.device_type == "cuda":
        startup = startup_run(
            config,
            runner,
            pools,
            state_pool=state_pool,
            key=key,
            plan=plan,
            baseline=baseline,
            baseline_reserved=baseline_reserved,
            chunk_tokens=scheduler.token_budget,
            running=scheduler.max_running,
            draft_bytes=speculation.draft_bytes,
        )
    return EngineCore(
        scheduler=scheduler,
        runner=runner,
        kv=kv,
        input_processor=InputProcessor(
            service,
            max_model_len=context,
            vocab_size=model_config.vocab_size,
            structured_backend=config.structured_backend,
            vision=model_config.vision_spec,
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
        device=device.device,
        dtype=device.dtype.label,
        backends=device.backends,
        cuda_graph=device.cuda_graph,
    )
