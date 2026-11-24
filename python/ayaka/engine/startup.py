"""The start-up run of an engine: measure, capture the graphs, fill the ledger, reconcile it.

``startup_run`` is the last step of :func:`ayaka.engine.build.build_engine` on a GPU. It measures
what the model needs beyond its weights and the cache, captures the CUDA graphs (the decode buckets
and, when speculation is on, the verify windows) and checks the memory plan against the device.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import replace
from typing import TYPE_CHECKING

from ayaka.engine.profile import (
    ProfileKey,
    StartupReport,
    measure_startup,
    profile_save_issues,
    reconcile,
    save_profile,
    should_save_profile,
)
from ayaka.memory.budget import MemoryBudget

if TYPE_CHECKING:
    from ayaka.configs.engine import EngineConfig
    from ayaka.execution.runner import ModelRunner
    from ayaka.kvcache.storage.pool import KVPool, LatentKVPool, RecurrentStatePool
    from ayaka.memory.budget import BudgetPlan

__all__ = ["startup_run", "verify_buckets"]

logger = logging.getLogger(__name__)


def startup_run(
    config: EngineConfig,
    runner: ModelRunner,
    pools: KVPool | LatentKVPool | Sequence[KVPool | LatentKVPool],
    *,
    state_pool: RecurrentStatePool | None = None,
    key: ProfileKey,
    plan: BudgetPlan,
    baseline: int,
    baseline_reserved: int,
    chunk_tokens: int,
    running: int,
    draft_bytes: int = 0,
) -> StartupReport:
    """Measure, capture the graphs, fill the ledger and check it against the device (GPU only).

    ``draft_bytes`` are the weights of a speculative drafter loaded beside the model: the ledger
    holds them apart from the model's own, so the reconciliation counts them once.
    """
    from ayaka.device.context import Device

    target = Device(config.device.device)
    buckets = tuple(bucket for bucket in config.device.graph_buckets if bucket <= running) or (1,)
    pool_seq = pools if isinstance(pools, Sequence) else (pools,)
    kv_total_bytes = sum(pool.nbytes for pool in pool_seq)
    state_bytes = state_pool.nbytes if state_pool is not None else 0
    profile = measure_startup(
        runner.model,
        runner.kv_buffers,
        key=key,
        device=target,
        baseline_bytes=baseline,
        baseline_reserved_bytes=baseline_reserved,
        kv_bytes=kv_total_bytes,
        extra_weights_bytes=draft_bytes,
        chunk_tokens=chunk_tokens,
        buckets=buckets,
        block_size=pool_seq[0].group.spec.block_size,
        usable_blocks=pool_seq[0].usable_blocks,
        latent_kv_buffers=getattr(runner, "latent_kv_buffers", {}),
        state_buffers=getattr(runner, "state_buffers", {}),
        group_layers=getattr(runner, "group_layers", ()),
        state_bytes=state_bytes,
    )
    budget = MemoryBudget(baseline)
    budget.reserve("weights", profile.weights_bytes)
    if draft_bytes > 0:
        budget.reserve("draft", min(draft_bytes, budget.free_bytes))
    budget.reserve("kv", profile.kv_bytes)
    if state_bytes:
        budget.reserve("state", min(state_bytes, budget.free_bytes))
    budget.reserve("runtime", min(profile.runtime_bytes, budget.free_bytes))
    budget.reserve("workspace", min(profile.workspace_bytes, budget.free_bytes))
    captured = ()
    if config.device.cuda_graph:
        windows = verify_buckets(config, runner)
        captured = tuple(runner.capture_graphs(buckets, verify_buckets=windows))
        graph_bytes = sum(item.bytes for item in captured) + runner.extra_graph_bytes
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


def verify_buckets(config: EngineConfig, runner: ModelRunner) -> tuple[int, ...]:
    """Total-row CUDA graph buckets to capture when speculative decoding is active."""
    if not config.speculative.enabled or runner.proposer is None:
        return ()
    max_rows = runner.builder.max_rows
    k = config.speculative.max_draft_tokens
    candidates = {2, 3, 4, 5, 6, 8, 9, 10, 12, 16, 24, 32, 48, 64, 1 + k}
    return tuple(sorted(b for b in candidates if 2 <= b <= max_rows))
