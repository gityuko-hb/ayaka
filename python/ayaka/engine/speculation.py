"""The speculative drafter of an engine: which proposer, which controller, how many look-ahead rows.

:func:`build_speculation` is the step of :func:`ayaka.engine.build.build_engine` that turns the
speculative configuration into a proposer (lookup, token recycling, or the multi-token-prediction
layer of a MiMo checkpoint) and a draft-length controller, after the memory plan has decided
whether a drafter with weights fits.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

from ayaka.speculative.ngram import LookupProposer
from ayaka.speculative.policy import CostAwareController, FixedController
from ayaka.speculative.token_recycling import TokenRecyclingProposer

if TYPE_CHECKING:
    from ayaka.configs.device import DeviceConfig
    from ayaka.configs.speculative import SpeculativeConfig
    from ayaka.engine.build import EnginePlan
    from ayaka.kvcache.storage.pool import KVPool, LatentKVPool
    from ayaka.kvcache.storage.spec import KVGroup, LatentKVGroup
    from ayaka.models.base import CausalLM
    from ayaka.ops.backend import OpsBackend
    from ayaka.speculative.policy import DraftController
    from ayaka.speculative.proposer import Proposer

__all__ = ["SpeculativeSetup", "build_speculation"]

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class SpeculativeSetup:
    """What :func:`build_speculation` decided.

    Attributes:
        proposer: The drafter, or ``None`` when speculation is off or refused.
        controller: The draft-length controller, or ``None`` without a proposer.
        lookahead: Draft rows a request may carry per step; the runner's buffers hold them.
        draft_bytes: Bytes of the drafter's own weights, loaded beside the model.
    """

    proposer: Proposer | None = None
    controller: DraftController | None = None
    lookahead: int = 0
    draft_bytes: int = 0


def build_speculation(
    spec_cfg: SpeculativeConfig,
    *,
    device: DeviceConfig,
    model: CausalLM,
    ops: OpsBackend,
    engine_plan: EnginePlan,
    pool: KVPool | LatentKVPool,
    group: KVGroup | LatentKVGroup,
    mtp_active: bool,
    has_recurrent_state: bool = False,
) -> SpeculativeSetup:
    """The proposer and controller ``spec_cfg`` asks for, or none, with the reason logged.

    ``pool`` and ``group`` are the model's KV pool and group, the prediction layer's KV included
    when ``mtp_active``. A model with recurrent-state layers refuses every drafter: a verify
    window advances the state through every draft row, and a rejected draft cannot be undone.
    """
    proposer: Proposer | None = None
    controller: DraftController | None = None
    lookahead = 0
    draft_bytes = 0
    if spec_cfg.enabled:
        if spec_cfg.method == "lookup":
            proposer = LookupProposer(
                min_match=spec_cfg.lookup_min_match,
                max_match=spec_cfg.lookup_max_match,
                candidates=spec_cfg.lookup_candidates,
                accept_prediction=spec_cfg.accept_prediction,
            )
        elif spec_cfg.method == "token_recycling":
            proposer = TokenRecyclingProposer()
        elif spec_cfg.method == "mtp":
            from ayaka.kvcache.storage.pool import KVPool

            # mtp_active means the plan put the prediction layer in the model's one paged group.
            if (
                mtp_active
                and isinstance(pool, KVPool)
                and hasattr(model, "rotary")
                and hasattr(model, "embed_tokens")
            ):
                from ayaka.models.mimo import load_mimo_mtp_layer
                from ayaka.speculative.mtp import MTPProposer

                mtp_layer = load_mimo_mtp_layer(
                    engine_plan.directory,
                    engine_plan.model_config,
                    rotary=model.rotary,  # type: ignore[attr-defined]  # MiMo/Qwen2/Llama share rotary
                    ops=ops,
                    dtype=device.dtype,
                    device=device.device,
                    layer_id=engine_plan.model_config.num_layers,
                    mtp_index=0,
                )
                draft_bytes = sum(p.numel() * p.element_size() for p in mtp_layer.parameters())
                kv_bufs = {lid: pool.buffers(lid) for lid in group.layer_ids}
                lm_head_mod = getattr(model, "lm_head", None)
                lm_head_w = getattr(lm_head_mod, "weight", None)
                proposer = MTPProposer(
                    mtp_layer,
                    model.embed_tokens,  # type: ignore[attr-defined]  # Llama/Qwen2/MiMo embed_tokens
                    model.compute_logits,
                    attention=ops.attention,
                    kv_buffers=kv_bufs,
                    mtp_layer_id=engine_plan.model_config.num_layers,
                    block_size=engine_plan.block_size,
                    vocab_size=engine_plan.model_config.vocab_size,
                    hidden_size=engine_plan.model_config.hidden_size,
                    dtype=device.dtype.torch_dtype,
                    device=device.device,
                    depth=spec_cfg.mtp_depth,
                    vocab_limit=spec_cfg.mtp_vocab_limit,
                    lookup_override_match=spec_cfg.lookup_override_match,
                    lm_head_weight=lm_head_w,
                )
            elif any(name == "mtp" for name, _ in engine_plan.budget.skipped):
                logger.warning("speculative mtp disabled: insufficient memory in budget")
            else:
                logger.warning(
                    "speculative mtp disabled: %s checkpoint has num_nextn_predict_layers=0",
                    engine_plan.model_config.architecture,
                )
        if proposer is not None:
            reason = proposer.supports(engine_plan.model_config)
            if reason is None and has_recurrent_state:
                # A verify window advances a recurrent state through every draft row, and a
                # rejected draft would leave it there: there is no rollback of the state.
                reason = "the model has recurrent-state layers, which a draft window cannot undo"
            if reason is not None:
                logger.warning("speculative %s disabled: %s", spec_cfg.method, reason)
                proposer = None
                lookahead = 0
            else:
                if lookahead == 0:
                    lookahead = spec_cfg.max_draft_tokens
                controller = (
                    FixedController(min(spec_cfg.fixed_k, lookahead))
                    if spec_cfg.controller == "fixed"
                    else CostAwareController(
                        max_k=lookahead,
                        min_gain=spec_cfg.min_gain,
                        ema=spec_cfg.ema,
                    )
                )
    return SpeculativeSetup(
        proposer=proposer, controller=controller, lookahead=lookahead, draft_bytes=draft_bytes
    )
