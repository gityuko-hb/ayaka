"""The speculative state of the requests a runner serves: who drafts, what, and why not.

:class:`SpeculationState` owns the proposer and the controller of a :class:`ModelRunner`, one
record per admitted request, the drafts planned for the next step and the reasons requests drafted
nothing, and the hooks a drafter with weights reads (the last hidden states, the features of
some layers). The runner delegates its ``admit_speculative``, ``commit_speculative``,
``release_speculative`` and ``plan_lookahead`` to it, and hands it each step's features.

A request with a penalty or a grammar does not speculate in this version; the step output says
why (``spec_disabled``).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from ayaka.execution.drafts import PlannedDraft
from ayaka.layers.context import FeatureSink
from ayaka.sampling.penalties import uses_penalties

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from ayaka.configs.speculative import SpeculativeConfig
    from ayaka.core.params import SamplingParams
    from ayaka.core.types import RequestId, TokenId
    from ayaka.execution.graph import GraphRunner
    from ayaka.sampling.structured.runtime import ConstraintRuntime
    from ayaka.scheduler.step import StepInput, StepOutput
    from ayaka.speculative.policy import DraftController
    from ayaka.speculative.proposer import Proposer

__all__ = ["SpeculationState"]

_LOG = logging.getLogger(__name__)

#: Draft tokens per round when no configuration says.
_DEFAULT_MAX_DRAFT_TOKENS = 8


@dataclass(slots=True)
class _RequestState:
    context_len: int
    disable_reason: str | None = None
    has_prediction: bool = False


class SpeculationState:
    """The proposer, the controller and the per-request draft state of one runner."""

    def __init__(
        self,
        constraints: ConstraintRuntime,
        *,
        speculative: SpeculativeConfig | None = None,
        proposer: Proposer | None = None,
        controller: DraftController | None = None,
    ) -> None:
        self._constraints = constraints
        self.speculative = speculative
        self._proposer = proposer
        self.controller = controller
        self.failed = False
        self._requests: dict[RequestId, _RequestState] = {}
        self.planned: dict[RequestId, PlannedDraft] = {}
        self.pending_disabled: dict[str, int] = {}
        self.needs_hidden = False
        self.feature_order: tuple[int, ...] = ()
        self.feature_sink: FeatureSink | None = None
        self._configure_hooks()

    def _configure_hooks(self) -> None:
        proposer = self._proposer
        self.needs_hidden = bool(getattr(proposer, "needs_hidden", False))
        order = tuple(getattr(proposer, "target_layer_ids", ()))
        self.feature_order = order
        self.feature_sink = FeatureSink(layer_ids=frozenset(order)) if order else None

    @property
    def proposer(self) -> Proposer | None:
        """The proposer, or ``None`` when there is none or it failed."""
        return None if self.failed else self._proposer

    @property
    def configured_proposer(self) -> Proposer | None:
        """The proposer as configured, even after it failed (it is still released)."""
        return self._proposer

    def replace(
        self,
        proposer: Proposer | None,
        controller: DraftController | None,
        speculative: SpeculativeConfig | None,
    ) -> None:
        """Install another proposer and controller; plans and counts start over."""
        self._proposer = proposer
        self.controller = controller
        if speculative is not None:
            self.speculative = speculative
        self.failed = False
        self.planned.clear()
        self.pending_disabled.clear()
        self._configure_hooks()

    def fail(self, where: str, exc: Exception) -> None:
        """Stop using the proposer for the life of the process; requests go on without drafts."""
        _LOG.warning("disabling speculative proposer after %s error: %s", where, exc)
        self.failed = True

    def admit(
        self,
        request_id: RequestId,
        prompt: Sequence[TokenId],
        *,
        sampling: SamplingParams,
        has_constraint: bool,
        prediction: Sequence[TokenId],
    ) -> None:
        """Register a request with the proposer and record whether it may draft."""
        proposer = self.proposer
        if proposer is None:
            return
        reason: str | None = None
        if uses_penalties(sampling):
            reason = "penalty"
        elif has_constraint:
            reason = "grammar"
        state = _RequestState(
            context_len=len(prompt),
            disable_reason=reason,
            has_prediction=bool(prediction),
        )
        self._requests[request_id] = state
        if reason is not None:
            return
        try:
            proposer.on_admit(
                request_id,
                prompt,
                prediction=prediction,
                allowed_tokens=sampling.allowed_token_ids,
            )
        except Exception as exc:
            self.fail("on_admit", exc)
            state.disable_reason = "proposer_error"

    def commit(self, request_id: RequestId, tokens: Sequence[TokenId]) -> None:
        """Feed committed tokens to the proposer; unverified drafts never enter."""
        state = self._requests.get(request_id)
        if state is None:
            return
        state.context_len += len(tokens)
        proposer = self.proposer
        if proposer is None or state.disable_reason is not None:
            return
        try:
            proposer.on_commit(request_id, tokens)
        except Exception as exc:
            self.fail("on_commit", exc)

    def release(self, request_id: RequestId) -> None:
        """Drop a request that finished or was aborted."""
        self._requests.pop(request_id, None)
        self.planned.pop(request_id, None)
        if self._proposer is not None:
            try:
                self._proposer.release(request_id)
            except Exception as exc:
                _LOG.warning("speculative proposer release failed for %r: %s", request_id, exc)

    def release_all(self) -> None:
        """Release every request (on shutdown)."""
        for request_id in tuple(self._requests):
            self.release(request_id)

    def plan(self) -> Mapping[RequestId, int]:
        """Ask the proposer and the controller for the draft rows of every request."""
        self.planned.clear()
        self.pending_disabled.clear()
        proposer = self.proposer
        if proposer is None or not self._requests:
            return {}
        max_k = (
            self.speculative.max_draft_tokens
            if self.speculative is not None
            else _DEFAULT_MAX_DRAFT_TOKENS
        )
        batch_size = len(self._requests)
        plan: dict[RequestId, int] = {}
        for request_id, state in tuple(self._requests.items()):
            if state.disable_reason is not None:
                self._disabled(state.disable_reason)
                continue
            try:
                proposal = proposer.propose(request_id, max_k)
            except Exception as exc:
                self.fail("propose", exc)
                self._disabled("proposer_error")
                break
            if not proposal.tokens:
                self._disabled("no_proposal")
                continue
            if self.controller is not None:
                k = self.controller.choose_k(
                    proposal, max_k, batch_size=batch_size, context_len=state.context_len
                )
            else:
                k = min(len(proposal.tokens), max_k)
            k = max(0, min(k, len(proposal.tokens), max_k))
            if k == 0:
                self._disabled("controller")
                continue
            self.planned[request_id] = PlannedDraft(proposal=proposal, requested_k=k)
            plan[request_id] = k
        return plan

    def notify(
        self,
        step: StepInput,
        output: StepOutput,
        hidden: torch.Tensor | None,
        graph: GraphRunner | None,
    ) -> None:
        """Hand a step's layer features to a drafter that reads them, then forget them.

        ``graph`` is the graph runner when the step replayed a graph (its features are in the
        graph's buffers), ``None`` when it ran eagerly (they are in :attr:`feature_sink`).
        """
        del hidden  # read by drafters that predict from the last hidden state
        proposer = self.proposer
        sink = self.feature_sink
        try:
            if (
                proposer is not None
                and self.feature_order
                and hasattr(proposer, "observe_features")
            ):
                if graph is not None:
                    feats = graph.features_view(self.feature_order, step.num_rows)
                elif sink is not None:
                    feats = torch.cat([sink.get(lid) for lid in self.feature_order], dim=-1)
                else:
                    feats = None
                if feats is not None:
                    emitted = output.tokens if output.tokens else ((),) * step.num_requests
                    proposer.observe_features(  # type: ignore[attr-defined]  # duck-typed block-drafter hook
                        step.request_ids,
                        step.start_positions,
                        step.rows_per_request,
                        emitted,
                        feats,
                    )
        except Exception as exc:
            self.fail("step update", exc)
        finally:
            if sink is not None:
                sink.clear()

    def _disabled(self, reason: str) -> None:
        self.pending_disabled[reason] = self.pending_disabled.get(reason, 0) + 1
