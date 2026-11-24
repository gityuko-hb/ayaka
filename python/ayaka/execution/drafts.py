"""The per-step bookkeeping of speculative drafts that the runner delegates.

``PlannedDraft`` is a proposal the runner asked a proposer for while planning look-ahead rows.
:func:`prepare_step_drafts` turns what was planned (or what a caller passed) into the clean draft
tokens of one step, and :func:`observe_step` tells the controller what the step cost and how many
drafts it accepted. Nothing here touches the model or the device.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from ayaka.core.types import ForwardMode
from ayaka.speculative.acceptance import sanitize_drafts
from ayaka.speculative.proposer import EMPTY_PROPOSAL, Proposal

if TYPE_CHECKING:
    from collections.abc import Sequence

    from ayaka.core.types import RequestId, TokenId
    from ayaka.scheduler.step import StepInput, StepOutput
    from ayaka.speculative.policy import DraftController

__all__ = ["PlannedDraft", "observe_decode", "observe_step", "prepare_step_drafts"]


@dataclass(frozen=True, slots=True)
class PlannedDraft:
    """A proposal planned for a request, and the number of draft rows it asked for."""

    proposal: Proposal
    requested_k: int


def prepare_step_drafts(
    step: StepInput,
    explicit_drafts: Sequence[Sequence[TokenId]] | None,
    *,
    plans: dict[RequestId, PlannedDraft],
    pending_disabled: dict[str, int],
    vocab: int,
) -> tuple[
    list[tuple[TokenId, ...]] | None,
    tuple[Proposal, ...],
    tuple[tuple[bool, ...], ...] | None,
    tuple[tuple[str, int], ...],
]:
    disabled = dict(pending_disabled)
    pending_disabled.clear()
    if step.mode is ForwardMode.PREFILL:
        return None, (), None, ()

    if explicit_drafts is not None:
        if len(explicit_drafts) != step.num_requests:
            return [tuple(d) for d in explicit_drafts], (), None, ()
        clean_list: list[tuple[TokenId, ...]] = []
        valid_list: list[tuple[bool, ...]] = []
        for index, raw in enumerate(explicit_drafts):
            wanted = step.draft_slots[index]
            clipped = tuple(raw[:wanted])
            fallback = step.tokens[index][-1]
            clean, valid = sanitize_drafts(clipped, vocab, fallback)
            clean_list.append(clean)
            valid_list.append(valid)
        return clean_list, (), tuple(valid_list), ()

    if step.mode is ForwardMode.DECODE:
        for request_id, planned in plans.items():
            if request_id in step.request_ids and planned.requested_k > 0:
                disabled["kv_shortage"] = disabled.get("kv_shortage", 0) + 1
        plans.clear()
        return None, (), None, tuple(sorted(disabled.items()))

    # ForwardMode.VERIFY with runner-supplied proposals.
    drafts_out: list[tuple[TokenId, ...]] = []
    proposals_out: list[Proposal] = []
    valid_out: list[tuple[bool, ...]] = []
    for index, request_id in enumerate(step.request_ids):
        wanted = step.draft_slots[index]
        planned = plans.get(request_id)
        if planned is None:
            proposals_out.append(EMPTY_PROPOSAL)
            drafts_out.append(())
            valid_out.append(())
            continue
        if wanted < planned.requested_k and wanted == 0:
            disabled["kv_shortage"] = disabled.get("kv_shortage", 0) + 1
        clipped = planned.proposal.tokens[:wanted]
        fallback = step.tokens[index][-1]
        clean, valid = sanitize_drafts(clipped, vocab, fallback)
        proposals_out.append(
            Proposal(
                tokens=clipped,
                confidence=planned.proposal.confidence,
                source=planned.proposal.source,
                probs=planned.proposal.probs[:wanted],
            )
        )
        drafts_out.append(clean)
        valid_out.append(valid)
    plans.clear()
    return drafts_out, tuple(proposals_out), tuple(valid_out), tuple(sorted(disabled.items()))


def observe_decode(controller: DraftController | None, step: StepInput, elapsed: float) -> None:
    if controller is None:
        return
    max_ctx = max(
        (s + r for s, r in zip(step.start_positions, step.rows_per_request, strict=True)),
        default=0,
    )
    controller.observe(
        source="",
        confidence=0,
        k=0,
        accepted=0,
        batch_size=step.num_requests,
        context_len=max_ctx,
        round_seconds=elapsed,
    )


def observe_step(
    controller: DraftController | None,
    step: StepInput,
    output: StepOutput,
    proposals: Sequence[Proposal],
    elapsed: float,
) -> None:
    if controller is None or not step.mode.is_decode_like:
        return
    if not proposals:
        observe_decode(controller, step, elapsed)
        return
    batch_size = step.num_requests
    for index, proposal in enumerate(proposals):
        k = len(proposal.tokens)
        ctx = step.start_positions[index] + step.rows_per_request[index]
        acc = output.accepted[index] if output.accepted else 0
        controller.observe(
            source=proposal.source,
            confidence=proposal.confidence,
            k=k,
            accepted=acc,
            batch_size=batch_size,
            context_len=ctx,
            round_seconds=elapsed,
        )
