"""Turn the tokens a step drew into the output the scheduler reads.

A decode request draws one token per logit row. :func:`answer_verify` applies the acceptance rule
of :mod:`ayaka.speculative.acceptance` to the draft tokens of each request and keeps the tokens
plain decoding would have emitted; :func:`answer_step` is the same for a step whose rows came with
their inputs. These functions hold no state and import no torch.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ayaka.sampling.logits import LogitRow
from ayaka.scheduler.step import StepOutput
from ayaka.speculative.acceptance import accept_one_point

if TYPE_CHECKING:
    from collections.abc import Sequence

    from ayaka.core.outputs import TokenLogprob
    from ayaka.core.types import TokenId
    from ayaka.execution.staged import ModelInputs
    from ayaka.scheduler.step import StepInput
    from ayaka.speculative.proposer import Proposal

__all__ = ["answer_step", "answer_verify", "logit_rows"]


def logit_rows(step: StepInput, logit_counts: Sequence[int]) -> list[LogitRow]:
    """One :class:`LogitRow` per logit row: row ``j`` of a request draws its token ``j``."""
    rows: list[LogitRow] = []
    for index, taken in enumerate(logit_counts):
        request_id = step.request_ids[index]
        params = step.sampling_params[index]
        produced = step.output_length(index)
        rows += [LogitRow(request_id, params, produced + j) for j in range(taken)]
    return rows


def answer_verify(
    step: StepInput,
    logit_counts: Sequence[int],
    draft_tokens: Sequence[Sequence[TokenId]],
    sampled: list[TokenId],
    logprobs: Sequence[TokenLogprob | None] = (),
    *,
    draft_valid: Sequence[Sequence[bool]] | None = None,
    vocab_size: int | None = None,
    used_proposals: Sequence[Proposal] = (),
    disabled: tuple[tuple[str, int], ...] = (),
) -> StepOutput:
    """Turn the drawn tokens into per-request tokens, applying the draft acceptance rule."""
    tokens: list[tuple[TokenId, ...]] = []
    accepted: list[int] = []
    drafted: list[int] = []
    pred_acc: list[int] = []
    pred_rej: list[int] = []
    per_request: list[tuple[TokenLogprob, ...] | None] = []
    cursor = 0
    for index, taken in enumerate(logit_counts):
        drawn = sampled[cursor : cursor + taken]
        window = logprobs[cursor : cursor + taken] if logprobs else ()
        cursor += taken
        drafts = draft_tokens[index]
        valid = draft_valid[index] if draft_valid is not None and index < len(draft_valid) else None
        result = accept_one_point(drawn, drafts, valid=valid, vocab_size=vocab_size)
        count = result.accepted
        tokens.append(result.tokens)
        accepted.append(count)
        drafted.append(len(drafts))
        if index < len(used_proposals) and used_proposals[index].source == "prediction":
            pred_acc.append(count)
            pred_rej.append(len(drafts) - count)
        else:
            pred_acc.append(0)
            pred_rej.append(0)
        kept = [entry for entry in window[: count + 1] if entry is not None]
        per_request.append(tuple(kept) if kept and len(kept) == len(result.tokens) else None)
    return StepOutput(
        step_id=step.step_id,
        request_ids=step.request_ids,
        tokens=tuple(tokens),
        accepted=tuple(accepted),
        logprobs=tuple(per_request) if any(entry is not None for entry in per_request) else (),
        spec_drafted=tuple(drafted) if any(drafted) else (),
        spec_prediction_accepted=tuple(pred_acc) if any(pred_acc) else (),
        spec_prediction_rejected=tuple(pred_rej) if any(pred_rej) else (),
        spec_disabled=disabled,
    )


def answer_step(
    step: StepInput,
    inputs: ModelInputs,
    sampled: list[TokenId],
    logprobs: Sequence[TokenLogprob | None] = (),
    *,
    draft_valid: Sequence[Sequence[bool]] | None = None,
    vocab_size: int | None = None,
    used_proposals: Sequence[Proposal] = (),
    disabled: tuple[tuple[str, int], ...] = (),
) -> StepOutput:
    return answer_verify(
        step,
        inputs.logit_counts,
        inputs.draft_tokens,
        sampled,
        logprobs,
        draft_valid=draft_valid,
        vocab_size=vocab_size,
        used_proposals=used_proposals,
        disabled=disabled,
    )
