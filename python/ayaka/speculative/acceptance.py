"""Exact one-point speculative verification and draft-token sanitization.

A request enters a verify step with ``1 + k`` rows: the pending token followed by ``k`` draft
tokens. Row ``j`` draws ``drawn[j]`` from the target's distribution using the position-keyed
``PICK`` random stream, so the token drawn at every position is independent of what draft window
it sits in.

For a point-mass draft ``q = delta_d`` (greedy or sampled with ``temperature > 0``):

* Accept ``a`` consecutive drafts ``j = 0 .. a - 1`` while ``drafts[j] == drawn[j]`` (and the draft
  token is valid and within the vocabulary).
* Emit ``drawn[:a + 1]`` (``a`` accepted drafts plus the bonus or correcting token ``drawn[a]``).

For ``q = delta_d``, ``P(accept) = p(d)`` and a rejection leaves ``p`` with ``d`` removed (drawn
from the exact residual distribution because ``drawn[j] ~ p`` was compared to ``d``). Thus the
emitted token distribution is identically ``p`` (Leviathan et al., arXiv:2211.17192) and a
drafter with no candidate list (``q = delta_d``) is never rejected outright.

This module imports no torch.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

    from ayaka.core.types import TokenId

__all__ = [
    "AcceptanceResult",
    "accept_one_point",
    "accept_one_point_batch",
    "sanitize_drafts",
]


@dataclass(frozen=True, slots=True)
class AcceptanceResult:
    """Outcome of verifying one request's draft window.

    Attributes:
        tokens: Emitted token ids (``a + 1`` tokens when ``drawn`` is non-empty).
        accepted: Number of draft tokens accepted (``a``).
        rejected: Number of draft tokens rejected (``len(drafts) - a``).
    """

    tokens: tuple[TokenId, ...]
    accepted: int
    rejected: int


def sanitize_drafts(
    drafts: Sequence[TokenId], vocab_size: int, fallback: TokenId
) -> tuple[tuple[TokenId, ...], tuple[bool, ...]]:
    """Replace out-of-vocabulary draft ids with ``fallback`` and record their validity.

    Out-of-vocabulary tokens cannot be passed to ``embed_tokens`` without failing the request.
    Replacing them with a safe fallback id keeps the forward pass valid while ``valid[j] = False``
    guarantees verification rejects at or before slot ``j``, preserving exact output equivalence.
    """
    if not drafts:
        return (), ()
    clean: list[TokenId] = []
    valid: list[bool] = []
    for token in drafts:
        ok = type(token) is int and 0 <= token < vocab_size
        clean.append(token if ok else fallback)
        valid.append(ok)
    return tuple(clean), tuple(valid)


def accept_one_point(
    drawn: Sequence[TokenId],
    drafts: Sequence[TokenId],
    *,
    valid: Sequence[bool] | None = None,
    vocab_size: int | None = None,
) -> AcceptanceResult:
    """Apply the one-point acceptance rule to a single request's drawn window.

    Args:
        drawn: Tokens sampled by the target for rows ``0 .. len(drafts)``.
        drafts: Draft tokens placed at rows ``1 .. len(drafts)``.
        valid: Optional per-draft validity flags (e.g. from :func:`sanitize_drafts`).
        vocab_size: Optional vocabulary bound; drafts outside ``[0, vocab_size)`` are rejected.
    """
    if not drawn:
        return AcceptanceResult(tokens=(), accepted=0, rejected=len(drafts))
    limit = min(len(drafts), len(drawn) - 1)
    count = 0
    while count < limit:
        if valid is not None and count < len(valid) and not valid[count]:
            break
        draft = drafts[count]
        if vocab_size is not None and not (0 <= draft < vocab_size):
            break
        if draft != drawn[count]:
            break
        count += 1
    return AcceptanceResult(
        tokens=tuple(drawn[: count + 1]),
        accepted=count,
        rejected=len(drafts) - count,
    )


def accept_one_point_batch(
    sampled: Sequence[TokenId],
    logit_counts: Sequence[int],
    draft_tokens: Sequence[Sequence[TokenId]],
    draft_valid: Sequence[Sequence[bool]] | None = None,
    *,
    vocab_size: int | None = None,
) -> tuple[AcceptanceResult, ...]:
    """Apply :func:`accept_one_point` across every request of a step."""
    results: list[AcceptanceResult] = []
    cursor = 0
    for index, taken in enumerate(logit_counts):
        drawn = sampled[cursor : cursor + taken]
        cursor += taken
        valid = draft_valid[index] if draft_valid is not None else None
        results.append(
            accept_one_point(drawn, draft_tokens[index], valid=valid, vocab_size=vocab_size)
        )
    return tuple(results)
