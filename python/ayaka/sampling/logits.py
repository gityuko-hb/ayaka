"""What a step's rows ask the sampler to do to their logits before it draws.

Most rows ask for nothing: their logits go straight to the draw, and :func:`build_controls`
returns ``None`` so the sampler keeps its short path (and a decode step can replay a CUDA
graph). A row that sets ``allowed_token_ids``, ``logit_bias``, ``min_tokens``, a penalty or
logprobs gets the transforms of :mod:`ayaka.ops.sampling` in their fixed order:

    1. one bitmask per row: the structural mask of an active constraint, allowed ids, and the
       stop ids (EOS unless ``ignore_eos``) while fewer than ``min_tokens`` tokens exist; the
       masks are intersected
    2. logit bias
    3. raw logprobs, when asked for (before penalties, temperature and filters)
    4. penalties, from :class:`~ayaka.sampling.penalties.PenaltyStates`
    5-8. temperature, filters, draw; logprobs of the sampling distribution, when asked for

Every transform is per row, so a row's token does not depend on the other rows of the step.

This module imports torch. Import it through ``ayaka.sampling``, which loads it on first use.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import numpy as np
import torch

from ayaka.core.params import LogprobMode
from ayaka.sampling.penalties import uses_penalties

if TYPE_CHECKING:
    from collections.abc import Collection, Sequence

    from numpy.typing import NDArray

    from ayaka.core.params import SamplingParams
    from ayaka.core.types import RequestId, TokenId
    from ayaka.sampling.penalties import PenaltyStates
    from ayaka.sampling.structured.runtime import ConstraintRuntime

__all__ = ["LogitControls", "LogitRow", "LogprobRequest", "build_controls", "needs_controls"]

_ALL_BITS = 0xFFFFFFFF


@dataclass(frozen=True, slots=True)
class LogitRow:
    """One row of logits and the request it draws for.

    Attributes:
        request_id: The request, for its penalty counts.
        params: Its sampling parameters.
        produced: Tokens the request emitted before this row's draw; ``min_tokens`` compares
            against it.
    """

    request_id: RequestId
    params: SamplingParams
    produced: int


@dataclass(frozen=True, slots=True)
class LogprobRequest:
    """What one row reports about its distribution.

    Attributes:
        top: How many of the most likely tokens to list (0: only the chosen one).
        token_ids: Explicit ids to list instead of the most likely ones.
        mode: ``RAW`` (before penalties and filters) or ``SAMPLING`` (the drawn distribution).
    """

    top: int
    token_ids: tuple[TokenId, ...] | None
    mode: LogprobMode


@dataclass(frozen=True, slots=True)
class LogitControls:
    """The tensors of a step's transforms; a field is ``None`` when no row needs it.

    Attributes:
        bitmask: ``[rows, ceil(vocab / 32)]`` int32 allowed-token bits.
        bias: ``(rows, tokens, biases)`` of the sparse logit bias.
        penalties: ``(prompt_and_output, output, repetition, frequency, presence)``.
        logprobs: Per row, what it reports, or ``None``.
    """

    bitmask: torch.Tensor | None
    bias: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None
    penalties: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None
    logprobs: tuple[LogprobRequest | None, ...]

    @property
    def any_raw_logprobs(self) -> bool:
        """Whether a row reports the raw distribution."""
        return any(r is not None and r.mode is LogprobMode.RAW for r in self.logprobs)

    @property
    def any_sampling_logprobs(self) -> bool:
        """Whether a row reports the distribution it was drawn from."""
        return any(r is not None and r.mode is LogprobMode.SAMPLING for r in self.logprobs)


def needs_controls(
    params: SamplingParams, produced: int | None = None, *, has_eos: bool = True
) -> bool:
    """Whether a request needs anything besides temperature, the filters and the draw.

    Args:
        params: The request's parameters.
        produced: Tokens it emitted before the draw; ``None`` asks whether it may ever need
            them. ``min_tokens`` needs a mask only until enough tokens exist, and only when
            there is a stop id or an EOS (not ignored) to hold back.
        has_eos: Whether the model has an end-of-sequence id.
    """
    if (
        params.allowed_token_ids is not None
        or bool(params.logit_bias)
        or uses_penalties(params)
        or params.logprobs is not None
        or params.logprob_token_ids is not None
    ):
        return True
    holds_back = bool(params.stop_token_ids) or (has_eos and not params.ignore_eos)
    if params.min_tokens == 0 or not holds_back:
        return False
    return produced is None or produced < params.min_tokens


def build_controls(
    rows: Sequence[LogitRow],
    *,
    vocab_size: int,
    eos_token_ids: Collection[TokenId],
    penalties: PenaltyStates | None,
    device: torch.device | str,
    constraints: ConstraintRuntime | None = None,
) -> LogitControls | None:
    """The transforms of a step, or ``None`` when no row asks for one.

    Args:
        rows: The step's logit rows, in order.
        vocab_size: Logits per row.
        eos_token_ids: The model's end-of-sequence ids, banned with the stop ids before
            ``min_tokens`` unless a request sets ``ignore_eos``.
        penalties: The requests' token counts.
        device: Where the logits are.
        constraints: The running output constraints; a row whose constraint is active gets its
            structural mask. Only the first row of a request's window has one (no speculation
            runs with a constraint).
    """
    has_eos = bool(eos_token_ids)
    structural: list[NDArray[np.uint32] | None] = [None] * len(rows)
    if constraints is not None and len(constraints):
        seen: set[RequestId] = set()
        for index, row in enumerate(rows):
            if row.request_id not in seen:
                seen.add(row.request_id)
                structural[index] = constraints.mask(row.request_id)
    if all(mask is None for mask in structural) and not any(
        needs_controls(row.params, row.produced, has_eos=has_eos) for row in rows
    ):
        return None
    return LogitControls(
        bitmask=_bitmask(rows, vocab_size, eos_token_ids, device, structural),
        bias=_bias(rows, device),
        penalties=_penalties(rows, penalties, device),
        logprobs=tuple(_logprob_request(row.params) for row in rows),
    )


def _bitmask(
    rows: Sequence[LogitRow],
    vocab: int,
    eos: Collection[TokenId],
    device: torch.device | str,
    structural: Sequence[NDArray[np.uint32] | None],
) -> torch.Tensor | None:
    words = (vocab + 31) // 32
    masks: list[np.ndarray | None] = []
    for row, grammar in zip(rows, structural, strict=True):
        params = row.params
        banned: set[int] = set()
        if row.produced < params.min_tokens:
            banned |= params.stop_token_ids
            if not params.ignore_eos:
                banned |= set(eos)
        if params.allowed_token_ids is None and not banned:
            masks.append(grammar[:words].astype(np.uint64) if grammar is not None else None)
            continue
        if params.allowed_token_ids is None:
            bits = np.full(words, _ALL_BITS, dtype=np.uint64)
        else:
            bits = np.zeros(words, dtype=np.uint64)
            allowed = np.fromiter(params.allowed_token_ids, dtype=np.int64)
            shifts = (allowed % 32).astype(np.uint64)
            np.bitwise_or.at(bits, allowed // 32, np.left_shift(np.uint64(1), shifts))
        for token in banned:
            if token < vocab:
                bits[token // 32] &= ~np.uint64(1 << (token % 32)) & np.uint64(_ALL_BITS)
        if grammar is not None:
            bits &= grammar[:words].astype(np.uint64)
        masks.append(bits)
    if all(mask is None for mask in masks):
        return None
    full = np.full(words, _ALL_BITS, dtype=np.uint64)
    table = np.stack([mask if mask is not None else full for mask in masks])
    signed = table.astype(np.uint32).view(np.int32)
    return torch.tensor(signed.tolist(), dtype=torch.int32, device=device)


def _bias(
    rows: Sequence[LogitRow], device: torch.device | str
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
    row_ids: list[int] = []
    tokens: list[int] = []
    biases: list[float] = []
    for index, row in enumerate(rows):
        # SamplingParams stores the bias as sorted (token, bias) pairs.
        pairs = cast("tuple[tuple[int, float], ...]", row.params.logit_bias or ())
        for token, value in pairs:
            row_ids.append(index)
            tokens.append(token)
            biases.append(value)
    if not row_ids:
        return None
    return (
        torch.tensor(row_ids, dtype=torch.int64, device=device),
        torch.tensor(tokens, dtype=torch.int64, device=device),
        torch.tensor(biases, dtype=torch.float32, device=device),
    )


def _penalties(
    rows: Sequence[LogitRow], states: PenaltyStates | None, device: torch.device | str
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None:
    if states is None or not any(uses_penalties(row.params) for row in rows):
        return None
    seen, output = states.rows(
        [row.request_id if uses_penalties(row.params) else None for row in rows]
    )

    def column(values: list[float]) -> torch.Tensor:
        return torch.tensor(values, dtype=torch.float32, device=device)

    return (
        seen,
        output,
        column([row.params.repetition_penalty for row in rows]),
        column([row.params.frequency_penalty for row in rows]),
        column([row.params.presence_penalty for row in rows]),
    )


def _logprob_request(params: SamplingParams) -> LogprobRequest | None:
    if params.logprobs is None and params.logprob_token_ids is None:
        return None
    ids = params.logprob_token_ids
    top = len(ids) if ids is not None else max(params.logprobs or 0, 0)
    return LogprobRequest(top=top, token_ids=ids, mode=params.logprob_mode)
