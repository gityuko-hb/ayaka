"""Logprobs of the drawn tokens, from the device to :class:`~ayaka.core.outputs.TokenLogprob`.

The sampler leaves, per row that asked, a ``[vocab]`` row of log-probabilities: the raw
distribution (after masks and bias, before penalties, temperature and filters) or the one the
token was drawn from. :func:`token_logprobs` reads the chosen token's value and rank and either
the most likely tokens or the explicit ids the request named, on the device, and brings the
result to the host in one copy after the token copy that already waits for the device.

Ties are broken by the smaller token id, in the rank and in the top list. A token outside the
distribution (masked, or dropped by a filter) is reported as ``-9999.0``, as the OpenAI API does,
not as ``-inf``, which JSON cannot carry. The decoded text of each token is filled later by the
output processor, which owns the tokenizer.

This module imports torch. Import it through ``ayaka.sampling``, which loads it on first use.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Final, cast

import torch

from ayaka.core.outputs import TokenLogprob
from ayaka.core.params import LogprobMode

if TYPE_CHECKING:
    from collections.abc import Sequence

    from ayaka.sampling.logits import LogprobRequest

__all__ = ["UNLIKELY_LOGPROB", "token_logprobs"]

#: What a token with no probability reports instead of ``-inf``.
UNLIKELY_LOGPROB: Final = -9999.0


def _finite(value: float) -> float:
    return value if math.isfinite(value) else UNLIKELY_LOGPROB


def _floats(tensor: torch.Tensor) -> list[float]:
    return cast("list[float]", tensor.tolist())  # pyright: ignore[reportUnknownMemberType]


def _ints(tensor: torch.Tensor) -> list[int]:
    return cast("list[int]", tensor.tolist())  # pyright: ignore[reportUnknownMemberType]


def _float_rows(tensor: torch.Tensor) -> list[list[float]]:
    return cast("list[list[float]]", tensor.tolist())  # pyright: ignore[reportUnknownMemberType]


def _int_rows(tensor: torch.Tensor) -> list[list[int]]:
    return cast("list[list[int]]", tensor.tolist())  # pyright: ignore[reportUnknownMemberType]


def token_logprobs(
    requests: Sequence[LogprobRequest | None],
    chosen: torch.Tensor,
    *,
    raw: torch.Tensor | None,
    drawn: torch.Tensor | None,
) -> list[TokenLogprob | None]:
    """One :class:`TokenLogprob` per row that asked for logprobs, ``None`` for the others.

    Args:
        requests: Per row, what it reports.
        chosen: ``[rows]`` int64, the drawn tokens.
        raw: ``[rows, vocab]`` raw log-probabilities, if a row asked for them.
        drawn: ``[rows, vocab]`` log-probabilities of the drawn distribution, if a row asked.
    """
    picked = [(index, request) for index, request in enumerate(requests) if request is not None]
    found: list[TokenLogprob | None] = [None] * len(requests)
    if not picked:
        return found
    sources: list[torch.Tensor] = []
    for index, request in picked:
        source = raw if request.mode is LogprobMode.RAW else drawn
        if source is None:
            raise ValueError(f"row {index} asks for {request.mode} logprobs that were not computed")
        sources.append(source[index])
    lp = torch.stack(sources)
    rows = torch.tensor([index for index, _ in picked], dtype=torch.int64, device=lp.device)
    tokens = chosen.index_select(0, rows)
    value = lp.gather(1, tokens[:, None])
    vocab = torch.arange(lp.shape[1], device=lp.device)
    rank = (lp > value).sum(dim=-1) + ((lp == value) & (vocab < tokens[:, None])).sum(dim=-1) + 1
    widest = max((r.top for _, r in picked if r.token_ids is None), default=0)
    top_values: list[list[float]] = []
    top_ids: list[list[int]] = []
    if widest > 0:
        ordered, order = torch.sort(lp, dim=-1, descending=True, stable=True)
        top_values = _float_rows(ordered[:, :widest])
        top_ids = _int_rows(order[:, :widest])
    chosen_ids = _ints(tokens)
    values = _floats(value[:, 0])
    ranks = _ints(rank)
    for position, (index, request) in enumerate(picked):
        if request.token_ids is not None:
            wanted = torch.tensor(request.token_ids, dtype=torch.int64, device=lp.device)
            listed = _floats(lp[position].index_select(0, wanted))
            top = tuple(zip(request.token_ids, map(_finite, listed), strict=True))
        else:
            count = request.top
            ids = top_ids[position][:count] if count else []
            logps = top_values[position][:count] if count else []
            top = tuple(zip(ids, map(_finite, logps), strict=True))
        found[index] = TokenLogprob(
            token_id=chosen_ids[position],
            logprob=_finite(values[position]),
            rank=ranks[position],
            top=top,
        )
    return found
    sources: list[torch.Tensor] = []
    for index in picked:
        request = requests[index]
        assert request is not None  # noqa: S101  # picked holds only rows with a request
        source = raw if request.mode is LogprobMode.RAW else drawn
        if source is None:
            raise ValueError(f"row {index} asks for {request.mode} logprobs that were not computed")
        sources.append(source[index])
    lp = torch.stack(sources)
    rows = torch.tensor(picked, dtype=torch.int64, device=lp.device)
    tokens = chosen.index_select(0, rows)
    value = lp.gather(1, tokens[:, None])
    vocab = torch.arange(lp.shape[1], device=lp.device)
    rank = (lp > value).sum(dim=-1) + ((lp == value) & (vocab < tokens[:, None])).sum(dim=-1) + 1
    widest = max(
        (requests[i].top for i in picked if requests[i].token_ids is None),  # type: ignore[union-attr]
        default=0,
    )
    top_values: list[list[float]] = []
    top_ids: list[list[int]] = []
    if widest > 0:
        ordered, order = torch.sort(lp, dim=-1, descending=True, stable=True)
        top_values = _float_rows(ordered[:, :widest])
        top_ids = _int_rows(order[:, :widest])
    explicit: list[list[float] | None] = []
    for position, index in enumerate(picked):
        ids = requests[index].token_ids  # type: ignore[union-attr]  # picked rows have one
        if ids is None:
            explicit.append(None)
        else:
            wanted = torch.tensor(ids, dtype=torch.int64, device=lp.device)
            explicit.append(_floats(lp[position].index_select(0, wanted)))
    chosen_ids = _ints(tokens)
    values = _floats(value[:, 0])
    ranks = _ints(rank)
    for position, index in enumerate(picked):
        request = requests[index]
        assert request is not None  # noqa: S101
        if request.token_ids is not None:
            listed = explicit[position] or []
            top = tuple(zip(request.token_ids, map(_finite, listed), strict=True))
        else:
            count = request.top
            top = tuple(
                zip(
                    top_ids[position][:count] if count else [],
                    map(_finite, top_values[position][:count] if count else []),
                    strict=True,
                )
            )
        found[index] = TokenLogprob(
            token_id=int(chosen_ids[position]),
            logprob=_finite(float(values[position])),
            rank=int(ranks[position]),
            top=top,
        )
    return found
