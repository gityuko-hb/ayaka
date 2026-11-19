"""Token counts for the repetition, frequency and presence penalties.

A penalty needs to know which tokens a request has seen and how often, so it is the one part of
sampling that keeps state per request. :class:`PenaltyStates` holds, for each request that uses a
penalty and only for those, two dense ``[vocab]`` int32 rows on the device:

* ``prompt_and_output``: every prompt token and every committed output token, for repetition;
* ``output``: the committed output tokens only, for frequency and presence.

The rows start from the prompt at admission and grow with each **committed** token, the tokens
the scheduler emitted after its stop and length cuts, never with a token the sampler drew and
the scheduler dropped. They are dropped when the request ends. A request that is preempted and
recomputed must rebuild them from its prompt and committed output; there is no preemption yet.

A row is about ``4 * vocab`` bytes (roughly 600 KB at a 150k vocabulary) per count; the sparse
alternative is an open decision that waits for a measurement.

This module imports torch. Import it through ``ayaka.sampling``, which loads it on first use.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from ayaka.core.params import SamplingParams
    from ayaka.core.types import RequestId, TokenId

__all__ = ["PenaltyState", "PenaltyStates", "uses_penalties"]


def uses_penalties(params: SamplingParams) -> bool:
    """Whether a request's parameters ask for any of the three penalties."""
    return (
        params.repetition_penalty != 1.0
        or params.frequency_penalty != 0.0
        or params.presence_penalty != 0.0
    )


@dataclass(frozen=True, slots=True)
class PenaltyState:
    """The two count rows of one request."""

    prompt_and_output: torch.Tensor
    output: torch.Tensor


class PenaltyStates:
    """The count rows of every request that uses a penalty.

    Args:
        vocab_size: Logits per row.
        device: Where the rows live: the model's device.
    """

    def __init__(self, vocab_size: int, device: torch.device | str) -> None:
        self._vocab = vocab_size
        self._device = torch.device(device)
        self._states: dict[RequestId, PenaltyState] = {}

    def __len__(self) -> int:
        return len(self._states)

    def __contains__(self, request_id: object) -> bool:
        return request_id in self._states

    def admit(
        self, request_id: RequestId, prompt: Sequence[TokenId], params: SamplingParams
    ) -> None:
        """Start counting for a request, if it uses a penalty; otherwise do nothing."""
        if not uses_penalties(params):
            return
        tokens = torch.tensor(list(prompt), dtype=torch.int64, device=self._device)
        prompt_and_output = torch.bincount(tokens, minlength=self._vocab).to(torch.int32)
        output = torch.zeros(self._vocab, dtype=torch.int32, device=self._device)
        self._states[request_id] = PenaltyState(prompt_and_output, output)

    def commit(self, request_id: RequestId, token_ids: Iterable[TokenId]) -> None:
        """Count tokens the request committed; ignored for a request without a state."""
        state = self._states.get(request_id)
        if state is None:
            return
        tokens = torch.tensor(list(token_ids), dtype=torch.int64, device=self._device)
        if tokens.numel() == 0:
            return
        ones = torch.ones_like(tokens, dtype=torch.int32)
        state.prompt_and_output.index_add_(0, tokens, ones)
        state.output.index_add_(0, tokens, ones)

    def drop(self, request_id: RequestId) -> None:
        """Forget a request that ended."""
        self._states.pop(request_id, None)

    def rows(self, request_ids: Sequence[RequestId | None]) -> tuple[torch.Tensor, torch.Tensor]:
        """``(prompt_and_output, output)`` as ``[len(request_ids), vocab]`` int32 tensors.

        ``None`` or a request without a state gives a row of zeros, which no penalty changes.
        """
        zeros = torch.zeros(self._vocab, dtype=torch.int32, device=self._device)
        states = [self._states.get(rid) if rid is not None else None for rid in request_ids]
        seen = torch.stack([s.prompt_and_output if s is not None else zeros for s in states])
        output = torch.stack([s.output if s is not None else zeros for s in states])
        return seen, output
