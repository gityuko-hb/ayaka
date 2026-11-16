"""The sampler: logits and a :class:`~ayaka.sampling.batch.SamplingBatch` in, tokens out.

The order of the steps is :mod:`ayaka.ops.sampling`'s (temperature, top-k, top-p, min-p,
then the inverse CDF of one random number per row). The random number is a pure function
of the row's seed and position, so a row gets the same token whatever else is in the step:
the batch size, the order of the rows and the other requests do not matter, and a
speculative window verifies each position with the number plain decoding would have used.

Penalties, logit bias, masks and logprobs come later and will be steps before and after
this one, in the order ``docs/strategy/07-sampling-output.md`` fixes.

This module imports torch. Import it through ``ayaka.sampling``, which loads it on first use.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from ayaka.ops.sampling import DrawStream

if TYPE_CHECKING:
    from ayaka.ops.sampling import SamplingOps
    from ayaka.sampling.batch import SamplingBatch

__all__ = ["Sampler"]


class Sampler:
    """Picks one token per row of a step.

    Args:
        ops: The sampling family of the backend chosen for this engine
            (``OpsBackend.sampling``).
    """

    def __init__(self, ops: SamplingOps) -> None:
        self._ops = ops

    def sample(self, logits: torch.Tensor, batch: SamplingBatch, out: torch.Tensor) -> None:
        """Write the token id of every row into ``out``.

        Args:
            logits: ``[rows, vocab]`` in a compute dtype, on the batch's device.
            batch: The step's parameters, with one entry per row of ``logits``.
            out: ``[rows]`` int64, allocated by the caller. Padding rows of the batch get a
                token too; the caller ignores it.

        Raises:
            ValueError: If the batch and the logits disagree on the number of rows.
            BackendError: If the backend cannot run the call, for example on logits it
                refuses (NaN, or a row with no finite value).
        """
        if logits.ndim != 2 or logits.shape[0] != batch.size:
            raise ValueError(
                f"logits must be [{batch.size}, vocab] to match the batch, got {list(logits.shape)}"
            )
        if batch.all_greedy:
            self._ops.argmax(logits, out)
            return
        u = torch.empty(batch.size, dtype=torch.float32, device=batch.device)
        self._ops.uniform(batch.seeds, batch.positions, batch.sequences, u, stream=DrawStream.PICK)
        self._ops.sample(logits, batch.temperature, batch.top_k, batch.top_p, batch.min_p, u, out)
