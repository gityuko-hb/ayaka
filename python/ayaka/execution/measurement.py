"""The costliest draw the sampler can make, run once at start-up so the memory plan counts it.

A verify step wider than every captured bucket samples all its rows eagerly, outside any graph
pool, so the scratch of that draw must be measured and kept free (``sampling_peak_bytes`` of the
start-up profile). This module imports torch.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from ayaka.core.params import LogprobMode, SamplingParams
from ayaka.core.types import RequestId
from ayaka.sampling.batch import SamplingBatch, SamplingRow
from ayaka.sampling.logits import LogitRow, build_controls

if TYPE_CHECKING:
    from collections.abc import Collection

    from ayaka.core.types import TokenId
    from ayaka.sampling.sampler import Sampler

__all__ = ["draw_for_measurement"]


def draw_for_measurement(
    sampler: Sampler,
    logits: torch.Tensor,
    *,
    device: torch.device,
    eos_token_ids: Collection[TokenId],
) -> None:
    """Sample every row of ``logits`` the costliest way ``sampler`` can, and drop the tokens.

    Half of the rows ask for raw logprobs and half for sampling logprobs, so the sampler holds
    its float32 copy of the logits and both logprob tensors at once.
    """
    rows = int(logits.shape[0])
    modes = (LogprobMode.RAW, LogprobMode.SAMPLING)
    params = [
        SamplingParams(temperature=1.0, seed=0, logprobs=1, logprob_mode=modes[row % 2])
        for row in range(rows)
    ]
    batch = SamplingBatch.build(
        [SamplingRow(params=row_params, seed=0, position=1) for row_params in params],
        device=device,
    )
    controls = build_controls(
        [LogitRow(RequestId(f"measure-{row}"), params[row], 0) for row in range(rows)],
        vocab_size=int(logits.shape[-1]),
        eos_token_ids=eos_token_ids,
        penalties=None,
        device=device,
    )
    out = torch.empty(rows, dtype=torch.int64, device=device)
    sampler.sample(logits, batch, out, controls)
