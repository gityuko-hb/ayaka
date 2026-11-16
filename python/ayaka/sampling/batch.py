"""The per-step snapshot of every row's sampling parameters, as tensors.

Sampling parameters belong to a request, but the sampler works on the rows of a
step. The runner describes each row with a :class:`SamplingRow` and
:meth:`SamplingBatch.build` turns the list into one tensor per parameter on the
device. The batch is a snapshot: nothing in it changes after it is built, and the
next step builds a new one.

A row is one token to draw: a decode step has one row per request, a window
that a speculative decoder verifies has ``1 + k``. ``core.SamplingParams`` has
already validated the parameters, so this module does not check them again; it
only converts them to the encoding the ops use (see the attributes of
:class:`SamplingBatch`).

This module imports torch. Import it through ``ayaka.sampling``, which loads it on first use.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from ayaka.sampling.rng import MAX_SEED, MAX_SEQUENCE

if TYPE_CHECKING:
    from collections.abc import Sequence

    from ayaka.core.params import SamplingParams

__all__ = ["SamplingBatch", "SamplingRow"]


@dataclass(frozen=True, slots=True, kw_only=True)
class SamplingRow:
    """What the sampler needs to know about one row of a step.

    Attributes:
        params: The request's sampling parameters.
        seed: The seed the request samples with, already resolved
            (:func:`ayaka.sampling.rng.resolve_seed`), in ``[0, MAX_SEED]``.
        position: Index in the sequence (prompt and output) of the token this row
            produces. This is one more than the position of the row's input token.
        sequence: Index of the output sequence for a request with ``n > 1``, else 0.
    """

    params: SamplingParams
    seed: int
    position: int
    sequence: int = 0

    def __post_init__(self) -> None:
        if not 0 <= self.seed <= MAX_SEED:
            raise ValueError(f"seed must be in [0, {MAX_SEED}], got {self.seed}")
        if not 0 <= self.position <= MAX_SEED:
            raise ValueError(f"position must be in [0, {MAX_SEED}], got {self.position}")
        if not 0 <= self.sequence <= MAX_SEQUENCE:
            raise ValueError(f"sequence must be in [0, {MAX_SEQUENCE}], got {self.sequence}")


@dataclass(frozen=True, slots=True, eq=False)
class SamplingBatch:
    """The sampling parameters of every row of one step. Immutable once built.

    Every tensor has one entry per row, on one device. A greedy row (temperature
    below ``1e-5``, or ``top_k == 1``) is stored with temperature 0 and the
    neutral value of every filter, so only its temperature matters.

    Attributes:
        seeds: int64, the resolved seed of the row's request.
        positions: int64, index of the token the row produces.
        sequences: int32, index of the output sequence.
        temperature: float32; 0 marks a greedy row.
        top_k: int32; 0 disables top-k.
        top_p: float32; 1 disables top-p.
        min_p: float32; 0 disables min-p.
        all_greedy: Whether every row is greedy, known on the host when the batch
            was built. The sampler then takes the argmax path: no random number,
            no softmax, no sort.
    """

    seeds: torch.Tensor
    positions: torch.Tensor
    sequences: torch.Tensor
    temperature: torch.Tensor
    top_k: torch.Tensor
    top_p: torch.Tensor
    min_p: torch.Tensor
    all_greedy: bool

    @property
    def size(self) -> int:
        """Number of rows, padding rows included."""
        return int(self.seeds.shape[0])

    @property
    def device(self) -> torch.device:
        """The device the tensors are on."""
        return self.seeds.device

    @classmethod
    def build(
        cls,
        rows: Sequence[SamplingRow],
        *,
        device: torch.device | str,
        size: int | None = None,
    ) -> SamplingBatch:
        """Build the batch of one step.

        Args:
            rows: One entry per row, in the order of the logits.
            device: Where the tensors go.
            size: Pad the batch to this many rows with greedy rows at position 0, so a step
                can run at a fixed shape. Their tokens are meaningless and must be dropped by the
                caller. ``None`` means no padding.

        Raises:
            ValueError: If ``size`` is smaller than the number of rows.
        """
        padding = 0
        if size is not None:
            if size < len(rows):
                raise ValueError(f"size {size} is smaller than the {len(rows)} rows given")
            padding = size - len(rows)

        seeds = [row.seed for row in rows] + [0] * padding
        positions = [row.position for row in rows] + [0] * padding
        sequences = [row.sequence for row in rows] + [0] * padding
        temperature = [0.0] * len(seeds)
        top_k = [0] * len(seeds)
        top_p = [1.0] * len(seeds)
        min_p = [0.0] * len(seeds)
        for index, row in enumerate(rows):
            params = row.params
            if params.is_greedy:
                continue
            temperature[index] = params.temperature
            top_k[index] = max(params.top_k, 0)  # TOP_K_DISABLED (-1) -> 0
            top_p[index] = params.top_p
            min_p[index] = params.min_p

        return cls(
            seeds=torch.tensor(seeds, dtype=torch.int64, device=device),
            positions=torch.tensor(positions, dtype=torch.int64, device=device),
            sequences=torch.tensor(sequences, dtype=torch.int32, device=device),
            temperature=torch.tensor(temperature, dtype=torch.float32, device=device),
            top_k=torch.tensor(top_k, dtype=torch.int32, device=device),
            top_p=torch.tensor(top_p, dtype=torch.float32, device=device),
            min_p=torch.tensor(min_p, dtype=torch.float32, device=device),
            all_greedy=not any(temperature),
        )
