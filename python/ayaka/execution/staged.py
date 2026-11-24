"""What :class:`~ayaka.execution.inputs.InputBuilder` hands out: the inputs of one step.

:class:`ModelInputs` are the inputs of an eager forward; :class:`StagedDecode` and
:class:`StagedVerify` are a decode step and a verify window written into the fixed buffers of a
captured graph. Each is a set of views of the builder's buffers, valid until its next step.

This module imports torch. Import it through ``ayaka.execution``, which loads it on first use.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch

    from ayaka.core.types import TokenId
    from ayaka.ops.attention import AttentionMetadata
    from ayaka.ops.linear_attention import LinearAttentionMetadata
    from ayaka.sampling.batch import SamplingBatch

__all__ = ["ModelInputs", "StagedDecode", "StagedVerify"]


@dataclass(frozen=True, slots=True, eq=False)
class StagedDecode:
    """A decode step written into the fixed buffers at a bucket's size, for a captured graph.

    Attributes:
        bucket: Rows the graph runs: the step's requests, then padding requests.
        count: The step's own requests; their rows come first.
        kv_lengths: ``[bucket]`` int32 CPU tensor, 0 for padding requests.
        block_rows: ``[bucket, max_blocks]`` int32 CPU tensor of group 0's block tables.
        all_greedy: Whether every request samples greedily, which picks the graph variant.
        group_block_rows: Per-group ``[bucket, max_blocks]`` int32 CPU tensors for all
            KV/latent groups.
        state_slots: Optional ``[bucket]`` int32 host/staged state slot ids.
    """

    bucket: int
    count: int
    kv_lengths: torch.Tensor
    block_rows: torch.Tensor
    all_greedy: bool
    group_block_rows: tuple[torch.Tensor, ...] = ()
    state_slots: torch.Tensor | None = None


@dataclass(frozen=True, slots=True, eq=False)
class StagedVerify:
    """A ``VERIFY`` step staged at a captured total-row bucket size."""

    bucket: int
    num_rows: int
    num_requests: int
    rows_per_request: torch.Tensor
    kv_lengths: torch.Tensor
    block_rows: torch.Tensor
    all_greedy: bool
    draft_tokens: tuple[tuple[TokenId, ...], ...]
    logit_counts: tuple[int, ...]


@dataclass(frozen=True, slots=True, eq=False)
class ModelInputs:
    """The inputs of one forward, as views of the builder's fixed buffers.

    Attributes:
        token_ids: ``[N]`` int64 input token of every row, requests in step order.
        attention: The planned step for group 0 (backward-compatible default).
        logit_rows: ``[L]`` int64 indices of the rows whose hidden states are projected and
            sampled: the last row of a prefill chunk that ends its prompt, every row of a decode
            or verify request.
        sampling: The sampling parameters of those ``L`` rows.
        sampled: ``[L]`` int64 buffer the sampler writes the chosen token ids into.
        logit_counts: Per request, how many of the ``L`` rows are its own (0 for a prefill chunk
            that does not end the prompt).
        draft_tokens: Per request, the tokens the step used for its draft rows.
        group_attentions: Planned attention metadata per KV/latent KV group.
        linear_attention: Planned linear-attention step metadata, or ``None`` when unused.
    """

    token_ids: torch.Tensor
    attention: AttentionMetadata
    logit_rows: torch.Tensor
    sampling: SamplingBatch
    sampled: torch.Tensor
    logit_counts: tuple[int, ...]
    draft_tokens: tuple[tuple[TokenId, ...], ...]
    group_attentions: tuple[AttentionMetadata, ...] = field(default=())
    linear_attention: LinearAttentionMetadata | None = None

    def __post_init__(self) -> None:
        if not self.group_attentions:
            object.__setattr__(self, "group_attentions", (self.attention,))
