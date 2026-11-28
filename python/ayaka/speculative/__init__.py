"""Lossless speculative decoding: acceptance, proposers, lookup and controllers.

Every public name resolves lazily from its defining submodule so importing ``ayaka.speculative``
loads no torch or heavy dependencies.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ayaka.utils.lazy_exports import build_lazy_exports

if TYPE_CHECKING:
    from ayaka.speculative.acceptance import (
        AcceptanceResult,
        accept_one_point,
        accept_one_point_batch,
        sanitize_drafts,
    )
    from ayaka.speculative.ngram import LookupProposer, NgramIndex, naive_ngram_match
    from ayaka.speculative.policy import (
        CostAwareController,
        DraftController,
        FixedController,
        match_bucket,
    )
    from ayaka.speculative.proposer import EMPTY_PROPOSAL, Proposal, Proposer
    from ayaka.speculative.token_recycling import TokenRecyclingProposer

__all__ = [
    "EMPTY_PROPOSAL",
    "AcceptanceResult",
    "CostAwareController",
    "DraftController",
    "FixedController",
    "LookupProposer",
    "NgramIndex",
    "Proposal",
    "Proposer",
    "TokenRecyclingProposer",
    "accept_one_point",
    "accept_one_point_batch",
    "match_bucket",
    "naive_ngram_match",
    "sanitize_drafts",
]

_EXPORTS = {
    "acceptance": (
        "AcceptanceResult",
        "accept_one_point",
        "accept_one_point_batch",
        "sanitize_drafts",
    ),
    "ngram": ("LookupProposer", "NgramIndex", "naive_ngram_match"),
    "policy": ("CostAwareController", "DraftController", "FixedController", "match_bucket"),
    "proposer": ("EMPTY_PROPOSAL", "Proposal", "Proposer"),
    "token_recycling": ("TokenRecyclingProposer",),
}

__getattr__, __dir__ = build_lazy_exports(__name__, _EXPORTS, globals())
