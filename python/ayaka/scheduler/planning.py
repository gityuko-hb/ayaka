"""Chunk, row and capacity planning helpers for :mod:`ayaka.scheduler.scheduler`."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ayaka.core.errors import InvalidRequestError
from ayaka.utils.validation import require_int

if TYPE_CHECKING:
    from ayaka.core.request import TokenizedRequest
    from ayaka.kvcache.manager import KVCacheManager
    from ayaka.kvcache.prefix.base import PrefixMatch
    from ayaka.scheduler.request import Request

__all__ = [
    "check_context",
    "check_kv_capacity",
    "grant_chunk",
    "grant_rows",
    "next_chunk",
]


def check_context(request: TokenizedRequest, limit: int | None) -> None:
    """Admission check: the prompt and ``max_tokens`` fit the model context."""
    if limit is None:
        return
    total = request.max_total_length
    if total is not None and total > limit:
        raise InvalidRequestError(
            f"prompt ({request.prompt_length}) plus max_tokens "
            f"({request.sampling.max_tokens}) must be <= the model context of {limit} tokens"
        )
    if request.prompt_length >= limit:
        raise InvalidRequestError(
            f"prompt ({request.prompt_length}) must be shorter than the model context "
            f"of {limit} tokens"
        )


def check_kv_capacity(request: TokenizedRequest, kv: KVCacheManager) -> None:
    """Admission check: the prompt fits the KV one request can hold on this machine.

    Read from fixed sizes only: :meth:`KVCacheManager.stats` walks the free queue to count cached
    blocks, which costs milliseconds on a large pool, on the engine thread, for every request.
    """
    capacity = kv.max_context_tokens
    if request.prompt_length > capacity:
        pool_tokens = min(kv.stats().usable_blocks) * kv.block_size
        raise InvalidRequestError(
            f"prompt ({request.prompt_length} tokens) exceeds the {capacity} tokens of KV "
            f"cache this engine can give one request (block table row: "
            f"{kv.max_tokens_per_request}, pool: {pool_tokens}); shorten the prompt or "
            "give the cache more memory"
        )


def next_chunk(prompt_length: int, computed: int, budget: int, block_size: int) -> int:
    """The next chunk of a prompt from ``computed``: ends on a block boundary unless the last."""
    remaining = prompt_length - computed
    if remaining <= budget:
        return remaining
    size = budget - ((computed + budget) % block_size)
    return size if size > 0 else budget


def grant_chunk(
    kv: KVCacheManager, request: Request, size: int, prefix: PrefixMatch | None = None
) -> int:
    """Allocate room for a chunk, or for as much of it as the free blocks allow.

    Args:
        kv: The cache manager.
        request: The request.
        size: Tokens of the chunk.
        prefix: For a request leaving the queue, the prefix match to attach first.

    Returns:
        The tokens the request now has room for, 0 if none.
    """
    request_id = request.request_id
    if kv.allocate(request_id, size, prefix):
        return size
    held = (
        kv.table.num_blocks(request_id, 0) * kv.block_size - request.computed
        if kv.has(request_id)
        else 0
    )
    shrunk = min(size, held + kv.free_blocks * kv.block_size)
    if 1 <= shrunk < size and kv.allocate(request_id, shrunk, prefix):
        return shrunk
    return 0


def grant_rows(kv: KVCacheManager, request: Request, drafts: int, left: int) -> int:
    """Rows to reserve for a decode step: ``1 + k`` if they fit, else 1, else 0.

    Args:
        kv: The cache manager.
        request: A request that finished its prefill.
        drafts: The look-ahead ``k`` the runner asked for.
        left: Tokens the request may still emit; ``k`` is capped so no row is wasted.
    """
    k = require_int(drafts, "lookahead", minimum=0)
    room = kv.max_tokens_per_request - request.computed
    if room < 1:
        return 0
    k = min(k, room - 1, left - 1)
    if kv.allocate(request.request_id, 1 + k):
        return 1 + k
    if k > 0 and kv.allocate(request.request_id, 1):
        return 1
    return 0
