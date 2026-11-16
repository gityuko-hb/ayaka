"""The one object the scheduler asks for KV blocks.

:class:`KVCacheManager` joins a :class:`~ayaka.kvcache.blocks.BlockPool` per group to one
:class:`~ayaka.kvcache.table.BlockTable`. It is bookkeeping: it decides which block ids a
request holds and never touches a tensor, so it runs and is tested without torch. The
engine builds the matching :class:`~ayaka.kvcache.storage.pool.KVPool` tensors from the
same groups and block counts.

There is no prefix reuse yet: every block belongs to one request and returns to the free
queue when the request is freed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from ayaka.core.errors import EngineError
from ayaka.kvcache.blocks import BlockPool
from ayaka.kvcache.table import BlockTable
from ayaka.utils.math_utils import div_ceil
from ayaka.utils.validation import require_int

if TYPE_CHECKING:
    from collections.abc import Sequence

    from ayaka.core.types import RequestId
    from ayaka.kvcache.storage.spec import KVGroup

__all__ = ["KVCacheManager", "KVCacheStats"]


@dataclass(frozen=True, slots=True)
class KVCacheStats:
    """A snapshot of the cache's occupancy.

    Attributes:
        num_requests: Requests that hold blocks.
        usable_blocks: Blocks a request can own, per group (the padding block excluded).
        free_blocks: Blocks not held by any request, per group.
        evictions: Cached blocks reused and so forgotten; always 0 until prefix reuse exists.
    """

    num_requests: int
    usable_blocks: tuple[int, ...]
    free_blocks: tuple[int, ...]
    evictions: int

    @property
    def used_blocks(self) -> tuple[int, ...]:
        """Blocks held by requests, per group."""
        return tuple(
            total - free for total, free in zip(self.usable_blocks, self.free_blocks, strict=True)
        )

    @property
    def utilization(self) -> float:
        """Share of the most loaded group's usable blocks that requests hold, in ``[0, 1]``."""
        return max(
            (
                used / total
                for used, total in zip(self.used_blocks, self.usable_blocks, strict=True)
            ),
            default=0.0,
        )


class KVCacheManager:
    """Hands blocks to requests, all groups or none.

    A request's cache grows in blocks. ``allocate`` makes sure the request has room for
    the tokens a step is about to write; ``commit`` records that tokens became real;
    ``free`` gives everything back. All groups share the block size, so every group
    holds the same number of blocks for a request, but each has its own pool and ids.

    Args:
        groups: The KV groups, one per distinct :class:`~ayaka.kvcache.storage.spec.KVSpec`.
            All must use the same ``block_size``.
        num_blocks: Blocks per group, the padding block included; one entry per group,
            normally ``BudgetPlan.kv_blocks``.
        max_requests: Requests that can hold blocks at once.
        max_blocks_per_request: Longest context a request can reach, in blocks.

    Raises:
        ValueError: If there is no group, the counts do not match the groups, the block
            sizes differ, or a number is out of range.
    """

    def __init__(
        self,
        groups: Sequence[KVGroup],
        num_blocks: Sequence[int],
        *,
        max_requests: int,
        max_blocks_per_request: int,
    ) -> None:
        if not groups:
            raise ValueError("at least one KV group is required")
        if len(num_blocks) != len(groups):
            raise ValueError(
                f"{len(groups)} groups need {len(groups)} block counts, got {len(num_blocks)}"
            )
        sizes = {group.spec.block_size for group in groups}
        if len(sizes) != 1:
            raise ValueError(f"every group must use the same block_size, got {sorted(sizes)}")
        self._groups = tuple(groups)
        self._block_size = sizes.pop()
        self._pools = tuple(BlockPool(count) for count in num_blocks)
        self._table = BlockTable(len(self._groups), max_requests, max_blocks_per_request)
        self._committed: dict[RequestId, int] = {}

    # -- reading ---------------------------------------------------------------------------

    @property
    def groups(self) -> tuple[KVGroup, ...]:
        """The KV groups, in the order every per-group value uses."""
        return self._groups

    @property
    def block_size(self) -> int:
        """Tokens per block, the same for every group."""
        return self._block_size

    @property
    def table(self) -> BlockTable:
        """The block table; read it to build a step's block rows. Only the manager writes it."""
        return self._table

    @property
    def max_tokens_per_request(self) -> int:
        """Longest context one request can hold: ``max_blocks_per_request * block_size``."""
        return self._table.max_blocks_per_request * self._block_size

    @property
    def free_blocks(self) -> int:
        """Blocks a request could still be given in the scarcest group."""
        return min(pool.free_count for pool in self._pools)

    def has(self, request_id: RequestId) -> bool:
        """Whether a request holds a block table row."""
        return request_id in self._committed

    def num_committed(self, request_id: RequestId) -> int:
        """Tokens of the request that are real, as recorded by :meth:`commit`.

        Raises:
            KeyError: If the request is unknown.
        """
        return self._committed[request_id]

    def blocks(self, request_id: RequestId) -> tuple[tuple[int, ...], ...]:
        """The block ids a request holds, per group, in token order.

        Raises:
            KeyError: If the request is unknown.
        """
        return tuple(self._table.blocks(request_id, group) for group in range(len(self._groups)))

    def blocks_needed(self, request_id: RequestId, num_tokens: int) -> int:
        """How many new blocks per group :meth:`allocate` would take; changes nothing.

        Args:
            request_id: A known request or a new one.
            num_tokens: Tokens to make room for beyond those already committed.
        """
        require_int(num_tokens, "num_tokens")
        if request_id in self._committed:
            wanted = div_ceil(self._committed[request_id] + num_tokens, self._block_size)
            return max(0, wanted - self._table.num_blocks(request_id, 0))
        return div_ceil(num_tokens, self._block_size)

    def stats(self) -> KVCacheStats:
        """The current occupancy."""
        return KVCacheStats(
            num_requests=len(self._committed),
            usable_blocks=tuple(pool.num_blocks - 1 for pool in self._pools),
            free_blocks=tuple(pool.free_count for pool in self._pools),
            evictions=sum(pool.evictions for pool in self._pools),
        )

    # -- changing --------------------------------------------------------------------------

    def allocate(self, request_id: RequestId, num_tokens: int) -> bool:
        """Make room for ``num_tokens`` more tokens of a request, in every group or in none.

        ``num_tokens`` counts every row the coming step will write, **look-ahead rows of
        a speculative step included**: ``1 + k`` for a decode step with ``k`` drafts. The
        room is measured from the committed length, so asking again before
        :meth:`commit` does not allocate twice. A new request gets a block table row.

        The call never evicts, never waits and never raises for a shortage. It returns
        ``False`` when any group has too few free blocks, when the request would outgrow a
        table row (:attr:`max_tokens_per_request`) or when no row is free; in every
        such case nothing has changed, and the scheduler decides what to do.

        Args:
            request_id: The request.
            num_tokens: Rows to make room for beyond the committed tokens; may be 0.

        Returns:
            ``True`` once the request has room; ``False`` if it could not be given.

        Raises:
            TypeError: If ``num_tokens`` is not an ``int``.
            ValueError: If ``num_tokens`` is negative.
        """
        need = self.blocks_needed(request_id, num_tokens)
        known = request_id in self._committed
        if not known and self._table.free_rows == 0:
            return False
        have = self._table.num_blocks(request_id, 0) if known else 0
        if have + need > self._table.max_blocks_per_request:
            return False
        if any(pool.free_count < need for pool in self._pools):
            return False
        # Every check is done: from here on the call cannot fail, so no group is left half done.
        if not known:
            self._table.acquire(request_id)
            self._committed[request_id] = 0
        if need:
            for group, pool in enumerate(self._pools):
                self._table.append(request_id, group, [pool.allocate() for _ in range(need)])
        return True

    def commit(self, request_id: RequestId, num_tokens: int) -> None:
        """Record that ``num_tokens`` more tokens of the request are real.

        Call it after a step with the number of tokens that stuck: a prefill chunk's length,
        1 for a decode step, the accepted drafts plus one for a speculative step. Rows past
        the committed length may hold rejected draft values; the next step overwrites them.

        Raises:
            KeyError: If the request is unknown.
            EngineError: If the tokens would exceed the room :meth:`allocate` made.
        """
        require_int(num_tokens, "num_tokens")
        committed = self._committed[request_id] + num_tokens
        room = self._table.num_blocks(request_id, 0) * self._block_size
        if committed > room:
            raise EngineError(
                f"request {request_id!r} commits {committed} tokens but holds room for {room}"
            )
        self._committed[request_id] = committed

    def free(self, request_id: RequestId) -> None:
        """Give back every block of a request, last block first, and its table row.

        A request the manager does not know is ignored, so an abort of a request that was
        never admitted needs no special case.
        """
        if request_id not in self._committed:
            return
        for group, pool in enumerate(self._pools):
            pool.free_chain(self._table.blocks(request_id, group))
        self._table.release(request_id)
        del self._committed[request_id]

    # -- checking --------------------------------------------------------------------------

    def check_integrity(self) -> None:
        """Verify the books against the table.

        Runs each pool's own check, then confirms that every group's reference counts equal
        how often each block appears in the table, that no block is held without a request
        or referenced twice (no sharing exists yet), that each request holds the same number
        of blocks in every group and enough room for its committed tokens, and that the
        table and the request set agree. It is for tests and debug builds, not the hot path.

        Raises:
            EngineError: Listing every problem found.
        """
        problems: list[str] = []
        for group, pool in enumerate(self._pools):
            try:
                pool.check_integrity()
            except EngineError as exc:
                problems.append(f"group {group}: {exc}")
            seen: dict[int, int] = {}
            for request_id in self._committed:
                for block_id in self._table.blocks(request_id, group):
                    seen[block_id] = seen.get(block_id, 0) + 1
            for block_id in range(1, pool.num_blocks):
                if pool.ref_count(block_id) != seen.get(block_id, 0):
                    problems.append(
                        f"group {group} block {block_id}: {pool.ref_count(block_id)} references, "
                        f"{seen.get(block_id, 0)} in the table"
                    )
                if seen.get(block_id, 0) > 1:
                    problems.append(f"group {group} block {block_id} is held by two requests")
        for request_id, committed in self._committed.items():
            counts = {self._table.num_blocks(request_id, g) for g in range(len(self._groups))}
            if len(counts) != 1:
                problems.append(f"request {request_id!r} holds uneven blocks across groups")
            elif committed > counts.pop() * self._block_size:
                problems.append(
                    f"request {request_id!r} committed more tokens than it has room for"
                )
        in_use = self._table.max_requests - self._table.free_rows
        if in_use != len(self._committed):
            problems.append(f"{in_use} table rows in use for {len(self._committed)} requests")
        if problems:
            raise EngineError("KV cache is inconsistent: " + "; ".join(problems))
