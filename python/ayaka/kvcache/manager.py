"""The one object the scheduler asks for KV blocks.

:class:`KVCacheManager` joins a :class:`~ayaka.kvcache.blocks.BlockPool` per group to one
:class:`~ayaka.kvcache.table.BlockTable`. It is bookkeeping: it decides which block ids a
request holds and never touches a tensor, so it runs and is tested without torch. The
engine builds the matching :class:`~ayaka.kvcache.storage.pool.KVPool` tensors from the
same groups and block counts.

With a :class:`~ayaka.kvcache.prefix.base.PrefixCache`, full blocks are shared. A new request
attaches the blocks of the longest prefix an earlier request computed (:meth:`match_prefix`,
then :meth:`allocate` with the match), and every block a request fills completely is registered
under its chain digest (:meth:`cache_blocks`) so later requests can attach it. A shared block
holds one reference per request. When the last one frees it, it returns to the free queue still
matchable, and it loses its digest only when the queue hands it out again.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from ayaka.core.errors import EngineError, OutOfMemoryError
from ayaka.kvcache.blocks import BlockPool
from ayaka.kvcache.prefix.base import PrefixMatch
from ayaka.kvcache.resize import plan_pool_resize
from ayaka.kvcache.table import BlockTable
from ayaka.utils.math_utils import div_ceil
from ayaka.utils.validation import require_int

if TYPE_CHECKING:
    from collections.abc import Sequence

    from ayaka.core.types import RequestId, TokenId
    from ayaka.kvcache.host_store import HostKVStore
    from ayaka.kvcache.prefix.base import PrefixCache
    from ayaka.kvcache.storage.pool import KVPool
    from ayaka.kvcache.storage.spec import KVGroup

__all__ = ["KVCacheManager", "KVCacheStats"]


@dataclass(frozen=True, slots=True)
class KVCacheStats:
    """A snapshot of the cache's occupancy.

    Attributes:
        num_requests: Requests that hold blocks.
        usable_blocks: Blocks a request can own, per group (the padding block excluded).
        free_blocks: Blocks not held by any request, per group; cached blocks included.
        evictions: Cached blocks handed out again and so forgotten.
        cached_blocks: Free blocks that can still be matched, per group.
        lookups: Requests admitted with a prefix lookup.
        hit_tokens: Prompt tokens those requests reused, in total.
    """

    num_requests: int
    usable_blocks: tuple[int, ...]
    free_blocks: tuple[int, ...]
    evictions: int
    cached_blocks: tuple[int, ...] = ()
    lookups: int = 0
    hit_tokens: int = 0

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


@dataclass(slots=True)
class _Chain:
    """The digests of a request's full blocks, and how many of its blocks are registered."""

    parent: bytes
    digests: list[bytes] = field(default_factory=list[bytes])
    registered: int = 0


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
        prefix: The prefix cache that names full blocks, or ``None`` for no reuse.

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
        prefix: PrefixCache | None = None,
        host_store: HostKVStore | None = None,
        pools: Sequence[KVPool] | None = None,
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
        if prefix is not None and prefix.block_size != self._block_size:
            raise ValueError(
                f"the prefix cache hashes blocks of {prefix.block_size} tokens, "
                f"the groups use {self._block_size}"
            )
        if pools is not None and len(pools) != len(groups):
            raise ValueError(
                f"{len(groups)} groups need {len(groups)} storage pools, got {len(pools)}"
            )
        self._pools = tuple(BlockPool(count) for count in num_blocks)
        self._storage_pools: tuple[KVPool, ...] | None = tuple(pools) if pools is not None else None
        self._host_store: HostKVStore | None = host_store
        self._table = BlockTable(len(self._groups), max_requests, max_blocks_per_request)
        self._committed: dict[RequestId, int] = {}
        self._prefix = prefix
        self._chains: dict[RequestId, _Chain] = {}
        self._lookups = 0
        self._hit_tokens = 0
        self._wire_eviction_hooks()

    def _wire_eviction_hooks(self) -> None:
        if self._host_store is None or self._storage_pools is None:
            for pool in self._pools:
                pool.set_eviction_hook(None)
            return
        host_store = self._host_store
        for group_idx, (pool, kv_pool) in enumerate(
            zip(self._pools, self._storage_pools, strict=True)
        ):

            def _hook(
                block_id: int,
                digest: bytes,
                g: int = group_idx,
                src: KVPool = kv_pool,
            ) -> None:
                host_store.offload(g, block_id, digest, src)

            pool.set_eviction_hook(_hook)

    def attach_storage(
        self,
        pools: Sequence[KVPool],
        host_store: HostKVStore | None = None,
    ) -> None:
        """Bind storage pools (and optional host-RAM tier) to this manager."""
        if len(pools) != len(self._groups):
            raise ValueError(
                f"{len(self._groups)} groups need {len(self._groups)} storage pools, "
                f"got {len(pools)}"
            )
        self._storage_pools = tuple(pools)
        if host_store is not None:
            self._host_store = host_store
        self._wire_eviction_hooks()

    @property
    def host_store(self) -> HostKVStore | None:
        """Attached host-RAM KV cache store, or ``None`` when disabled."""
        return self._host_store

    @property
    def storage_pools(self) -> tuple[KVPool, ...] | None:
        """Attached tensor storage pools, or ``None`` for bookkeeping-only instances."""
        return self._storage_pools

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
    def prefix_caching(self) -> bool:
        """Whether full blocks are shared between requests."""
        return self._prefix is not None

    @property
    def max_tokens_per_request(self) -> int:
        """Longest context one request can hold: ``max_blocks_per_request * block_size``."""
        return self._table.max_blocks_per_request * self._block_size

    @property
    def max_context_tokens(self) -> int:
        """Longest context one request can reach: its table row, or every usable block of the
        scarcest pool (the padding block is never given out), whichever is shorter."""
        usable = min(pool.num_blocks for pool in self._pools) - 1
        return min(self.max_tokens_per_request, usable * self._block_size)

    @property
    def free_blocks(self) -> int:
        """Blocks a request could still be given in the scarcest group."""
        return min(pool.free_count for pool in self._pools)

    @property
    def free_counts(self) -> tuple[int, ...]:
        """Free blocks per group; cheap, unlike :meth:`stats`, which walks the free queues."""
        return tuple(pool.free_count for pool in self._pools)

    @property
    def usable_counts(self) -> tuple[int, ...]:
        """Usable blocks per group (excluding the padding block), in O(groups) time."""
        return tuple(pool.num_blocks - 1 for pool in self._pools)

    @property
    def utilization(self) -> float:
        """Share of usable blocks currently held across the scarcest group, in ``[0, 1]``."""
        return max(
            (
                (total - free) / total
                for total, free in zip(self.usable_counts, self.free_counts, strict=True)
                if total > 0
            ),
            default=0.0,
        )

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
            cached_blocks=tuple(pool.cached_count for pool in self._pools),
            lookups=self._lookups,
            hit_tokens=self._hit_tokens,
        )

    def match_prefix(self, token_ids: Sequence[TokenId], salt: str | None) -> PrefixMatch | None:
        """The longest prefix of ``token_ids`` whose blocks are cached; changes no block.

        The match is valid until the next call that allocates or frees, so pass it to
        :meth:`allocate` in the same scheduling pass.

        Returns:
            ``None`` without a prefix cache; otherwise a match, possibly of zero blocks, that
            also carries the digests of every full prompt block.
        """
        if self._prefix is None:
            return None
        return self._prefix.match(self._pools, token_ids, salt)

    def match_and_promote_prefix(
        self, token_ids: Sequence[TokenId], salt: str | None
    ) -> PrefixMatch | None:
        """Match prefix blocks in VRAM and promote contiguous suffix blocks from host RAM.

        Falls back identically to :meth:`match_prefix` when no host store is attached or when
        host promotion cannot proceed (no free VRAM block or missing digest).
        """
        match = self.match_prefix(token_ids, salt)
        if match is None or self._host_store is None or self._storage_pools is None:
            return match
        digests = match.digests
        limit = (
            len(digests) - 1 if len(digests) * self._block_size >= len(token_ids) else len(digests)
        )
        matched = [list(ids) for ids in match.blocks]
        to_release: list[tuple[BlockPool, int]] = []
        # The blocks matched in VRAM hold no reference yet: allocate() below takes the least
        # recently used free block, which may be one of them, and a promotion would overwrite
        # it while the match still names it as a prefix block. Hold them until the end.
        for pool, ids in zip(self._pools, matched, strict=True):
            for block_id in ids:
                pool.touch(block_id)
                to_release.append((pool, block_id))
        idx = match.num_blocks
        while idx < limit:
            digest = digests[idx]
            if not self._host_store.has(digest) or any(pool.free_count < 1 for pool in self._pools):
                break
            new_ids = [pool.allocate() for pool in self._pools]
            if not self._host_store.promote(digest, new_ids, self._storage_pools):
                for pool, block_id in zip(self._pools, new_ids, strict=True):
                    pool.free(block_id)
                break
            for group, (pool, block_id) in enumerate(zip(self._pools, new_ids, strict=True)):
                if not pool.register(block_id, digest):
                    for undo_pool, undo_id in zip(self._pools, new_ids, strict=True):
                        undo_pool.free(undo_id)
                    break
                matched[group].append(block_id)
                to_release.append((pool, block_id))
            else:
                idx += 1
                continue
            break
        for pool, block_id in to_release:
            pool.free(block_id)
        count = len(matched[0]) if matched else 0
        if count == match.num_blocks:
            return match
        return PrefixMatch(
            num_tokens=count * self._block_size,
            num_blocks=count,
            root=match.root,
            digests=match.digests,
            blocks=tuple(tuple(ids) for ids in matched),
        )

    def _prefix_block_ids(
        self, token_ids: Sequence[TokenId], salt: str | None = None
    ) -> list[list[int]]:
        if self._prefix is None:
            return []
        digests = self._prefix.digests(self._prefix.root(salt), token_ids)
        matched: list[list[int]] = []
        for digest in digests:
            ids = [
                block_id for pool in self._pools if (block_id := pool.lookup(digest)) is not None
            ]
            if len(ids) != len(self._pools):
                break
            matched.append(ids)
        return matched

    def cached_prefix_blocks(self, token_ids: Sequence[TokenId], salt: str | None = None) -> int:
        """Number of full blocks of ``token_ids`` currently cached across every VRAM pool."""
        return len(self._prefix_block_ids(token_ids, salt))

    def pin_prefix(self, token_ids: Sequence[TokenId], salt: str | None = None) -> int:
        """Pin every currently cached prefix block of ``token_ids`` in VRAM."""
        matched = self._prefix_block_ids(token_ids, salt)
        for ids in matched:
            for pool, block_id in zip(self._pools, ids, strict=True):
                pool.pin(block_id)
        return len(matched)

    def unpin_prefix(self, token_ids: Sequence[TokenId], salt: str | None = None) -> int:
        """Release one pin from every pinned cached prefix block of ``token_ids``."""
        matched = self._prefix_block_ids(token_ids, salt)
        for ids in matched:
            for pool, block_id in zip(self._pools, ids, strict=True):
                if pool.is_pinned(block_id):
                    pool.unpin(block_id)
        return len(matched)

    # -- changing --------------------------------------------------------------------------

    def allocate(
        self, request_id: RequestId, num_tokens: int, prefix: PrefixMatch | None = None
    ) -> bool:
        """Make room for ``num_tokens`` more tokens of a request, in every group or in none.

        ``num_tokens`` counts every row the coming step will write, **look-ahead rows of
        a speculative step included**: ``1 + k`` for a decode step with ``k`` drafts. The
        room is measured from the committed length, so asking again before
        :meth:`commit` does not allocate twice. A new request gets a block table row.

        ``prefix``, a result of :meth:`match_prefix`, is only for a request that holds nothing
        yet. Its matched blocks are attached first and count as committed, and ``num_tokens``
        is measured after them. It also makes the request's full blocks cacheable through
        :meth:`cache_blocks`; a request admitted without one never registers a block.

        The call never evicts, never waits and never raises for a shortage. It returns
        ``False`` when any group has too few free blocks, when the request would outgrow a
        table row (:attr:`max_tokens_per_request`) or when no row is free; in every
        such case nothing has changed, and the scheduler decides what to do.

        Args:
            request_id: The request.
            num_tokens: Rows to make room for beyond the committed tokens; may be 0.
            prefix: The match to attach to a new request.

        Returns:
            ``True`` once the request has room; ``False`` if it could not be given.

        Raises:
            TypeError: If ``num_tokens`` is not an ``int``.
            ValueError: If ``num_tokens`` is negative.
            EngineError: If ``prefix`` is passed for a request that already holds blocks.
        """
        known = request_id in self._committed
        if prefix is not None and known:
            raise EngineError(f"request {request_id!r} already holds blocks; a prefix needs none")
        shared = prefix.num_blocks if prefix is not None else 0
        need = self.blocks_needed(request_id, num_tokens)
        if shared:
            need = div_ceil(shared * self._block_size + num_tokens, self._block_size) - shared
        if not known and self._table.free_rows == 0:
            return False
        have = self._table.num_blocks(request_id, 0) if known else shared
        if have + need > self._table.max_blocks_per_request:
            return False
        if prefix is not None and shared:
            # Attaching a cached block takes it out of the free queue, so it no longer counts.
            for pool, ids in zip(self._pools, prefix.blocks, strict=True):
                revived = sum(1 for block_id in ids if pool.is_free(block_id))
                if pool.free_count - revived < need:
                    return False
        elif any(pool.free_count < need for pool in self._pools):
            return False
        # Every check is done: from here on the call cannot fail, so no group is left half done.
        if not known:
            self._table.acquire(request_id)
            self._committed[request_id] = 0
            if prefix is not None and self._prefix is not None:
                self._lookups += 1
                self._hit_tokens += prefix.num_tokens
                self._chains[request_id] = _Chain(
                    parent=prefix.root, digests=list(prefix.digests), registered=shared
                )
        if prefix is not None and shared:
            for group, (pool, ids) in enumerate(zip(self._pools, prefix.blocks, strict=True)):
                for block_id in ids:
                    pool.touch(block_id)
                self._table.append(request_id, group, ids)
            self._committed[request_id] = prefix.num_tokens
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

    def cacheable_blocks(self, request_id: RequestId) -> tuple[int, int]:
        """``(registered, full)``: blocks of the request already named, and blocks now full.

        A block is full once all its tokens are committed. When ``full`` passes ``registered``,
        the caller hands :meth:`cache_blocks` the tokens of blocks ``registered .. full``. A
        request that cannot cache (no prefix cache, or admitted without a match) gives
        ``(0, 0)``.

        Raises:
            KeyError: If the request is unknown.
        """
        committed = self._committed[request_id]
        chain = self._chains.get(request_id)
        if chain is None:
            return 0, 0
        return chain.registered, committed // self._block_size

    def cache_blocks(self, request_id: RequestId, token_ids: Sequence[TokenId]) -> int:
        """Register the request's newly full blocks so later requests can attach them.

        Args:
            request_id: The request.
            token_ids: The tokens of blocks ``registered .. full`` (see :meth:`cacheable_blocks`),
                from the first block not yet registered. A partial last block is ignored.

        Returns:
            Blocks registered by this call. A block whose digest another block already holds
            stays private to the request: there is no deduplication.

        Raises:
            KeyError: If the request is unknown.
            EngineError: If ``token_ids`` cover blocks that are not full yet.
        """
        chain = self._chains.get(request_id)
        if chain is None or self._prefix is None:
            return 0
        start = chain.registered
        full = self._committed[request_id] // self._block_size
        count = len(token_ids) // self._block_size
        if start + count > full:
            raise EngineError(
                f"request {request_id!r}: only {full} blocks are full, cannot cache "
                f"blocks {start}..{start + count}"
            )
        if count == 0:
            return 0
        known = len(chain.digests)
        if known < start + count:
            parent = chain.digests[known - 1] if known else chain.parent
            offset = (known - start) * self._block_size
            chain.digests.extend(
                self._prefix.digests(parent, token_ids[offset : count * self._block_size])
            )
        rows = self.blocks(request_id)
        added = 0
        for index in range(start, start + count):
            digest = chain.digests[index]
            if any(pool.lookup(digest) is not None for pool in self._pools):
                continue
            for pool, ids in zip(self._pools, rows, strict=True):
                pool.register(ids[index], digest)
            added += 1
        chain.registered = start + count
        return added

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
        self._chains.pop(request_id, None)

    def resize(self, new_num_blocks_per_group: Sequence[int]) -> tuple[KVPool, ...]:
        """Atomically grow or compact-shrink every KV group to ``new_num_blocks_per_group``.

        Validates active + pinned capacity up front and allocates replacement storage pools before
        mutating any bookkeeping. If storage reallocation fails with an out-of-memory error, the
        previous pools and cache remain completely intact.
        """
        if len(new_num_blocks_per_group) != len(self._pools):
            raise ValueError(
                f"{len(self._pools)} groups need {len(self._pools)} target block counts, "
                f"got {len(new_num_blocks_per_group)}"
            )
        plans = [
            plan_pool_resize(pool, target)
            for pool, target in zip(self._pools, new_num_blocks_per_group, strict=True)
        ]
        if self._storage_pools is not None and self._host_store is not None:
            any_evicted = False
            for group_idx, (plan, kv_pool) in enumerate(
                zip(plans, self._storage_pools, strict=True)
            ):
                for old_id, digest in plan.evicted_digests:
                    self._host_store.offload(group_idx, old_id, digest, kv_pool)
                    any_evicted = True
            if any_evicted:
                self._host_store.flush()

        new_pools: tuple[KVPool, ...] = ()
        if self._storage_pools is not None:
            allocated: list[KVPool] = []
            try:
                for kv_pool, plan in zip(self._storage_pools, plans, strict=True):
                    allocated.append(kv_pool.reallocate(plan.new_num_blocks, plan.remap))
            except Exception as exc:
                allocated.clear()
                if "out of memory" in str(exc).lower() or type(exc).__name__ == "OutOfMemoryError":
                    raise OutOfMemoryError(f"KV pool resize ran out of memory: {exc}") from exc
                raise
            new_pools = tuple(allocated)

        for group_idx, (pool, plan) in enumerate(zip(self._pools, plans, strict=True)):
            pool.apply_resize(plan.new_num_blocks, plan.remap)
            self._table.remap_blocks(group_idx, plan.remap)
        if new_pools:
            self._storage_pools = new_pools
            self._wire_eviction_hooks()
        self.check_integrity()
        return new_pools

    # -- checking --------------------------------------------------------------------------

    def check_integrity(self) -> None:
        """Verify the books against the table.

        Runs each pool's own check, then confirms that every group's reference counts equal
        how often each block appears in the table (a shared block once per request holding
        it), that no request holds a block twice, that a request's registered blocks carry its
        chain digests in every group, that each request holds the same number of blocks in
        every group and enough room for its committed tokens, and that the table and the
        request set agree. It is for tests and debug builds, not the hot path.

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
                chain = self._table.blocks(request_id, group)
                if len(set(chain)) != len(chain):
                    problems.append(f"group {group}: request {request_id!r} holds a block twice")
                for block_id in chain:
                    seen[block_id] = seen.get(block_id, 0) + 1
            for block_id in range(1, pool.num_blocks):
                if pool.ref_count(block_id) != seen.get(block_id, 0):
                    problems.append(
                        f"group {group} block {block_id}: {pool.ref_count(block_id)} references, "
                        f"{seen.get(block_id, 0)} in the table"
                    )
        for request_id, committed in self._committed.items():
            counts = {self._table.num_blocks(request_id, g) for g in range(len(self._groups))}
            if len(counts) != 1:
                problems.append(f"request {request_id!r} holds uneven blocks across groups")
            elif committed > counts.pop() * self._block_size:
                problems.append(
                    f"request {request_id!r} committed more tokens than it has room for"
                )
        for request_id, chain in self._chains.items():
            if request_id not in self._committed:
                problems.append(f"request {request_id!r} has a digest chain but no blocks")
                continue
            rows = self.blocks(request_id)
            for index in range(min(chain.registered, len(chain.digests))):
                held = {
                    pool.digest_of(ids[index]) for pool, ids in zip(self._pools, rows, strict=True)
                }
                if len(held) != 1:
                    problems.append(
                        f"request {request_id!r} block {index} has different digests per group"
                    )
                elif None not in held and held != {chain.digests[index]}:
                    problems.append(
                        f"request {request_id!r} block {index} is registered under another digest"
                    )
        in_use = self._table.max_requests - self._table.free_rows
        if in_use != len(self._committed):
            problems.append(f"{in_use} table rows in use for {len(self._committed)} requests")
        if problems:
            raise EngineError("KV cache is inconsistent: " + "; ".join(problems))
