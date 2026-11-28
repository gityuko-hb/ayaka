"""The one object the scheduler asks for KV blocks and recurrent state slots.

:class:`KVCacheManager` joins a :class:`~ayaka.kvcache.blocks.BlockPool` per KV
(or latent KV) group to one :class:`~ayaka.kvcache.table.BlockTable`, and
optionally pairs them with a :class:`~ayaka.kvcache.state_slots.StateSlotPool`
and :class:`~ayaka.kvcache.state_slots.StateCheckpointStore` for hybrid
linear-attention models. It is pure bookkeeping: it never touches a tensor.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from ayaka.core.errors import EngineError
from ayaka.kvcache.blocks import BlockPool
from ayaka.kvcache.prefix.base import PrefixMatch
from ayaka.kvcache.state_slots import PADDING_STATE_SLOT, StateCheckpointStore, StateSlotPool
from ayaka.kvcache.table import BlockTable
from ayaka.utils.math_utils import div_ceil
from ayaka.utils.validation import require_int

if TYPE_CHECKING:
    from collections.abc import Sequence

    from ayaka.core.types import RequestId, TokenId
    from ayaka.kvcache.prefix.base import PrefixCache
    from ayaka.kvcache.storage.spec import KVGroup, LatentKVGroup

__all__ = ["KVCacheManager", "KVCacheStats"]


@dataclass(frozen=True, slots=True)
class KVCacheStats:
    """A snapshot of the cache's occupancy."""

    num_requests: int
    usable_blocks: tuple[int, ...]
    free_blocks: tuple[int, ...]
    evictions: int
    cached_blocks: tuple[int, ...] = ()
    lookups: int = 0
    hit_tokens: int = 0
    state_slots_used: int = 0
    state_slots_free: int = 0
    state_checkpoint_bytes: int = 0
    state_checkpoint_hits: int = 0
    state_checkpoint_misses: int = 0

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
    """Hands blocks and recurrent state slots to requests, all groups or none."""

    def __init__(
        self,
        groups: Sequence[KVGroup | LatentKVGroup],
        num_blocks: Sequence[int],
        *,
        max_requests: int,
        max_blocks_per_request: int,
        prefix: PrefixCache | None = None,
        state_slots: StateSlotPool | None = None,
        state_checkpoints: StateCheckpointStore | None = None,
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
        if state_slots is not None and state_slots.usable_slots < max_requests:
            raise ValueError(
                f"state slot pool has {state_slots.usable_slots} usable slots, "
                f"fewer than max_requests={max_requests}"
            )
        self._pools = tuple(BlockPool(count) for count in num_blocks)
        self._table = BlockTable(len(self._groups), max_requests, max_blocks_per_request)
        self._committed: dict[RequestId, int] = {}
        self._prefix = prefix
        self._state_slots = state_slots
        self._state_checkpoints = state_checkpoints
        self._chains: dict[RequestId, _Chain] = {}
        self._lookups = 0
        self._hit_tokens = 0

    # -- reading ---------------------------------------------------------------------------

    @property
    def groups(self) -> tuple[KVGroup | LatentKVGroup, ...]:
        """The KV or latent KV groups, in the order every per-group value uses."""
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
    def state_slots(self) -> StateSlotPool | None:
        """Recurrent state slot allocator for hybrid linear-attention models, or ``None``."""
        return self._state_slots

    @property
    def state_checkpoints(self) -> StateCheckpointStore | None:
        """Host checkpoint store for recurrent state prefix hits, or ``None``."""
        return self._state_checkpoints

    @property
    def max_tokens_per_request(self) -> int:
        """Longest context one request can hold: ``max_blocks_per_request * block_size``."""
        return self._table.max_blocks_per_request * self._block_size

    @property
    def max_context_tokens(self) -> int:
        """Longest context one request can reach."""
        usable = min(pool.num_blocks for pool in self._pools) - 1
        return min(self.max_tokens_per_request, usable * self._block_size)

    @property
    def free_blocks(self) -> int:
        """Blocks a request could still be given in the scarcest group."""
        return min(pool.free_count for pool in self._pools)

    @property
    def free_counts(self) -> tuple[int, ...]:
        """Free blocks per group."""
        return tuple(pool.free_count for pool in self._pools)

    def has(self, request_id: RequestId) -> bool:
        """Whether a request holds a block table row."""
        return request_id in self._committed

    def state_slot(self, request_id: RequestId) -> int:
        """The recurrent state slot owned by ``request_id``, or padding slot 0 when unused."""
        if self._state_slots is None:
            return PADDING_STATE_SLOT
        return self._state_slots.slot_of(request_id)

    def num_committed(self, request_id: RequestId) -> int:
        """Tokens of the request that are real, as recorded by :meth:`commit`."""
        return self._committed[request_id]

    def blocks(self, request_id: RequestId) -> tuple[tuple[int, ...], ...]:
        """The block ids a request holds, per group, in token order."""
        return tuple(self._table.blocks(request_id, group) for group in range(len(self._groups)))

    def chain_digests(self, request_id: RequestId) -> tuple[bytes, ...]:
        """Registered/known block chain digests for ``request_id`` (empty if uncached)."""
        chain = self._chains.get(request_id)
        return tuple(chain.digests) if chain is not None else ()

    def preview_state_checkpoints(
        self,
        request_id: RequestId,
        prefix_tokens: Sequence[TokenId],
        start_position: int,
        chunk_len: int,
    ) -> tuple[tuple[int, bytes], ...]:
        """Return ``(chunk_offset, digest)`` pairs where recurrent state should snapshot.

        Extends the request's internal block digest chain up to
        ``(start_position + chunk_len) // block_size`` so every checkpoint boundary
        aligned to ``state_checkpoints.stride_blocks`` can be captured during the
        prefill forward pass.
        """
        if (
            self._state_slots is None
            or self._state_checkpoints is None
            or self._prefix is None
            or chunk_len <= 0
        ):
            return ()
        chain = self._chains.get(request_id)
        if chain is None:
            return ()
        block_size = self._block_size
        stop_pos = start_position + chunk_len
        last_full = stop_pos // block_size
        first_block = start_position // block_size
        if last_full <= first_block:
            return ()
        known = len(chain.digests)
        if known < last_full:
            parent = chain.digests[known - 1] if known else chain.parent
            chain.digests.extend(
                self._prefix.digests(
                    parent, prefix_tokens[known * block_size : last_full * block_size]
                )
            )
        targets: list[tuple[int, bytes]] = []
        for block_idx in range(first_block, last_full):
            end_pos = (block_idx + 1) * block_size
            if start_position < end_pos <= stop_pos and self._state_checkpoints.should_checkpoint(
                block_idx
            ):
                targets.append((end_pos - start_position, chain.digests[block_idx]))
        return tuple(targets)

    def blocks_needed(self, request_id: RequestId, num_tokens: int) -> int:
        """How many new blocks per group :meth:`allocate` would take; changes nothing."""
        require_int(num_tokens, "num_tokens")
        if request_id in self._committed:
            wanted = div_ceil(self._committed[request_id] + num_tokens, self._block_size)
            return max(0, wanted - self._table.num_blocks(request_id, 0))
        return div_ceil(num_tokens, self._block_size)

    def stats(self) -> KVCacheStats:
        """The current occupancy."""
        slots_used = self._state_slots.used_count if self._state_slots is not None else 0
        slots_free = self._state_slots.free_count if self._state_slots is not None else 0
        ckpt_bytes = (
            self._state_checkpoints.current_bytes if self._state_checkpoints is not None else 0
        )
        ckpt_hits = self._state_checkpoints.hits if self._state_checkpoints is not None else 0
        ckpt_misses = self._state_checkpoints.misses if self._state_checkpoints is not None else 0
        return KVCacheStats(
            num_requests=len(self._committed),
            usable_blocks=tuple(pool.num_blocks - 1 for pool in self._pools),
            free_blocks=tuple(pool.free_count for pool in self._pools),
            evictions=sum(pool.evictions for pool in self._pools),
            cached_blocks=tuple(pool.cached_count for pool in self._pools),
            lookups=self._lookups,
            hit_tokens=self._hit_tokens,
            state_slots_used=slots_used,
            state_slots_free=slots_free,
            state_checkpoint_bytes=ckpt_bytes,
            state_checkpoint_hits=ckpt_hits,
            state_checkpoint_misses=ckpt_misses,
        )

    def match_prefix(self, token_ids: Sequence[TokenId], salt: str | None) -> PrefixMatch | None:
        """The longest prefix of ``token_ids`` whose blocks (and state checkpoint) are cached.

        When the manager carries recurrent state slots, a KV prefix hit is
        aligned back to the longest cached block whose digest also holds a
        recurrent-state checkpoint in :attr:`state_checkpoints`, guaranteeing exact
        recurrent state equivalence on warm prefix hits.
        """
        if self._prefix is None:
            return None
        match = self._prefix.match(self._pools, token_ids, salt)
        if self._state_slots is None:
            return match
        if self._state_checkpoints is None or match.num_blocks == 0:
            return PrefixMatch.miss(match.root, match.digests, len(self._pools))
        for idx in range(match.num_blocks - 1, -1, -1):
            digest = match.digests[idx]
            if self._state_checkpoints.contains(digest):
                count = idx + 1
                return PrefixMatch(
                    num_tokens=count * self._block_size,
                    num_blocks=count,
                    root=match.root,
                    digests=match.digests,
                    blocks=tuple(ids[:count] for ids in match.blocks),
                    checkpoint_digest=digest,
                )
        return PrefixMatch.miss(match.root, match.digests, len(self._pools))

    # -- changing --------------------------------------------------------------------------

    def allocate(
        self, request_id: RequestId, num_tokens: int, prefix: PrefixMatch | None = None
    ) -> bool:
        """Make room for ``num_tokens`` more tokens of a request, in every group/slot or none."""
        known = request_id in self._committed
        if prefix is not None and known:
            raise EngineError(f"request {request_id!r} already holds blocks; a prefix needs none")
        shared = prefix.num_blocks if prefix is not None else 0
        need = self.blocks_needed(request_id, num_tokens)
        if shared:
            need = div_ceil(shared * self._block_size + num_tokens, self._block_size) - shared
        if not known and self._table.free_rows == 0:
            return False
        if not known and self._state_slots is not None and self._state_slots.free_count == 0:
            return False
        have = self._table.num_blocks(request_id, 0) if known else shared
        if have + need > self._table.max_blocks_per_request:
            return False
        if prefix is not None and shared:
            for pool, ids in zip(self._pools, prefix.blocks, strict=True):
                revived = sum(1 for block_id in ids if pool.is_free(block_id))
                if pool.free_count - revived < need:
                    return False
        elif any(pool.free_count < need for pool in self._pools):
            return False
        # Every check is done: from here on the call cannot fail.
        if not known:
            self._table.acquire(request_id)
            if self._state_slots is not None:
                assigned = self._state_slots.allocate(request_id)
                if assigned is None:  # pragma: no cover - guarded above
                    self._table.release(request_id)
                    return False
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
        """Record that ``num_tokens`` more tokens of the request are real."""
        require_int(num_tokens, "num_tokens")
        committed = self._committed[request_id] + num_tokens
        room = self._table.num_blocks(request_id, 0) * self._block_size
        if committed > room:
            raise EngineError(
                f"request {request_id!r} commits {committed} tokens but holds room for {room}"
            )
        self._committed[request_id] = committed

    def cacheable_blocks(self, request_id: RequestId) -> tuple[int, int]:
        """``(registered, full)``: blocks already named, and blocks now full."""
        committed = self._committed[request_id]
        chain = self._chains.get(request_id)
        if chain is None:
            return 0, 0
        return chain.registered, committed // self._block_size

    def cache_blocks(self, request_id: RequestId, token_ids: Sequence[TokenId]) -> int:
        """Register the request's newly full blocks so later requests can attach them."""
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
        """Give back every block and state slot of a request, and its table row."""
        if request_id not in self._committed:
            return
        for group, pool in enumerate(self._pools):
            pool.free_chain(self._table.blocks(request_id, group))
        self._table.release(request_id)
        if self._state_slots is not None:
            self._state_slots.release(request_id)
        del self._committed[request_id]
        self._chains.pop(request_id, None)

    # -- checking --------------------------------------------------------------------------

    def check_integrity(self) -> None:
        """Verify the books, state slots, and checkpoint store against the table."""
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
        if self._state_slots is not None:
            try:
                self._state_slots.check_integrity()
            except EngineError as exc:
                problems.append(str(exc))
            if self._state_slots.used_count != len(self._committed):
                problems.append(
                    f"{self._state_slots.used_count} state slots in use for "
                    f"{len(self._committed)} requests"
                )
            for request_id in self._committed:
                if not self._state_slots.has(request_id):
                    problems.append(f"request {request_id!r} holds KV without a state slot")
        if self._state_checkpoints is not None:
            try:
                self._state_checkpoints.check_integrity()
            except EngineError as exc:
                problems.append(str(exc))
        if problems:
            raise EngineError("KV cache is inconsistent: " + "; ".join(problems))
