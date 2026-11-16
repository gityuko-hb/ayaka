"""The books of one group's blocks: who is free, referenced or cached.

A :class:`BlockPool` is bookkeeping, not memory: it never touches a tensor.
Block ids index the tensors of the matching :class:`~ayaka.kvcache.storage.pool.KVPool`.
Per block it keeps a reference count, an optional digest and a place in the free
queue, all in flat integer arrays, so there is no Python object per block and
:meth:`BlockPool.check_integrity` can scan everything in one pass.

The free queue is the eviction order. Allocation takes the head. A block returned
with a digest (a cached block that can still be matched) goes to the tail, so it
is the last to be reused; a block returned without one has no value to keep and
goes to the head, so it is reused first. A chain of blocks is returned from its
last block to its first, which puts leaves of a cached chain ahead of its root in
the queue: a chain is eaten from the leaf, never from the root, so the blocks
behind an evicted root do not linger unreachable.

Block 0 is the padding block (:data:`~ayaka.kvcache.storage.spec.PADDING_BLOCK_ID`).
It is never queued, allocated, referenced or cached.
"""

from __future__ import annotations

from array import array
from dataclasses import dataclass
from typing import TYPE_CHECKING

from ayaka.core.errors import CapacityError, EngineError
from ayaka.kvcache.storage.spec import PADDING_BLOCK_ID
from ayaka.utils.validation import require_int

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

__all__ = ["Block", "BlockPool"]

_NONE = -1


@dataclass(frozen=True, slots=True)
class Block:
    """A read-only view of one block, for tests and debugging; hot paths use ids."""

    pool: BlockPool
    id: int

    @property
    def ref_count(self) -> int:
        """Requests (and pins) currently referencing the block."""
        return self.pool.ref_count(self.id)

    @property
    def digest(self) -> bytes | None:
        """The digest the block is registered under, or ``None``."""
        return self.pool.digest_of(self.id)

    @property
    def is_free(self) -> bool:
        """Whether the block is in the free queue."""
        return self.pool.is_free(self.id)


class BlockPool:
    """Reference counts, digests and the free queue of ``num_blocks`` blocks.

    Args:
        num_blocks: Blocks in the group's pool, the padding block included; at least 2.
            Block ids ``1 .. num_blocks - 1`` start free, in ascending order.

    Raises:
        TypeError: If ``num_blocks`` is not an ``int``.
        ValueError: If ``num_blocks`` is below 2.

    Block ids passed to the methods are not type-checked: they come from this pool
    and the manager, and a bad one is a programming error caught by the range check.
    """

    def __init__(self, num_blocks: int) -> None:
        require_int(num_blocks, "num_blocks", minimum=2)
        self._num_blocks = num_blocks
        self._ref = array("i", [0]) * num_blocks
        self._prev = array("i", [_NONE] * num_blocks)
        self._next = array("i", [_NONE] * num_blocks)
        self._queued = bytearray(num_blocks)
        self._digest: list[bytes | None] = [None] * num_blocks
        self._by_digest: dict[bytes, int] = {}
        self._head = _NONE
        self._tail = _NONE
        self._free_count = 0
        self._evictions = 0
        for block_id in range(PADDING_BLOCK_ID + 1, num_blocks):
            self._push_tail(block_id)

    # -- reading ---------------------------------------------------------------------------

    @property
    def num_blocks(self) -> int:
        """Blocks tracked, the padding block included."""
        return self._num_blocks

    @property
    def free_count(self) -> int:
        """Blocks in the free queue: those a request could be given."""
        return self._free_count

    @property
    def cached_count(self) -> int:
        """Free blocks that still hold a registered digest and could be matched again."""
        return sum(1 for block_id in self.free_queue() if self._digest[block_id] is not None)

    @property
    def evictions(self) -> int:
        """How many registered blocks have been reused, losing their digest."""
        return self._evictions

    def block(self, block_id: int) -> Block:
        """A view of one block."""
        self._check_id(block_id)
        return Block(self, block_id)

    def ref_count(self, block_id: int) -> int:
        """References held on a block."""
        self._check_id(block_id)
        return self._ref[block_id]

    def digest_of(self, block_id: int) -> bytes | None:
        """The digest a block is registered under, or ``None``."""
        self._check_id(block_id)
        return self._digest[block_id]

    def is_free(self, block_id: int) -> bool:
        """Whether a block is in the free queue."""
        self._check_id(block_id)
        return bool(self._queued[block_id])

    def lookup(self, digest: bytes) -> int | None:
        """The block registered under ``digest``, or ``None``. Read-only."""
        return self._by_digest.get(digest)

    def free_queue(self) -> list[int]:
        """Block ids from the head (reused first) to the tail (reused last)."""
        return list(self._walk())

    def _walk(self) -> Iterator[int]:
        block_id = self._head
        while block_id != _NONE:
            yield block_id
            block_id = self._next[block_id]

    # -- changing --------------------------------------------------------------------------

    def allocate(self) -> int:
        """Take the head of the free queue for a request: reference count 1.

        A block that was cached loses its digest here; this is the moment a cached
        block is evicted.

        Raises:
            CapacityError: If the queue is empty. Callers check :attr:`free_count`
                first; ``KVCacheManager.allocate`` turns that into ``False``.
        """
        if self._free_count == 0:
            raise CapacityError("no free block")
        block_id = self._head
        self._unlink(block_id)
        digest = self._digest[block_id]
        if digest is not None:
            del self._by_digest[digest]
            self._digest[block_id] = None
            self._evictions += 1
        self._ref[block_id] = 1
        return block_id

    def touch(self, block_id: int) -> None:
        """Add a reference to a block a request now shares; a cached block leaves the queue.

        Raises:
            EngineError: If the block is unreferenced and not in the queue.
        """
        self._check_id(block_id)
        if self._ref[block_id] == 0:
            if not self._queued[block_id]:
                raise EngineError(f"block {block_id} has no references and is not free")
            self._unlink(block_id)
        self._ref[block_id] += 1

    def free(self, block_id: int) -> None:
        """Drop one reference; at zero the block rejoins the queue.

        With a digest it goes to the tail (reused last); without one, to the head
        (reused first).

        Raises:
            EngineError: If the block holds no reference.
        """
        self._check_id(block_id)
        if self._ref[block_id] <= 0:
            raise EngineError(f"block {block_id} is already free")
        self._ref[block_id] -= 1
        if self._ref[block_id] == 0:
            if self._digest[block_id] is None:
                self._push_head(block_id)
            else:
                self._push_tail(block_id)

    def free_chain(self, block_ids: Sequence[int]) -> None:
        """Drop one reference on each block of a request's chain, last block first."""
        for block_id in reversed(block_ids):
            self.free(block_id)

    def register(self, block_id: int, digest: bytes) -> bool:
        """Make a referenced block matchable under ``digest``.

        Returns:
            ``False`` when another block already holds ``digest``; the block stays
            private to its request. ``True`` once registered.

        Raises:
            EngineError: If the block is not referenced or already has a digest.
        """
        self._check_id(block_id)
        if self._ref[block_id] <= 0:
            raise EngineError(f"block {block_id} is not referenced and cannot be registered")
        if self._digest[block_id] is not None:
            raise EngineError(f"block {block_id} is already registered")
        if digest in self._by_digest:
            return False
        self._by_digest[digest] = block_id
        self._digest[block_id] = digest
        return True

    # -- the queue -------------------------------------------------------------------------

    def _check_id(self, block_id: int) -> None:
        if block_id == PADDING_BLOCK_ID:
            raise ValueError("block 0 is the padding block and has no bookkeeping")
        if not 0 < block_id < self._num_blocks:
            raise ValueError(f"block id {block_id} is outside 1..{self._num_blocks - 1}")

    def _push_head(self, block_id: int) -> None:
        self._prev[block_id] = _NONE
        self._next[block_id] = self._head
        if self._head != _NONE:
            self._prev[self._head] = block_id
        else:
            self._tail = block_id
        self._head = block_id
        self._queued[block_id] = 1
        self._free_count += 1

    def _push_tail(self, block_id: int) -> None:
        self._next[block_id] = _NONE
        self._prev[block_id] = self._tail
        if self._tail != _NONE:
            self._next[self._tail] = block_id
        else:
            self._head = block_id
        self._tail = block_id
        self._queued[block_id] = 1
        self._free_count += 1

    def _unlink(self, block_id: int) -> None:
        before, after = self._prev[block_id], self._next[block_id]
        if before != _NONE:
            self._next[before] = after
        else:
            self._head = after
        if after != _NONE:
            self._prev[after] = before
        else:
            self._tail = before
        self._prev[block_id] = self._next[block_id] = _NONE
        self._queued[block_id] = 0
        self._free_count -= 1

    # -- checking --------------------------------------------------------------------------

    def check_integrity(self) -> None:
        """Verify the books against themselves.

        Checks that the queue links are consistent and acyclic and its length is
        :attr:`free_count`; that every queued block has no references and every
        unreferenced block is queued; that no reference count is negative; that the
        digest table and the per-block digests agree both ways; and that the
        padding block is untouched.

        Raises:
            EngineError: Listing every problem found.
        """
        problems: list[str] = []
        order: list[int] = []
        seen: set[int] = set()
        block_id = self._head
        previous = _NONE
        while block_id != _NONE and len(order) <= self._num_blocks:
            if not 0 < block_id < self._num_blocks:
                problems.append(f"queue link points outside the pool: {block_id}")
                break
            if block_id in seen:
                problems.append(f"queue has a cycle at block {block_id}")
                break
            if self._prev[block_id] != previous:
                problems.append(f"block {block_id} has prev {self._prev[block_id]}, not {previous}")
            seen.add(block_id)
            order.append(block_id)
            previous, block_id = block_id, self._next[block_id]
        else:
            if previous != self._tail:
                problems.append(f"queue ends at {previous} but the tail is {self._tail}")
        if len(order) != self._free_count:
            problems.append(f"queue holds {len(order)} blocks but free_count is {self._free_count}")
        for candidate in range(PADDING_BLOCK_ID + 1, self._num_blocks):
            refs = self._ref[candidate]
            queued = bool(self._queued[candidate])
            if refs < 0:
                problems.append(f"block {candidate} has negative references: {refs}")
            if queued != (candidate in seen):
                problems.append(f"block {candidate} queued flag disagrees with the queue")
            if queued and refs != 0:
                problems.append(f"free block {candidate} has {refs} references")
            if not queued and refs == 0:
                problems.append(f"block {candidate} has no references and is not free (leaked)")
        for digest, owner in self._by_digest.items():
            if not 0 < owner < self._num_blocks or self._digest[owner] != digest:
                problems.append(f"digest table points at block {owner}, which disagrees")
        registered = sum(1 for held in self._digest if held is not None)
        if registered != len(self._by_digest):
            problems.append(
                f"{registered} blocks hold digests, the table has {len(self._by_digest)}"
            )
        if (
            self._ref[PADDING_BLOCK_ID]
            or self._queued[PADDING_BLOCK_ID]
            or self._digest[PADDING_BLOCK_ID] is not None
        ):
            problems.append("the padding block has bookkeeping")
        if problems:
            raise EngineError("block pool is inconsistent: " + "; ".join(problems))
