"""Recurrent state slot allocation and block-aligned state checkpointing.

Pure Python bookkeeping with no torch dependency: manages fixed per-request
recurrent state slots alongside paged KV blocks and an LRU host checkpoint store
keyed by block-chain digests for exact linear-attention prefix-cache hits.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import TYPE_CHECKING

from ayaka.core.errors import EngineError
from ayaka.utils.validation import require_int

if TYPE_CHECKING:
    from ayaka.core.types import RequestId

__all__ = ["PADDING_STATE_SLOT", "StateCheckpointStore", "StateSlotPool"]

#: Slot 0 is always reserved for padding requests, just like KV block 0.
PADDING_STATE_SLOT = 0


class StateSlotPool:
    """Fixed-capacity allocator mapping live requests to recurrent state slots.

    Slots ``1 .. max_slots - 1`` can be assigned to requests; slot ``0`` is
    permanently reserved as the padding slot and is never assigned.

    Args:
        max_slots: Total slots in the backing pool, including padding slot 0;
            must be at least 2.
    """

    def __init__(self, max_slots: int) -> None:
        require_int(max_slots, "max_slots", minimum=2)
        self._max_slots = max_slots
        # Popped from the end so slots are handed out 1, 2, ...
        self._free: list[int] = list(range(max_slots - 1, 0, -1))
        self._slots: dict[RequestId, int] = {}

    @property
    def max_slots(self) -> int:
        """Total slots tracked, including padding slot 0."""
        return self._max_slots

    @property
    def usable_slots(self) -> int:
        """Slots assignable to requests (``max_slots - 1``)."""
        return self._max_slots - 1

    @property
    def free_count(self) -> int:
        """Slots currently free."""
        return len(self._free)

    @property
    def used_count(self) -> int:
        """Slots currently assigned to requests."""
        return len(self._slots)

    def has(self, request_id: RequestId) -> bool:
        """Whether ``request_id`` holds a state slot."""
        return request_id in self._slots

    def slot_of(self, request_id: RequestId) -> int:
        """Return the state slot assigned to ``request_id``."""
        return self._slots[request_id]

    def allocate(self, request_id: RequestId) -> int | None:
        """Assign a free slot to ``request_id``, or return existing slot if already assigned.

        Returns ``None`` if no slot is free without mutating state.
        """
        existing = self._slots.get(request_id)
        if existing is not None:
            return existing
        if not self._free:
            return None
        slot = self._free.pop()
        self._slots[request_id] = slot
        return slot

    def release(self, request_id: RequestId) -> int | None:
        """Release ``request_id``'s slot back to the pool; returns the released slot or ``None``."""
        slot = self._slots.pop(request_id, None)
        if slot is None:
            return None
        self._free.append(slot)
        return slot

    def check_integrity(self) -> None:
        """Verify no slot is leaked, duplicated, or out of range."""
        problems: list[str] = []
        used = list(self._slots.values())
        free = self._free
        if len(set(used)) != len(used):
            problems.append("two requests hold the same state slot")
        if len(set(free)) != len(free):
            problems.append("free state slot appears twice")
        overlap = set(used) & set(free)
        if overlap:
            problems.append(f"state slots {sorted(overlap)} are both used and free")
        all_slots = set(used) | set(free)
        expected = set(range(1, self._max_slots))
        if all_slots != expected:
            problems.append(
                f"state slots account for {len(all_slots)} slots instead of {len(expected)}"
            )
        if problems:
            raise EngineError("state slot pool is inconsistent: " + "; ".join(problems))


class StateCheckpointStore:
    """Bounded LRU store of recurrent state checkpoints indexed by block-chain digest.

    Only checkpoints at block indices matching ``stride_blocks`` are retained,
    allowing callers to trade host RAM overhead against bounded suffix recompute
    on prefix-cache hits.

    Args:
        max_bytes: Maximum host memory budget in bytes for stored checkpoints.
        stride_blocks: Store a checkpoint every ``stride_blocks`` blocks (>= 1).
    """

    def __init__(self, max_bytes: int, stride_blocks: int = 1) -> None:
        require_int(max_bytes, "max_bytes", minimum=0)
        require_int(stride_blocks, "stride_blocks", minimum=1)
        self._max_bytes = max_bytes
        self._stride_blocks = stride_blocks
        self._entries: OrderedDict[bytes, tuple[object, int]] = OrderedDict()
        self._current_bytes = 0
        self._hits = 0
        self._misses = 0
        self._evictions = 0

    @property
    def max_bytes(self) -> int:
        """Configured byte ceiling."""
        return self._max_bytes

    @property
    def stride_blocks(self) -> int:
        """Block interval between state checkpoints."""
        return self._stride_blocks

    @property
    def current_bytes(self) -> int:
        """Total bytes currently held in the checkpoint store."""
        return self._current_bytes

    @property
    def count(self) -> int:
        """Number of checkpoints currently stored."""
        return len(self._entries)

    @property
    def hits(self) -> int:
        """Successful checkpoint lookups."""
        return self._hits

    @property
    def misses(self) -> int:
        """Missed checkpoint lookups."""
        return self._misses

    @property
    def evictions(self) -> int:
        """Checkpoints evicted to stay within ``max_bytes``."""
        return self._evictions

    def should_checkpoint(self, block_index: int) -> bool:
        """Whether 0-based ``block_index`` lands on a checkpoint boundary."""
        return (block_index + 1) % self._stride_blocks == 0

    def contains(self, digest: bytes) -> bool:
        """Read-only check whether ``digest`` has a checkpoint."""
        return digest in self._entries

    def get(self, digest: bytes) -> object | None:
        """Retrieve checkpoint payload for ``digest`` and promote it in LRU order."""
        entry = self._entries.get(digest)
        if entry is None:
            self._misses += 1
            return None
        self._entries.move_to_end(digest)
        self._hits += 1
        return entry[0]

    def put(self, digest: bytes, payload: object, nbytes: int) -> bool:
        """Insert or update a checkpoint payload under ``digest``.

        Evicts oldest entries until ``nbytes`` fits within ``max_bytes``.
        Returns ``False`` without storing if ``nbytes > max_bytes`` or
        ``nbytes < 0``.
        """
        require_int(nbytes, "nbytes", minimum=0)
        if nbytes > self._max_bytes:
            return False
        existing = self._entries.pop(digest, None)
        if existing is not None:
            self._current_bytes -= existing[1]
        while self._entries and self._current_bytes + nbytes > self._max_bytes:
            _, (_, evicted_bytes) = self._entries.popitem(last=False)
            self._current_bytes -= evicted_bytes
            self._evictions += 1
        self._entries[digest] = (payload, nbytes)
        self._current_bytes += nbytes
        return True

    def clear(self) -> None:
        """Drop all stored checkpoints."""
        self._entries.clear()
        self._current_bytes = 0

    def check_integrity(self) -> None:
        """Verify accounting invariants of the checkpoint store."""
        total = sum(nbytes for _, nbytes in self._entries.values())
        if total != self._current_bytes:
            raise EngineError(
                f"state checkpoint store bytes mismatch: tracked {self._current_bytes}, "
                f"actual {total}"
            )
        if self._current_bytes > self._max_bytes:
            raise EngineError(
                f"state checkpoint store exceeds max_bytes: "
                f"{self._current_bytes} > {self._max_bytes}"
            )
