"""Atomic KV pool resize planning: grow in place or compact/evict cleanly on shrink."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from ayaka.core.errors import CapacityError
from ayaka.kvcache.storage.spec import PADDING_BLOCK_ID
from ayaka.utils.validation import require_int

if TYPE_CHECKING:
    from ayaka.kvcache.blocks import BlockPool

__all__ = ["ResizePlan", "plan_pool_resize"]


@dataclass(frozen=True, slots=True)
class ResizePlan:
    """Plan for moving one KV group from ``old_num_blocks`` to ``new_num_blocks``.

    Attributes:
        old_num_blocks: Existing pool block count (including padding block 0).
        new_num_blocks: Target pool block count (including padding block 0).
        remap: Mapping ``old_block_id -> new_block_id`` for every preserved block.
        evicted_digests: Cached free blocks dropped during shrink as ``(old_block_id, digest)``,
            eligible for write-behind offload to host RAM before tensors are released.
    """

    old_num_blocks: int
    new_num_blocks: int
    remap: dict[int, int]
    evicted_digests: tuple[tuple[int, bytes], ...]


def plan_pool_resize(pool: BlockPool, new_num_blocks: int) -> ResizePlan:
    """Plan a grow or compacted shrink of ``pool`` to ``new_num_blocks``.

    Active (``ref_count > 0``) and pinned blocks are always preserved. On shrink, most
    recently used cached free blocks are retained up to remaining capacity and older
    cached blocks are marked in ``evicted_digests``.

    Raises:
        ValueError: If ``new_num_blocks < 2``.
        CapacityError: If active and pinned blocks exceed ``new_num_blocks - 1``.
    """
    target = require_int(new_num_blocks, "new_num_blocks", minimum=2)
    old = pool.num_blocks
    usable_target = target - 1

    retained = [
        b for b in range(PADDING_BLOCK_ID + 1, old) if pool.ref_count(b) > 0 or pool.is_pinned(b)
    ]
    if len(retained) > usable_target:
        raise CapacityError(
            f"cannot shrink KV pool to {usable_target} usable blocks: "
            f"{len(retained)} blocks are active or pinned"
        )

    if target >= old:
        remap = {b: b for b in range(PADDING_BLOCK_ID + 1, old)}
        return ResizePlan(
            old_num_blocks=old,
            new_num_blocks=target,
            remap=remap,
            evicted_digests=(),
        )

    free_order = pool.free_queue()
    remaining_slots = usable_target - len(retained)
    cached_mru_first = [b for b in reversed(free_order) if pool.digest_of(b) is not None]
    kept_cached = list(reversed(cached_mru_first[:remaining_slots]))

    shrink_remap: dict[int, int] = {}
    next_id = PADDING_BLOCK_ID + 1
    for old_id in retained:
        shrink_remap[old_id] = next_id
        next_id += 1
    for old_id in kept_cached:
        shrink_remap[old_id] = next_id
        next_id += 1

    evicted: list[tuple[int, bytes]] = []
    for old_id in free_order:
        if old_id in shrink_remap:
            continue
        digest = pool.digest_of(old_id)
        if digest is not None:
            evicted.append((old_id, digest))

    return ResizePlan(
        old_num_blocks=old,
        new_num_blocks=target,
        remap=shrink_remap,
        evicted_digests=tuple(evicted),
    )
