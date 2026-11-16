"""Which blocks a request holds, as the rows a kernel reads.

A :class:`BlockTable` gives each running request a row. Row ``r`` of group ``g``
lists, in order, the blocks that hold the request's positions: entry ``i`` covers
tokens ``[i * block_size, (i + 1) * block_size)``. Unused entries are 0, the
padding block, which is how :mod:`ayaka.ops.attention` reads "no block".

The table lives on the host, in NumPy, so it needs no torch. The scheduler side
appends ids to a row; the execution side copies the rows a step needs (see
:meth:`BlockTable.gather`) or only the ranges that changed since the last step
(:meth:`BlockTable.take_dirty`) to the device. A finished request's row is reused,
never reallocated.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from ayaka.core.errors import CapacityError, EngineError
from ayaka.utils.validation import require_int

if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy.typing import NDArray

    from ayaka.core.types import RequestId

__all__ = ["BlockTable"]


class BlockTable:
    """Rows of block ids for up to ``max_requests`` requests, one set of rows per group.

    Args:
        num_groups: KV groups; every group has its own block ids for the same request.
        max_requests: Requests that can hold a row at once.
        max_blocks_per_request: Entries in a row; the longest context a request can
            reach is ``max_blocks_per_request * block_size``.

    Raises:
        TypeError: If an argument is not an ``int``.
        ValueError: If an argument is below 1.
    """

    def __init__(self, num_groups: int, max_requests: int, max_blocks_per_request: int) -> None:
        require_int(num_groups, "num_groups", minimum=1)
        require_int(max_requests, "max_requests", minimum=1)
        require_int(max_blocks_per_request, "max_blocks_per_request", minimum=1)
        self._rows: NDArray[np.int32] = np.zeros(
            (num_groups, max_requests, max_blocks_per_request), dtype=np.int32
        )
        self._lengths: NDArray[np.int32] = np.zeros((num_groups, max_requests), dtype=np.int32)
        # Popped from the end: rows go out as 0, 1, 2, ... and a released row is reused first.
        self._free_rows: list[int] = list(range(max_requests - 1, -1, -1))
        self._row_of: dict[RequestId, int] = {}
        self._dirty: list[tuple[int, int, int, int]] = []

    @property
    def num_groups(self) -> int:
        """KV groups the table holds rows for."""
        return int(self._rows.shape[0])

    @property
    def max_requests(self) -> int:
        """Requests that can hold a row at once."""
        return int(self._rows.shape[1])

    @property
    def max_blocks_per_request(self) -> int:
        """Entries in one row."""
        return int(self._rows.shape[2])

    @property
    def free_rows(self) -> int:
        """Rows no request holds."""
        return len(self._free_rows)

    @property
    def rows(self) -> NDArray[np.int32]:
        """The live ``[num_groups, max_requests, max_blocks_per_request]`` int32 array.

        Read it, do not write it: writes bypass the lengths and the dirty ranges.
        """
        return self._rows

    @property
    def lengths(self) -> NDArray[np.int32]:
        """The live ``[num_groups, max_requests]`` int32 array of blocks in use per row."""
        return self._lengths

    def has(self, request_id: RequestId) -> bool:
        """Whether a request holds a row."""
        return request_id in self._row_of

    def row_of(self, request_id: RequestId) -> int:
        """The row a request holds.

        Raises:
            KeyError: If the request holds none.
        """
        return self._row_of[request_id]

    def acquire(self, request_id: RequestId) -> int:
        """Give a request an empty row in every group.

        Raises:
            EngineError: If the request already holds a row.
            CapacityError: If every row is taken.
        """
        if request_id in self._row_of:
            raise EngineError(f"request {request_id!r} already holds a block table row")
        if not self._free_rows:
            raise CapacityError(f"all {self.max_requests} block table rows are in use")
        row = self._free_rows.pop()
        self._row_of[request_id] = row
        return row

    def release(self, request_id: RequestId) -> None:
        """Clear a request's row in every group and make it available again.

        Raises:
            KeyError: If the request holds no row.
        """
        row = self._row_of.pop(request_id)
        for group in range(self.num_groups):
            used = int(self._lengths[group, row])
            if used:
                self._rows[group, row, :used] = 0
                self._lengths[group, row] = 0
                self._dirty.append((group, row, 0, used))
        self._free_rows.append(row)

    def num_blocks(self, request_id: RequestId, group: int) -> int:
        """Blocks a request holds in a group."""
        return int(self._lengths[group, self._row_of[request_id]])

    def blocks(self, request_id: RequestId, group: int) -> tuple[int, ...]:
        """The block ids a request holds in a group, in token order."""
        row = self._row_of[request_id]
        return tuple(int(b) for b in self._rows[group, row, : self._lengths[group, row]])

    def room(self, request_id: RequestId, group: int) -> int:
        """How many more blocks a request's row can take in a group."""
        return self.max_blocks_per_request - self.num_blocks(request_id, group)

    def append(self, request_id: RequestId, group: int, block_ids: Sequence[int]) -> None:
        """Add block ids to the end of a request's row in a group.

        Raises:
            CapacityError: If the row has no room for all of them; nothing is written.
            EngineError: If an id is 0, the padding block.
        """
        count = len(block_ids)
        if count == 0:
            return
        if any(block_id <= 0 for block_id in block_ids):
            raise EngineError("a block table cannot hold the padding block or a negative id")
        row = self._row_of[request_id]
        start = int(self._lengths[group, row])
        stop = start + count
        if stop > self.max_blocks_per_request:
            raise CapacityError(
                f"request {request_id!r} needs {stop} blocks in group {group}, "
                f"but a row holds {self.max_blocks_per_request}"
            )
        self._rows[group, row, start:stop] = block_ids
        self._lengths[group, row] = stop
        self._dirty.append((group, row, start, stop))

    def gather(self, group: int, request_ids: Sequence[RequestId]) -> NDArray[np.int32]:
        """Copy the rows of the given requests, in order, as a ``[B, max_blocks]`` int32 array.

        This is the ``block_rows`` argument of :meth:`ayaka.ops.attention.AttentionOps.plan`.
        """
        indices = [self._row_of[request_id] for request_id in request_ids]
        return self._rows[group, indices]

    def take_dirty(self) -> list[tuple[int, int, int, int]]:
        """Return the ``(group, row, start, stop)`` ranges changed since the last call.

        The ranges are cleared by the call, so each change is reported once.
        """
        changed, self._dirty = self._dirty, []
        return changed
