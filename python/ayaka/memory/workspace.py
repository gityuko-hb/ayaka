"""Scratch memory of a step: one fixed buffer handed out by bumping an offset.

A backend or a layer that needs temporary memory inside a step (a kernel
library's split-K scratch, a gather buffer) takes it from a :class:`Workspace`
instead of allocating. The buffer is allocated once, at start-up, sized by the
memory plan, and charged to the ``workspace`` owner of the
:class:`~ayaka.memory.budget.MemoryBudget`; a step resets the offset and takes
what it needs. Nothing is freed piecewise, so taking is a few integer
operations, the addresses are the same in every step (which a CUDA graph
needs), and running out is an error with the numbers rather than a silent
allocation.

torch is imported when a workspace is built, never when this module is imported.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Final

from ayaka.core.errors import CapacityError
from ayaka.utils.math_utils import div_ceil
from ayaka.utils.torch_utils import require_torch, resolve_device
from ayaka.utils.validation import require_int

if TYPE_CHECKING:
    from collections.abc import Sequence

__all__ = ["WORKSPACE_ALIGNMENT", "Workspace"]

#: Every slice starts on a multiple of this many bytes: enough for any vector load.
WORKSPACE_ALIGNMENT: Final = 256


class Workspace:
    """A bump allocator over one byte buffer on one device.

    Args:
        nbytes: Size of the buffer; at least one byte.
        device: Where to allocate it.

    Raises:
        TypeError: If ``nbytes`` is not an ``int``.
        ValueError: If ``nbytes`` is below 1.
        CapabilityError: If torch or the device is missing.
    """

    def __init__(self, nbytes: int, *, device: Any) -> None:
        require_int(nbytes, "nbytes", minimum=1)
        torch = require_torch(capability="workspace")
        self._buffer: Any = torch.empty(
            nbytes, dtype=torch.uint8, device=resolve_device(device, capability="workspace")
        )
        self._offset = 0
        self._high_water = 0

    @property
    def nbytes(self) -> int:
        """Size of the buffer."""
        return int(self._buffer.numel())

    @property
    def used_bytes(self) -> int:
        """Bytes handed out since the last :meth:`reset`, alignment included."""
        return self._offset

    @property
    def high_water_bytes(self) -> int:
        """The most bytes any one step has used: what the buffer could shrink to."""
        return self._high_water

    @property
    def device(self) -> Any:
        """The ``torch.device`` of the buffer."""
        return self._buffer.device

    def reset(self) -> None:
        """Hand the whole buffer out again; every slice taken before is now free to reuse."""
        self._offset = 0

    def take(self, shape: Sequence[int], dtype: Any) -> Any:
        """A tensor of ``shape`` and ``dtype`` over the next free bytes; not initialised.

        Raises:
            CapacityError: If the rest of the buffer is too small, with the
                bytes needed and the bytes left.
            ValueError: If a dimension is negative.
        """
        torch = require_torch(capability="workspace")
        count = 1
        for size in shape:
            count *= require_int(size, "shape entry")
        itemsize = int(torch.empty((), dtype=dtype).element_size())
        need = count * itemsize
        start = div_ceil(self._offset, WORKSPACE_ALIGNMENT) * WORKSPACE_ALIGNMENT
        if start + need > self.nbytes:
            have = max(self.nbytes - start, 0)
            raise CapacityError(
                f"workspace: need {need} bytes, have {have}, short {need - have}",
                context={"what": "workspace", "need": need, "have": have, "short": need - have},
            )
        self._offset = start + need
        self._high_water = max(self._high_water, self._offset)
        return self._buffer[start : start + need].view(dtype).view(tuple(shape))
