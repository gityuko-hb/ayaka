"""The tensors of one KV group.

A :class:`KVPool` allocates the keys and values of one :class:`~ayaka.kvcache.storage.spec.KVGroup`
once, at start-up, as two tensors, and hands out one layer's slice as
:class:`~ayaka.ops.kvcache.KVBuffers`. It owns memory and nothing else: which
block belongs to which request is :class:`~ayaka.kvcache.blocks.BlockPool` and
:class:`~ayaka.kvcache.table.BlockTable`'s business.

torch is imported when a pool is built, never when this module is imported.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ayaka.core.types import KVLayoutKind
from ayaka.ops.kvcache import KVBuffers
from ayaka.utils.torch_utils import require_torch, resolve_device
from ayaka.utils.validation import require_int

if TYPE_CHECKING:
    from torch import Tensor

    from ayaka.kvcache.storage.spec import KVGroup

__all__ = ["KVPool"]


class KVPool:
    """Keys and values of every layer of one group, in one allocation each.

    The tensors have shape ``[num_layers, num_blocks, block_size, kv_heads, head_dim]``
    (``NHD``) or ``[num_layers, num_blocks, kv_heads, block_size, head_dim]``
    (``HND``), so one layer's slice is exactly the layout
    :mod:`ayaka.ops.kvcache` defines. Block 0 is the padding block and is
    zero-filled; it is counted in :attr:`nbytes` because it occupies memory.

    Args:
        group: The layers and the geometry to allocate for.
        num_blocks: Blocks per layer, padding block included; at least 2.
        device: Where to allocate: a ``torch.device``, a string, a CUDA index, or
            ``None`` for the best available device.

    Raises:
        ValueError: If ``num_blocks`` is below 2.
        CapabilityError: If torch is missing or the device is unavailable.
    """

    def __init__(self, group: KVGroup, num_blocks: int, *, device: Any = None) -> None:
        require_int(num_blocks, "num_blocks", minimum=2)
        spec = group.spec
        torch = require_torch(capability="kv_storage")
        resolved = resolve_device(device, capability="kv_storage")
        dtype = spec.dtype.torch_dtype
        if spec.layout is KVLayoutKind.NHD:
            shape = (
                spec.num_layers,
                num_blocks,
                spec.block_size,
                spec.num_kv_heads,
                spec.head_dim,
            )
        else:
            shape = (
                spec.num_layers,
                num_blocks,
                spec.num_kv_heads,
                spec.block_size,
                spec.head_dim,
            )
        self._group = group
        self._num_blocks = num_blocks
        self._k: Tensor = torch.zeros(shape, dtype=dtype, device=resolved)
        self._v: Tensor = torch.zeros(shape, dtype=dtype, device=resolved)

    @property
    def group(self) -> KVGroup:
        """The group this pool stores."""
        return self._group

    @property
    def num_blocks(self) -> int:
        """Blocks per layer, the padding block included."""
        return self._num_blocks

    @property
    def usable_blocks(self) -> int:
        """Blocks a request can own: every block but the padding block."""
        return self._num_blocks - 1

    @property
    def device(self) -> Any:
        """The ``torch.device`` the tensors live on."""
        return self._k.device

    @property
    def nbytes(self) -> int:
        """Bytes allocated: ``num_blocks * spec.bytes_per_block()``, the padding block included."""
        return int(self._k.numel() * self._k.element_size() * 2)

    def buffers(self, layer_id: int) -> KVBuffers:
        """The keys and values of one model layer, as views into the pool's tensors.

        Args:
            layer_id: A model layer id that belongs to this group.

        Raises:
            KeyError: If the layer is not in this group.
        """
        index = self._group.layer_index(layer_id)
        spec = self._group.spec
        return KVBuffers(self._k[index], self._v[index], spec.layout, spec.codec)
