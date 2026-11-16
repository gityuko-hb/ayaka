"""The forward context: what one model step tells every layer.

A step is planned once, by the model runner, and then run through every layer.
The runner builds a :class:`ForwardContext` from the planned
:class:`~ayaka.ops.attention.AttentionMetadata` and the cache tensors of each
layer, and passes it down. A layer never builds or edits one, and nothing in it
changes during the step.

This module does not import torch.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING

from ayaka.core.errors import EngineError

if TYPE_CHECKING:
    from collections.abc import Mapping

    from torch import Tensor

    from ayaka.core.types import ForwardMode
    from ayaka.ops.attention import AttentionMetadata
    from ayaka.ops.kvcache import KVBuffers

__all__ = ["ForwardContext"]


@dataclass(frozen=True, slots=True, eq=False)
class ForwardContext:
    """An immutable snapshot of one step's metadata and cache tensors.

    The step's slot mapping, block tables and the query and context lengths of
    each request are in ``attention`` (see
    :class:`~ayaka.ops.attention.AttentionMetadata` for their meaning and for
    the paged layout). The context itself is frozen and its mapping is
    read-only; the tensors it points to are shared with the runner and must not
    be modified by a layer, except that the cache tensors are written through
    :meth:`~ayaka.ops.kvcache.KVCacheOps.store`.

    Attributes:
        attention: The planned step, shared by every layer.
        kv_buffers: The cache tensors of each layer that has a KV cache, by
            layer id. Layers of one group share a layout and block size, not
            tensors.
    """

    attention: AttentionMetadata
    kv_buffers: Mapping[int, KVBuffers]

    def __post_init__(self) -> None:
        # A private copy: the runner may keep reusing the dict it built this from.
        object.__setattr__(self, "kv_buffers", MappingProxyType(dict(self.kv_buffers)))

    @property
    def mode(self) -> ForwardMode:
        """How the step uses its rows."""
        return self.attention.mode

    @property
    def positions(self) -> Tensor:
        """``[N]`` int32 position of each row in its sequence, for position embeddings."""
        return self.attention.positions

    def buffers(self, layer_id: int) -> KVBuffers:
        """Return the cache tensors of one layer.

        Raises:
            EngineError: If the runner built this context without that layer.
        """
        try:
            return self.kv_buffers[layer_id]
        except KeyError:
            raise EngineError(
                f"the forward context has no KV buffers for layer {layer_id}; "
                f"it has them for layers {sorted(self.kv_buffers)}",
                context={"layer_id": layer_id},
            ) from None
