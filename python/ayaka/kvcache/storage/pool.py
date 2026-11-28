"""The tensors of KV, latent KV, and recurrent state pools.

``KVPool``, ``LatentKVPool``, and ``RecurrentStatePool`` allocate device storage
from their pure-Python specs and hand out per-layer tensor views. torch is
imported when a pool is instantiated, never when this module is imported.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ayaka.core.types import KVLayoutKind
from ayaka.ops.kvcache import KVBuffers, LatentKVBuffers
from ayaka.ops.linear_attention import RecurrentStateBuffers
from ayaka.utils.torch_utils import require_torch, resolve_device
from ayaka.utils.validation import require_int

if TYPE_CHECKING:
    from torch import Tensor

    from ayaka.kvcache.storage.spec import KVGroup, LatentKVGroup, RecurrentStateGroup

__all__ = ["KVPool", "LatentKVPool", "RecurrentStatePool"]


class KVPool:
    """Keys and values of every layer of one group, in one allocation each.

    The tensors have shape ``[num_layers, num_blocks, block_size, kv_heads, head_dim]``
    (``NHD``) or ``[num_layers, num_blocks, kv_heads, block_size, head_dim]``
    (``HND``). Block 0 is the padding block and is zero-filled.
    """

    def __init__(self, group: KVGroup, num_blocks: int, *, device: Any = None) -> None:
        require_int(num_blocks, "num_blocks", minimum=2)
        spec = group.spec
        torch = require_torch(capability="kv_storage")
        resolved = resolve_device(device, capability="kv_storage")
        dtype = spec.storage_dtype.torch_dtype
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
        # torch.zeros on float8_e4m3fn creates zero-initialized FP8 KV tensors directly.
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
        """The keys and values of one model layer, as views into the pool's tensors."""
        index = self._group.layer_index(layer_id)
        spec = self._group.spec
        return KVBuffers(
            self._k[index],
            self._v[index],
            spec.layout,
            spec.codec,
            k_scale=spec.kv_scale,
            v_scale=spec.kv_scale,
        )


class LatentKVPool:
    """Compressed latent KV and decoupled RoPE keys for Multi-Head Latent Attention."""

    def __init__(self, group: LatentKVGroup, num_blocks: int, *, device: Any = None) -> None:
        require_int(num_blocks, "num_blocks", minimum=2)
        spec = group.spec
        torch = require_torch(capability="kv_storage")
        resolved = resolve_device(device, capability="kv_storage")
        dtype = spec.storage_dtype.torch_dtype
        self._group = group
        self._num_blocks = num_blocks
        self._ckv: Tensor = torch.zeros(
            (spec.num_layers, num_blocks, spec.block_size, spec.kv_lora_rank),
            dtype=dtype,
            device=resolved,
        )
        self._kpe: Tensor = torch.zeros(
            (spec.num_layers, num_blocks, spec.block_size, spec.qk_rope_head_dim),
            dtype=dtype,
            device=resolved,
        )

    @property
    def group(self) -> LatentKVGroup:
        """The latent KV group this pool stores."""
        return self._group

    @property
    def num_blocks(self) -> int:
        """Blocks per layer, the padding block included."""
        return self._num_blocks

    @property
    def usable_blocks(self) -> int:
        """Blocks a request can own."""
        return self._num_blocks - 1

    @property
    def device(self) -> Any:
        """The ``torch.device`` the tensors live on."""
        return self._ckv.device

    @property
    def nbytes(self) -> int:
        """Total bytes allocated across ``ckv`` and ``kpe``."""
        return int(
            self._ckv.numel() * self._ckv.element_size()
            + self._kpe.numel() * self._kpe.element_size()
        )

    def buffers(self, layer_id: int) -> LatentKVBuffers:
        """Views of one layer's compressed KV latent and rotary key tensors."""
        index = self._group.layer_index(layer_id)
        spec = self._group.spec
        return LatentKVBuffers(
            self._ckv[index],
            self._kpe[index],
            codec=spec.codec,
            ckv_scale=spec.kv_scale,
            kpe_scale=spec.kv_scale,
        )


class RecurrentStatePool:
    """Fixed per-request recurrent SSM and causal conv1d state pool for linear-attention layers.

    Slot 0 is permanently reserved as a zero-initialized padding slot.
    """

    def __init__(self, group: RecurrentStateGroup, max_slots: int, *, device: Any = None) -> None:
        require_int(max_slots, "max_slots", minimum=2)
        spec = group.spec
        torch = require_torch(capability="kv_storage")
        resolved = resolve_device(device, capability="kv_storage")
        self._group = group
        self._max_slots = max_slots
        self._ssm: Tensor = torch.zeros(
            (spec.num_layers, max_slots, spec.num_v_heads, spec.key_head_dim, spec.value_head_dim),
            dtype=spec.state_dtype.torch_dtype,
            device=resolved,
        )
        self._conv: Tensor = torch.zeros(
            (spec.num_layers, max_slots, spec.conv_dim, spec.conv_kernel_size),
            dtype=spec.conv_dtype.torch_dtype,
            device=resolved,
        )

    @property
    def group(self) -> RecurrentStateGroup:
        """The recurrent state group this pool stores."""
        return self._group

    @property
    def max_slots(self) -> int:
        """Total slots allocated, including padding slot 0."""
        return self._max_slots

    @property
    def usable_slots(self) -> int:
        """Slots available to requests."""
        return self._max_slots - 1

    @property
    def device(self) -> Any:
        """The ``torch.device`` the state tensors live on."""
        return self._ssm.device

    @property
    def nbytes(self) -> int:
        """Total bytes allocated across SSM and conv1d states."""
        return int(
            self._ssm.numel() * self._ssm.element_size()
            + self._conv.numel() * self._conv.element_size()
        )

    def buffers(self, layer_id: int) -> RecurrentStateBuffers:
        """Per-layer state views into the pool."""
        index = self._group.layer_index(layer_id)
        return RecurrentStateBuffers(self._ssm[index], self._conv[index])

    def zero_slot(self, slot_id: int) -> None:
        """Reset one slot's SSM and conv1d state across every layer."""
        require_int(slot_id, "slot_id", minimum=0)
        self._ssm[:, slot_id].zero_()
        self._conv[:, slot_id].zero_()

    def snapshot_slot(self, slot_id: int) -> tuple[Tensor, Tensor]:
        """Copy one slot's state across every layer onto CPU for prefix checkpointing."""
        require_int(slot_id, "slot_id", minimum=1)
        return (
            self._ssm[:, slot_id].detach().cpu().clone(),
            self._conv[:, slot_id].detach().cpu().clone(),
        )

    def restore_slot(self, slot_id: int, snapshot: tuple[Tensor, Tensor]) -> None:
        """Restore a CPU state snapshot into device slot ``slot_id``."""
        require_int(slot_id, "slot_id", minimum=1)
        ssm, conv = snapshot
        self._ssm[:, slot_id].copy_(ssm)
        self._conv[:, slot_id].copy_(conv)
