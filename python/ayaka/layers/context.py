"""The forward context: what one model step tells every layer.

A step is planned once, by the model runner, and then run through every layer.
The runner builds a :class:`ForwardContext` from the planned
:class:`~ayaka.ops.attention.AttentionMetadata` and the cache tensors of each
layer, and passes it down. A layer never builds or edits one, and nothing in it
changes during the step.

This module does not import torch.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import TYPE_CHECKING

from ayaka.core.errors import EngineError

if TYPE_CHECKING:
    from collections.abc import Mapping

    from torch import Tensor

    from ayaka.core.types import ForwardMode
    from ayaka.ops.attention import AttentionMetadata
    from ayaka.ops.kvcache import KVBuffers, LatentKVBuffers
    from ayaka.ops.linear_attention import LinearAttentionMetadata, RecurrentStateBuffers

__all__ = ["ForwardContext", "StateCheckpointCapture"]

_EMPTY_PROXY: Mapping[int, object] = MappingProxyType({})


def _empty_attention_map() -> Mapping[int, AttentionMetadata]:
    return {}


def _empty_latent_map() -> Mapping[int, LatentKVBuffers]:
    return {}


def _empty_state_map() -> Mapping[int, RecurrentStateBuffers]:
    return {}


@dataclass(frozen=True, slots=True, eq=False)
class StateCheckpointCapture:
    """Prefill token offsets where linear-attention layers should snapshot state.

    Attributes:
        boundaries: Per-request sequence of ``(token_offset_within_chunk, digest)``
            pairs (strictly ascending token offsets in ``1..chunk_len``).
        snapshots: Mutable sink populated during the forward pass, keyed by
            ``(layer_id, digest)`` -> ``(ssm_state_tensor, conv_state_tensor)``.
    """

    boundaries: tuple[tuple[tuple[int, bytes], ...], ...]
    snapshots: dict[tuple[int, bytes], tuple[Tensor, Tensor]]


@dataclass(frozen=True, slots=True, eq=False)
class ForwardContext:
    """An immutable snapshot of one step's metadata and cache tensors.

    The step's slot mapping, block tables and the query and context lengths of
    each request are in ``attention`` (see
    :class:`~ayaka.ops.attention.AttentionMetadata` for their meaning and for
    the paged layout). Multi-group models also supply ``layer_attention``,
    ``latent_kv_buffers``, ``state_buffers``, and ``linear_attention``. The context
    itself is frozen and its mappings are read-only.

    Attributes:
        attention: Default planned step (group 0).
        kv_buffers: Standard paged KV cache tensors by layer id.
        layer_attention: Optional per-layer attention metadata overriding
            ``attention`` for models with multiple KV or latent KV groups.
        latent_kv_buffers: Compressed MLA latent KV tensors by layer id.
        state_buffers: Recurrent linear-attention state tensors by layer id.
        linear_attention: Step metadata for recurrent linear-attention layers.
        checkpoint_capture: Optional prefill checkpoint boundary hook.
        input_embeds: Optional precomputed token/vision embeddings ``[N, hidden]``.
    """

    attention: AttentionMetadata
    kv_buffers: Mapping[int, KVBuffers]
    layer_attention: Mapping[int, AttentionMetadata] = field(default_factory=_empty_attention_map)
    latent_kv_buffers: Mapping[int, LatentKVBuffers] = field(default_factory=_empty_latent_map)
    state_buffers: Mapping[int, RecurrentStateBuffers] = field(default_factory=_empty_state_map)
    linear_attention: LinearAttentionMetadata | None = None
    checkpoint_capture: StateCheckpointCapture | None = None
    input_embeds: Tensor | None = None

    def __post_init__(self) -> None:
        # Private read-only copies: the runner may keep reusing the dicts it built this from.
        object.__setattr__(self, "kv_buffers", MappingProxyType(dict(self.kv_buffers)))
        object.__setattr__(self, "layer_attention", MappingProxyType(dict(self.layer_attention)))
        object.__setattr__(
            self, "latent_kv_buffers", MappingProxyType(dict(self.latent_kv_buffers))
        )
        object.__setattr__(self, "state_buffers", MappingProxyType(dict(self.state_buffers)))

    @property
    def mode(self) -> ForwardMode:
        """How the step uses its rows."""
        return self.attention.mode

    @property
    def positions(self) -> Tensor:
        """``[N]`` int32 position of each row in its sequence, for position embeddings."""
        return self.attention.positions

    def attention_for(self, layer_id: int) -> AttentionMetadata:
        """Return the attention metadata governing ``layer_id``."""
        return self.layer_attention.get(layer_id, self.attention)

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

    def latent_buffers(self, layer_id: int) -> LatentKVBuffers:
        """Return the latent MLA cache tensors of one layer."""
        try:
            return self.latent_kv_buffers[layer_id]
        except KeyError:
            raise EngineError(
                f"the forward context has no latent KV buffers for layer {layer_id}; "
                f"it has them for layers {sorted(self.latent_kv_buffers)}",
                context={"layer_id": layer_id},
            ) from None

    def state_for(self, layer_id: int) -> RecurrentStateBuffers:
        """Return the recurrent state tensors of one linear-attention layer."""
        try:
            return self.state_buffers[layer_id]
        except KeyError:
            raise EngineError(
                f"the forward context has no state buffers for layer {layer_id}; "
                f"it has them for layers {sorted(self.state_buffers)}",
                context={"layer_id": layer_id},
            ) from None
