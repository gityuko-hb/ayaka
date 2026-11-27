"""Linear-attention (Gated DeltaNet and Kimi Delta Attention) op contract.

Defines the layer-facing contract for recurrent state buffers, causal 1D
depthwise convolutions, and delta-rule state recurrences. Importing this module
does not import torch.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, NamedTuple, Protocol

if TYPE_CHECKING:
    from torch import Tensor

__all__ = ["LinearAttentionMetadata", "LinearAttentionOps", "RecurrentStateBuffers"]


class RecurrentStateBuffers(NamedTuple):
    """Per-layer recurrent state tensors indexed by request slot.

    Slot 0 is always reserved as a padding slot so padded batch rows write
    safely without branching.

    Attributes:
        ssm_state: Recurrent matrix state of shape
            ``[max_slots, num_v_heads, key_head_dim, value_head_dim]`` in fp32.
        conv_state: Causal 1D convolution ring buffer of shape
            ``[max_slots, conv_dim, conv_kernel_size]`` in compute dtype.
    """

    ssm_state: Tensor
    conv_state: Tensor

    @property
    def max_slots(self) -> int:
        """Total state slots allocated, including padding slot 0."""
        return int(self.ssm_state.shape[0])

    @property
    def num_v_heads(self) -> int:
        """Number of value heads per layer."""
        return int(self.ssm_state.shape[1])

    @property
    def key_head_dim(self) -> int:
        """Key head dimension."""
        return int(self.ssm_state.shape[2])

    @property
    def value_head_dim(self) -> int:
        """Value head dimension."""
        return int(self.ssm_state.shape[3])


@dataclass(frozen=True, slots=True, eq=False)
class LinearAttentionMetadata:
    """Step metadata for recurrent linear-attention layers.

    Attributes:
        is_prefill: True for ragged prefill steps, False for single-row decode steps.
        cu_seqlens: ``[B + 1]`` int32/int64 cumulative query row offsets for ragged
            prefill; ignored in single-row decode.
        state_slots: ``[B]`` int32/int64 slot index per request (0 for padding rows).
        rows_per_request: ``[B]`` int32/int64 query row counts per request.
    """

    is_prefill: bool
    cu_seqlens: Tensor
    state_slots: Tensor
    rows_per_request: Tensor


class LinearAttentionOps(Protocol):
    """Causal convolution and delta-rule recurrence ops over recurrent state slots."""

    def causal_conv1d(
        self,
        x: Tensor,
        weight: Tensor,
        conv_state: Tensor,
        state_slots: Tensor,
        cu_seqlens: Tensor,
        *,
        is_prefill: bool,
        out: Tensor,
        bias: Tensor | None = None,
    ) -> None:
        """Apply depthwise causal 1D convolution with SiLU and update ``conv_state`` in place.

        Args:
            x: Input activations of shape ``[N, conv_dim]``.
            weight: Depthwise convolution weights of shape ``[conv_dim, kernel_size]``
                (or ``[conv_dim, 1, kernel_size]``).
            conv_state: State buffer ``[max_slots, conv_dim, kernel_size]``, updated in place
                at ``state_slots``.
            state_slots: ``[B]`` slot ids; slot 0 is a padding request whose state updates
                stay isolated in slot 0 and whose outputs are zeroed.
            cu_seqlens: ``[B + 1]`` cumulative row offsets for prefill.
            is_prefill: Prefill vs single-row decode mode.
            out: Preallocated output tensor ``[N, conv_dim]`` (may equal ``x`` only when
                specified; callers normally pass a separate tensor).
            bias: Optional ``[conv_dim]`` bias vector.
        """
        ...

    def gated_delta_rule(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        g: Tensor,
        beta: Tensor,
        ssm_state: Tensor,
        state_slots: Tensor,
        cu_seqlens: Tensor,
        *,
        is_prefill: bool,
        scale: float | None,
        use_qk_l2norm: bool,
        out: Tensor,
    ) -> None:
        """Run Gated DeltaNet recurrence and update ``ssm_state`` in place.

        Args:
            q: Queries of shape ``[N, num_k_heads, key_head_dim]``.
            k: Keys of shape ``[N, num_k_heads, key_head_dim]``.
            v: Values of shape ``[N, num_v_heads, value_head_dim]``.
            g: Scalar log-decay per value head of shape ``[N, num_v_heads]`` (float32).
            beta: Update gate of shape ``[N, num_v_heads]`` in ``(0, 1)``.
            ssm_state: Recurrent state ``[max_slots, num_v_heads, key_head_dim, value_head_dim]``
                in float32, updated in place at ``state_slots``.
            state_slots: ``[B]`` slot ids (0 for padding).
            cu_seqlens: ``[B + 1]`` cumulative row offsets for prefill.
            is_prefill: Prefill vs single-row decode mode.
            scale: Query scaling factor; ``None`` uses ``key_head_dim ** -0.5``.
            use_qk_l2norm: Whether to L2-normalize ``q`` and ``k`` along the head dimension.
            out: Preallocated output tensor ``[N, num_v_heads, value_head_dim]``.
        """
        ...

    def kda_recurrence(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        g: Tensor,
        beta: Tensor,
        ssm_state: Tensor,
        state_slots: Tensor,
        cu_seqlens: Tensor,
        *,
        is_prefill: bool,
        scale: float | None,
        use_qk_l2norm: bool,
        out: Tensor,
    ) -> None:
        """Run Kimi Delta Attention recurrence with per-key-channel log-decay ``g``.

        Same contract as :meth:`gated_delta_rule`, except ``g`` has shape
        ``[N, num_v_heads, key_head_dim]``.
        """
        ...
