"""Multi-Head Latent Attention (MLA) op contract.

MLA compresses keys and values into a shared low-rank latent vector ``ckv``
paired with a small decoupled rotary key ``kpe``, reducing KV cache bytes per
token by more than half compared with standard GQA/MHA. Importing this module
does not import torch.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from torch import Tensor

    from ayaka.ops.attention import AttentionMetadata
    from ayaka.ops.kvcache import LatentKVBuffers

__all__ = ["MLAOps"]


class MLAOps(Protocol):
    """Writing and attending over compressed latent KV buffers."""

    def store_latent(
        self,
        ckv: Tensor,
        kpe: Tensor,
        buffers: LatentKVBuffers,
        meta: AttentionMetadata,
    ) -> None:
        """Write compressed ``ckv`` and rotary ``kpe`` rows into the paged latent cache.

        Args:
            ckv: Compressed KV latent of shape ``[N, kv_lora_rank]``.
            kpe: Decoupled rotary key of shape ``[N, qk_rope_head_dim]``.
            buffers: Layer latent cache tensors.
            meta: Planned step metadata providing ``slot_mapping``.
        """
        ...

    def attend_mla(
        self,
        q_nope: Tensor,
        q_pe: Tensor,
        buffers: LatentKVBuffers,
        meta: AttentionMetadata,
        out: Tensor,
        *,
        scale: float,
    ) -> None:
        """Attend absorbed queries over the compressed latent KV cache.

        Args:
            q_nope: Absorbed non-rotary queries of shape
                ``[N, num_heads, kv_lora_rank]``.
            q_pe: Rotary queries of shape ``[N, num_heads, qk_rope_head_dim]``.
            buffers: Layer latent cache tensors.
            meta: Planned step metadata.
            out: Output tensor of shape ``[N, num_heads, kv_lora_rank]`` in compute dtype.
            scale: Softmax logit scale factor.
        """
        ...
