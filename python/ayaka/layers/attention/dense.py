"""Multi-head and grouped-query attention over the paged KV cache."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import nn

from ayaka.layers.linear import Linear

if TYPE_CHECKING:
    from ayaka.layers.context import ForwardContext
    from ayaka.ops.backend import OpsBackend

__all__ = ["Attention"]


class Attention(nn.Module):
    """Project to queries, keys and values, write the cache, attend, project back.

    ``qkv_proj`` is one :class:`~ayaka.layers.linear.Linear` whose output rows
    are the query rows, then the key rows, then the value rows. ``o_proj``
    maps the heads back to ``hidden_size``. With ``num_kv_heads < num_heads``
    several query heads share one key/value head (grouped-query attention);
    equal counts are plain multi-head attention.

    The layer holds no cache. It reads the tensors of ``layer_id`` from the
    :class:`~ayaka.layers.context.ForwardContext` of the step, stores this
    step's keys and values there, then attends over everything the step's
    requests have cached. A step may carry several rows per request, so
    prefill, decode and verification all go through :meth:`forward`.

    Args:
        hidden_size: Width of the input and the output.
        num_heads: Query heads.
        num_kv_heads: Key/value heads; divides ``num_heads``.
        head_dim: Width of a head.
        layer_id: Which entry of the context's cache tensors this layer owns.
        bias: Whether ``qkv_proj`` and ``o_proj`` have a bias.
        ops: The backend whose ``kvcache``, ``attention`` and ``linear`` ops do the work.
        dtype: Compute dtype of the parameters.
        device: Where the parameters live.

    Raises:
        ValueError: If ``num_kv_heads`` does not divide ``num_heads``.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        *,
        layer_id: int,
        bias: bool = True,
        ops: OpsBackend,
        dtype: torch.dtype,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__()
        if num_heads % num_kv_heads:
            raise ValueError(f"num_kv_heads ({num_kv_heads}) must divide num_heads ({num_heads})")
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.layer_id = layer_id
        self.scale = head_dim**-0.5
        self.qkv_proj = Linear(
            hidden_size,
            (num_heads + 2 * num_kv_heads) * head_dim,
            bias=bias,
            ops=ops,
            dtype=dtype,
            device=device,
        )
        self.o_proj = Linear(
            num_heads * head_dim, hidden_size, bias=bias, ops=ops, dtype=dtype, device=device
        )
        self._kvcache = ops.kvcache
        self._attention = ops.attention

    def forward(self, hidden: torch.Tensor, context: ForwardContext) -> torch.Tensor:
        """Attend for the rows of one step.

        Args:
            hidden: ``[N, hidden_size]``, the rows of every request in the step, in
                request order, as planned in ``context``.
            context: The step's forward context.

        Returns:
            ``[N, hidden_size]``.
        """
        rows = hidden.shape[0]
        q_width, kv_width = self.num_heads * self.head_dim, self.num_kv_heads * self.head_dim
        q, k, v = self.qkv_proj(hidden).split([q_width, kv_width, kv_width], dim=-1)
        # The ops take dense tensors; ``split`` returns strided views.
        q = q.reshape(rows, self.num_heads, self.head_dim).contiguous()
        k = k.reshape(rows, self.num_kv_heads, self.head_dim).contiguous()
        v = v.reshape(rows, self.num_kv_heads, self.head_dim).contiguous()

        buffers = context.buffers(self.layer_id)
        self._kvcache.store(k, v, buffers, context.attention)
        attended = torch.empty_like(q)
        self._attention.attend(q, buffers, context.attention, attended, scale=self.scale)
        return self.o_proj(attended.reshape(rows, q_width))
