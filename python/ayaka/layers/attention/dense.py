"""Multi-head and grouped-query attention over the paged KV cache."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import torch
from torch import nn

from ayaka.layers.linear import Linear
from ayaka.layers.norm import RMSNorm

if TYPE_CHECKING:
    from ayaka.layers.context import ForwardContext
    from ayaka.layers.quantization.method import QuantMethod
    from ayaka.layers.rope import RotaryEmbedding
    from ayaka.ops.backend import OpsBackend

__all__ = ["Attention", "QKNorm"]

type QKNorm = Literal["head", "full"]
"""Where an attention layer normalises its queries and keys before the rotary embedding.

``"head"`` is one RMSNorm of width ``head_dim`` shared by every head (Qwen3); ``"full"`` is
one RMSNorm over all the query rows and one over all the key rows of a token (OLMoE).
"""


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
    prefill, decode and verification all go through :meth:`forward`. With a
    rotary embedding, queries and keys are rotated by ``context.positions``
    before the keys are stored, so the cache holds rotated keys. With ``qk_norm``,
    queries and keys are RMS-normalised (``q_norm``, ``k_norm``) before that rotation.

    Args:
        hidden_size: Width of the input and the output.
        num_heads: Query heads.
        num_kv_heads: Key/value heads; divides ``num_heads``.
        head_dim: Width of a head.
        layer_id: Which entry of the context's cache tensors this layer owns.
        bias: Whether ``qkv_proj`` has a bias.
        out_bias: Whether ``o_proj`` has a bias; ``None`` follows ``bias``.
        rotary: The model's rotary embedding, the same module in every layer; ``None``
            for a model with learned position embeddings.
        qk_norm: Normalise queries and keys per head or over the whole projection
            (:data:`QKNorm`), or ``None`` for neither.
        norm_eps: Epsilon of those norms; read only with ``qk_norm``.
        ops: The backend whose ``kvcache``, ``attention`` and ``linear`` ops do the work.
        dtype: Compute dtype of the parameters.
        device: Where the parameters live.
        quant: How ``qkv_proj`` and ``o_proj`` store and multiply their weights; ``None``
            for the compute dtype.

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
        out_bias: bool | None = None,
        rotary: RotaryEmbedding | None = None,
        qk_norm: QKNorm | None = None,
        norm_eps: float = 1e-6,
        window: int | None = None,
        use_sinks: bool = False,
        sink_dtype: torch.dtype | None = None,
        ops: OpsBackend,
        dtype: torch.dtype,
        device: torch.device | str | None = None,
        quant: QuantMethod | None = None,
    ) -> None:
        super().__init__()
        if num_heads % num_kv_heads:
            raise ValueError(f"num_kv_heads ({num_kv_heads}) must divide num_heads ({num_heads})")
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.layer_id = layer_id
        self.scale = head_dim**-0.5
        self.window = window
        self.qkv_proj = Linear(
            hidden_size,
            (num_heads + 2 * num_kv_heads) * head_dim,
            bias=bias,
            ops=ops,
            dtype=dtype,
            device=device,
            quant=quant,
        )
        self.o_proj = Linear(
            num_heads * head_dim,
            hidden_size,
            bias=bias if out_bias is None else out_bias,
            ops=ops,
            dtype=dtype,
            device=device,
            quant=quant,
        )
        self.qk_norm = qk_norm
        self.q_norm: RMSNorm | None = None
        self.k_norm: RMSNorm | None = None
        if qk_norm is not None:
            q_norm_width, k_norm_width = (
                (head_dim, head_dim)
                if qk_norm == "head"
                else (num_heads * head_dim, num_kv_heads * head_dim)
            )
            self.q_norm = RMSNorm(q_norm_width, eps=norm_eps, ops=ops, dtype=dtype, device=device)
            self.k_norm = RMSNorm(k_norm_width, eps=norm_eps, ops=ops, dtype=dtype, device=device)
        # Every layer of a model holds the same module; it has no parameters and its table is
        # a non-persistent buffer, so sharing it adds nothing to a checkpoint.
        self.rotary = rotary
        self.sinks: nn.Parameter | None = (
            nn.Parameter(
                torch.empty(num_heads, dtype=sink_dtype or dtype, device=device),
                requires_grad=False,
            )
            if use_sinks
            else None
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
        if self.q_norm is not None and self.k_norm is not None and self.qk_norm == "full":
            q, k = self.q_norm(q.contiguous()), self.k_norm(k.contiguous())
        # The ops take dense tensors; ``split`` returns strided views.
        q = q.reshape(rows, self.num_heads, self.head_dim).contiguous()
        k = k.reshape(rows, self.num_kv_heads, self.head_dim).contiguous()
        v = v.reshape(rows, self.num_kv_heads, self.head_dim).contiguous()
        if self.q_norm is not None and self.k_norm is not None and self.qk_norm == "head":
            q, k = self.q_norm(q), self.k_norm(k)
        if self.rotary is not None:
            self.rotary(context.positions, q, k)

        buffers = context.buffers(self.layer_id)
        meta = context.attention_for(self.layer_id)
        self._kvcache.store(k, v, buffers, meta)
        attended = torch.empty_like(q)
        self._attention.attend(
            q,
            buffers,
            meta,
            attended,
            scale=self.scale,
            window=self.window,
            sinks=self.sinks,
        )
        return self.o_proj(attended.reshape(rows, q_width))
