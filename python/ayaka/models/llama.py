"""Llama: rotary positions, grouped-query attention, RMSNorm and a SiLU-gated feed-forward.

The block is pre-norm with the residual kept apart, so each norm also adds::

    h, residual = input_layernorm(x + residual)          # fused add + RMSNorm
    h = attn(h)                                          # rotary q and k, GQA, no bias
    h, residual = post_attention_layernorm(h + residual)
    h = mlp(h)                                           # down(silu(gate) * up)
    ...
    hidden = norm(h + residual)                          # logits = hidden @ lm_head.T

This is the arithmetic of Hugging Face's ``LlamaForCausalLM``. The config knobs
it reads are ``rope_theta``, ``rope_scaling`` (``linear``, ``llama3``, ``yarn``),
``attention_bias``, ``mlp_bias``, ``head_dim`` and ``tie_word_embeddings``;
``hidden_act`` must be ``silu``. A ``partial_rotary_factor`` below 1 is refused
when the config is read (``ModelConfig.from_hf_config``).

Checkpoint layout: Hugging Face stores ``q_proj``, ``k_proj`` and ``v_proj``
separately, and ``gate_proj`` and ``up_proj``; this model fuses each group into
one :class:`~ayaka.layers.linear.Linear` (``qkv_proj``, ``gate_up_proj``) and
lists the parts with :meth:`LlamaForCausalLM.stored_parts`. Projections are
``[out, in]`` in the checkpoint already, so nothing is transposed. The query and
key weights are in the Hugging Face order, which rotates halves (not
interleaved pairs).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

from torch import nn

from ayaka.core.errors import ModelLoadError
from ayaka.layers.attention.dense import Attention
from ayaka.layers.embedding.token import LMHead, TokenEmbedding
from ayaka.layers.ffn.dense import GatedMLP
from ayaka.layers.norm import RMSNorm
from ayaka.layers.rope import RotaryEmbedding
from ayaka.models.base import CausalLM
from ayaka.weights.mapping import WeightMapping

if TYPE_CHECKING:
    import torch

    from ayaka.configs.model import ModelConfig
    from ayaka.layers.context import ForwardContext
    from ayaka.ops.backend import OpsBackend
    from ayaka.utils.dtypes import DType

__all__ = ["LlamaForCausalLM"]

_RENAMES: Final = (
    (r"model\.embed_tokens\.weight", r"embed_tokens.weight"),
    (r"model\.norm\.weight", r"norm.weight"),
    (
        r"model\.layers\.(\d+)\.(input_layernorm|post_attention_layernorm)\.weight",
        r"layers.\1.\2.weight",
    ),
    (
        r"model\.layers\.(\d+)\.self_attn\.([qkvo])_proj\.(weight|bias)",
        r"layers.\1.attn.\2_proj.\3",
    ),
    (r"model\.layers\.(\d+)\.mlp\.(gate|up|down)_proj\.(weight|bias)", r"layers.\1.mlp.\2_proj.\3"),
)
#: Older conversions store each layer's inverse frequencies; ours are computed.
_ROTARY_BUFFERS: Final = (r"model\.layers\.\d+\.self_attn\.rotary_emb\.inv_freq",)


class _Block(nn.Module):
    def __init__(
        self,
        config: ModelConfig,
        layer_id: int,
        *,
        rotary: RotaryEmbedding,
        ops: OpsBackend,
        dtype: torch.dtype,
        device: torch.device | str | None,
    ) -> None:
        super().__init__()
        hidden = config.hidden_size
        self.input_layernorm = RMSNorm(
            hidden, eps=config.norm_eps, ops=ops, dtype=dtype, device=device
        )
        self.attn = Attention(
            hidden,
            config.num_attention_heads,
            config.num_kv_heads,
            config.head_dim,
            layer_id=layer_id,
            bias=config.attention_bias,
            rotary=rotary,
            ops=ops,
            dtype=dtype,
            device=device,
        )
        self.post_attention_layernorm = RMSNorm(
            hidden, eps=config.norm_eps, ops=ops, dtype=dtype, device=device
        )
        self.mlp = GatedMLP(
            hidden,
            config.intermediate_size,
            activation="silu",
            bias=config.mlp_bias,
            ops=ops,
            dtype=dtype,
            device=device,
        )

    def forward(
        self, x: torch.Tensor, residual: torch.Tensor | None, context: ForwardContext
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            residual = x
            h = self.input_layernorm(x)
        else:
            self.input_layernorm.fused_add(x, residual)
            h = x
        h = self.attn(h, context)
        self.post_attention_layernorm.fused_add(h, residual)
        return self.mlp(h), residual


class LlamaForCausalLM(CausalLM):
    """Llama (and the checkpoints that share its layout) with a language-model head.

    Parameters are ``embed_tokens``,
    ``layers.N.{input_layernorm, attn.qkv_proj, attn.o_proj, post_attention_layernorm,
    mlp.gate_up_proj, mlp.down_proj}``, ``norm`` and ``lm_head``. The head shares
    ``embed_tokens``' table when the config ties them. One
    :class:`~ayaka.layers.rope.RotaryEmbedding`, ``rotary``, serves every layer.

    Raises:
        ModelLoadError: If the config is not one this implementation runs: no
            ``rope_theta``, an activation other than ``silu``, or an odd ``head_dim``.
    """

    def __init__(
        self,
        config: ModelConfig,
        *,
        ops: OpsBackend,
        dtype: DType,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__(config, ops=ops, dtype=dtype, device=device)
        _check_supported(config)
        torch_dtype = dtype.torch_dtype
        hidden = config.hidden_size
        theta = config.rope_theta
        if theta is None:  # _check_supported refused this; the branch narrows the type
            raise ModelLoadError("Llama needs rope_theta")
        self.rotary = RotaryEmbedding(
            config.head_dim,
            max_positions=config.max_position_embeddings,
            theta=theta,
            scaling=config.rope_scaling,
            ops=ops,
            device=device,
        )
        self.embed_tokens = TokenEmbedding(
            config.vocab_size, hidden, ops=ops, dtype=torch_dtype, device=device
        )
        self.layers = nn.ModuleList(
            _Block(config, layer_id, rotary=self.rotary, ops=ops, dtype=torch_dtype, device=device)
            for layer_id in range(config.num_layers)
        )
        self.norm = RMSNorm(hidden, eps=config.norm_eps, ops=ops, dtype=torch_dtype, device=device)
        self.lm_head = LMHead(hidden, config.vocab_size, ops=ops, dtype=torch_dtype, device=device)
        self.tie_weights()

    @classmethod
    def weight_mapping(cls, config: ModelConfig) -> WeightMapping:
        """Names in a Hugging Face ``LlamaForCausalLM`` checkpoint."""
        ignore = (
            (*_ROTARY_BUFFERS, r"lm_head\.weight")
            if config.tie_word_embeddings
            else _ROTARY_BUFFERS
        )
        return WeightMapping(renames=_RENAMES, ignore=ignore)

    def tie_weights(self) -> None:
        """Share ``embed_tokens``' table with ``lm_head`` when the config ties them."""
        if self.config.tie_word_embeddings:
            self.lm_head.tie_weights(self.embed_tokens)

    def stored_parts(self, name: str) -> tuple[tuple[str, int], ...]:
        """``qkv_proj`` is stored as ``q_proj``, ``k_proj``, ``v_proj``; ``gate_up_proj`` as two."""
        config = self.config
        prefix, dot, suffix = name.rpartition(".")
        if not dot:
            return ()
        if prefix.endswith(".attn.qkv_proj"):
            base = prefix.removesuffix("qkv_proj")
            q_rows = config.num_attention_heads * config.head_dim
            kv_rows = config.num_kv_heads * config.head_dim
            return (
                (f"{base}q_proj.{suffix}", q_rows),
                (f"{base}k_proj.{suffix}", kv_rows),
                (f"{base}v_proj.{suffix}", kv_rows),
            )
        if prefix.endswith(".mlp.gate_up_proj"):
            base = prefix.removesuffix("gate_up_proj")
            rows = config.intermediate_size
            return ((f"{base}gate_proj.{suffix}", rows), (f"{base}up_proj.{suffix}", rows))
        return ()

    def forward(self, token_ids: torch.Tensor, context: ForwardContext) -> torch.Tensor:
        """Embed the tokens, run the blocks and normalise; positions only rotate q and k.

        Returns ``[N, hidden_size]``; see :meth:`CausalLM.forward`.
        """
        x = self.embed_tokens(token_ids)
        residual: torch.Tensor | None = None
        for layer in self.layers:
            x, residual = layer(x, residual, context)
        if residual is None:  # a model with no layers
            return self.norm(x)
        self.norm.fused_add(x, residual)
        return x

    def compute_logits(self, hidden_rows: torch.Tensor) -> torch.Tensor:
        """Project ``[R, hidden_size]`` rows onto the vocabulary."""
        return self.lm_head(hidden_rows)


def _check_supported(config: ModelConfig) -> None:
    problems: list[str] = []
    if config.rope_theta is None:
        problems.append("it has no rope_theta")
    if config.hidden_act != "silu":
        problems.append(f"hidden_act is {config.hidden_act!r}, not 'silu'")
    if config.head_dim % 2:
        problems.append(f"head_dim {config.head_dim} is odd, so it cannot be rotated in pairs")
    if problems:
        raise ModelLoadError(
            f"cannot build {config.architecture} from this config: " + "; ".join(problems),
            context={"architecture": config.architecture},
        )
