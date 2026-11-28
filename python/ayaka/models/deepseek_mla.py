"""DeepSeek V2/V3 Multi-Head Latent Attention (MLA) causal language models."""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar, Final

from torch import nn

from ayaka.core.errors import ModelLoadError
from ayaka.kvcache.storage.spec import LatentKVGroup, LatentKVSpec
from ayaka.layers.attention.mla import MultiHeadLatentAttention
from ayaka.layers.embedding.token import LMHead, TokenEmbedding
from ayaka.layers.ffn.dense import GatedMLP
from ayaka.layers.ffn.moe import MoEBlock
from ayaka.layers.norm import RMSNorm
from ayaka.layers.rope import RotaryEmbedding
from ayaka.models._moe import expert_parts
from ayaka.models.base import CausalLM
from ayaka.ops.kvcache import RAW_CODEC
from ayaka.weights.mapping import WeightMapping

if TYPE_CHECKING:
    import torch

    from ayaka.configs.model import ModelConfig
    from ayaka.core.types import KVLayoutKind
    from ayaka.layers.context import ForwardContext
    from ayaka.ops.backend import OpsBackend
    from ayaka.utils.dtypes import DType
    from ayaka.weights.quant import QuantFormat

__all__ = ["DeepseekV2ForCausalLM", "DeepseekV3ForCausalLM"]

_RENAMES: Final = (
    (r"model\.embed_tokens\.weight", r"embed_tokens.weight"),
    (r"model\.norm\.weight", r"norm.weight"),
    (
        r"model\.layers\.(\d+)\.(input_layernorm|post_attention_layernorm)\.weight",
        r"layers.\1.\2.weight",
    ),
    (
        r"model\.layers\.(\d+)\.self_attn\."
        r"(q_proj|q_a_proj|q_a_layernorm|q_b_proj|kv_a_proj_with_mqa|kv_a_layernorm|kv_b_proj|o_proj)"
        r"\.weight",
        r"layers.\1.attn.\2.weight",
    ),
    (r"model\.layers\.(\d+)\.mlp\.(gate|up|down)_proj\.weight", r"layers.\1.mlp.\2_proj.weight"),
    (r"model\.layers\.(\d+)\.mlp\.gate\.weight", r"layers.\1.mlp.routed.gate.weight"),
    (
        r"model\.layers\.(\d+)\.mlp\.experts\.(\d+)\.(gate|up|down)_proj\.weight",
        r"layers.\1.mlp.routed.experts.\2.\3_proj.weight",
    ),
    (
        r"model\.layers\.(\d+)\.mlp\.shared_experts\.(gate|up|down)_proj\.weight",
        r"layers.\1.mlp.shared.\2_proj.weight",
    ),
)
_ROTARY_BUFFERS: Final = (r"model\.layers\.\d+\.self_attn\.rotary_emb\.inv_freq",)


class _DeepseekMoE(nn.Module):
    def __init__(
        self,
        config: ModelConfig,
        *,
        ops: OpsBackend,
        dtype: torch.dtype,
        device: torch.device | str | None,
    ) -> None:
        super().__init__()
        self.routed = MoEBlock(
            config.hidden_size,
            config.moe_intermediate_size,
            config.num_experts,
            config.experts_per_token,
            renormalize=config.norm_topk_prob,
            ops=ops,
            dtype=dtype,
            device=device,
        )
        if config.num_shared_experts > 0:
            self.shared: GatedMLP | None = GatedMLP(
                config.hidden_size,
                config.moe_intermediate_size * config.num_shared_experts,
                activation="silu",
                bias=False,
                ops=ops,
                dtype=dtype,
                device=device,
            )
        else:
            self.shared = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.routed(x)
        if self.shared is not None:
            out = out + self.shared(x)
        return out


class _DeepseekBlock(nn.Module):
    def __init__(
        self,
        config: ModelConfig,
        layer_id: int,
        rotary: RotaryEmbedding,
        *,
        ops: OpsBackend,
        dtype: torch.dtype,
        device: torch.device | str | None,
    ) -> None:
        super().__init__()
        self.input_layernorm = RMSNorm(
            config.hidden_size, eps=config.norm_eps, ops=ops, dtype=dtype, device=device
        )
        self.attn = MultiHeadLatentAttention(
            config.hidden_size,
            config.num_attention_heads,
            q_lora_rank=config.q_lora_rank,
            kv_lora_rank=config.kv_lora_rank,
            qk_nope_head_dim=config.qk_nope_head_dim,
            qk_rope_head_dim=config.qk_rope_head_dim,
            v_head_dim=config.v_head_dim,
            layer_id=layer_id,
            rotary=rotary,
            norm_eps=config.norm_eps,
            ops=ops,
            dtype=dtype,
            device=device,
        )
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.norm_eps, ops=ops, dtype=dtype, device=device
        )
        if config.is_moe and layer_id not in config.dense_layers:
            self.mlp: nn.Module = _DeepseekMoE(config, ops=ops, dtype=dtype, device=device)
        else:
            self.mlp = GatedMLP(
                config.hidden_size,
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


class DeepseekV2ForCausalLM(CausalLM):
    """DeepSeek V2 causal language model backed by absorbed Multi-Head Latent Attention."""

    checkpoint_renames: ClassVar[tuple[tuple[str, str], ...]] = _RENAMES
    quantizable: ClassVar[bool] = False

    def __init__(
        self,
        config: ModelConfig,
        *,
        ops: OpsBackend,
        dtype: DType,
        device: torch.device | str | None = None,
        quant: QuantFormat | None = None,
    ) -> None:
        super().__init__(config, ops=ops, dtype=dtype, device=device, quant=quant)
        theta = config.rope_theta
        if theta is None:
            raise ModelLoadError(f"{config.architecture} requires rope_theta")
        if config.kv_lora_rank <= 0 or config.qk_nope_head_dim <= 0 or config.qk_rope_head_dim <= 0:
            raise ModelLoadError(
                f"{config.architecture} requires kv_lora_rank, qk_nope_head_dim, qk_rope_head_dim"
            )
        if config.v_head_dim <= 0:
            raise ModelLoadError(f"{config.architecture} requires v_head_dim > 0")
        torch_dtype = dtype.torch_dtype
        self.rotary = RotaryEmbedding(
            config.qk_rope_head_dim,
            max_positions=config.max_position_embeddings,
            theta=theta,
            scaling=config.rope_scaling,
            interleaved=config.rope_interleave,
            ops=ops,
            device=device,
        )
        self.embed_tokens = TokenEmbedding(
            config.vocab_size, config.hidden_size, ops=ops, dtype=torch_dtype, device=device
        )
        self.layers = nn.ModuleList(
            _DeepseekBlock(config, layer_id, self.rotary, ops=ops, dtype=torch_dtype, device=device)
            for layer_id in range(config.num_layers)
        )
        self.norm = RMSNorm(
            config.hidden_size, eps=config.norm_eps, ops=ops, dtype=torch_dtype, device=device
        )
        self.lm_head = LMHead(
            config.hidden_size, config.vocab_size, ops=ops, dtype=torch_dtype, device=device
        )
        self.tie_weights()

    @classmethod
    def weight_mapping(cls, config: ModelConfig) -> WeightMapping:
        ignore = (
            (*_ROTARY_BUFFERS, r"lm_head\.weight")
            if config.tie_word_embeddings
            else _ROTARY_BUFFERS
        )
        return WeightMapping(renames=cls.checkpoint_renames, ignore=ignore)

    def tie_weights(self) -> None:
        if self.config.tie_word_embeddings:
            self.lm_head.tie_weights(self.embed_tokens)

    def stored_parts(self, name: str) -> tuple[tuple[str, int], ...]:
        if ".routed.experts." in name:
            mapped = name.replace(".mlp.routed.experts.", ".mlp.experts.")
            parts = expert_parts(self.config, mapped)
            if parts:
                return parts
        prefix, dot, suffix = name.rpartition(".")
        if not dot:
            return ()
        if prefix.endswith(".mlp.gate_up_proj"):
            base = prefix.removesuffix("gate_up_proj")
            rows = self.config.intermediate_size
            return ((f"{base}gate_proj.{suffix}", rows), (f"{base}up_proj.{suffix}", rows))
        if prefix.endswith(".shared.gate_up_proj"):
            base = prefix.removesuffix("gate_up_proj")
            rows = self.config.moe_intermediate_size * self.config.num_shared_experts
            return ((f"{base}gate_proj.{suffix}", rows), (f"{base}up_proj.{suffix}", rows))
        return ()

    def kv_specs(
        self,
        *,
        block_size: int,
        layout: KVLayoutKind,
        codec: str = RAW_CODEC,
        kv_scale: float = 1.0,
    ) -> tuple[LatentKVGroup, ...]:
        del layout
        spec = LatentKVSpec(
            num_layers=self.config.num_layers,
            kv_lora_rank=self.config.kv_lora_rank,
            qk_rope_head_dim=self.config.qk_rope_head_dim,
            block_size=block_size,
            dtype=self.compute_dtype,
            codec=codec,
            kv_scale=kv_scale,
        )
        return (LatentKVGroup(tuple(range(self.config.num_layers)), spec),)

    def forward(self, token_ids: torch.Tensor, context: ForwardContext) -> torch.Tensor:
        x = self.embed_tokens(token_ids)
        residual: torch.Tensor | None = None
        for layer in self.layers:
            x, residual = layer(x, residual, context)
        if residual is None:
            return self.norm(x)
        self.norm.fused_add(x, residual)
        return x

    def compute_logits(self, hidden_rows: torch.Tensor) -> torch.Tensor:
        return self.lm_head(hidden_rows)


class DeepseekV3ForCausalLM(DeepseekV2ForCausalLM):
    """DeepSeek V3 causal language model sharing the absorbed MLA architecture."""
