"""OpenAI gpt-oss causal language model (alternating sliding/full sinks, YaRN, MXFP4 MoE)."""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar, Final

import torch
from torch import nn
from torch.nn import functional

from ayaka.core.errors import ModelLoadError
from ayaka.kvcache.storage.spec import KVGroup, KVSpec
from ayaka.layers.attention.dense import Attention
from ayaka.layers.embedding.token import LMHead, TokenEmbedding
from ayaka.layers.linear import Linear
from ayaka.layers.norm import RMSNorm
from ayaka.layers.rope import RotaryEmbedding
from ayaka.models.base import CausalLM
from ayaka.ops.kvcache import RAW_CODEC
from ayaka.weights.mapping import WeightMapping
from ayaka.weights.mxfp4 import dequantize_mxfp4, mxfp4_lut

if TYPE_CHECKING:
    from ayaka.configs.model import ModelConfig
    from ayaka.core.types import KVLayoutKind
    from ayaka.layers.context import ForwardContext
    from ayaka.ops.backend import OpsBackend
    from ayaka.utils.dtypes import DType
    from ayaka.weights.quant import QuantFormat

__all__ = ["GptOssForCausalLM"]

_RENAMES: Final = (
    (r"model\.embed_tokens\.weight", r"embed_tokens.weight"),
    (r"model\.norm\.weight", r"norm.weight"),
    (
        r"model\.layers\.(\d+)\.(input_layernorm|post_attention_layernorm)\.weight",
        r"layers.\1.\2.weight",
    ),
    (
        r"model\.layers\.(\d+)\.self_attn\.([qkvo]_proj)\.(weight|bias)",
        r"layers.\1.attn.\2.\3",
    ),
    (r"model\.layers\.(\d+)\.self_attn\.sinks", r"layers.\1.attn.sinks"),
    (r"model\.layers\.(\d+)\.mlp\.router\.(weight|bias)", r"layers.\1.mlp.router.\2"),
    (
        r"model\.layers\.(\d+)\.mlp\.experts\.(gate_up_proj|down_proj|"
        r"gate_up_proj_blocks|gate_up_proj_scales|gate_up_proj_bias|"
        r"down_proj_blocks|down_proj_scales|down_proj_bias)",
        r"layers.\1.mlp.experts.\2",
    ),
)
_ROTARY_BUFFERS: Final = (r"model\.layers\.\d+\.self_attn\.rotary_emb\.inv_freq",)


def _gpt_oss_swiglu(gate_up: torch.Tensor, *, alpha: float, limit: float | None) -> torch.Tensor:
    gate, up = gate_up[..., ::2], gate_up[..., 1::2]
    if limit is not None:
        gate = gate.clamp(max=limit)
        up = up.clamp(min=-limit, max=limit)
    glu = gate * torch.sigmoid(gate * alpha)
    return (up + 1.0) * glu


class _GptOssExperts(nn.Module):
    """Expert FFN block supporting both BF16 dense tensors and MXFP4 packed tensors."""

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_experts: int,
        *,
        alpha: float,
        limit: float | None,
        mxfp4: bool,
        dtype: torch.dtype,
        device: torch.device | str | None,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_experts = num_experts
        self.alpha = alpha
        self.limit = limit
        self.mxfp4 = mxfp4
        self.compute_dtype = dtype

        if mxfp4:
            if hidden_size % 32 or intermediate_size % 32:
                raise ModelLoadError("mxfp4 requires hidden and intermediate sizes divisible by 32")
            in_groups = hidden_size // 32
            mid_groups = intermediate_size // 32
            self.gate_up_proj_blocks = nn.Parameter(
                torch.empty(
                    (num_experts, 2 * intermediate_size, in_groups, 16),
                    dtype=torch.uint8,
                    device=device,
                ),
                requires_grad=False,
            )
            self.gate_up_proj_scales = nn.Parameter(
                torch.empty(
                    (num_experts, 2 * intermediate_size, in_groups),
                    dtype=torch.uint8,
                    device=device,
                ),
                requires_grad=False,
            )
            self.down_proj_blocks = nn.Parameter(
                torch.empty(
                    (num_experts, hidden_size, mid_groups, 16),
                    dtype=torch.uint8,
                    device=device,
                ),
                requires_grad=False,
            )
            self.down_proj_scales = nn.Parameter(
                torch.empty(
                    (num_experts, hidden_size, mid_groups),
                    dtype=torch.uint8,
                    device=device,
                ),
                requires_grad=False,
            )
        else:
            self.gate_up_proj = nn.Parameter(
                torch.empty(
                    (num_experts, hidden_size, 2 * intermediate_size), dtype=dtype, device=device
                ),
                requires_grad=False,
            )
            self.down_proj = nn.Parameter(
                torch.empty(
                    (num_experts, intermediate_size, hidden_size), dtype=dtype, device=device
                ),
                requires_grad=False,
            )

        self.gate_up_proj_bias = nn.Parameter(
            torch.empty((num_experts, 2 * intermediate_size), dtype=dtype, device=device),
            requires_grad=False,
        )
        self.down_proj_bias = nn.Parameter(
            torch.empty((num_experts, hidden_size), dtype=dtype, device=device),
            requires_grad=False,
        )

    def _gate_up_linear(self, x: torch.Tensor, expert_idx: int | torch.Tensor) -> torch.Tensor:
        if self.mxfp4:
            w = dequantize_mxfp4(
                self.gate_up_proj_blocks[expert_idx],
                self.gate_up_proj_scales[expert_idx],
                dtype=x.dtype,
            )
            bias = self.gate_up_proj_bias[expert_idx]
            if w.ndim == 2:
                return functional.linear(x, w, bias)
            return torch.einsum("rkd,rkod->rko", x, w) + bias
        w = self.gate_up_proj[expert_idx]
        bias = self.gate_up_proj_bias[expert_idx]
        if w.ndim == 2:
            return x @ w + bias
        return torch.einsum("rkd,rkdo->rko", x, w) + bias

    def _down_linear(self, x: torch.Tensor, expert_idx: int | torch.Tensor) -> torch.Tensor:
        if self.mxfp4:
            w = dequantize_mxfp4(
                self.down_proj_blocks[expert_idx],
                self.down_proj_scales[expert_idx],
                dtype=x.dtype,
            )
            bias = self.down_proj_bias[expert_idx]
            if w.ndim == 2:
                return functional.linear(x, w, bias)
            return torch.einsum("rkd,rkod->rko", x, w) + bias
        w = self.down_proj[expert_idx]
        bias = self.down_proj_bias[expert_idx]
        if w.ndim == 2:
            return x @ w + bias
        return torch.einsum("rkd,rkdo->rko", x, w) + bias

    def forward(
        self,
        hidden: torch.Tensor,
        router_indices: torch.Tensor,
        routing_weights: torch.Tensor,
    ) -> torch.Tensor:
        rows = hidden.shape[0]
        if rows == 0:
            return hidden
        capturing = hidden.is_cuda and torch.cuda.is_current_stream_capturing()
        if capturing or rows <= 8:
            # Fixed-shape gathered execution avoids any dynamic nonzero() host sync inside graphs.
            x_exp = hidden.unsqueeze(1).expand(-1, router_indices.shape[1], -1)
            gate_up = self._gate_up_linear(x_exp, router_indices)
            gated = _gpt_oss_swiglu(gate_up, alpha=self.alpha, limit=self.limit)
            out = self._down_linear(gated, router_indices)
            return (out * routing_weights.unsqueeze(-1)).sum(dim=1).to(hidden.dtype)

        next_states = torch.zeros_like(hidden)
        with torch.no_grad():
            expert_mask = functional.one_hot(router_indices, num_classes=self.num_experts).permute(
                2, 1, 0
            )
            expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()
        for hit in expert_hit:
            expert_idx = int(hit[0].item())
            if expert_idx >= self.num_experts:
                continue
            top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
            current_state = hidden[token_idx]
            gate_up = self._gate_up_linear(current_state, expert_idx)
            gated = _gpt_oss_swiglu(gate_up, alpha=self.alpha, limit=self.limit)
            out = self._down_linear(gated, expert_idx)
            weighted = out * routing_weights[token_idx, top_k_pos, None]
            next_states.index_add_(0, token_idx, weighted.to(hidden.dtype))
        return next_states


class _GptOssMoE(nn.Module):
    def __init__(
        self,
        config: ModelConfig,
        *,
        mxfp4: bool,
        ops: OpsBackend,
        dtype: torch.dtype,
        device: torch.device | str | None,
    ) -> None:
        super().__init__()
        self.top_k = config.experts_per_token
        self.router = Linear(
            config.hidden_size,
            config.num_experts,
            bias=True,
            ops=ops,
            dtype=dtype,
            device=device,
        )
        self.experts = _GptOssExperts(
            config.hidden_size,
            config.moe_intermediate_size,
            config.num_experts,
            alpha=config.swiglu_alpha,
            limit=config.swiglu_limit,
            mxfp4=mxfp4,
            dtype=dtype,
            device=device,
        )

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        logits = self.router(hidden)
        top_vals, top_idx = torch.topk(logits, self.top_k, dim=-1)
        weights = functional.softmax(top_vals, dim=-1, dtype=top_vals.dtype)
        return self.experts(hidden, top_idx, weights)


class _GptOssBlock(nn.Module):
    def __init__(
        self,
        config: ModelConfig,
        layer_id: int,
        rotary: RotaryEmbedding,
        *,
        window: int | None,
        mxfp4: bool,
        ops: OpsBackend,
        dtype: torch.dtype,
        device: torch.device | str | None,
    ) -> None:
        super().__init__()
        self.input_layernorm = RMSNorm(
            config.hidden_size, eps=config.norm_eps, ops=ops, dtype=dtype, device=device
        )
        self.attn = Attention(
            config.hidden_size,
            config.num_attention_heads,
            config.num_kv_heads,
            config.head_dim,
            layer_id=layer_id,
            bias=config.attention_bias,
            out_bias=config.attention_bias,
            rotary=rotary,
            window=window,
            use_sinks=True,
            sink_dtype=dtype,
            ops=ops,
            dtype=dtype,
            device=device,
        )
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.norm_eps, ops=ops, dtype=dtype, device=device
        )
        self.mlp = _GptOssMoE(config, mxfp4=mxfp4, ops=ops, dtype=dtype, device=device)

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


class GptOssForCausalLM(CausalLM):
    """OpenAI gpt-oss alternating sliding+sink/full+sink MoE causal language model."""

    checkpoint_renames: ClassVar[tuple[tuple[str, str], ...]] = _RENAMES
    quantizable: ClassVar[bool] = True
    supported_quants: ClassVar[tuple[str, ...]] = ("mxfp4",)

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
        if not config.is_moe:
            raise ModelLoadError(f"{config.architecture} requires MoE configuration")
        theta = config.rope_theta
        if theta is None:
            raise ModelLoadError(f"{config.architecture} requires rope_theta")
        types = config.layer_types or tuple(
            "sliding_attention" if idx % 2 == 0 else "full_attention"
            for idx in range(config.num_layers)
        )
        self._sliding_layers = tuple(i for i, t in enumerate(types) if t == "sliding_attention")
        self._full_layers = tuple(i for i, t in enumerate(types) if t == "full_attention")
        mxfp4 = quant is not None and quant.producer == "mxfp4"

        torch_dtype = dtype.torch_dtype
        self.rotary = RotaryEmbedding(
            config.head_dim,
            max_positions=config.max_position_embeddings,
            theta=theta,
            scaling=config.rope_scaling,
            ops=ops,
            device=device,
        )
        self.embed_tokens = TokenEmbedding(
            config.vocab_size, config.hidden_size, ops=ops, dtype=torch_dtype, device=device
        )
        self.layers = nn.ModuleList(
            _GptOssBlock(
                config,
                idx,
                self.rotary,
                window=config.sliding_window if types[idx] == "sliding_attention" else None,
                mxfp4=mxfp4,
                ops=ops,
                dtype=torch_dtype,
                device=device,
            )
            for idx in range(config.num_layers)
        )
        self.norm = RMSNorm(
            config.hidden_size, eps=config.norm_eps, ops=ops, dtype=torch_dtype, device=device
        )
        self.lm_head = LMHead(
            config.hidden_size, config.vocab_size, ops=ops, dtype=torch_dtype, device=device
        )
        if mxfp4 and self.embed_tokens.weight.device.type != "meta":
            mxfp4_lut(self.embed_tokens.weight.device, torch_dtype)
        self.tie_weights()

    def initialize_buffers(self) -> None:
        super().initialize_buffers()
        if self.quant_format is not None and self.quant_format.producer == "mxfp4":
            mxfp4_lut(self.embed_tokens.weight.device, self.compute_dtype.torch_dtype)

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
        prefix, dot, suffix = name.rpartition(".")
        if not dot or not prefix.endswith(".attn.qkv_proj"):
            return ()
        base = prefix.removesuffix("qkv_proj")
        q_rows = self.config.num_attention_heads * self.config.head_dim
        kv_rows = self.config.num_kv_heads * self.config.head_dim
        return (
            (f"{base}q_proj.{suffix}", q_rows),
            (f"{base}k_proj.{suffix}", kv_rows),
            (f"{base}v_proj.{suffix}", kv_rows),
        )

    def kv_specs(
        self,
        *,
        block_size: int,
        layout: KVLayoutKind,
        codec: str = RAW_CODEC,
        kv_scale: float = 1.0,
    ) -> tuple[KVGroup, ...]:
        groups: list[KVGroup] = []
        if self._sliding_layers:
            groups.append(
                KVGroup(
                    self._sliding_layers,
                    KVSpec(
                        num_layers=len(self._sliding_layers),
                        num_kv_heads=self.config.num_kv_heads,
                        head_dim=self.config.head_dim,
                        block_size=block_size,
                        dtype=self.compute_dtype,
                        layout=layout,
                        codec=codec,
                        kv_scale=kv_scale,
                        window=self.config.sliding_window,
                    ),
                )
            )
        if self._full_layers:
            groups.append(
                KVGroup(
                    self._full_layers,
                    KVSpec(
                        num_layers=len(self._full_layers),
                        num_kv_heads=self.config.num_kv_heads,
                        head_dim=self.config.head_dim,
                        block_size=block_size,
                        dtype=self.compute_dtype,
                        layout=layout,
                        codec=codec,
                        kv_scale=kv_scale,
                    ),
                )
            )
        return tuple(groups)

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
