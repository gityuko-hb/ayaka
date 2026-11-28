"""Qwen3.5 hybrid linear-attention (Gated DeltaNet) and full-attention language model."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, ClassVar, Final

import torch
from torch import nn

from ayaka.core.errors import ModelLoadError
from ayaka.kvcache.storage.spec import (
    KVGroup,
    KVSpec,
    RecurrentStateGroup,
    RecurrentStateSpec,
)
from ayaka.layers.attention.linear import GatedDeltaNetAttention
from ayaka.layers.embedding.token import LMHead, TokenEmbedding
from ayaka.layers.ffn.dense import GatedMLP
from ayaka.layers.linear import Linear
from ayaka.layers.norm import RMSNorm
from ayaka.layers.rope import RotaryEmbedding
from ayaka.models._vision_common import VisionLifecycle
from ayaka.models._vision_qwen3_5 import Qwen3_5VisionTower
from ayaka.models.base import CausalLM
from ayaka.ops.kvcache import RAW_CODEC
from ayaka.utils.dtypes import DType
from ayaka.weights.mapping import WeightMapping

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from ayaka.configs.model import ModelConfig
    from ayaka.core.multimodal import ImageInput, ImageSpan
    from ayaka.core.types import KVLayoutKind
    from ayaka.layers.context import ForwardContext
    from ayaka.ops.backend import OpsBackend
    from ayaka.weights.quant import QuantFormat

__all__ = ["Qwen3_5ForConditionalGeneration"]

_PREFIX: Final = r"model\.(?:language_model\.)?"
_RENAMES: Final = (
    (rf"{_PREFIX}embed_tokens\.weight", r"embed_tokens.weight"),
    (rf"{_PREFIX}norm\.weight", r"norm.weight"),
    (
        rf"{_PREFIX}layers\.(\d+)\.(input_layernorm|post_attention_layernorm)\.weight",
        r"layers.\1.\2.weight",
    ),
    (
        rf"{_PREFIX}layers\.(\d+)\.self_attn\.(q_proj|k_proj|v_proj|o_proj|q_norm|k_norm)\.weight",
        r"layers.\1.attn.\2.weight",
    ),
    (
        rf"{_PREFIX}layers\.(\d+)\.linear_attn\.(in_proj_qkv|in_proj_z|in_proj_a|in_proj_b|out_proj)\.weight",
        r"layers.\1.attn.\2.weight",
    ),
    (rf"{_PREFIX}layers\.(\d+)\.linear_attn\.conv1d\.weight", r"layers.\1.attn.conv1d_weight"),
    (rf"{_PREFIX}layers\.(\d+)\.linear_attn\.A_log", r"layers.\1.attn.a_log"),
    (rf"{_PREFIX}layers\.(\d+)\.linear_attn\.dt_bias", r"layers.\1.attn.dt_bias"),
    (rf"{_PREFIX}layers\.(\d+)\.linear_attn\.norm\.weight", r"layers.\1.attn.norm_weight"),
    (
        rf"{_PREFIX}layers\.(\d+)\.mlp\.(gate|up|down)_proj\.weight",
        r"layers.\1.mlp.\2_proj.weight",
    ),
)
_IGNORE: Final = (
    r"model\.visual\..*",
    r"mtp\..*",
    rf"{_PREFIX}layers\.\d+\.self_attn\.rotary_emb\.inv_freq",
)


class _Qwen35FullAttention(nn.Module):
    """Full GQA attention with optional Qwen3.5 sigmoid output gating and partial RoPE."""

    def __init__(
        self,
        config: ModelConfig,
        *,
        layer_id: int,
        rotary: RotaryEmbedding,
        ops: OpsBackend,
        dtype: torch.dtype,
        device: torch.device | str | None,
    ) -> None:
        super().__init__()
        self.layer_id = layer_id
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_kv_heads
        self.head_dim = config.head_dim
        self.output_gate = config.attn_output_gate
        self.scale = config.head_dim**-0.5
        self.rotary = rotary
        self.mrope_section = config.mrope_section or (11, 11, 10)

        q_mult = 2 if config.attn_output_gate else 1
        self.q_proj = Linear(
            config.hidden_size,
            q_mult * config.num_attention_heads * config.head_dim,
            bias=config.attention_bias,
            ops=ops,
            dtype=dtype,
            device=device,
        )
        self.k_proj = Linear(
            config.hidden_size,
            config.num_kv_heads * config.head_dim,
            bias=config.attention_bias,
            ops=ops,
            dtype=dtype,
            device=device,
        )
        self.v_proj = Linear(
            config.hidden_size,
            config.num_kv_heads * config.head_dim,
            bias=config.attention_bias,
            ops=ops,
            dtype=dtype,
            device=device,
        )
        self.o_proj = Linear(
            config.num_attention_heads * config.head_dim,
            config.hidden_size,
            bias=config.attention_bias,
            ops=ops,
            dtype=dtype,
            device=device,
        )
        self.q_norm = RMSNorm(
            config.head_dim,
            eps=config.norm_eps,
            unit_offset=True,
            ops=ops,
            dtype=dtype,
            device=device,
        )
        self.k_norm = RMSNorm(
            config.head_dim,
            eps=config.norm_eps,
            unit_offset=True,
            ops=ops,
            dtype=dtype,
            device=device,
        )
        self._kvcache = ops.kvcache
        self._attention = ops.attention

    def forward(self, hidden: torch.Tensor, context: ForwardContext) -> torch.Tensor:
        rows = hidden.shape[0]
        if rows == 0:
            return hidden
        q_raw = self.q_proj(hidden)
        gate: torch.Tensor | None = None
        if self.output_gate:
            q_part, gate_part = q_raw.reshape(rows, self.num_heads, 2 * self.head_dim).chunk(
                2, dim=-1
            )
            q = q_part.contiguous()
            gate = gate_part.reshape(rows, self.num_heads * self.head_dim)
        else:
            q = q_raw.reshape(rows, self.num_heads, self.head_dim).contiguous()
        k = self.k_proj(hidden).reshape(rows, self.num_kv_heads, self.head_dim).contiguous()
        v = self.v_proj(hidden).reshape(rows, self.num_kv_heads, self.head_dim).contiguous()
        q = self.q_norm(q)
        k = self.k_norm(k)
        meta = context.attention_for(self.layer_id)
        if context.mrope_positions is not None:
            self.rotary.apply_mrope(context.mrope_positions, q, k, self.mrope_section)
        elif context.rope_position_offsets is not None:
            self.rotary(meta.positions + context.rope_position_offsets, q, k)
        else:
            self.rotary(meta.positions, q, k)
        buffers = context.buffers(self.layer_id)
        self._kvcache.store(k, v, buffers, meta)
        attended = torch.empty_like(q)
        self._attention.attend(q, buffers, meta, attended, scale=self.scale)
        out = attended.reshape(rows, self.num_heads * self.head_dim)
        if gate is not None:
            out = out * torch.sigmoid(gate)
        return self.o_proj(out)


class _Qwen35Block(nn.Module):
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
        kind = (
            config.layer_types[layer_id] if layer_id < len(config.layer_types) else "full_attention"
        )
        self.input_layernorm = RMSNorm(
            config.hidden_size,
            eps=config.norm_eps,
            unit_offset=True,
            ops=ops,
            dtype=dtype,
            device=device,
        )
        if kind == "linear_attention":
            self.attn: nn.Module = GatedDeltaNetAttention(
                config.hidden_size,
                config.linear_num_key_heads,
                config.linear_num_value_heads,
                config.linear_key_head_dim,
                config.linear_value_head_dim,
                config.linear_conv_kernel_dim,
                layer_id=layer_id,
                use_qk_l2norm=True,
                norm_eps=config.norm_eps,
                ops=ops,
                dtype=dtype,
                device=device,
            )
        else:
            self.attn = _Qwen35FullAttention(
                config,
                layer_id=layer_id,
                rotary=rotary,
                ops=ops,
                dtype=dtype,
                device=device,
            )
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size,
            eps=config.norm_eps,
            unit_offset=True,
            ops=ops,
            dtype=dtype,
            device=device,
        )
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


class Qwen3_5ForConditionalGeneration(CausalLM):  # noqa: N801 -- matches Hugging Face config architecture name
    """Qwen3.5 hybrid Gated DeltaNet + full attention causal language model."""

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
            raise ModelLoadError(
                f"{config.architecture} requires rope_theta",
                context={"architecture": config.architecture},
            )
        layer_types = config.layer_types or tuple(
            "full_attention" for _ in range(config.num_layers)
        )
        self._full_layers = tuple(i for i, t in enumerate(layer_types) if t == "full_attention")
        self._linear_layers = tuple(i for i, t in enumerate(layer_types) if t == "linear_attention")
        if self._linear_layers and (
            config.linear_num_key_heads <= 0
            or config.linear_num_value_heads <= 0
            or config.linear_key_head_dim <= 0
            or config.linear_value_head_dim <= 0
            or config.linear_conv_kernel_dim <= 0
        ):
            raise ModelLoadError(
                f"{config.architecture} missing linear-attention geometry fields",
                context={"architecture": config.architecture},
            )

        rotary_dim = max(2, int(config.head_dim * config.partial_rotary_factor))
        if rotary_dim % 2:
            raise ModelLoadError(f"rotary_dim must be even, got {rotary_dim}")
        torch_dtype = dtype.torch_dtype
        self.rotary = RotaryEmbedding(
            rotary_dim,
            max_positions=config.max_position_embeddings,
            theta=theta,
            scaling=config.rope_scaling,
            ops=ops,
            device=device,
        )
        self.embed_tokens = TokenEmbedding(
            config.vocab_size, config.hidden_size, ops=ops, dtype=torch_dtype, device=device
        )
        spec = config.vision_spec
        self._spatial_merge = spec.spatial_merge_size if spec is not None else 2
        vision_tower = (
            Qwen3_5VisionTower(config, dtype=torch_dtype, device="meta")
            if config.has_vision
            else None
        )
        self._vision = VisionLifecycle(
            vision_tower,
            spec,
            dtype=torch_dtype,
            device_fn=lambda: self.embed_tokens.weight.device,
        )
        self.layers = nn.ModuleList(
            _Qwen35Block(config, layer_id, self.rotary, ops=ops, dtype=torch_dtype, device=device)
            for layer_id in range(config.num_layers)
        )
        self.norm = RMSNorm(
            config.hidden_size,
            eps=config.norm_eps,
            unit_offset=True,
            ops=ops,
            dtype=torch_dtype,
            device=device,
        )
        self.lm_head = LMHead(
            config.hidden_size, config.vocab_size, ops=ops, dtype=torch_dtype, device=device
        )
        self.tie_weights()

    @classmethod
    def weight_mapping(cls, config: ModelConfig) -> WeightMapping:
        ignore = (*_IGNORE, r"lm_head\.weight") if config.tie_word_embeddings else _IGNORE
        return WeightMapping(renames=cls.checkpoint_renames, ignore=ignore)

    def tie_weights(self) -> None:
        if self.config.tie_word_embeddings:
            self.lm_head.tie_weights(self.embed_tokens)

    def stored_parts(self, name: str) -> tuple[tuple[str, int], ...]:
        if name.endswith(".mlp.gate_up_proj.weight") or ".mlp.gate_up_proj." in name:
            prefix, dot, suffix = name.rpartition(".")
            if dot and prefix.endswith(".mlp.gate_up_proj"):
                base = prefix.removesuffix("gate_up_proj")
                rows = self.config.intermediate_size
                return ((f"{base}gate_proj.{suffix}", rows), (f"{base}up_proj.{suffix}", rows))
        return ()

    def kv_specs(
        self,
        *,
        block_size: int,
        layout: KVLayoutKind,
        codec: str = RAW_CODEC,
        kv_scale: float = 1.0,
    ) -> tuple[KVGroup, ...]:
        layers = self._full_layers
        if not layers:
            return ()
        spec = KVSpec(
            num_layers=len(layers),
            num_kv_heads=self.config.num_kv_heads,
            head_dim=self.config.head_dim,
            block_size=block_size,
            dtype=self.compute_dtype,
            layout=layout,
            codec=codec,
            kv_scale=kv_scale,
        )
        return (KVGroup(layers, spec),)

    def state_spec(self) -> RecurrentStateGroup | None:
        if not self._linear_layers:
            return None
        conv_dim = (
            2 * self.config.linear_num_key_heads * self.config.linear_key_head_dim
            + self.config.linear_num_value_heads * self.config.linear_value_head_dim
        )
        spec = RecurrentStateSpec(
            num_layers=len(self._linear_layers),
            conv_dim=conv_dim,
            conv_kernel_size=self.config.linear_conv_kernel_dim,
            num_v_heads=self.config.linear_num_value_heads,
            key_head_dim=self.config.linear_key_head_dim,
            value_head_dim=self.config.linear_value_head_dim,
            state_dtype=DType.FP32,
            conv_dtype=self.compute_dtype,
        )
        return RecurrentStateGroup(self._linear_layers, spec)

    def forward(self, token_ids: torch.Tensor, context: ForwardContext) -> torch.Tensor:
        x = (
            context.input_embeds
            if context.input_embeds is not None
            else self.embed_tokens(token_ids)
        )
        residual: torch.Tensor | None = None
        for layer in self.layers:
            x, residual = layer(x, residual, context)
        if residual is None:
            return self.norm(x)
        self.norm.fused_add(x, residual)
        return x

    def compute_logits(self, hidden_rows: torch.Tensor) -> torch.Tensor:
        return self.lm_head(hidden_rows)

    def has_vision_tower(self) -> bool:
        return self._vision.has_tower

    def uses_multimodal_decode_offsets(self) -> bool:
        return self._vision.has_tower

    def vision_loaded(self) -> bool:
        return self._vision.is_loaded

    def vision_weight_bytes(self) -> int:
        return self._vision.weight_bytes()

    def vision_activation_bytes(self) -> int:
        return self._vision.activation_bytes()

    def attach_vision_source(self, directory: Path, *, lazy: bool) -> None:
        self._vision.attach(directory, lazy=lazy)

    def ensure_vision_loaded(self) -> None:
        self._vision.ensure_loaded()

    def release_vision(self) -> None:
        self._vision.release()

    def encode_images(self, images: Sequence[ImageInput]) -> tuple[torch.Tensor, ...]:
        return self._vision.encode(images)

    def embed_input_ids(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(token_ids)

    def multimodal_position_ids(
        self,
        prompt_lengths: Sequence[int],
        spans: Sequence[Sequence[ImageSpan]],
        starts: Sequence[int],
        rows: Sequence[int],
        *,
        is_prefill: bool,
        device: torch.device,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        if is_prefill:
            chunks = [
                _qwen3_5_positions(prompt_len, req_spans, self._spatial_merge, device)[
                    :, start : start + count
                ]
                for prompt_len, req_spans, start, count in zip(
                    prompt_lengths, spans, starts, rows, strict=True
                )
            ]
            return torch.cat(chunks, dim=1), None
        offsets: list[int] = []
        for prompt_len, req_spans, count in zip(prompt_lengths, spans, rows, strict=True):
            delta = _qwen3_5_delta(prompt_len, req_spans, self._spatial_merge)
            offsets.extend([delta] * count)
        if not any(offsets):
            return None, None
        return None, torch.tensor(offsets, dtype=torch.int64, device=device)

    def _check_part_pattern(self, name: str) -> bool:
        return bool(re.fullmatch(r"layers\.\d+\.mlp\.gate_up_proj\.(weight|bias)", name))


def _qwen3_5_delta(prompt_len: int, spans: Sequence[ImageSpan], merge: int) -> int:
    pos = 0
    cursor = 0
    for span in spans:
        pos += span.token_offset - cursor
        _, gh, gw = span.grid_thw
        pos += max(gh // merge, gw // merge)
        cursor = span.token_offset + span.token_count
    pos += prompt_len - cursor
    return pos - prompt_len


def _qwen3_5_positions(
    prompt_len: int, spans: Sequence[ImageSpan], merge: int, device: torch.device
) -> torch.Tensor:
    parts: list[torch.Tensor] = []
    pos = 0
    cursor = 0
    for span in spans:
        if span.token_offset > cursor:
            text_len = span.token_offset - cursor
            parts.append(
                torch.arange(text_len, dtype=torch.int64, device=device).view(1, -1).expand(3, -1)
                + pos
            )
            pos += text_len
        gt, gh, gw = span.grid_thw
        lt, lh, lw = max(gt, 1), gh // merge, gw // merge
        tg, hg, wg = torch.meshgrid(
            torch.arange(lt, dtype=torch.int64, device=device) + pos,
            torch.arange(lh, dtype=torch.int64, device=device) + pos,
            torch.arange(lw, dtype=torch.int64, device=device) + pos,
            indexing="ij",
        )
        parts.append(torch.stack((tg, hg, wg), dim=0).reshape(3, -1))
        pos += max(lh, lw)
        cursor = span.token_offset + span.token_count
    if prompt_len > cursor:
        text_len = prompt_len - cursor
        parts.append(
            torch.arange(text_len, dtype=torch.int64, device=device).view(1, -1).expand(3, -1) + pos
        )
    return (
        torch.cat(parts, dim=1) if parts else torch.empty((3, 0), dtype=torch.int64, device=device)
    )
