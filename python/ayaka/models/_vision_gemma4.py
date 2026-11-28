"""Gemma 4 Vision Transformer tower matching ``model.vision_tower`` and ``model.embed_vision``."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch
import torch.nn.functional as functional
from torch import nn

from ayaka.core.errors import ModelLoadError
from ayaka.models._vision_common import count_module_tensor_bytes, load_prefixed_weights
from ayaka.multimodal.preprocess import preprocess_gemma4

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from pathlib import Path

    from ayaka.configs.model import ModelConfig
    from ayaka.core.multimodal import ImageInput, VisionSpec

__all__ = ["Gemma4VisionTower"]


class _ClippableLinear(nn.Module):
    input_min: torch.Tensor
    input_max: torch.Tensor
    output_min: torch.Tensor
    output_max: torch.Tensor

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        use_clipped_linears: bool,
        dtype: torch.dtype,
        device: torch.device | str | None,
    ) -> None:
        super().__init__()
        self.use_clipped_linears = use_clipped_linears
        self.linear = nn.Linear(in_features, out_features, bias=False, dtype=dtype, device=device)
        if use_clipped_linears:
            self.register_buffer(
                "input_min", torch.tensor(-float("inf"), dtype=dtype, device=device)
            )
            self.register_buffer(
                "input_max", torch.tensor(float("inf"), dtype=dtype, device=device)
            )
            self.register_buffer(
                "output_min", torch.tensor(-float("inf"), dtype=dtype, device=device)
            )
            self.register_buffer(
                "output_max", torch.tensor(float("inf"), dtype=dtype, device=device)
            )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.use_clipped_linears:
            hidden_states = torch.clamp(hidden_states, self.input_min, self.input_max)
        hidden_states = self.linear(hidden_states)
        if self.use_clipped_linears:
            hidden_states = torch.clamp(hidden_states, self.output_min, self.output_max)
        return hidden_states


class _Gemma4VisionRMSNorm(nn.Module):
    def __init__(
        self,
        dim: int,
        *,
        eps: float,
        with_scale: bool,
        dtype: torch.dtype,
        device: torch.device | str | None,
    ) -> None:
        super().__init__()
        self.eps = eps
        self.with_scale = with_scale
        if with_scale:
            self.weight = nn.Parameter(
                torch.empty(dim, dtype=dtype, device=device), requires_grad=False
            )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        wide = hidden_states.float()
        mean_sq = wide.pow(2).mean(dim=-1, keepdim=True) + self.eps
        normed = wide * torch.pow(mean_sq, -0.5)
        if self.with_scale:
            normed = normed * self.weight.float()
        return normed.type_as(hidden_states)


class _PatchEmbedder(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        patch_size: int,
        position_embedding_size: int,
        *,
        dtype: torch.dtype,
        device: torch.device | str | None,
    ) -> None:
        super().__init__()
        self.input_proj = nn.Linear(
            3 * patch_size * patch_size, hidden_size, bias=False, dtype=dtype, device=device
        )
        self.position_embedding_table = nn.Parameter(
            torch.empty(2, position_embedding_size, hidden_size, dtype=dtype, device=device),
            requires_grad=False,
        )

    def forward(
        self,
        pixel_values: torch.Tensor,
        pixel_position_ids: torch.Tensor,
        padding_positions: torch.Tensor,
    ) -> torch.Tensor:
        scaled = 2.0 * (pixel_values - 0.5)
        target_dtype = self.input_proj.weight.dtype
        if target_dtype.is_floating_point:
            scaled = scaled.to(target_dtype)
        hidden = self.input_proj(scaled)
        clamped = pixel_position_ids.clamp(min=0)
        x_emb = functional.embedding(clamped[..., 0], self.position_embedding_table[0])
        y_emb = functional.embedding(clamped[..., 1], self.position_embedding_table[1])
        pos = torch.where(padding_positions.unsqueeze(-1), 0.0, x_emb + y_emb).to(hidden.dtype)
        return hidden + pos


class _VisionRotaryEmbedding(nn.Module):
    inv_freq: torch.Tensor

    def __init__(self, head_dim: int, theta: float, device: torch.device | str | None) -> None:
        super().__init__()
        self._spatial_dim = head_dim // 2
        self._theta = theta
        self.register_buffer(
            "inv_freq",
            torch.empty(self._spatial_dim // 2, dtype=torch.float32, device=device),
            persistent=False,
        )
        if self.inv_freq.device.type != "meta":
            self.reset_cache(self.inv_freq.device)

    def reset_cache(self, device: torch.device) -> None:
        idx = torch.arange(0, self._spatial_dim, 2, dtype=torch.float32, device=device)
        inv = 1.0 / (self._theta ** (idx / self._spatial_dim))
        self.inv_freq.copy_(inv)

    def forward(
        self, hidden_states: torch.Tensor, position_ids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        inv = self.inv_freq.to(device=hidden_states.device, dtype=torch.float32)
        freqs = position_ids[..., None].float() * inv
        cos = freqs.cos()
        sin = freqs.sin()
        cos_hhww = torch.cat((cos[:, :, 0], cos[:, :, 0], cos[:, :, 1], cos[:, :, 1]), dim=-1)
        sin_hhww = torch.cat((sin[:, :, 0], sin[:, :, 0], sin[:, :, 1], sin[:, :, 1]), dim=-1)
        return cos_hhww.to(hidden_states.dtype), sin_hhww.to(hidden_states.dtype)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def _apply_2d_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    x_parts = (x[..., :half], x[..., half:])
    c_parts = (cos[..., :half].unsqueeze(2), cos[..., half:].unsqueeze(2))
    s_parts = (sin[..., :half].unsqueeze(2), sin[..., half:].unsqueeze(2))
    out = [
        (xp * cp) + (_rotate_half(xp) * sp)
        for xp, cp, sp in zip(x_parts, c_parts, s_parts, strict=True)
    ]
    return torch.cat(out, dim=-1)


class _VisionAttention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        eps: float,
        use_clipped_linears: bool,
        *,
        dtype: torch.dtype,
        device: torch.device | str | None,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.q_proj = _ClippableLinear(
            hidden_size,
            num_heads * head_dim,
            use_clipped_linears=use_clipped_linears,
            dtype=dtype,
            device=device,
        )
        self.k_proj = _ClippableLinear(
            hidden_size,
            num_kv_heads * head_dim,
            use_clipped_linears=use_clipped_linears,
            dtype=dtype,
            device=device,
        )
        self.v_proj = _ClippableLinear(
            hidden_size,
            num_kv_heads * head_dim,
            use_clipped_linears=use_clipped_linears,
            dtype=dtype,
            device=device,
        )
        self.o_proj = _ClippableLinear(
            num_heads * head_dim,
            hidden_size,
            use_clipped_linears=use_clipped_linears,
            dtype=dtype,
            device=device,
        )
        self.q_norm = _Gemma4VisionRMSNorm(
            head_dim, eps=eps, with_scale=True, dtype=dtype, device=device
        )
        self.k_norm = _Gemma4VisionRMSNorm(
            head_dim, eps=eps, with_scale=True, dtype=dtype, device=device
        )
        self.v_norm = _Gemma4VisionRMSNorm(
            head_dim, eps=eps, with_scale=False, dtype=dtype, device=device
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        batch, length, _ = hidden_states.shape
        q = self.q_norm(
            self.q_proj(hidden_states).view(batch, length, self.num_heads, self.head_dim)
        )
        k = self.k_norm(
            self.k_proj(hidden_states).view(batch, length, self.num_kv_heads, self.head_dim)
        )
        v = self.v_norm(
            self.v_proj(hidden_states).view(batch, length, self.num_kv_heads, self.head_dim)
        )
        q = _apply_2d_rope(q, cos, sin).transpose(1, 2)
        k = _apply_2d_rope(k, cos, sin).transpose(1, 2)
        v = v.transpose(1, 2)
        if self.num_kv_heads != self.num_heads:
            repeat = self.num_heads // self.num_kv_heads
            k = k.repeat_interleave(repeat, dim=1)
            v = v.repeat_interleave(repeat, dim=1)
        attn_mask = valid_mask[:, None, None, :].expand(batch, 1, length, length)
        out = functional.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, is_causal=False, scale=1.0
        )
        return self.o_proj(out.transpose(1, 2).reshape(batch, length, -1).contiguous())


class _VisionMLP(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        use_clipped_linears: bool,
        *,
        dtype: torch.dtype,
        device: torch.device | str | None,
    ) -> None:
        super().__init__()
        self.gate_proj = _ClippableLinear(
            hidden_size,
            intermediate_size,
            use_clipped_linears=use_clipped_linears,
            dtype=dtype,
            device=device,
        )
        self.up_proj = _ClippableLinear(
            hidden_size,
            intermediate_size,
            use_clipped_linears=use_clipped_linears,
            dtype=dtype,
            device=device,
        )
        self.down_proj = _ClippableLinear(
            intermediate_size,
            hidden_size,
            use_clipped_linears=use_clipped_linears,
            dtype=dtype,
            device=device,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        gate = functional.gelu(self.gate_proj(hidden_states), approximate="tanh")
        return self.down_proj(gate * self.up_proj(hidden_states))


class _VisionEncoderLayer(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        eps: float,
        use_clipped_linears: bool,
        *,
        dtype: torch.dtype,
        device: torch.device | str | None,
    ) -> None:
        super().__init__()
        self.self_attn = _VisionAttention(
            hidden_size,
            num_heads,
            num_kv_heads,
            head_dim,
            eps,
            use_clipped_linears,
            dtype=dtype,
            device=device,
        )
        self.mlp = _VisionMLP(
            hidden_size, intermediate_size, use_clipped_linears, dtype=dtype, device=device
        )
        self.input_layernorm = _Gemma4VisionRMSNorm(
            hidden_size, eps=eps, with_scale=True, dtype=dtype, device=device
        )
        self.post_attention_layernorm = _Gemma4VisionRMSNorm(
            hidden_size, eps=eps, with_scale=True, dtype=dtype, device=device
        )
        self.pre_feedforward_layernorm = _Gemma4VisionRMSNorm(
            hidden_size, eps=eps, with_scale=True, dtype=dtype, device=device
        )
        self.post_feedforward_layernorm = _Gemma4VisionRMSNorm(
            hidden_size, eps=eps, with_scale=True, dtype=dtype, device=device
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        attn_out = self.self_attn(self.input_layernorm(hidden_states), cos, sin, valid_mask)
        hidden_states = hidden_states + self.post_attention_layernorm(attn_out)
        mlp_out = self.mlp(self.pre_feedforward_layernorm(hidden_states))
        return hidden_states + self.post_feedforward_layernorm(mlp_out)


class _VisionEncoder(nn.Module):
    def __init__(
        self,
        num_layers: int,
        hidden_size: int,
        intermediate_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        eps: float,
        rope_theta: float,
        use_clipped_linears: bool,
        *,
        dtype: torch.dtype,
        device: torch.device | str | None,
    ) -> None:
        super().__init__()
        self.rotary_emb = _VisionRotaryEmbedding(head_dim, rope_theta, device)
        self.layers = nn.ModuleList(
            [
                _VisionEncoderLayer(
                    hidden_size,
                    intermediate_size,
                    num_heads,
                    num_kv_heads,
                    head_dim,
                    eps,
                    use_clipped_linears,
                    dtype=dtype,
                    device=device,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        valid_mask: torch.Tensor,
        pixel_position_ids: torch.Tensor,
    ) -> torch.Tensor:
        hidden = inputs_embeds
        cos, sin = self.rotary_emb(hidden, pixel_position_ids)
        for layer in self.layers:
            hidden = layer(hidden, cos, sin, valid_mask)
        return hidden


class _VisionModel(nn.Module):
    std_bias: torch.Tensor
    std_scale: torch.Tensor

    def __init__(
        self,
        vc: Mapping[str, object],
        *,
        dtype: torch.dtype,
        device: torch.device | str | None,
    ) -> None:
        super().__init__()
        hidden_size = int(cast("int", vc.get("hidden_size", 768)))
        intermediate_size = int(cast("int", vc.get("intermediate_size", 3072)))
        num_layers = int(cast("int", vc.get("num_hidden_layers", 16)))
        num_heads = int(cast("int", vc.get("num_attention_heads", 12)))
        num_kv_heads = int(cast("int", vc.get("num_key_value_heads", num_heads)))
        head_dim = int(cast("int", vc.get("head_dim") or (hidden_size // num_heads)))
        patch_size = int(cast("int", vc.get("patch_size", 16)))
        pos_emb_size = int(cast("int", vc.get("position_embedding_size", 10240)))
        eps = float(cast("float", vc.get("rms_norm_eps", 1e-6)))
        rope_raw = vc.get("rope_parameters")
        rope_theta = (
            float(cast("float", cast("Mapping[str, object]", rope_raw).get("rope_theta", 100.0)))
            if isinstance(rope_raw, dict)
            else 100.0
        )
        use_clipped = bool(vc.get("use_clipped_linears", False))
        self.pooling_kernel_size = int(cast("int", vc.get("pooling_kernel_size", 3)))
        self.standardize = bool(vc.get("standardize", False))
        self.root_hidden_size = hidden_size**0.5
        self.patch_embedder = _PatchEmbedder(
            hidden_size, patch_size, pos_emb_size, dtype=dtype, device=device
        )
        self.encoder = _VisionEncoder(
            num_layers,
            hidden_size,
            intermediate_size,
            num_heads,
            num_kv_heads,
            head_dim,
            eps,
            rope_theta,
            use_clipped,
            dtype=dtype,
            device=device,
        )
        if self.standardize:
            self.register_buffer("std_bias", torch.empty(hidden_size, dtype=dtype, device=device))
            self.register_buffer("std_scale", torch.empty(hidden_size, dtype=dtype, device=device))

    def _pool(
        self,
        hidden_states: torch.Tensor,
        pixel_position_ids: torch.Tensor,
        padding_positions: torch.Tensor,
        output_length: int,
    ) -> torch.Tensor:
        hidden = hidden_states.masked_fill(padding_positions.unsqueeze(-1), 0.0)
        input_seq_len = hidden.shape[1]
        k = self.pooling_kernel_size
        k_squared = k * k
        if input_seq_len != output_length:
            clamped = pixel_position_ids.clamp(min=0)
            max_x = clamped[..., 0].max(dim=-1, keepdim=True)[0] + 1
            k_idxs = torch.div(clamped, k, rounding_mode="floor")
            flat_idx = k_idxs[..., 0] + (max_x // k) * k_idxs[..., 1]
            weights = functional.one_hot(flat_idx.long(), output_length).float() / k_squared
            pooled = (weights.transpose(1, 2) @ hidden.float()).to(hidden.dtype)
            mask = ~((weights == 0).all(dim=1))
        else:
            pooled = hidden
            mask = ~padding_positions
        scaled = pooled.float() * self.root_hidden_size
        kept = scaled[mask]
        if self.standardize:
            kept = (kept - self.std_bias.float()) * self.std_scale.float()
        return kept.to(hidden_states.dtype)

    def forward(self, pixel_values: torch.Tensor, pixel_position_ids: torch.Tensor) -> torch.Tensor:
        output_length = pixel_values.shape[-2] // (self.pooling_kernel_size**2)
        padding_positions = (pixel_position_ids == -1).all(dim=-1)
        embeds = self.patch_embedder(pixel_values, pixel_position_ids, padding_positions)
        encoded = self.encoder(embeds, ~padding_positions, pixel_position_ids)
        return self._pool(encoded, pixel_position_ids, padding_positions, output_length)


class _VisionEmbedder(nn.Module):
    def __init__(
        self,
        vc: Mapping[str, object],
        text_hidden_size: int,
        *,
        dtype: torch.dtype,
        device: torch.device | str | None,
    ) -> None:
        super().__init__()
        mm_hidden = int(cast("int", vc.get("output_proj_dims", vc.get("hidden_size", 768))))
        eps = float(cast("float", vc.get("rms_norm_eps", 1e-6)))
        self.embedding_pre_projection_norm = _Gemma4VisionRMSNorm(
            mm_hidden, eps=eps, with_scale=False, dtype=dtype, device=device
        )
        self.embedding_projection = nn.Linear(
            mm_hidden, text_hidden_size, bias=False, dtype=dtype, device=device
        )

    def forward(self, inputs_embeds: torch.Tensor) -> torch.Tensor:
        return self.embedding_projection(self.embedding_pre_projection_norm(inputs_embeds))


class Gemma4VisionTower(nn.Module):
    """Standalone Gemma 4 vision encoder and language-space projector."""

    def __init__(
        self,
        config: ModelConfig,
        *,
        dtype: torch.dtype,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__()
        vc: Mapping[str, object] = config.vision_config or {}
        if not vc:
            raise ModelLoadError(f"{config.architecture} missing vision_config")
        self.dtype = dtype
        self.patch_size = int(cast("int", vc.get("patch_size", 16)))
        self.hidden_size = int(cast("int", vc.get("hidden_size", 768)))
        self.vision_tower = _VisionModel(vc, dtype=dtype, device=device)
        self.embed_vision = _VisionEmbedder(vc, config.hidden_size, dtype=dtype, device=device)

    def reset_buffers(self, device: torch.device) -> None:
        self.vision_tower.encoder.rotary_emb.reset_cache(device)

    @property
    def is_loaded(self) -> bool:
        return self.embed_vision.embedding_projection.weight.device.type != "meta"

    def weight_bytes(self) -> int:
        return count_module_tensor_bytes(self, self.dtype.itemsize)

    def activation_bytes(self, height: int = 768, width: int = 1024) -> int:
        del height, width
        patches = 280 * 9
        return patches * self.hidden_size * self.dtype.itemsize * 12

    def load_from_safetensors(
        self, directory: Path, *, device: torch.device | str | None, dtype: torch.dtype
    ) -> None:
        load_prefixed_weights(
            self,
            directory,
            prefixes=("model.vision_tower.", "model.embed_vision."),
            strip_prefixes=("model.",),
            device=device,
            dtype=dtype,
            reset_buffers=self.vision_tower.encoder.rotary_emb,
        )

    def forward(self, pixel_values: torch.Tensor, pixel_position_ids: torch.Tensor) -> torch.Tensor:
        soft_tokens = self.vision_tower(pixel_values, pixel_position_ids)
        return self.embed_vision(soft_tokens)

    @torch.inference_mode()
    def encode_images(
        self, images: Sequence[ImageInput], spec: VisionSpec
    ) -> tuple[torch.Tensor, ...]:
        device = self.embed_vision.embedding_projection.weight.device
        pixels, pos_ids, soft_counts = preprocess_gemma4(images, spec, device=device)
        projected = self.forward(pixels, pos_ids)
        return tuple(torch.split(projected, list(soft_counts), dim=0))
