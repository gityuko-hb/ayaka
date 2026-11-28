"""Qwen3.5 Vision Transformer tower matching Hugging Face ``Qwen3_5VisionModel``."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, cast

import torch
import torch.nn.functional as functional
from torch import nn

from ayaka.core.errors import ModelLoadError
from ayaka.models._vision_common import count_module_tensor_bytes, load_prefixed_weights
from ayaka.multimodal.preprocess import preprocess_qwen3_5

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from pathlib import Path

    from ayaka.configs.model import ModelConfig
    from ayaka.core.multimodal import ImageInput, VisionSpec

__all__ = ["Qwen3_5VisionTower"]


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


class _VisionRotaryEmbedding(nn.Module):
    inv_freq: torch.Tensor

    def __init__(self, head_dim: int, theta: float, device: torch.device | str | None) -> None:
        super().__init__()
        self._spatial_dim = head_dim // 2
        self._theta = theta
        half_spatial = self._spatial_dim // 2
        self.register_buffer(
            "inv_freq",
            torch.empty(half_spatial, dtype=torch.float32, device=device),
            persistent=False,
        )
        if self.inv_freq.device.type != "meta":
            self.reset_cache(self.inv_freq.device)

    def reset_cache(self, device: torch.device) -> None:
        idx = torch.arange(0, self._spatial_dim, 2, dtype=torch.float32, device=device)
        inv = 1.0 / (self._theta ** (idx / self._spatial_dim))
        self.inv_freq.copy_(inv)

    def forward(self, position_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        inv = self.inv_freq.to(device=position_ids.device, dtype=torch.float32)
        freqs = position_ids[..., None].float() * inv
        cos = freqs.cos()
        sin = freqs.sin()
        cos_hw = torch.cat((cos[:, 0], cos[:, 1]), dim=-1)
        sin_hw = torch.cat((sin[:, 0], sin[:, 1]), dim=-1)
        return torch.cat((cos_hw, cos_hw), dim=-1), torch.cat((sin_hw, sin_hw), dim=-1)


class _PatchEmbed(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_size: int,
        temporal_patch_size: int,
        patch_size: int,
        *,
        dtype: torch.dtype,
        device: torch.device | str | None,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.temporal_patch_size = temporal_patch_size
        self.patch_size = patch_size
        self.hidden_size = hidden_size
        kernel = (temporal_patch_size, patch_size, patch_size)
        self.proj = nn.Conv3d(
            in_channels,
            hidden_size,
            kernel_size=kernel,
            stride=kernel,
            bias=True,
            dtype=dtype,
            device=device,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        target_dtype = self.proj.weight.dtype
        patches = hidden_states.view(
            -1, self.in_channels, self.temporal_patch_size, self.patch_size, self.patch_size
        )
        return self.proj(patches.to(dtype=target_dtype)).view(-1, self.hidden_size)


class _VisionMLP(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        *,
        dtype: torch.dtype,
        device: torch.device | str | None,
    ) -> None:
        super().__init__()
        self.linear_fc1 = nn.Linear(
            hidden_size, intermediate_size, bias=True, dtype=dtype, device=device
        )
        self.linear_fc2 = nn.Linear(
            intermediate_size, hidden_size, bias=True, dtype=dtype, device=device
        )
        self._approximate = "tanh" if hidden_act in ("gelu_pytorch_tanh", "gelu_new") else "none"

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden = functional.gelu(self.linear_fc1(hidden_states), approximate=self._approximate)
        return self.linear_fc2(hidden)


class _VisionAttention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        *,
        dtype: torch.dtype,
        device: torch.device | str | None,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.scaling = self.head_dim**-0.5
        self.qkv = nn.Linear(hidden_size, hidden_size * 3, bias=True, dtype=dtype, device=device)
        self.proj = nn.Linear(hidden_size, hidden_size, bias=True, dtype=dtype, device=device)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        seq_length = hidden_states.shape[0]
        qkv = self.qkv(hidden_states).reshape(seq_length, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.permute(1, 0, 2, 3).unbind(0)

        cos_u = cos.unsqueeze(-2)
        sin_u = sin.unsqueeze(-2)
        orig_dtype = q.dtype
        q_rot = ((q.float() * cos_u) + (_rotate_half(q.float()) * sin_u)).to(orig_dtype)
        k_rot = ((k.float() * cos_u) + (_rotate_half(k.float()) * sin_u)).to(orig_dtype)

        diffs = cu_seqlens[1:] - cu_seqlens[:-1]
        lengths = [int(diffs[i].item()) for i in range(diffs.shape[0])]
        if len(lengths) == 1:
            attn = functional.scaled_dot_product_attention(
                q_rot.transpose(0, 1).unsqueeze(0),
                k_rot.transpose(0, 1).unsqueeze(0),
                v.transpose(0, 1).unsqueeze(0),
                attn_mask=None,
                is_causal=False,
                scale=self.scaling,
            )
            merged = attn.squeeze(0).transpose(0, 1)
        else:
            q_parts = torch.split(q_rot, lengths, dim=0)
            k_parts = torch.split(k_rot, lengths, dim=0)
            v_parts = torch.split(v, lengths, dim=0)
            outs = [
                functional.scaled_dot_product_attention(
                    qp.transpose(0, 1).unsqueeze(0),
                    kp.transpose(0, 1).unsqueeze(0),
                    vp.transpose(0, 1).unsqueeze(0),
                    attn_mask=None,
                    is_causal=False,
                    scale=self.scaling,
                )
                .squeeze(0)
                .transpose(0, 1)
                for qp, kp, vp in zip(q_parts, k_parts, v_parts, strict=True)
            ]
            merged = torch.cat(outs, dim=0)

        return self.proj(merged.reshape(seq_length, -1).contiguous())


class _VisionBlock(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_heads: int,
        hidden_act: str,
        *,
        dtype: torch.dtype,
        device: torch.device | str | None,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, eps=1e-6, dtype=dtype, device=device)
        self.norm2 = nn.LayerNorm(hidden_size, eps=1e-6, dtype=dtype, device=device)
        self.attn = _VisionAttention(hidden_size, num_heads, dtype=dtype, device=device)
        self.mlp = _VisionMLP(
            hidden_size, intermediate_size, hidden_act, dtype=dtype, device=device
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        hidden_states = hidden_states + self.attn(self.norm1(hidden_states), cu_seqlens, cos, sin)
        return hidden_states + self.mlp(self.norm2(hidden_states))


class _PatchMerger(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        out_hidden_size: int,
        spatial_merge_size: int,
        *,
        dtype: torch.dtype,
        device: torch.device | str | None,
    ) -> None:
        super().__init__()
        merged = hidden_size * (spatial_merge_size**2)
        self.merged_size = merged
        self.norm = nn.LayerNorm(hidden_size, eps=1e-6, dtype=dtype, device=device)
        self.linear_fc1 = nn.Linear(merged, merged, bias=True, dtype=dtype, device=device)
        self.linear_fc2 = nn.Linear(merged, out_hidden_size, bias=True, dtype=dtype, device=device)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        normed = self.norm(hidden_states).view(-1, self.merged_size)
        return self.linear_fc2(functional.gelu(self.linear_fc1(normed)))


def _vision_grid_coords(
    grid_thw: torch.Tensor, spatial_merge_size: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    device = grid_thw.device
    merge = spatial_merge_size
    rows_all: list[torch.Tensor] = []
    cols_all: list[torch.Tensor] = []
    h_all: list[torch.Tensor] = []
    w_all: list[torch.Tensor] = []
    for i in range(grid_thw.shape[0]):
        t = int(grid_thw[i, 0].item())
        h = int(grid_thw[i, 1].item())
        w = int(grid_thw[i, 2].item())
        hpos, wpos = torch.meshgrid(
            torch.arange(h, device=device),
            torch.arange(w, device=device),
            indexing="ij",
        )
        shape = (h // merge, merge, w // merge, merge)
        h_flat = hpos.reshape(shape).transpose(1, 2).flatten()
        w_flat = wpos.reshape(shape).transpose(1, 2).flatten()
        if t > 1:
            h_flat = h_flat.repeat(t)
            w_flat = w_flat.repeat(t)
        rows_all.append(h_flat)
        cols_all.append(w_flat)
        h_all.append(torch.full_like(h_flat, h))
        w_all.append(torch.full_like(w_flat, w))
    return (
        torch.cat(rows_all, dim=0),
        torch.cat(cols_all, dim=0),
        torch.cat(h_all, dim=0),
        torch.cat(w_all, dim=0),
    )


def _bilinear_interp_indices(
    row: torch.Tensor, col: torch.Tensor, heights: torch.Tensor, widths: torch.Tensor, side: int
) -> tuple[torch.Tensor, torch.Tensor]:
    def axis(index: torch.Tensor, size: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        src = index.float() * (side - 1) / torch.clamp(size.float() - 1.0, min=1.0)
        floor = torch.floor(src)
        offsets = torch.arange(2, device=index.device)
        taps = (floor.long()[:, None] + offsets).clamp(0, side - 1)
        weights = (1.0 - (src[:, None] - floor[:, None] - offsets).abs()).clamp(min=0.0)
        return taps, weights

    h_taps, h_weights = axis(row, heights)
    w_taps, w_weights = axis(col, widths)
    indices = (h_taps[:, :, None] * side + w_taps[:, None, :]).reshape(-1, 4)
    weights = (h_weights[:, :, None] * w_weights[:, None, :]).reshape(-1, 4)
    return indices, weights


class Qwen3_5VisionTower(nn.Module):  # noqa: N801 -- matches Hugging Face family name
    """Standalone Qwen3.5 vision encoder matching Hugging Face ``model.visual`` weights."""

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
        hidden_size = int(cast("int", vc.get("hidden_size", 1152)))
        intermediate_size = int(cast("int", vc.get("intermediate_size", 4304)))
        depth = int(cast("int", vc.get("depth", 27)))
        num_heads = int(cast("int", vc.get("num_heads", 16)))
        in_channels = int(cast("int", vc.get("in_channels", 3)))
        patch_size = int(cast("int", vc.get("patch_size", 16)))
        temporal_patch_size = int(cast("int", vc.get("temporal_patch_size", 2)))
        spatial_merge_size = int(cast("int", vc.get("spatial_merge_size", 2)))
        num_pos = int(cast("int", vc.get("num_position_embeddings", 2304)))
        out_hidden_size = int(cast("int", vc.get("out_hidden_size", config.hidden_size)))
        hidden_act = str(vc.get("hidden_act", "gelu_pytorch_tanh"))
        rope_raw = vc.get("rope_parameters")
        rope_theta = (
            float(cast("float", cast("Mapping[str, object]", rope_raw).get("rope_theta", 10000.0)))
            if isinstance(rope_raw, dict)
            else 10000.0
        )

        self.dtype = dtype
        self.spatial_merge_size = spatial_merge_size
        self.patch_size = patch_size
        self.num_grid_per_side = math.isqrt(num_pos)
        self.patch_embed = _PatchEmbed(
            in_channels,
            hidden_size,
            temporal_patch_size,
            patch_size,
            dtype=dtype,
            device=device,
        )
        self.pos_embed = nn.Embedding(num_pos, hidden_size, dtype=dtype, device=device)
        self.rotary_pos_emb = _VisionRotaryEmbedding(hidden_size // num_heads, rope_theta, device)
        self.blocks = nn.ModuleList(
            [
                _VisionBlock(
                    hidden_size,
                    intermediate_size,
                    num_heads,
                    hidden_act,
                    dtype=dtype,
                    device=device,
                )
                for _ in range(depth)
            ]
        )
        self.merger = _PatchMerger(
            hidden_size, out_hidden_size, spatial_merge_size, dtype=dtype, device=device
        )

    def reset_buffers(self, device: torch.device) -> None:
        self.rotary_pos_emb.reset_cache(device)

    @property
    def is_loaded(self) -> bool:
        return self.patch_embed.proj.weight.device.type != "meta"

    def weight_bytes(self) -> int:
        return count_module_tensor_bytes(self, self.dtype.itemsize)

    def activation_bytes(self, height: int = 768, width: int = 1024) -> int:
        patches = (height // self.patch_size) * (width // self.patch_size)
        hidden = self.patch_embed.hidden_size
        elem = self.dtype.itemsize
        # Patch hidden + QKV/FFN peak workspace heuristic per image.
        return patches * hidden * elem * 12

    def load_from_safetensors(
        self, directory: Path, *, device: torch.device | str | None, dtype: torch.dtype
    ) -> None:
        load_prefixed_weights(
            self,
            directory,
            prefixes=("model.visual.", "visual."),
            strip_prefixes=("model.visual.", "visual."),
            device=device,
            dtype=dtype,
            reset_buffers=self.rotary_pos_emb,
        )

    def forward(self, pixel_values: torch.Tensor, grid_thw: torch.Tensor) -> torch.Tensor:
        row, col, heights, widths = _vision_grid_coords(grid_thw, self.spatial_merge_size)
        indices, weights = _bilinear_interp_indices(
            row, col, heights, widths, self.num_grid_per_side
        )
        pos_embeds = (self.pos_embed(indices) * weights[:, :, None]).sum(dim=1)
        hidden = self.patch_embed(pixel_values)
        hidden = hidden + pos_embeds.to(hidden.dtype)

        position_ids = torch.stack((row, col), dim=-1)
        cos, sin = self.rotary_pos_emb(position_ids)
        seqlens = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0])
        cu_seqlens = functional.pad(seqlens.cumsum(dim=0, dtype=torch.int32), (1, 0), value=0)

        for block in self.blocks:
            hidden = block(hidden, cu_seqlens, cos, sin)
        return self.merger(hidden)

    @torch.inference_mode()
    def encode_images(
        self, images: Sequence[ImageInput], spec: VisionSpec
    ) -> tuple[torch.Tensor, ...]:
        device = self.patch_embed.proj.weight.device
        pixels, grid_thw = preprocess_qwen3_5(images, spec, device=device)
        merged = self.forward(pixels, grid_thw)
        merge = self.spatial_merge_size
        counts = [
            int(grid_thw[i, 0].item())
            * (int(grid_thw[i, 1].item()) // merge)
            * (int(grid_thw[i, 2].item()) // merge)
            for i in range(grid_thw.shape[0])
        ]
        return tuple(torch.split(merged, counts, dim=0))
