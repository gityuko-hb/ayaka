"""Deterministic pure-PyTorch image preprocessors matching HF vision processors."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

from ayaka.multimodal.png import decode_png_rgb

if TYPE_CHECKING:
    from collections.abc import Sequence

    from torch import Tensor

    from ayaka.core.multimodal import ImageInput, VisionSpec

__all__ = [
    "aspect_ratio_preserving_size",
    "image_token_count",
    "preprocess_gemma4",
    "preprocess_qwen3_5",
    "smart_resize",
]


def smart_resize(
    height: int,
    width: int,
    *,
    factor: int = 32,
    min_pixels: int = 65536,
    max_pixels: int = 16777216,
) -> tuple[int, int]:
    """Compute Qwen2-VL/Qwen3.5 aspect-preserving dimensions divisible by ``factor``."""
    if min(height, width) <= 0:
        raise ValueError(f"invalid image dimensions {height}x{width}")
    ratio = max(height, width) / min(height, width)
    if ratio > 200:
        raise ValueError(f"absolute aspect ratio must be smaller than 200, got {ratio}")
    h_bar = round(height / factor) * factor
    w_bar = round(width / factor) * factor
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = max(factor, math.floor(height / beta / factor) * factor)
        w_bar = max(factor, math.floor(width / beta / factor) * factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    return int(h_bar), int(w_bar)


def aspect_ratio_preserving_size(
    height: int,
    width: int,
    *,
    patch_size: int = 16,
    max_patches: int = 2520,
    pooling_kernel_size: int = 3,
) -> tuple[int, int]:
    """Compute Gemma 4 target size fitting within ``max_patches`` and divisible by ``k * P``."""
    if min(height, width) <= 0:
        raise ValueError(f"invalid image dimensions {height}x{width}")
    total_px = height * width
    target_px = max_patches * (patch_size**2)
    factor = math.sqrt(target_px / total_px)
    ideal_height = factor * height
    ideal_width = factor * width
    side_mult = pooling_kernel_size * patch_size

    target_height = math.floor(ideal_height / side_mult) * side_mult
    target_width = math.floor(ideal_width / side_mult) * side_mult

    if target_height == 0 and target_width == 0:
        raise ValueError(f"cannot resize {height}x{width} image to 0x0 with multiple {side_mult}")

    max_side_length = (max_patches // (pooling_kernel_size**2)) * side_mult
    if target_height == 0:
        target_height = side_mult
        target_width = min(math.floor(width / height) * side_mult, max_side_length)
    elif target_width == 0:
        target_width = side_mult
        target_height = min(math.floor(height / width) * side_mult, max_side_length)

    if target_height * target_width > target_px:
        raise ValueError(
            f"resizing [{height}x{width}] to [{target_height}x{target_width}] exceeds "
            f"{max_patches} patches with patch_size={patch_size}"
        )
    return int(target_height), int(target_width)


def image_token_count(
    height: int, width: int, spec: VisionSpec
) -> tuple[int, tuple[int, int, int]]:
    """Compute visual token count and ``(T, H_p, W_p)`` patch grid for ``(height, width)``."""
    p = spec.patch_size
    m = spec.spatial_merge_size
    if spec.family == "qwen3_5":
        rh, rw = smart_resize(
            height,
            width,
            factor=p * m,
            min_pixels=spec.min_pixels,
            max_pixels=spec.max_pixels,
        )
        gh, gw = rh // p, rw // p
        tokens = (gh // m) * (gw // m)
        return tokens, (1, gh, gw)
    if spec.family == "gemma4":
        max_patches = spec.max_soft_tokens * (m**2)
        rh, rw = aspect_ratio_preserving_size(
            height,
            width,
            patch_size=p,
            max_patches=max_patches,
            pooling_kernel_size=m,
        )
        gh, gw = rh // p, rw // p
        tokens = (gh * gw) // (m**2)
        return tokens, (1, gh, gw)
    raise ValueError(f"unsupported vision family {spec.family!r}")


def _decode_to_chw_tensor(data: bytes) -> Tensor:
    import torch

    height, width, rgb_bytes = decode_png_rgb(data)
    # Copy from mutable bytearray so torch does not warn about non-writable buffers.
    buf = bytearray(rgb_bytes)
    tensor = torch.frombuffer(buf, dtype=torch.uint8).reshape(height, width, 3)
    return tensor.permute(2, 0, 1).contiguous()


def _bicubic_resize(img: Tensor, target_h: int, target_w: int) -> Tensor:
    import torch.nn.functional as functional

    _, h, w = img.shape
    if (h, w) == (target_h, target_w):
        return img.float()
    resized = functional.interpolate(
        img.unsqueeze(0).float(),
        size=(target_h, target_w),
        mode="bicubic",
        align_corners=False,
        antialias=True,
    )
    return resized.clamp(0.0, 255.0).squeeze(0)


def preprocess_qwen3_5(
    images: Sequence[ImageInput],
    spec: VisionSpec,
    *,
    image_mean: tuple[float, float, float] = (0.5, 0.5, 0.5),
    image_std: tuple[float, float, float] = (0.5, 0.5, 0.5),
    dtype: Any = None,
    device: Any = None,
) -> tuple[Tensor, Tensor]:
    """Preprocess PNG images into Qwen3.5 ``(pixel_values, image_grid_thw)`` tensors."""
    import torch

    p = spec.patch_size
    m = spec.spatial_merge_size
    t = spec.temporal_patch_size
    mean = torch.tensor(image_mean, dtype=torch.float32).view(3, 1, 1)
    std = torch.tensor(image_std, dtype=torch.float32).view(3, 1, 1)

    all_patches: list[Tensor] = []
    grids: list[list[int]] = []
    for image in images:
        raw = _decode_to_chw_tensor(image.data)
        _, h, w = raw.shape
        rh, rw = smart_resize(
            h, w, factor=p * m, min_pixels=spec.min_pixels, max_pixels=spec.max_pixels
        )
        x = _bicubic_resize(raw, rh, rw)
        x = (x / 255.0 - mean) / std
        gh, gw = rh // p, rw // p
        patches = x.reshape(1, 3, gh // m, m, p, gw // m, m, p).permute(0, 2, 5, 3, 6, 1, 4, 7)
        flat = (
            patches.unsqueeze(6)
            .expand(-1, -1, -1, -1, -1, -1, t, -1, -1)
            .reshape(gh * gw, 3 * t * p * p)
        )
        all_patches.append(flat)
        grids.append([1, gh, gw])

    pixel_values = torch.cat(all_patches, dim=0)
    if dtype is not None:
        pixel_values = pixel_values.to(dtype=dtype)
    if device is not None:
        pixel_values = pixel_values.to(device=device)
    grid_thw = torch.tensor(grids, dtype=torch.long, device=device)
    return pixel_values, grid_thw


def preprocess_gemma4(
    images: Sequence[ImageInput],
    spec: VisionSpec,
    *,
    dtype: Any = None,
    device: Any = None,
) -> tuple[Tensor, Tensor, tuple[int, ...]]:
    """Preprocess PNG images into Gemma 4 ``(pixel_values, image_position_ids, soft_counts)``."""
    import torch
    import torch.nn.functional as functional

    p = spec.patch_size
    k = spec.spatial_merge_size
    max_patches = spec.max_soft_tokens * (k**2)

    pixel_batches: list[Tensor] = []
    pos_batches: list[Tensor] = []
    soft_counts: list[int] = []
    for image in images:
        raw = _decode_to_chw_tensor(image.data)
        _, h, w = raw.shape
        th, tw = aspect_ratio_preserving_size(
            h, w, patch_size=p, max_patches=max_patches, pooling_kernel_size=k
        )
        x = _bicubic_resize(raw, th, tw) / 255.0
        ph, pw = th // p, tw // p
        patches = x.reshape(3, ph, p, pw, p).permute(1, 3, 2, 4, 0).reshape(ph * pw, 3 * p * p)
        soft_counts.append((ph * pw) // (k**2))
        grid = torch.meshgrid(
            torch.arange(pw, dtype=torch.long),
            torch.arange(ph, dtype=torch.long),
            indexing="xy",
        )
        positions = torch.stack(grid, dim=-1).reshape(ph * pw, 2)
        pad_len = max_patches - patches.shape[0]
        if pad_len > 0:
            patches = functional.pad(patches, (0, 0, 0, pad_len), mode="constant", value=0.0)
            positions = functional.pad(positions, (0, 0, 0, pad_len), mode="constant", value=-1)
        pixel_batches.append(patches)
        pos_batches.append(positions)

    pixel_values = torch.stack(pixel_batches, dim=0)
    position_ids = torch.stack(pos_batches, dim=0)
    if dtype is not None:
        pixel_values = pixel_values.to(dtype=dtype)
    if device is not None:
        pixel_values = pixel_values.to(device=device)
        position_ids = position_ids.to(device=device)
    return pixel_values, position_ids, tuple(soft_counts)
