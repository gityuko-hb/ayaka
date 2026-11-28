"""MXFP4 (E2M1 with UE8M0 block-32 scales) weight unpacking and dequantization."""

from __future__ import annotations

from typing import Final

import torch

__all__ = ["FP4_E2M1_LUT", "dequantize_mxfp4", "mxfp4_lut"]

#: Ordered 4-bit E2M1 value lookup table matching OpenAI gpt-oss / Hugging Face MXFP4 encoding.
FP4_E2M1_LUT: Final[tuple[float, ...]] = (
    0.0,
    0.5,
    1.0,
    1.5,
    2.0,
    3.0,
    4.0,
    6.0,
    -0.0,
    -0.5,
    -1.0,
    -1.5,
    -2.0,
    -3.0,
    -4.0,
    -6.0,
)

_E2M1_EXPONENT_BIAS: Final = 127
_LUT_CACHE: dict[tuple[torch.device, torch.dtype], torch.Tensor] = {}


def mxfp4_lut(device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Return the cached 16-entry FP4 LUT tensor on ``device`` without H2D copies inside graphs."""
    key = (device, dtype)
    cached = _LUT_CACHE.get(key)
    if cached is None:
        with torch.inference_mode(False):
            cached = torch.tensor(FP4_E2M1_LUT, dtype=dtype, device=device)
        _LUT_CACHE[key] = cached
    return cached


def dequantize_mxfp4(
    blocks: torch.Tensor,
    scales: torch.Tensor,
    *,
    dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Unpack uint8 MXFP4 ``blocks`` ``[..., G, 16]`` and UE8M0 ``scales`` ``[..., G]``.

    Returns dense floating weights ``[..., G * 32]`` in ``dtype`` matching
    :func:`transformers.integrations.mxfp4.convert_moe_packed_tensors` element values.
    """
    if blocks.dtype != torch.uint8:
        raise ValueError(f"mxfp4 blocks must have uint8 dtype, got {blocks.dtype}")
    if scales.dtype != torch.uint8:
        raise ValueError(f"mxfp4 scales must have uint8 dtype, got {scales.dtype}")
    if blocks.shape[:-1] != scales.shape:
        raise ValueError(
            f"mxfp4 blocks leading shape {tuple(blocks.shape[:-1])} must match scales "
            f"{tuple(scales.shape)}"
        )

    lut = mxfp4_lut(blocks.device, dtype)
    *prefix, groups, bytes_per_group = blocks.shape
    rows_total = scales.numel()
    flat_blocks = blocks.reshape(rows_total, bytes_per_group)
    flat_scales = (scales.to(torch.int32) - _E2M1_EXPONENT_BIAS).reshape(rows_total, 1)

    out = torch.empty(rows_total, bytes_per_group * 2, dtype=dtype, device=blocks.device)
    out[:, 0::2] = lut[(flat_blocks & 0x0F).to(torch.int64)]
    out[:, 1::2] = lut[(flat_blocks >> 4).to(torch.int64)]
    torch.ldexp(out, flat_scales, out=out)
    return out.reshape(*prefix, groups * bytes_per_group * 2)
