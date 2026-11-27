"""Unit and reference-parity tests for pure-Python PNG codec, preprocessing, and placeholders."""

from __future__ import annotations

import base64
import importlib
import struct
import zlib
from typing import Any

import pytest

torch: Any = pytest.importorskip("torch")

from ayaka.core.errors import InvalidRequestError  # noqa: E402
from ayaka.core.multimodal import ImageInput, VisionSpec  # noqa: E402
from ayaka.multimodal.png import (  # noqa: E402
    decode_data_url_image,
    decode_png_rgb,
    encode_png_rgb,
    make_ide_screenshot_png,
    read_png_dimensions,
)
from ayaka.multimodal.preprocess import (  # noqa: E402
    aspect_ratio_preserving_size,
    image_token_count,
    preprocess_gemma4,
    preprocess_qwen3_5,
    smart_resize,
)
from ayaka.multimodal.tokens import expand_image_placeholders  # noqa: E402

pytestmark = pytest.mark.unit


def _make_png_with_type(
    width: int, height: int, color_type: int, bit_depth: int = 8, interlace: int = 0
) -> bytes:
    """Build synthetic PNGs across grayscale, RGB, RGBA, and indexed color types."""
    channels = {0: 1, 2: 3, 3: 1, 4: 2, 6: 4}[color_type]
    rows = bytearray()
    for y in range(height):
        rows.append(y % 5 if interlace == 0 else 0)
        for x in range(width):
            vals = [(x * 17 + y * 31 + c * 53) & 0xFF for c in range(channels)]
            rows.extend(vals)
    # Apply filter 0 simply if interlace == 0 and overwrite filter byte
    if interlace == 0:
        raw = bytearray()
        for y in range(height):
            raw.append(0)
            for x in range(width):
                raw.extend([(x * 17 + y * 31 + c * 53) & 0xFF for c in range(channels)])
    else:
        raw = rows

    def chunk(tag: bytes, payload: bytes) -> bytes:
        crc = zlib.crc32(tag + payload) & 0xFFFFFFFF
        return struct.pack(">I", len(payload)) + tag + payload + struct.pack(">I", crc)

    ihdr = struct.pack(">IIBBBBB", width, height, bit_depth, color_type, 0, 0, interlace)
    plte = b""
    if color_type == 3:
        palette = bytearray(256 * 3)
        for idx in range(256):
            palette[idx * 3] = idx
            palette[idx * 3 + 1] = (idx * 3) & 0xFF
            palette[idx * 3 + 2] = (idx * 7) & 0xFF
        plte = chunk(b"PLTE", bytes(palette))
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", ihdr)
        + plte
        + chunk(b"IDAT", zlib.compress(bytes(raw)))
        + chunk(b"IEND", b"")
    )


def test_png_roundtrip_and_color_types() -> None:
    raw_rgb = bytes((x * 3 + y * 5) & 0xFF for y in range(12) for x in range(16 * 3))
    png_data = encode_png_rgb(16, 12, raw_rgb)
    assert read_png_dimensions(png_data) == (12, 16)
    h, w, decoded = decode_png_rgb(png_data)
    assert (h, w) == (12, 16)
    assert decoded == raw_rgb

    # Grayscale, Gray+Alpha, RGBA, Indexed
    for color_type in (0, 3, 4, 6):
        variant = _make_png_with_type(10, 8, color_type)
        vh, vw, vrgb = decode_png_rgb(variant)
        assert (vh, vw) == (8, 10)
        assert len(vrgb) == 8 * 10 * 3


def test_png_validation_caps_and_screenshot_1024x768() -> None:
    screenshot = make_ide_screenshot_png(1024, 768)
    h, w = read_png_dimensions(screenshot)
    assert (h, w) == (768, 1024)
    dh, dw, rgb = decode_png_rgb(screenshot)
    assert (dh, dw) == (768, 1024)
    assert len(rgb) == 768 * 1024 * 3

    with pytest.raises(InvalidRequestError, match="PNG header"):
        ImageInput(data=b"not-a-png", media_type="image/png")

    corrupted = bytearray(screenshot)
    corrupted[len(corrupted) // 2] ^= 0xFF
    with pytest.raises(InvalidRequestError, match="corrupt"):
        decode_png_rgb(bytes(corrupted))

    # Refuse remote HTTP/HTTPS URLs cleanly
    with pytest.raises(InvalidRequestError, match="refused"):
        decode_data_url_image("https://example.com/cat.png")
    with pytest.raises(InvalidRequestError, match="refused"):
        decode_data_url_image("http://127.0.0.1/cat.png")

    b64 = base64.b64encode(screenshot).decode("ascii")
    parsed = decode_data_url_image(f"data:image/png;base64,{b64}")
    assert parsed.data == screenshot


def test_qwen3_5_and_gemma4_geometry_and_pixel_parity() -> None:
    png_1024 = make_ide_screenshot_png(1024, 768)
    img = ImageInput(data=png_1024, media_type="image/png")

    # Closed-form Qwen3.5 and Gemma 4 resize geometry checks
    assert smart_resize(768, 1024, factor=32, min_pixels=65536, max_pixels=16777216) == (768, 1024)
    assert smart_resize(240, 320, factor=32, min_pixels=65536, max_pixels=16777216) == (256, 320)
    assert smart_resize(500, 900, factor=32, min_pixels=65536, max_pixels=16777216) == (512, 896)
    for h, w in ((768, 1024), (240, 320), (500, 900)):
        gh, gw = aspect_ratio_preserving_size(
            h, w, patch_size=16, max_patches=2520, pooling_kernel_size=3
        )
        assert gh % 48 == 0
        assert gw % 48 == 0
        assert (gh // 16) * (gw // 16) <= 2520

    qspec = VisionSpec(family="qwen3_5", image_token_id=248056, patch_size=16, spatial_merge_size=2)
    q_count, q_grid = image_token_count(768, 1024, qspec)
    assert q_grid == (1, 768 // 16, 1024 // 16)
    assert q_count == (48 // 2) * (64 // 2) == 768
    q_pixels, q_thw = preprocess_qwen3_5([img], qspec)
    assert tuple(q_pixels.shape) == (48 * 64, 3 * 2 * 16 * 16)
    assert q_thw.tolist() == [[1, 48, 64]]

    # Exact Qwen2VL patch ordering check against raw RGB tensor (no resize needed at 768x1024)
    _, _, rgb_raw = decode_png_rgb(png_1024)
    chw = (
        torch.frombuffer(bytearray(rgb_raw), dtype=torch.uint8)
        .reshape(768, 1024, 3)
        .permute(2, 0, 1)
        .float()
    )
    normed = (chw / 255.0 - 0.5) / 0.5
    ref_p = normed.reshape(1, 3, 24, 2, 16, 32, 2, 16).permute(0, 2, 5, 3, 6, 1, 4, 7)
    ref_flat = ref_p.unsqueeze(6).expand(-1, -1, -1, -1, -1, -1, 2, -1, -1).reshape(48 * 64, -1)
    torch.testing.assert_close(q_pixels, ref_flat, atol=1e-6, rtol=1e-6)

    gspec = VisionSpec(
        family="gemma4",
        image_token_id=258880,
        boi_token_id=255999,
        eoi_token_id=258882,
        patch_size=16,
        spatial_merge_size=3,
        max_soft_tokens=280,
    )
    g_count, g_grid = image_token_count(768, 1024, gspec)
    g_pixels, g_pos, g_soft = preprocess_gemma4([img], gspec)
    assert g_soft == (g_count,)
    assert tuple(g_pixels.shape) == (1, 280 * 9, 3 * 16 * 16)
    assert tuple(g_pos.shape) == (1, 280 * 9, 2)
    # Match patch layout and zero-padding for the exact resized image
    nh, nw = g_grid[1], g_grid[2]
    th, tw = nh * 16, nw * 16
    resized = (
        torch.nn.functional.interpolate(
            chw.unsqueeze(0), size=(th, tw), mode="bicubic", align_corners=False, antialias=True
        )
        .clamp(0.0, 255.0)
        .squeeze(0)
        / 255.0
    )
    patches = (
        resized.reshape(3, nh, 16, nw, 16).permute(1, 3, 2, 4, 0).reshape(nh * nw, 16 * 16 * 3)
    )
    ref_patches = torch.zeros(280 * 9, 16 * 16 * 3, dtype=patches.dtype)
    ref_patches[: nh * nw] = patches
    grid = torch.meshgrid(
        torch.arange(nw, dtype=torch.long),
        torch.arange(nh, dtype=torch.long),
        indexing="xy",
    )
    ref_pos = torch.full((280 * 9, 2), -1, dtype=torch.long)
    ref_pos[: nh * nw] = torch.stack(grid, dim=-1).reshape(-1, 2)
    torch.testing.assert_close(g_pixels[0], ref_patches, atol=1e-6, rtol=1e-6)
    assert torch.equal(g_pos[0], ref_pos)


@pytest.mark.integration
def test_placeholder_expansion_and_geometry_match_transformers() -> None:
    transformers: Any = pytest.importorskip("transformers")
    get_aspect_ratio_preserving_size = importlib.import_module(
        "transformers.models.gemma4.processing_gemma4"
    ).get_aspect_ratio_preserving_size

    for h, w in ((768, 1024), (240, 320), (500, 900)):
        assert aspect_ratio_preserving_size(
            h, w, patch_size=16, max_patches=2520, pooling_kernel_size=3
        ) == get_aspect_ratio_preserving_size(
            height=h, width=w, patch_size=16, max_patches=2520, pooling_kernel_size=3
        )

    shot = ImageInput(data=make_ide_screenshot_png(128, 96), media_type="image/png")

    q_tok: Any = transformers.AutoTokenizer.from_pretrained(
        "Qwen/Qwen3.5-9B", local_files_only=True
    )
    q_rendered = q_tok.apply_chat_template(
        [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "hi"}]}],
        tokenize=False,
        add_generation_prompt=True,
    )
    q_ids = q_tok.encode(q_rendered, add_special_tokens=False)
    qspec = VisionSpec(family="qwen3_5", image_token_id=248056, patch_size=16, spatial_merge_size=2)
    q_exp, q_spans = expand_image_placeholders(q_ids, [shot], qspec)
    assert len(q_spans) == 1
    assert q_spans[0].token_count > 1
    assert all(q_exp[q_spans[0].token_offset + k] == 248056 for k in range(q_spans[0].token_count))

    g_tok: Any = transformers.AutoTokenizer.from_pretrained(
        "google/gemma-4-E2B-it", local_files_only=True
    )
    g_rendered = g_tok.apply_chat_template(
        [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "hi"}]}],
        tokenize=False,
        add_generation_prompt=True,
    )
    g_ids = g_tok.encode(g_rendered, add_special_tokens=False)
    gspec = VisionSpec(
        family="gemma4",
        image_token_id=258880,
        boi_token_id=255999,
        eoi_token_id=258882,
        patch_size=16,
        spatial_merge_size=3,
        max_soft_tokens=280,
    )
    g_exp, g_spans = expand_image_placeholders(g_ids, [shot], gspec)
    assert len(g_spans) == 1
    assert g_exp[g_spans[0].token_offset - 1] == 255999
    assert g_exp[g_spans[0].token_offset + g_spans[0].token_count] == 258882
    assert all(g_exp[g_spans[0].token_offset + k] == 258880 for k in range(g_spans[0].token_count))
