"""Pure-Python multimodal PNG codec and deterministic vision preprocessing."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ayaka.utils.lazy_exports import build_lazy_exports

if TYPE_CHECKING:
    from .png import (
        decode_data_url_image,
        decode_png_rgb,
        encode_png_rgb,
        make_ide_screenshot_png,
        read_png_dimensions,
    )
    from .preprocess import (
        aspect_ratio_preserving_size,
        image_token_count,
        preprocess_gemma4,
        preprocess_qwen3_5,
        smart_resize,
    )
    from .tokens import expand_image_placeholders

__all__ = [
    "aspect_ratio_preserving_size",
    "decode_data_url_image",
    "decode_png_rgb",
    "encode_png_rgb",
    "expand_image_placeholders",
    "image_token_count",
    "make_ide_screenshot_png",
    "preprocess_gemma4",
    "preprocess_qwen3_5",
    "read_png_dimensions",
    "smart_resize",
]

_EXPORTS = {
    "png": (
        "decode_data_url_image",
        "decode_png_rgb",
        "encode_png_rgb",
        "make_ide_screenshot_png",
        "read_png_dimensions",
    ),
    "preprocess": (
        "aspect_ratio_preserving_size",
        "image_token_count",
        "preprocess_gemma4",
        "preprocess_qwen3_5",
        "smart_resize",
    ),
    "tokens": ("expand_image_placeholders",),
}

__getattr__, __dir__ = build_lazy_exports(__name__, _EXPORTS, globals())
