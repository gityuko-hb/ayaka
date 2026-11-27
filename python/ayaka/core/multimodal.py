"""Dependency-free multimodal request and layout contracts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

from ayaka.core.errors import InvalidRequestError
from ayaka.utils.validation import require_frozen, require_int, require_text

__all__ = [
    "MAX_IMAGE_BYTES",
    "MAX_IMAGE_PIXELS",
    "SUPPORTED_IMAGE_MEDIA_TYPES",
    "ImageInput",
    "ImageSpan",
    "VisionSpec",
]

#: Maximum accepted encoded payload size per image (20 MiB).
MAX_IMAGE_BYTES: Final[int] = 20 * 2**20

#: Maximum decoded pixel count per image to bound decompression work.
MAX_IMAGE_PIXELS: Final[int] = 16_777_216

#: Image MIME types accepted by the built-in decoder.
SUPPORTED_IMAGE_MEDIA_TYPES: Final[frozenset[str]] = frozenset({"image/png"})

#: PNG file signature bytes.
_PNG_SIGNATURE: Final[bytes] = b"\x89PNG\r\n\x1a\n"


@dataclass(frozen=True, slots=True)
class ImageInput:
    """Validated encoded image bytes attached to a prompt or turn.

    Attributes:
        data: Raw encoded image file bytes (PNG).
        media_type: Lower-case MIME media type, defaulting to ``"image/png"``.
    """

    data: bytes
    media_type: str = "image/png"

    def __post_init__(self) -> None:
        require_frozen(self, "image")
        if type(self.data) is not bytes:
            raise TypeError("image data must be bytes")
        if not self.data:
            raise InvalidRequestError("image data must not be empty")
        if len(self.data) > MAX_IMAGE_BYTES:
            raise InvalidRequestError(
                f"image payload exceeds {MAX_IMAGE_BYTES} byte limit ({len(self.data)} bytes)"
            )
        require_text(self.media_type, "image media_type")
        normalized = self.media_type.lower()
        if normalized != self.media_type:
            object.__setattr__(self, "media_type", normalized)
        if normalized not in SUPPORTED_IMAGE_MEDIA_TYPES:
            raise InvalidRequestError(
                f"unsupported image media type {self.media_type!r}; "
                f"supported: {sorted(SUPPORTED_IMAGE_MEDIA_TYPES)}"
            )
        if normalized == "image/png" and not self.data.startswith(_PNG_SIGNATURE):
            raise InvalidRequestError("image/png payload is missing a valid PNG header")


@dataclass(frozen=True, slots=True)
class ImageSpan:
    """Range of expanded image placeholder tokens inside ``TokenizedRequest``.

    Attributes:
        image_index: Zero-based index into ``TokenizedRequest.images``.
        token_offset: Start token index in ``TokenizedRequest.prompt_token_ids``.
        token_count: Number of placeholder tokens owned by this image.
        grid_thw: ``(temporal, height_patches, width_patches)`` patch grid for rotary mapping.
    """

    image_index: int
    token_offset: int
    token_count: int
    grid_thw: tuple[int, int, int] = (1, 1, 1)

    def __post_init__(self) -> None:
        require_frozen(self, "image_span")
        require_int(self.image_index, "image_index", minimum=0)
        require_int(self.token_offset, "token_offset", minimum=0)
        require_int(self.token_count, "token_count", minimum=1)
        if type(self.grid_thw) is not tuple or len(self.grid_thw) != 3:
            raise TypeError("grid_thw must be a 3-tuple of positive ints")
        for axis in self.grid_thw:
            require_int(axis, "grid_thw entry", minimum=1)


@dataclass(frozen=True, slots=True)
class VisionSpec:
    """Static multimodal tokenization and image-geometry specification of a loaded model.

    Attributes:
        family: Preprocessing/tower family (``"qwen3_5"`` or ``"gemma4"``).
        image_token_id: Vocabulary token id repeated for each visual token.
        boi_token_id: Optional token id inserted before visual tokens (e.g. Gemma 4).
        eoi_token_id: Optional token id inserted after visual tokens (e.g. Gemma 4).
        patch_size: Spatial patch edge in pixels.
        temporal_patch_size: Temporal patch depth (frames per patch).
        spatial_merge_size: Merge/pooling edge factor per visual output token.
        min_pixels: Minimum pixel budget (Qwen3.5 smart resize).
        max_pixels: Maximum pixel budget (Qwen3.5 smart resize).
        max_soft_tokens: Soft-token budget (Gemma 4 aspect-ratio resize).
        mrope_section: Optional 3D rotary section splits for text+vision M-RoPE.
    """

    family: str
    image_token_id: int
    boi_token_id: int | None = None
    eoi_token_id: int | None = None
    patch_size: int = 16
    temporal_patch_size: int = 2
    spatial_merge_size: int = 2
    min_pixels: int = 65536
    max_pixels: int = 16777216
    max_soft_tokens: int = 280
    mrope_section: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        require_frozen(self, "vision_spec")
        if self.family not in ("qwen3_5", "gemma4"):
            raise ValueError(f"unsupported vision family {self.family!r}")
        require_int(self.image_token_id, "image_token_id", minimum=0)
        if self.boi_token_id is not None:
            require_int(self.boi_token_id, "boi_token_id", minimum=0)
        if self.eoi_token_id is not None:
            require_int(self.eoi_token_id, "eoi_token_id", minimum=0)
        require_int(self.patch_size, "patch_size", minimum=1)
        require_int(self.temporal_patch_size, "temporal_patch_size", minimum=1)
        require_int(self.spatial_merge_size, "spatial_merge_size", minimum=1)
        require_int(self.min_pixels, "min_pixels", minimum=1)
        require_int(self.max_pixels, "max_pixels", minimum=1)
        require_int(self.max_soft_tokens, "max_soft_tokens", minimum=1)
