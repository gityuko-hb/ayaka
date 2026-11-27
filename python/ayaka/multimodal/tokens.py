"""Image placeholder token expansion matching Hugging Face processor contracts."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ayaka.core.errors import InvalidRequestError
from ayaka.core.multimodal import ImageSpan
from ayaka.multimodal.png import read_png_dimensions
from ayaka.multimodal.preprocess import image_token_count

if TYPE_CHECKING:
    from collections.abc import Sequence

    from ayaka.core.multimodal import ImageInput, VisionSpec

__all__ = ["expand_image_placeholders"]


def expand_image_placeholders(
    token_ids: Sequence[int],
    images: Sequence[ImageInput],
    spec: VisionSpec,
) -> tuple[tuple[int, ...], tuple[ImageSpan, ...]]:
    """Replace each image placeholder token with its exact visual token sequence.

    For Qwen3.5, each occurrence of ``spec.image_token_id`` is expanded to ``N`` copies of
    ``spec.image_token_id``. For Gemma 4, each occurrence of ``spec.image_token_id`` is
    expanded to ``[boi_token_id, *(image_token_id * N), eoi_token_id]`` so ``ImageSpan``
    covers only the ``N`` soft image tokens between the image boundary tokens.

    Raises:
        InvalidRequestError: If the number of image placeholders differs from ``len(images)``.
    """
    if not images:
        return tuple(token_ids), ()

    placeholder_id = spec.image_token_id
    placeholder_count = sum(1 for tok in token_ids if tok == placeholder_id)
    if placeholder_count != len(images):
        raise InvalidRequestError(
            f"prompt contains {placeholder_count} image placeholder tokens, "
            f"but request carries {len(images)} image(s)"
        )

    counts_and_grids = [
        image_token_count(*read_png_dimensions(image.data), spec) for image in images
    ]
    expanded: list[int] = []
    spans: list[ImageSpan] = []
    img_idx = 0
    for tok in token_ids:
        if tok != placeholder_id:
            expanded.append(int(tok))
            continue
        token_count, grid_thw = counts_and_grids[img_idx]
        if spec.boi_token_id is not None:
            expanded.append(spec.boi_token_id)
        offset = len(expanded)
        expanded.extend([placeholder_id] * token_count)
        if spec.eoi_token_id is not None:
            expanded.append(spec.eoi_token_id)
        spans.append(
            ImageSpan(
                image_index=img_idx,
                token_offset=offset,
                token_count=token_count,
                grid_thw=grid_thw,
            )
        )
        img_idx += 1
    return tuple(expanded), tuple(spans)
