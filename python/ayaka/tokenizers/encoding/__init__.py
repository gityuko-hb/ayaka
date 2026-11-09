"""Helpers around the encode path that are not tied to one adapter.

``chunked``  newline splitting of very long prompts (not wired into the service yet).
``media``    tokenizing chat that contains embedded media placeholders.

Public names are resolved from their submodule on first access.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ayaka.utils.lazy_exports import build_lazy_exports

if TYPE_CHECKING:
    from ayaka.tokenizers.encoding.chunked import (
        encode_chunked,
        safe_split_points,
        split_into_chunks,
    )
    from ayaka.tokenizers.encoding.media import encode_media_chat

__all__ = [
    "encode_chunked",
    "encode_media_chat",
    "safe_split_points",
    "split_into_chunks",
]

_EXPORTS = {
    "chunked": ("encode_chunked", "safe_split_points", "split_into_chunks"),
    "media": ("encode_media_chat",),
}

__getattr__, __dir__ = build_lazy_exports(__name__, _EXPORTS, globals())
