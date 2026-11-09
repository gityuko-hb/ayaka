"""Concrete tokenizers behind :class:`~ayaka.tokenizers.ports.TokenizerLike`.

``hf``       HuggingFace fast or slow tokenizers (needs ``transformers``).
``tiktoken`` OpenAI-style BPE loaded from a local vocabulary file (needs ``tiktoken``).

Third-party packages are imported lazily inside the adapters, so importing this
package costs nothing and a missing package fails with a clear, actionable error.
Public names are resolved from their submodule on first access.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ayaka.utils.lazy_exports import build_lazy_exports

if TYPE_CHECKING:
    from ayaka.tokenizers.adapters.hf import HfTokenizer
    from ayaka.tokenizers.adapters.tiktoken import (
        DEFAULT_PAT_STR,
        TiktokenTokenizer,
        load_tiktoken_vocab,
    )

__all__ = [
    "DEFAULT_PAT_STR",
    "HfTokenizer",
    "TiktokenTokenizer",
    "load_tiktoken_vocab",
]

_EXPORTS = {
    "hf": ("HfTokenizer",),
    "tiktoken": ("DEFAULT_PAT_STR", "TiktokenTokenizer", "load_tiktoken_vocab"),
}

__getattr__, __dir__ = build_lazy_exports(__name__, _EXPORTS, globals())
