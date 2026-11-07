"""Newline splitting for very long prompts, gated by an adapter's equivalence claim.

A newline is only a *candidate* boundary: normalization, added tokens and BPE merges
can cross it. So a text is split only when the adapter says
``supports_chunked_encode`` and special tokens are off; otherwise the whole prompt
goes through the adapter in one call::

    encode_chunked(text)
        |- add_special_tokens or not supports_chunked_encode --> encode whole (1 chunk)
        |- text shorter than ``chunk_chars``                  --> encode whole (1 chunk)
        '- else  split_into_chunks --> encode_batch --> concatenate ids

Status: nothing in the service calls this yet, so ``chunk_chars`` and
``long_prompt_chars`` in the tokenizer config have no effect.
"""

from __future__ import annotations

from itertools import pairwise
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ayaka.tokenizers.ports import TokenizerLike

__all__ = ["encode_chunked", "safe_split_points", "split_into_chunks"]


def safe_split_points(text: str, target_chars: int) -> list[int]:
    """Offsets of candidate cuts, each just after a newline followed by non-space.

    Cuts are at least ``target_chars`` apart. The name is historical: these are
    candidates, not a proof that cutting there leaves the token ids unchanged.

    Args:
        text: The text to split.
        target_chars: Minimum chunk length; ``<= 0`` or a shorter text gives no cuts.

    Returns:
        Strictly increasing offsets inside ``(0, len(text))``.
    """
    if target_chars <= 0 or len(text) <= target_chars:
        return []
    points: list[int] = []
    last = 0
    while last + target_chars < len(text):
        probe = last + target_chars
        cut = -1
        while probe < len(text):
            newline = text.find("\n", probe)
            if newline == -1 or newline + 1 >= len(text):
                break
            if not text[newline + 1].isspace():
                cut = newline + 1
                break
            probe = newline + 1
        if cut == -1:
            break
        points.append(cut)
        last = cut
    return points


def split_into_chunks(text: str, target_chars: int) -> list[str]:
    """Split ``text`` at :func:`safe_split_points`; joining the chunks restores it exactly."""
    points = [0, *safe_split_points(text, target_chars), len(text)]
    return [text[start:end] for start, end in pairwise(points)]


def encode_chunked(
    tokenizer: TokenizerLike,
    text: str,
    *,
    chunk_chars: int,
    add_special_tokens: bool = True,
) -> tuple[list[int], int]:
    """Encode ``text``, in independent chunks only when the adapter guarantees it is safe.

    Args:
        tokenizer: The adapter; must report ``supports_chunked_encode`` to be chunked.
        text: The text to encode.
        chunk_chars: Target chunk length in characters.
        add_special_tokens: Special tokens force the whole-prompt path.

    Returns:
        The token ids and how many chunks were encoded independently.

    Raises:
        ValueError: If ``encode_batch`` returns the wrong number of results.
    """
    if add_special_tokens or not tokenizer.supports_chunked_encode:
        return tokenizer.encode(text, add_special_tokens=add_special_tokens), 1
    chunks = split_into_chunks(text, chunk_chars)
    if len(chunks) == 1:
        return tokenizer.encode(text, add_special_tokens=add_special_tokens), 1
    parts = tokenizer.encode_batch(chunks, add_special_tokens=False)
    if len(parts) != len(chunks):
        raise ValueError("encode_batch cardinality mismatch")
    return [token for part in parts for token in part], len(chunks)
