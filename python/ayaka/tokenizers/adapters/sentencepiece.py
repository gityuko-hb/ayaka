"""What each token of a SentencePiece vocabulary writes in context, read from its token strings.

SentencePiece marks a word's leading space with ``▁`` (U+2581) inside the token string, and its
decoders turn the marker back into a space; with byte fallback, a character the vocabulary lacks
is spelled as ``<0xNN>`` tokens, one per byte. ``decode([token])`` hides both: alone, a
word-initial token loses its space and a fallback byte becomes the replacement character. A
grammar needs the bytes a token adds after other text, which the token string gives directly.

Nothing here needs ``tokenizers`` or ``transformers``: the layout is read from the serialized
``tokenizer.json`` and the table from the vocabulary mapping.
"""

from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING, Final, cast

if TYPE_CHECKING:
    from collections.abc import Mapping

__all__ = ["sentencepiece_layout", "sentencepiece_table"]

#: SentencePiece's word marker, which its decoders turn into a space.
_METASPACE: Final = "▁"
#: A byte-fallback token (``<0x0A>``) stands for that one byte.
_BYTE_FALLBACK: Final = re.compile(r"<0x([0-9A-Fa-f]{2})>")


def _mapping(value: object) -> dict[str, object]:
    """A JSON object; anything else reads as empty."""
    return cast("dict[str, object]", value) if isinstance(value, dict) else {}


def sentencepiece_layout(tokenizer_json: str) -> tuple[bool, bool]:
    """Whether the decoder turns ``▁`` into a space, and whether the vocabulary has byte fallback.

    The marker is a ``Metaspace`` decoder (SentencePiece's own) or a ``Replace`` of ``▁`` by a
    space (Llama 2's ``Sequence``). Byte fallback is a ``ByteFallback`` decoder step or the
    model's ``byte_fallback`` flag. Text that is not a JSON object answers ``(False, False)``.
    """
    try:
        spec = _mapping(json.loads(tokenizer_json))
    except ValueError:
        return False, False
    decoder = _mapping(spec.get("decoder"))
    nested = decoder.get("decoders")
    steps: list[object] = (
        list(cast("list[object]", nested)) if isinstance(nested, list) else [decoder]
    )
    marker = byte_fallback = False
    for raw_step in steps:
        step = _mapping(raw_step)
        kind = step.get("type")
        pattern = _mapping(step.get("pattern"))
        if kind == "Metaspace" or (
            kind == "Replace" and pattern.get("String") == _METASPACE and step.get("content") == " "
        ):
            marker = True
        elif kind == "ByteFallback":
            byte_fallback = True
    if _mapping(spec.get("model")).get("byte_fallback") is True:
        byte_fallback = True
    return marker, byte_fallback


def sentencepiece_table(
    vocab: Mapping[str, int],
    added: Mapping[str, int],
    size: int,
    *,
    byte_fallback: bool,
) -> list[bytes]:
    """Each id's bytes: ``▁`` as a space, ``<0xNN>`` as that byte, added tokens as written.

    Args:
        vocab: Token string to id, added tokens included.
        added: The added tokens, written literally (a decoder never rewrites them).
        size: Ids from 0 to ``size - 1`` get an entry; an id without a token gets ``b""``.
        byte_fallback: Whether ``<0xNN>`` tokens stand for bytes.
    """
    table = [b""] * size
    for token, ident in vocab.items():
        if not 0 <= ident < size:
            continue
        byte = _BYTE_FALLBACK.fullmatch(token) if byte_fallback else None
        if byte is not None:
            table[ident] = bytes([int(byte.group(1), 16)])
        else:
            table[ident] = token.replace(_METASPACE, " ").encode()
    for token, ident in added.items():
        if 0 <= ident < size:
            table[ident] = token.encode()
    return table
