"""What a grammar is told each token writes is what the token writes after other text.

A grammar masks tokens by the bytes they add. On a SentencePiece vocabulary ``decode([token])``
drops the space a word-initial token (``▁hello``) writes in context, and a byte-fallback token
(``<0xC3>``) decodes alone to the replacement character; a grammar built from it lets the model
write spaces inside JSON keys and names. Checked on real tokenizers of the three layouts
(ByteLevel, Metaspace, byte fallback).
"""

from __future__ import annotations

import random
from typing import TYPE_CHECKING, Any

import pytest

from ayaka.tokenizers.factory import DefaultTokenizerFactory
from real_tokenizers import KINDS, build_hf_dir_of, hf_config

if TYPE_CHECKING:
    from pathlib import Path

pytestmark = pytest.mark.integration

_WORDS = ("hello", "world", "héllo", "wörld", "ça", "va?", "中文", "текст", "🙂", "ok,", "tiny")


def _load(tmp_path: Path, kind: str) -> Any:
    directory = build_hf_dir_of(tmp_path / kind, kind)
    tokenizer = DefaultTokenizerFactory.from_config(hf_config(directory)).tokenizer
    assert tokenizer is not None
    return tokenizer


@pytest.mark.parametrize("kind", KINDS)
def test_the_table_concatenates_to_the_decoded_text_after_a_context(
    tmp_path: Path, kind: str
) -> None:
    tokenizer = _load(tmp_path, kind)
    table = tokenizer.grammar_token_bytes()
    assert table, f"no grammar byte table for the {kind} layout"

    rng = random.Random(0)  # noqa: S311  # a fixed seed, not security
    context = tokenizer.encode("ok", add_special_tokens=False)
    for _ in range(50):
        text = " ".join(rng.choice(_WORDS) for _ in range(8))
        ids = context + tokenizer.encode(" " + text, add_special_tokens=False)
        joined = b"".join(table[token] for token in ids).decode("utf-8", errors="replace")
        decoded = tokenizer.decode(ids)
        # A SentencePiece decoder strips the one space it prepended at the very start.
        assert joined in (decoded, " " + decoded), (kind, text)


@pytest.mark.parametrize("kind", ["metaspace", "byte_fallback"])
def test_the_grammar_sees_the_space_a_word_initial_token_writes(tmp_path: Path, kind: str) -> None:
    from ayaka.sampling.structured.llguidance_backend import (  # pyright: ignore[reportPrivateUsage]
        _Vocabulary,
    )

    tokenizer = _load(tmp_path, kind)
    word = next(token for token in ("▁hello", "▁wo", "▁h") if token in tokenizer.get_vocab())
    ident = tokenizer.get_vocab()[word]
    vocabulary = _Vocabulary(tokenizer, tokenizer.vocab_size, eos=tokenizer.eos_token_ids)

    assert tokenizer.decode([ident]) == word[1:]  # alone, the space is gone
    assert vocabulary.tokens[ident] == (" " + word[1:]).encode()  # in context it is written


def test_a_byte_fallback_token_is_its_byte(tmp_path: Path) -> None:
    tokenizer = _load(tmp_path, "byte_fallback")
    vocab = tokenizer.get_vocab()
    table = tokenizer.grammar_token_bytes()

    assert table[vocab["<0xC3>"]] == b"\xc3"
    assert table[vocab["<0x0A>"]] == b"\n"


@pytest.mark.parametrize("kind", KINDS)
def test_the_grammar_tokenizes_a_text_into_tokens_that_spell_it(tmp_path: Path, kind: str) -> None:
    """llguidance trusts these ids for the bytes it forces, such as ``{"`` at an answer's start."""
    from ayaka.sampling.structured.llguidance_backend import (  # pyright: ignore[reportPrivateUsage]
        _Vocabulary,
    )

    tokenizer = _load(tmp_path, kind)
    vocabulary = _Vocabulary(tokenizer, tokenizer.vocab_size, eos=tokenizer.eos_token_ids)

    for text in ("hello", "world", "hello, world", "tiny"):
        ids = vocabulary(text)
        assert b"".join(vocabulary.tokens[token] for token in ids) == text.encode(), (kind, text)
