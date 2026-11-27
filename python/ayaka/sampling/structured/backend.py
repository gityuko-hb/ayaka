"""Backend-neutral contracts and vocabulary helpers for structured output compilers."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, Protocol

if TYPE_CHECKING:
    from collections.abc import Collection, Sequence

    import numpy as np
    from numpy.typing import NDArray

    from ayaka.core.constraints import StructuredParams
    from ayaka.tokenizers.ports import TokenizerLike

__all__ = [
    "StructuredOutputBackend",
    "StructuredOutputBackendName",
    "StructuredOutputGrammar",
    "byte_table",
    "decode_token_bytes",
    "json_openers",
    "reserved_marker_tokens",
    "vocabulary_bytes",
]

type StructuredOutputBackendName = Literal["llguidance", "xgrammar"]


class StructuredOutputGrammar(Protocol):
    """Compiled grammar state machine supporting masking, validation, and rollback."""

    def compute_bitmask(self) -> bytes:
        """Return the allowed next-token bitmask as little-endian bytes."""
        ...

    def consume_token(self, sampled_token: int) -> bool:
        """Advance state by one sampled token; return ``False`` on grammar error."""
        ...

    def accept_tokens(self, tokens: Sequence[int]) -> bool:
        """Advance state by ``tokens``; return ``False`` on grammar error."""
        ...

    def validate_tokens(self, tokens: Sequence[int]) -> bool:
        """Check whether ``tokens`` are valid from the current state without mutating it."""
        ...

    def rollback(self, num_tokens: int) -> None:
        """Undo the last ``num_tokens`` accepted tokens."""
        ...

    def fill_bitmask(self) -> NDArray[np.uint32]:
        """Return the allowed next-token bitmask packed as ``uint32`` words."""
        ...

    def is_stopped(self) -> bool:
        """Whether the grammar reached a stop state."""
        ...

    def is_terminated(self) -> bool:
        """Whether the grammar reached an accepting end state."""
        ...

    def is_error(self) -> bool:
        """Whether the grammar is in an invalid/refused state."""
        ...

    def get_error(self) -> str:
        """Human-readable error reason when :meth:`is_error` is ``True``."""
        ...

    def reset(self) -> None:
        """Reset back to the compiled start state."""
        ...

    def deep_copy(self) -> StructuredOutputGrammar:
        """Return an independent copy starting at this grammar's current state."""
        ...


class StructuredOutputBackend(Protocol):
    """Compiler backend producing :class:`StructuredOutputGrammar` templates."""

    @property
    def name(self) -> StructuredOutputBackendName:
        """Backend identifier (``"llguidance"`` or ``"xgrammar"``)."""
        ...

    def compile(
        self,
        params: StructuredParams,
        *,
        reserved: frozenset[int] = frozenset(),
    ) -> StructuredOutputGrammar:
        """Compile ``params`` into a fresh template grammar."""
        ...

    def openers(self) -> tuple[frozenset[int], frozenset[int]]:
        """Return ``(brace_tokens, skip_tokens)`` for optional bare-JSON tool calls."""
        ...

    def close(self) -> None:
        """Release any background compilation resources."""
        ...


def reserved_marker_tokens(params: StructuredParams) -> frozenset[int]:
    """Tool-call marker token ids reserved from ordinary text emissions."""
    syntax = params.syntax
    if syntax is None:
        return frozenset()
    return frozenset(t for t in (syntax.open_token, syntax.close_token) if t is not None)


def byte_table(tokenizer: TokenizerLike) -> tuple[bytes, ...]:
    """Return raw token-bytes table exposed by a tokenizer adapter when available."""
    table = getattr(tokenizer, "grammar_token_bytes", None) or getattr(
        tokenizer, "token_bytes_table", None
    )
    if table is None:
        return ()
    try:
        return tuple(table())
    except (TypeError, ValueError):
        return ()


def decode_token_bytes(tokenizer: TokenizerLike, token: int) -> bytes:
    """Decode one token id to UTF-8 bytes as fallback when no table exists."""
    try:
        return tokenizer.decode([token]).encode()
    except (ValueError, KeyError, IndexError):
        return b""


def vocabulary_bytes(
    tokenizer: TokenizerLike,
    vocab_size: int,
    reserved: Collection[int] = (),
) -> tuple[list[bytes], list[int]]:
    """Build per-token byte spellings and sorted special token ids up to ``vocab_size``."""
    table = list(byte_table(tokenizer))
    specials = set(tokenizer.all_special_ids) | set(reserved)
    tokens = [b""] * vocab_size
    for token in range(vocab_size):
        if token < len(table) and table[token]:
            tokens[token] = table[token]
        elif token not in specials:
            tokens[token] = decode_token_bytes(tokenizer, token)
        if token in specials and not tokens[token]:
            tokens[token] = decode_token_bytes(tokenizer, token) or f"<[{token}]>".encode()
    special_ids = sorted(token for token in specials if token < vocab_size)
    return tokens, special_ids


def json_openers(
    tokens: Sequence[bytes], specials: Collection[int]
) -> tuple[frozenset[int], frozenset[int]]:
    """Find tokens starting a JSON object (after optional whitespace) and skippable tokens."""
    brace: set[int] = set()
    blank: set[int] = set(specials)
    for token, text in enumerate(tokens):
        stripped = text.lstrip(b" \t\r\n")
        if not text:
            continue
        if not stripped:
            blank.add(token)
        elif stripped.startswith(b"{"):
            brace.add(token)
    return frozenset(brace), frozenset(blank)
