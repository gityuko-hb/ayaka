"""Structural interfaces (``Protocol``) that the rest of the engine depends on.

The engine never imports a concrete tokenizer; it only sees these shapes, so an
adapter can be swapped or faked without touching the service::

    TokenizerLike ----------- required by every adapter
        |  new_decode_stream() -> DecodeStreamLike | None
        |
        +-- also satisfies --> BytesTokenizerLike       exact bytes per token id
        |                        \\__ ByteTokenizerLike  = TokenizerLike + BytesTokenizerLike
        |
        +-- also satisfies --> RenderedEncoderLike      parses control tokens in rendered text

    DetokenizerLike ---------- what the engine calls per request (in or out of process)
    TokenizerFactory --------- builds a LoadedTokenizer from a TokenizerConfig

Dependency direction (arrows point at what is imported)::

    adapters, decoding, service  -->  ports  -->  configs, contracts (types only)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from ayaka.configs.tokenizer import DetokUpdate, TokenizerConfig
    from ayaka.tokenizers.contracts import LoadedTokenizer

__all__ = [
    "ByteTokenizerLike",
    "BytesTokenizerLike",
    "DecodeStreamLike",
    "DetokenizerLike",
    "RenderedEncoderLike",
    "TokenizerFactory",
    "TokenizerLike",
]


@runtime_checkable
class DecodeStreamLike(Protocol):
    """Stateful, incremental decoder for one stream of generated tokens.

    It accumulates partial UTF-8 sequences between calls and releases text only
    at a valid character boundary.
    """

    def step(self, token_id: int) -> str | None:
        """Feed one token id.

        Returns:
            The text that became decodable, or ``None`` while the bytes seen so
            far do not yet complete a character.
        """
        ...


@runtime_checkable
class TokenizerLike(Protocol):
    """The tokenizer surface the engine requires.

    Groups: metadata (``name_or_path``, special ids, sizes), encoding (``encode``,
    ``encode_batch``, ``apply_chat_template``), decoding (``decode``,
    ``convert_*``, ``new_decode_stream``) and vocabulary access.

    ``vocab_size`` is the logical size reported by the tokenizer. ``max_token_id``
    is the highest id it can produce, which exceeds ``vocab_size - 1`` when tokens
    were added beyond the base vocabulary.
    """

    @property
    def name_or_path(self) -> str:
        """Hub id or local directory the tokenizer was loaded from."""
        ...

    def fingerprint(self) -> str:
        """Hash of everything that changes token ids; cache keys must include it."""
        ...

    @property
    def eos_token_id(self) -> int | None:
        """Primary end-of-sequence id, or ``None``."""
        ...

    @property
    def eos_token_ids(self) -> frozenset[int]:
        """Every id that ends generation (the primary EOS plus any extra stop ids)."""
        ...

    @property
    def bos_token_id(self) -> int | None:
        """Beginning-of-sequence id, or ``None``."""
        ...

    @property
    def pad_token_id(self) -> int | None:
        """Padding id, or ``None``."""
        ...

    @property
    def all_special_ids(self) -> frozenset[int]:
        """Ids of all special tokens."""
        ...

    @property
    def all_special_tokens(self) -> tuple[str, ...]:
        """Strings of all special tokens."""
        ...

    @property
    def vocab_size(self) -> int:
        """Logical vocabulary size."""
        ...

    @property
    def max_token_id(self) -> int:
        """Highest id the tokenizer can produce."""
        ...

    @property
    def valid_token_ids(self) -> frozenset[int]:
        """Every id that maps to a token, computed once and shared (never copied per call)."""
        ...

    @property
    def max_chars_per_token(self) -> int:
        """Longest token, measured by the adapter (characters for HF, bytes for tiktoken)."""
        ...

    @property
    def is_fast(self) -> bool:
        """Whether the implementation is native (Rust) rather than pure Python."""
        ...

    @property
    def supports_chunked_encode(self) -> bool:
        """Whether splitting a long text at newline candidates gives identical ids.

        Adapters answer ``True`` only when they can show equivalence for their own
        normalizer, pre-tokenizer, merges and added vocabulary.
        """
        ...

    def encode(self, text: str, *, add_special_tokens: bool = True) -> list[int]:
        """Encode one text into token ids."""
        ...

    def encode_batch(
        self, texts: Sequence[str], *, add_special_tokens: bool = True
    ) -> list[list[int]]:
        """Encode several texts at once, using the native batch path when there is one."""
        ...

    def apply_chat_template(
        self,
        messages: Sequence[Mapping[str, Any]],
        *,
        tools: Sequence[Mapping[str, Any]] | None = None,
        add_generation_prompt: bool = True,
        chat_template: str | None = None,
        **kwargs: Any,
    ) -> str:
        """Render a conversation into a prompt string (not yet tokenized).

        ``chat_template`` overrides the template shipped with the tokenizer.
        Extra keyword arguments are forwarded to the template.
        """
        ...

    def decode(
        self,
        ids: Sequence[int] | int,
        *,
        skip_special_tokens: bool = False,
        spaces_between_special_tokens: bool = True,
    ) -> str:
        """Decode token ids to text.

        Adapters without a notion of spacing around special tokens accept
        ``spaces_between_special_tokens`` and ignore it.
        """
        ...

    def convert_ids_to_tokens(
        self, ids: Sequence[int], *, skip_special_tokens: bool = False
    ) -> list[str]:
        """Map ids to token strings, keeping token boundaries."""
        ...

    def convert_tokens_to_string(self, tokens: Sequence[str]) -> str:
        """Merge token strings (as returned by ``convert_ids_to_tokens``) into text."""
        ...

    def new_decode_stream(self, *, skip_special_tokens: bool = False) -> DecodeStreamLike | None:
        """Create a fresh incremental decode stream, or ``None`` if unsupported."""
        ...

    def get_vocab(self) -> dict[str, int]:
        """A copy of the token-string to id mapping. Copies are O(vocab): not for hot paths."""
        ...

    def get_added_vocab(self) -> dict[str, int]:
        """A copy of the tokens added on top of the base vocabulary."""
        ...

    def get_decoded_vocab(self) -> Sequence[bytes]:
        """Bytes of every token indexed by id (empty ``bytes`` for gaps), read-only and shared."""
        ...


@runtime_checkable
class BytesTokenizerLike(Protocol):
    """Optional: a tokenizer that exposes exact bytes per token id.

    Lets the byte detokenizer skip the reference decode on every token, but only
    after :meth:`verify_byte_path` confirmed the table reproduces ``decode``.
    """

    @property
    def byte_path_verified(self) -> bool:
        """``True`` once :meth:`verify_byte_path` has passed on this instance."""
        ...

    def token_bytes_table(self) -> Sequence[bytes]:
        """Bytes of each token indexed by id; empty if not reconstructible. Shared, read-only."""
        ...

    def verify_byte_path(self, samples: Sequence[str] = ()) -> bool:
        """Probe the byte table against ``decode``; record and return the result."""
        ...


@runtime_checkable
class ByteTokenizerLike(TokenizerLike, BytesTokenizerLike, Protocol):
    """A tokenizer that is both a :class:`TokenizerLike` and a :class:`BytesTokenizerLike`."""


@runtime_checkable
class RenderedEncoderLike(Protocol):
    """Optional: a tokenizer whose plain ``encode`` ignores control tokens.

    Such an adapter (tiktoken) never produces special ids from user text, so text
    that a chat template rendered must go through this method for its control
    tokens to become real special tokens.
    """

    def encode_rendered_batch(self, texts: Sequence[str]) -> list[list[int]]:
        """Encode template-rendered prompts, turning control tokens into special ids."""
        ...


class DetokenizerLike(Protocol):
    """Per-request incremental detokenizer, whether it runs in-process or in a worker."""

    def update(self, new_token_ids: Sequence[int], *, check_stops: bool = True) -> DetokUpdate:
        """Consume new tokens and return the text they produced."""
        ...

    def finish(self) -> DetokUpdate:
        """Flush what is still buffered; later calls return an empty update."""
        ...


@runtime_checkable
class TokenizerFactory(Protocol):
    """Builds a :class:`~ayaka.tokenizers.contracts.LoadedTokenizer` at start-up.

    The composition root calls it once; the result lives for the process.
    """

    @classmethod
    def from_config(cls, config: TokenizerConfig) -> LoadedTokenizer:
        """Load the tokenizer ``config`` describes and report its real capabilities."""
        ...
