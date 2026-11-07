"""HuggingFace tokenizer adapter.

:class:`HfTokenizer` wraps any ``PreTrainedTokenizer`` or ``PreTrainedTokenizerFast``
and presents the :class:`~ayaka.tokenizers.ports.TokenizerLike` surface. For
byte-level BPE vocabularies it also satisfies
:class:`~ayaka.tokenizers.ports.BytesTokenizerLike`::

    AutoTokenizer ------> HfTokenizer ------------------> TokenizerLike
    (transformers)           |                                (encode, decode, ...)
                             +-- new_decode_stream ------> _DecodeStreamAdapter
                             |      (Rust DecodeStream)        = DecodeStreamLike
                             +-- token_bytes_table -----> BytesTokenizerLike
                                    verify_byte_path()        (fast byte detokenizer)

Modes handled here: ``auto`` and ``hf`` load a fast tokenizer, ``slow`` loads the
pure-Python one. ``tiktoken`` and ``mistral`` need their own adapters.

The byte path is opt-in per instance: call :meth:`HfTokenizer.verify_byte_path`
after loading, and only a passing probe lets the byte detokenizer be selected.

``transformers`` is imported lazily, so importing this module needs no extras.
"""

from __future__ import annotations

import hashlib
import json
import logging
from functools import cache
from typing import TYPE_CHECKING, Any, cast

from ayaka.tokenizers.decoding.byte import verify_byte_table
from ayaka.utils.hf_utils import optional_file, parse_eos_token_ids, read_json
from ayaka.utils.import_utils import require_module

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from ayaka.configs.tokenizer import TokenizerConfig
    from ayaka.tokenizers.ports import DecodeStreamLike

__all__ = ["HfTokenizer"]

_LOGGER = logging.getLogger(__name__)

# Pre-tokenizer class names that mark a byte-level BPE vocabulary: candidates for
# byte-table reconstruction, always subject to verification.
_BYTE_LEVEL_TYPES = frozenset({"ByteLevel"})

_CHATML_TEMPLATE = (
    "{% for message in messages %}"
    "{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>\n'}}"
    "{% endfor %}"
    "{% if add_generation_prompt %}{{'<|im_start|>assistant\n'}}{% endif %}"
)

# Texts the byte table must reproduce exactly; each targets a distinct failure.
_DEFAULT_PROBES: tuple[str, ...] = (
    # Leading, trailing and repeated whitespace.
    "Hello world",
    "  Leading,   multiple   spaces, and trailing.  ",
    # Source code: tabs, mixed indentation, CRLF.
    'def foo(x: int = 42) -> str:\n\t"""Docstring."""\n\r\n    return f\'{x}\\n\'\n',
    # Vietnamese: Latin Extended, 2-3 bytes per character.
    "Tiếng Việt: Xin chào thế giới! Phở bò, đường xá, ệ, ữ, ỗ, ậ, ỷ, ỹ.",
    # CJK: 3 bytes per character, no spaces between words.
    "日本語テキスト\uff08漢字・ひらがな・カタカナ\uff09。中文测试\uff0c自然语言处理。한국어.",
    # Right-to-left and other scripts.
    "مرحبا بالعالم! Привет мир! Ελληνικά.",
    # 4-byte characters and ZWJ / variation-selector sequences.
    "Emoji 4-byte: 🚀 🦄 🔥 👨\u200d👩\u200d👧\u200d👦 🏳️\u200d🌈 \u2192 ⚡",
    # Symbols, currency and contractions (pre-tokenizer split rules).
    "Prices: $1,234.56 or €99.99! They're, couldn't, isn't (π ≈ 3.14159 >= 0).",
    # Markup that looks like special tokens.
    '<|im_start|>user\n<tag attr="val">content</tag><|endoftext|>',
    # Zero-width and non-breaking spaces.
    "Zero\u200bWidth\u200cJoiner\u00a0NBSP and\u202fNarrowNBSP.",
)


@cache
def _byte_level_char_to_byte() -> dict[str, int]:
    """Invert GPT-2's byte-to-printable-character map (character to byte value).

    Byte-level BPE vocabularies spell every byte as a printable stand-in character.
    Bytes with a natural printable glyph map to themselves; the other 68 are shifted
    to U+0100 and above.
    """
    printable = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
    code_points = printable[:]
    shift = 0
    for byte in range(256):
        if byte not in printable:
            printable.append(byte)
            code_points.append(256 + shift)
            shift += 1
    return {chr(cp): byte for byte, cp in zip(printable, code_points, strict=True)}


class _DecodeStreamAdapter:
    """Bridges the Rust ``tokenizers.decoders.DecodeStream`` to ``DecodeStreamLike``.

    Attributes:
        _raw: The Rust ``tokenizers.Tokenizer`` that ``DecodeStream.step`` needs.
        _stream: The active ``DecodeStream``.
    """

    __slots__ = ("_raw", "_stream")

    def __init__(self, raw_tokenizer: Any, skip_special_tokens: bool) -> None:
        decoders = require_module("tokenizers.decoders", capability="tokenizer.hf.stream")
        self._raw = raw_tokenizer
        self._stream = decoders.DecodeStream(skip_special_tokens=skip_special_tokens)

    def step(self, token_id: int) -> str | None:
        """Feed one id; ``None`` until the bytes seen so far complete a character."""
        return self._stream.step(self._raw, token_id)


class HfTokenizer:
    """``TokenizerLike`` and ``BytesTokenizerLike`` over a HuggingFace tokenizer.

    Everything expensive is computed once at construction and shared read-only:
    the vocabulary snapshot, ``valid_token_ids`` and the fingerprint. The byte table
    is built on first use. Nothing here truncates; length limits belong to the
    service, so the fingerprint covers only what changes token ids.

    Attributes:
        byte_path_verified: ``True`` once :meth:`verify_byte_path` passed.
        _tok: The wrapped HuggingFace tokenizer.
        _raw: The Rust tokenizer behind it, or ``None`` for a slow tokenizer.
        _name: ``name_or_path`` captured at construction.
        _fp: BLAKE2b-16 fingerprint of the tokenizer state.
        _eos: Primary EOS id, or ``None``.
        _eos_token_ids: Primary EOS plus any extra stop ids.
        _bos: BOS id, or ``None``.
        _pad: PAD id, or ``None``.
        _special_ids: Ids of all special tokens.
        _special_toks: Strings of all special tokens.
        _vocab: Full token-to-id mapping (snapshot, added tokens included).
        _added: Added-token mapping (snapshot).
        _valid_ids: Every id present in ``_vocab``.
        _vocab_size: Logical vocabulary size reported by the tokenizer.
        _max_token_id: Highest id in the vocabulary (never negative).
        _max_chars: Length of the longest token string.
        _is_fast: Whether the tokenizer is Rust-backed.
        _byte_level: Whether the pre-tokenizer is byte-level BPE.
        _decoded_vocab: Cached ``get_decoded_vocab()``, built lazily.
        _byte_table: Cached ``token_bytes_table()``, built lazily.
    """

    __slots__ = (
        "_added",
        "_bos",
        "_byte_level",
        "_byte_table",
        "_decoded_vocab",
        "_eos",
        "_eos_token_ids",
        "_fp",
        "_is_fast",
        "_max_chars",
        "_max_token_id",
        "_name",
        "_pad",
        "_raw",
        "_special_ids",
        "_special_toks",
        "_tok",
        "_valid_ids",
        "_vocab",
        "_vocab_size",
        "byte_path_verified",
    )

    @classmethod
    def from_config(cls, config: TokenizerConfig) -> HfTokenizer:
        """Load the tokenizer ``config`` describes.

        ``mode='slow'`` loads the pure-Python tokenizer; ``auto`` and ``hf`` ask
        for the fast one, and ``hf`` fails if only a slow one exists. Files next
        to the tokenizer are applied as described in
        :mod:`ayaka.utils.hf_utils`: added tokens missing from
        ``tokenizer_config.json``, a chat template stored in ``chat_template.json``
        and extra stop ids from ``generation_config.json``.

        Raises:
            ValueError: For token-ID-only mode or a mode this adapter does not handle,
                or if ``hf`` was requested and only a slow tokenizer exists.
            CapabilityError: If ``transformers`` is not installed.
        """
        if config.skip_tokenizer_init:
            raise ValueError("use DefaultTokenizerFactory for token-ID-only mode")
        if config.mode not in ("auto", "hf", "slow"):
            raise ValueError(f"unsupported tokenizer mode: {config.mode}")
        transformers = require_module(
            "transformers", capability="tokenizer.hf", remedy="pip install transformers"
        )
        source = str(config.tokenizer)
        tok = transformers.AutoTokenizer.from_pretrained(
            source,
            revision=config.revision,
            trust_remote_code=config.trust_remote_code,
            cache_dir=config.download_dir,
            use_fast=config.mode != "slow",
            local_files_only=config.offline,
        )

        def find(filename: str) -> Any:
            return optional_file(
                source,
                filename,
                revision=config.revision,
                cache_dir=config.download_dir,
                offline=config.offline,
            )

        _sync_added_tokens(tok, transformers, find("tokenizer.json"))
        if not getattr(tok, "chat_template", None):
            _load_chat_template(tok, find("chat_template.json"))
        if not getattr(tok, "chat_template", None):
            _apply_chatml_fallback(tok)

        generation_config = read_json(find("generation_config.json"))
        eos_ids = parse_eos_token_ids(
            generation_config if isinstance(generation_config, dict) else None  # pyright: ignore[reportUnknownArgumentType]
        )
        result = cls(tok, eos_token_ids=eos_ids)
        if config.mode == "hf" and not result.is_fast:
            raise ValueError("mode='hf' requires a fast tokenizer")
        if config.mode == "slow" and result.is_fast:
            raise ValueError("mode='slow' requires a slow tokenizer")
        return result

    def __init__(
        self,
        tok: Any,
        *,
        eos_token_ids: frozenset[int] | Sequence[int] | None = None,
    ) -> None:
        """Wrap an existing HuggingFace tokenizer and snapshot its state.

        Args:
            tok: A ``PreTrainedTokenizer`` or ``PreTrainedTokenizerFast``.
            eos_token_ids: Extra stop ids to add to the tokenizer's own EOS.
        """
        self._tok = tok
        self._name = str(getattr(tok, "name_or_path", "<unknown>"))
        self._is_fast = bool(getattr(tok, "is_fast", False))
        self._raw = getattr(tok, "_tokenizer", None) or getattr(tok, "backend_tokenizer", None)
        if self._raw is not None:
            try:
                self._raw.no_truncation()
                self._raw.no_padding()
            except (AttributeError, TypeError):
                _LOGGER.debug("could not disable built-in truncation/padding", exc_info=True)

        self._vocab: dict[str, int] = dict(tok.get_vocab())
        self._added: dict[str, int] = dict(tok.get_added_vocab())
        self._vocab.update(self._added)
        self._valid_ids = frozenset(self._vocab.values())
        # The tokenizer's own vocab_size is the logical size; len(vocab) would
        # also count added tokens.
        self._vocab_size: int = getattr(tok, "vocab_size", len(self._vocab))
        self._max_token_id = max(0, max(self._valid_ids, default=0))
        self._max_chars = max((len(t) for t in self._vocab), default=1)

        self._eos = getattr(tok, "eos_token_id", None)
        self._bos = getattr(tok, "bos_token_id", None)
        self._pad = getattr(tok, "pad_token_id", None)
        # ``all_special_ids`` lists only the named special tokens, but ``decode`` skips every
        # added token flagged ``special``. Take the union so both agree.
        flagged: dict[int, str] = {
            int(ident): str(added.content)
            for ident, added in dict(getattr(tok, "added_tokens_decoder", None) or {}).items()
            if getattr(added, "special", False)
        }
        named_ids = {i for i in (getattr(tok, "all_special_ids", ()) or ()) if i is not None}
        self._special_ids = frozenset(named_ids | flagged.keys())
        named_toks: list[str] = list(getattr(tok, "all_special_tokens", ()) or ())
        extra_toks = [text for _, text in sorted(flagged.items()) if text not in named_toks]
        self._special_toks = tuple(named_toks + extra_toks)
        stop_ids = set(eos_token_ids or ())
        if self._eos is not None:
            stop_ids.add(int(self._eos))
        self._eos_token_ids = frozenset(stop_ids)

        self._byte_level = self._detect_byte_level()
        self._decoded_vocab: tuple[bytes, ...] | None = None
        self._byte_table: tuple[bytes, ...] | None = None
        self.byte_path_verified = False
        self._fp = self._compute_fingerprint()

    # -- construction helpers ------------------------------------------------

    def _detect_byte_level(self) -> bool:
        """Whether the pre-tokenizer is, or contains, a ``ByteLevel`` step.

        A candidate test only; :meth:`verify_byte_path` is the real check. Slow
        tokenizers have no Rust pre-tokenizer to inspect and answer ``False``.
        """
        pre_tokenizer = getattr(self._raw, "pre_tokenizer", None)
        if pre_tokenizer is None:
            return False
        name = type(pre_tokenizer).__name__
        if name in _BYTE_LEVEL_TYPES:
            return True
        if name == "Sequence":
            try:
                return any(type(step).__name__ in _BYTE_LEVEL_TYPES for step in pre_tokenizer)
            except TypeError:
                return False
        return False

    def _compute_fingerprint(self) -> str:
        """BLAKE2b-16 over everything that changes token ids or rendered prompts.

        Covers the class, special tokens, the chat template and the serialized Rust
        tokenizer (or, for slow tokenizers, the vocabulary in id order).
        """
        digest = hashlib.blake2b(digest_size=16)
        header = {
            "class": type(self._tok).__qualname__,
            "specials": self._special_toks,
            "special_ids": sorted(self._special_ids),
            "chat_template": getattr(self._tok, "chat_template", None),
        }
        digest.update(json.dumps(header, sort_keys=True, ensure_ascii=False).encode())
        if self._raw is not None:
            try:
                digest.update(self._raw.to_str().encode())
            except (AttributeError, TypeError, ValueError):
                _LOGGER.debug("tokenizer is not serializable; hashing vocab", exc_info=True)
            else:
                return digest.hexdigest()
        for token, ident in sorted(self._vocab.items(), key=lambda item: item[1]):
            digest.update(f"{ident}\0{token}\0".encode())
        return digest.hexdigest()

    # -- metadata ------------------------------------------------------------

    @property
    def name_or_path(self) -> str:
        """Hub id or directory the tokenizer was loaded from."""
        return self._name

    def fingerprint(self) -> str:
        """32-character hex digest that changes when token ids or prompts could change."""
        return self._fp

    @property
    def eos_token_id(self) -> int | None:
        """Primary EOS id, or ``None``."""
        return self._eos

    @property
    def eos_token_ids(self) -> frozenset[int]:
        """Primary EOS plus extra stop ids from ``generation_config.json``."""
        return self._eos_token_ids

    @property
    def chat_template(self) -> str | None:
        """The chat template in effect, or ``None``."""
        return getattr(self._tok, "chat_template", None)

    @property
    def bos_token_id(self) -> int | None:
        """BOS id, or ``None``."""
        return self._bos

    @property
    def pad_token_id(self) -> int | None:
        """PAD id, or ``None``."""
        return self._pad

    @property
    def all_special_ids(self) -> frozenset[int]:
        """Ids of all special tokens."""
        return self._special_ids

    @property
    def all_special_tokens(self) -> tuple[str, ...]:
        """Strings of all special tokens."""
        return self._special_toks

    @property
    def vocab_size(self) -> int:
        """Base vocabulary size, excluding tokens added beyond it."""
        return self._vocab_size

    @property
    def max_token_id(self) -> int:
        """Highest id in the vocabulary; can exceed ``vocab_size - 1`` with added tokens."""
        return self._max_token_id

    @property
    def valid_token_ids(self) -> frozenset[int]:
        """Every id present in the vocabulary; one shared object, never copied."""
        return self._valid_ids

    @property
    def max_chars_per_token(self) -> int:
        """Length in characters of the longest token string."""
        return self._max_chars

    @property
    def is_fast(self) -> bool:
        """Whether the tokenizer is Rust-backed."""
        return self._is_fast

    @property
    def supports_chunked_encode(self) -> bool:
        """Always ``False``: byte-level detection does not prove chunk equivalence."""
        return False

    # -- encoding ------------------------------------------------------------

    def encode(self, text: str, *, add_special_tokens: bool = True) -> list[int]:
        """Encode one text; special tokens written in ``text`` are parsed as such."""
        return self._tok.encode(text, add_special_tokens=add_special_tokens)

    def encode_batch(
        self, texts: Sequence[str], *, add_special_tokens: bool = True
    ) -> list[list[int]]:
        """Encode several texts in one call (native batching for fast tokenizers)."""
        result = self._tok(
            list(texts),
            add_special_tokens=add_special_tokens,
            padding=False,
            truncation=False,
            return_attention_mask=False,
        )
        return result["input_ids"]

    def apply_chat_template(
        self,
        messages: Sequence[Mapping[str, Any]],
        *,
        tools: Sequence[Mapping[str, Any]] | None = None,
        add_generation_prompt: bool = True,
        chat_template: str | None = None,
        **kwargs: Any,
    ) -> str:
        """Render a conversation with the tokenizer's (or the given) template.

        Raises:
            TypeError: If the template renders anything but a single string.
        """
        rendered = self._tok.apply_chat_template(
            list(messages),
            tools=list(tools) if tools else None,
            add_generation_prompt=add_generation_prompt,
            chat_template=chat_template,
            tokenize=False,
            **kwargs,
        )
        if not isinstance(rendered, str):
            raise TypeError("chat template must render exactly one string")
        return rendered

    # -- decoding ------------------------------------------------------------

    def decode(
        self,
        ids: Sequence[int] | int,
        *,
        skip_special_tokens: bool = False,
        spaces_between_special_tokens: bool = True,
    ) -> str:
        """Decode ids to text without HuggingFace's tokenization-space cleanup."""
        if isinstance(ids, int):
            ids = [ids]
        return self._tok.decode(
            list(ids),
            skip_special_tokens=skip_special_tokens,
            spaces_between_special_tokens=spaces_between_special_tokens,
            clean_up_tokenization_spaces=False,
        )

    def convert_ids_to_tokens(
        self, ids: Sequence[int], *, skip_special_tokens: bool = False
    ) -> list[str]:
        """Map ids to token strings; ``None`` entries become empty strings."""
        out = self._tok.convert_ids_to_tokens(list(ids), skip_special_tokens=skip_special_tokens)
        if isinstance(out, str):
            return [out]
        return [t if t is not None else "" for t in out]

    def convert_tokens_to_string(self, tokens: Sequence[str]) -> str:
        """Merge token strings into text, handling byte-fallback tokens."""
        return self._tok.convert_tokens_to_string(list(tokens))

    def new_decode_stream(self, *, skip_special_tokens: bool = False) -> DecodeStreamLike | None:
        """A fresh native decode stream, or ``None`` for slow tokenizers.

        Also ``None`` when the installed ``tokenizers`` package has no ``DecodeStream``.
        """
        if self._raw is None or not self._is_fast:
            return None
        try:
            return _DecodeStreamAdapter(self._raw, skip_special_tokens)
        except ImportError:
            return None

    # -- vocabulary ----------------------------------------------------------

    def get_vocab(self) -> dict[str, int]:
        """A copy of the full mapping, added tokens included. O(vocab): not for hot paths."""
        return dict(self._vocab)

    def get_added_vocab(self) -> dict[str, int]:
        """A copy of the tokens added on top of the base vocabulary."""
        return dict(self._added)

    def get_decoded_vocab(self) -> tuple[bytes, ...]:
        """UTF-8 bytes of each token by id (empty for gaps); built once, shared read-only.

        Base tokens are first merged to text and re-encoded; added tokens are taken
        literally, since they were inserted after BPE training.
        """
        if self._decoded_vocab is None:
            table: list[bytes] = [b""] * (self._max_token_id + 1)
            for token, ident in self._vocab.items():
                if 0 <= ident <= self._max_token_id:
                    table[ident] = self._tok.convert_tokens_to_string([token]).encode()
            for token, ident in self._added.items():
                if 0 <= ident <= self._max_token_id:
                    table[ident] = token.encode()
            self._decoded_vocab = tuple(table)
        return self._decoded_vocab

    # -- byte table ----------------------------------------------------------

    def token_bytes_table(self) -> tuple[bytes, ...]:
        """Raw bytes of each token by id; built once, shared read-only.

        Empty unless the vocabulary is byte-level BPE. A token spelled with
        characters outside the byte alphabet (a special token stored in the base
        vocabulary) falls back to its UTF-8 text. Added tokens are always literal.
        """
        if self._byte_table is None:
            self._byte_table = self._build_byte_table() if self._byte_level else ()
        return self._byte_table

    def _build_byte_table(self) -> tuple[bytes, ...]:
        char_to_byte = _byte_level_char_to_byte()
        table: list[bytes] = [b""] * (self._max_token_id + 1)
        for token, ident in self._vocab.items():
            if not 0 <= ident <= self._max_token_id:
                continue
            try:
                table[ident] = bytes(char_to_byte[ch] for ch in token)
            except KeyError:
                table[ident] = token.encode()
        for token, ident in self._added.items():
            if 0 <= ident <= self._max_token_id:
                table[ident] = token.encode()
        return tuple(table)

    def verify_byte_path(self, samples: Sequence[str] = ()) -> bool:
        """Probe the byte table against ``decode`` and record the result.

        Args:
            samples: Extra texts to check; the built-in multilingual probes are
                used when empty.

        Returns:
            ``True`` when every probe matched; ``False`` for an empty table or any
            mismatch.
        """
        if not self.token_bytes_table():
            self.byte_path_verified = False
            return False
        self.byte_path_verified = verify_byte_table(self, samples or _DEFAULT_PROBES)
        return self.byte_path_verified


def _sync_added_tokens(tok: Any, transformers: Any, tokenizer_json: Any) -> None:
    """Add tokens declared in ``tokenizer.json`` that ``tokenizer_config.json`` omitted.

    Some checkpoints (for example Qwen3) leave ``<think>`` or ``<tool_response>`` out
    of ``added_tokens_decoder``, so the loaded tokenizer cannot see them.
    """
    data = read_json(tokenizer_json)
    if not isinstance(data, dict):
        return
    declared = cast("list[dict[str, Any]]", cast("dict[str, Any]", data).get("added_tokens") or [])
    if not declared:
        return
    known = set(tok.get_vocab().values())
    missing = [
        transformers.AddedToken(
            entry["content"],
            rstrip=entry.get("rstrip", False),
            lstrip=entry.get("lstrip", False),
            single_word=entry.get("single_word", False),
            normalized=entry.get("normalized", False),
            special=entry.get("special", False),
        )
        for entry in sorted(declared, key=lambda e: e.get("id", 0))
        if entry.get("id") not in known
    ]
    if missing:
        _LOGGER.info("adding %d token(s) declared only in tokenizer.json", len(missing))
        tok.add_tokens(missing)


def _load_chat_template(tok: Any, template_file: Any) -> None:
    """Use ``chat_template.json`` when the tokenizer config carries no template."""
    data = read_json(template_file)
    if isinstance(data, dict) and "chat_template" in data:
        tok.chat_template = data["chat_template"]
    elif isinstance(data, str):
        tok.chat_template = data


def _apply_chatml_fallback(tok: Any) -> None:
    """Install a plain ChatML template when the vocabulary looks like ChatML.

    A guess, not a certainty: a model that merely defines these two tokens gets
    ChatML prompts. It is logged so a wrong guess is visible; supply an explicit
    ``chat_template`` to avoid relying on it.
    """
    vocab: Any = tok.get_vocab() if hasattr(tok, "get_vocab") else {}
    if "<|im_start|>" in vocab and "<|im_end|>" in vocab:
        _LOGGER.warning("no chat template found; falling back to ChatML for %s", tok.name_or_path)
        tok.chat_template = _CHATML_TEMPLATE
