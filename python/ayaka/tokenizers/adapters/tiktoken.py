"""tiktoken adapter: OpenAI-style BPE from a local vocabulary file.

:class:`TiktokenTokenizer` implements :class:`~ayaka.tokenizers.ports.TokenizerLike`,
:class:`~ayaka.tokenizers.ports.BytesTokenizerLike` and
:class:`~ayaka.tokenizers.ports.RenderedEncoderLike`::

    vocab file --> load_tiktoken_vocab --> TiktokenTokenizer
    (+ tokenizer_config.json                  |
       added_tokens_decoder)                  +-- encode(text)             user text: never
                                              |                            produces special ids
                                              +-- apply_chat_template      sandboxed Jinja render
                                              |
                                              '-- encode_rendered_batch    template output: control
                                                                           tokens become special ids

Two encode paths exist on purpose. Text a user typed must never turn into control
tokens, so ``encode`` ignores them. Text a *template* rendered must keep them, so
the service sends it through ``encode_rendered_batch``. The service, not this adapter,
keeps literal special tokens in chat *content* from reaching the template (see
:mod:`ayaka.tokenizers.security`), so ``apply_chat_template`` here renders what it is given.

Only local files are read; a hub id is rejected with a clear error. ``tiktoken`` and
``jinja2`` are imported lazily.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import json
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from ayaka.tokenizers.decoding.byte import verify_byte_table
from ayaka.utils.import_utils import require_module

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from ayaka.configs.tokenizer import TokenizerConfig
    from ayaka.tokenizers.ports import DecodeStreamLike

__all__ = ["DEFAULT_PAT_STR", "TiktokenTokenizer", "load_tiktoken_vocab"]

#: OpenAI ``cl100k`` regex split pattern; used unless the vocabulary file names its own.
DEFAULT_PAT_STR = (
    r"""(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}|"""
    r""" ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"""
)

_VOCAB_FILE_NAMES = ("tiktoken.model", "tokenizer.model", "tokenizer.tiktoken")
_UTF8_BOM = b"\xef\xbb\xbf"
_UTF8 = "utf-8"
_PROBES: tuple[str, ...] = (
    "Hello world",
    "  spaces   and\ttabs\r\nand newlines\n\n",
    "Tiếng Việt, 日本語, Привет, 🚀",
)


@lru_cache(maxsize=32)
def _compile_template(chat_template: str) -> Any:
    """Compile a chat template in a sandbox, cached by source text.

    The template can come from a request (``ChatInput.chat_template``), so it must
    not be able to reach Python internals. The environment mirrors HuggingFace's
    (``trim_blocks``, ``lstrip_blocks``, ``raise_exception``) so templates written
    for it render the same.
    """
    sandbox = require_module(
        "jinja2.sandbox", capability="tokenizer.chat_template", remedy="pip install jinja2"
    )

    def raise_exception(message: str) -> None:
        raise ValueError(message)

    env = sandbox.ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)
    env.globals["raise_exception"] = raise_exception
    return env.from_string(chat_template)


def load_tiktoken_vocab(
    path: Path,
) -> tuple[dict[bytes, int], dict[str, int], str | None, int | None]:
    """Read a vocabulary file in any of the supported layouts.

    Supported: the xtok JSON object (``regular_tokens`` plus ``special_tokens``), a
    flat JSON ``{token_b64: rank}`` mapping, and plain ``<base64> <rank>`` lines.

    Returns:
        ``(mergeable_ranks, special_tokens, pat_str, explicit_n_vocab)``; the last
        two are ``None`` when the file does not specify them.

    Raises:
        ValueError: If the file is empty or not in a supported layout (a
            SentencePiece ``tokenizer.model`` lands here too).
    """
    head = path.read_bytes()[:16]
    head = head.removeprefix(_UTF8_BOM).lstrip()
    try:
        text = path.read_text(encoding="utf-8-sig")
    except UnicodeDecodeError as exc:
        raise ValueError(
            f"{path}: not a tiktoken vocabulary (is it a binary SentencePiece model?)"
        ) from exc

    if head[:1] == b"{":
        obj = cast("dict[str, Any]", json.loads(text))
        if "regular_tokens" in obj:
            merge = {bytes(it["bytes"]): it["token"] for it in obj["regular_tokens"]}
            special = {
                bytes(it["bytes"]).decode(): it["token"] for it in obj.get("special_tokens", [])
            }
            return merge, special, obj.get("pat_str"), obj.get("vocab_size")
        try:
            return {base64.b64decode(k): int(v) for k, v in obj.items()}, {}, None, None
        except (binascii.Error, ValueError) as exc:
            raise ValueError(
                f"{path}: JSON is neither the xtok format nor a flat {{token_b64: rank}} mapping"
            ) from exc

    ranks: dict[bytes, int] = {}
    for lineno, raw_line in enumerate(text.splitlines(), 1):
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 2:
            raise ValueError(f"{path}:{lineno}: expected '<base64> <rank>', got {line[:40]!r}")
        ranks[base64.b64decode(parts[0])] = int(parts[1])
    if not ranks:
        raise ValueError(f"{path}: vocabulary file is empty")
    return ranks, {}, None, None


def _first_special(special: Mapping[str, int], *names: str) -> int | None:
    """Id of the first of ``names`` that is a special token (``0`` counts as present)."""
    for name in names:
        if name in special:
            return special[name]
    return None


class TiktokenTokenizer:
    """``TokenizerLike`` over a tiktoken ``Encoding``.

    EOS, BOS and PAD are guessed from conventional token names (``<|im_end|>``,
    ``<|endoftext|>``, ``</s>``, ``<s>``, ``<|pad|>`` ...) because a tiktoken
    vocabulary does not declare them. Check ``eos_token_id`` for a model whose
    names differ.

    Attributes:
        _enc: The tiktoken ``Encoding``.
        _name: Name or path reported by :attr:`name_or_path`.
        _fp: BLAKE2b-16 fingerprint of pattern, vocabulary and specials.
        _tbl: Bytes of every token indexed by id (shared, never copied).
        _special_str2id: Special token string to id.
        _special_ids: Ids of the special tokens.
        _max_token_id: Highest id in the vocabulary.
        _max_chars: Longest token, in bytes.
        _eos: Guessed EOS id, or ``None``.
        _bos: Guessed BOS id, or ``None``.
        _pad: Guessed PAD id (falls back to EOS), or ``None``.
        _valid_ids: Ids that map to a non-empty token.
        _vocab_str: Cached string-keyed vocabulary, built lazily.
        _chunk_safe: Whether the split pattern is the one chunking was reasoned for.
        byte_path_verified: ``True`` once :meth:`verify_byte_path` passed.
    """

    __slots__ = (
        "_bos",
        "_chunk_safe",
        "_enc",
        "_eos",
        "_fp",
        "_max_chars",
        "_max_token_id",
        "_name",
        "_pad",
        "_special_ids",
        "_special_str2id",
        "_tbl",
        "_valid_ids",
        "_vocab_str",
        "byte_path_verified",
    )

    @classmethod
    def from_config(cls, config: TokenizerConfig) -> TiktokenTokenizer:
        """Load a vocabulary file from ``config.tokenizer`` (a file or a directory).

        A directory is searched for ``tiktoken.model``, ``tokenizer.model`` then
        ``tokenizer.tiktoken``. Extra special tokens are taken from a sibling
        ``tokenizer_config.json`` (``added_tokens_decoder``).

        Raises:
            ValueError: If ``config.tokenizer`` is not a local path (this adapter
                never downloads) or the file is not a tiktoken vocabulary.
            FileNotFoundError: If no vocabulary file is found.
        """
        path = Path(config.tokenizer)
        if path.is_dir():
            for name in _VOCAB_FILE_NAMES:
                if (path / name).is_file():
                    path = path / name
                    break
            else:
                raise FileNotFoundError(
                    f"no tiktoken vocabulary in {path} (looked for {', '.join(_VOCAB_FILE_NAMES)})"
                )
        elif not path.is_file():
            raise ValueError(
                f"tiktoken loads local files only; {str(config.tokenizer)!r} is not a file "
                "or directory"
            )

        merge, special, pat, n_vocab = load_tiktoken_vocab(path)
        cfg_path = path.parent / "tokenizer_config.json"
        if cfg_path.is_file():
            cfg = cast("dict[str, Any]", json.loads(cfg_path.read_text(encoding="utf-8-sig")))
            added = cast("dict[str, dict[str, Any] | None]", cfg.get("added_tokens_decoder") or {})
            for token_id, info in added.items():
                content = (info or {}).get("content")
                if content:
                    special.setdefault(content, int(token_id))

        return cls(
            mergeable_ranks=merge,
            special_tokens=special,
            pat_str=pat or DEFAULT_PAT_STR,
            explicit_n_vocab=n_vocab,
            name=str(config.tokenizer),
        )

    def __init__(
        self,
        *,
        mergeable_ranks: Mapping[bytes, int],
        special_tokens: Mapping[str, int],
        pat_str: str = DEFAULT_PAT_STR,
        explicit_n_vocab: int | None = None,
        name: str = "tiktoken",
    ) -> None:
        """Build the encoding and every table once.

        Args:
            mergeable_ranks: Token bytes to rank (the base BPE vocabulary).
            special_tokens: Special token strings to ids.
            pat_str: Regex that splits text before BPE.
            explicit_n_vocab: Vocabulary size to enforce, if the file declares one.
            name: Name reported by :attr:`name_or_path`.
        """
        tiktoken = require_module(
            "tiktoken", capability="tokenizer.tiktoken", remedy="pip install tiktoken"
        )
        options: dict[str, Any] = {
            "name": name,
            "pat_str": pat_str,
            "mergeable_ranks": dict(mergeable_ranks),
            "special_tokens": dict(special_tokens),
        }
        if explicit_n_vocab is not None:
            options["explicit_n_vocab"] = explicit_n_vocab
        self._enc = tiktoken.Encoding(**options)
        self._name = name
        self._special_str2id = dict(special_tokens)
        self._special_ids = frozenset(special_tokens.values())
        self._chunk_safe = pat_str == DEFAULT_PAT_STR

        self._max_token_id = max(
            max(mergeable_ranks.values(), default=-1),
            max(special_tokens.values(), default=-1),
        )
        table: list[bytes] = [b""] * (self._max_token_id + 1)
        for token_bytes, rank in mergeable_ranks.items():
            table[rank] = token_bytes
        for text, ident in special_tokens.items():
            table[ident] = text.encode(_UTF8)
        self._tbl = tuple(table)
        self._valid_ids = frozenset(i for i, b in enumerate(self._tbl) if b)
        self._max_chars = max(
            max((len(b) for b in mergeable_ranks), default=1),
            max((len(s.encode(_UTF8)) for s in special_tokens), default=1),
        )

        self._eos = _first_special(special_tokens, "<|im_end|>", "<|endoftext|>", "</s>")
        self._bos = _first_special(special_tokens, "<|im_start|>", "<s>")
        pad = _first_special(special_tokens, "<|pad|>", "<pad>", "[PAD]")
        self._pad = pad if pad is not None else self._eos
        self._vocab_str: dict[str, int] | None = None
        self.byte_path_verified = False

        digest = hashlib.blake2b(digest_size=16)
        digest.update(pat_str.encode(_UTF8) + b"\0")
        if explicit_n_vocab is not None:
            digest.update(explicit_n_vocab.to_bytes(4, "little"))
        for token_bytes, rank in sorted(mergeable_ranks.items(), key=lambda kv: kv[1]):
            digest.update(rank.to_bytes(4, "little") + token_bytes + b"\0")
        for text, ident in sorted(special_tokens.items(), key=lambda kv: kv[1]):
            digest.update(ident.to_bytes(4, "little") + text.encode(_UTF8) + b"\0")
        self._fp = digest.hexdigest()

    # -- metadata ------------------------------------------------------------

    @property
    def name_or_path(self) -> str:
        """Name or path this tokenizer was loaded under."""
        return self._name

    def fingerprint(self) -> str:
        """Hex digest of the pattern, vocabulary and special tokens."""
        return self._fp

    @property
    def eos_token_id(self) -> int | None:
        """Guessed EOS id, or ``None``."""
        return self._eos

    @property
    def eos_token_ids(self) -> frozenset[int]:
        """The guessed EOS id as a set (empty if none was found)."""
        return frozenset() if self._eos is None else frozenset({self._eos})

    @property
    def bos_token_id(self) -> int | None:
        """Guessed BOS id, or ``None``."""
        return self._bos

    @property
    def pad_token_id(self) -> int | None:
        """Guessed PAD id (EOS when no pad token exists), or ``None``."""
        return self._pad

    @property
    def all_special_ids(self) -> frozenset[int]:
        """Ids of the special tokens."""
        return self._special_ids

    @property
    def all_special_tokens(self) -> tuple[str, ...]:
        """Strings of the special tokens."""
        return tuple(self._special_str2id)

    @property
    def vocab_size(self) -> int:
        """``n_vocab`` of the encoding: highest id plus one, or the declared size."""
        return int(self._enc.n_vocab)

    @property
    def max_token_id(self) -> int:
        """Highest id in the vocabulary."""
        return self._max_token_id

    @property
    def valid_token_ids(self) -> frozenset[int]:
        """Ids that map to a non-empty token; one shared object, never copied."""
        return self._valid_ids

    @property
    def max_chars_per_token(self) -> int:
        """Length of the longest token, in bytes."""
        return self._max_chars

    @property
    def is_fast(self) -> bool:
        """Always ``True``: tiktoken is native."""
        return True

    @property
    def supports_chunked_encode(self) -> bool:
        """``True`` only for the default split pattern chunking was reasoned for.

        A custom ``pat_str`` from a vocabulary file may cross newline boundaries,
        so the adapter does not claim equivalence for it.
        """
        return self._chunk_safe

    # -- encoding ------------------------------------------------------------

    def encode(self, text: str, *, add_special_tokens: bool = True) -> list[int]:
        """Encode user text; it never produces special ids, so injection is impossible.

        ``add_special_tokens`` is accepted for protocol compatibility and ignored:
        tiktoken adds no BOS/EOS on its own.
        """
        del add_special_tokens
        return self._enc.encode(text, allowed_special=set(), disallowed_special=())

    def encode_batch(
        self, texts: Sequence[str], *, add_special_tokens: bool = True
    ) -> list[list[int]]:
        """Batch form of :meth:`encode`; special tokens in the text stay plain text."""
        del add_special_tokens
        return self._enc.encode_batch(list(texts), allowed_special=set(), disallowed_special=())

    def encode_rendered(self, rendered: str) -> list[int]:
        """Encode template output, turning control tokens into special ids.

        Only call this with text a template produced from content the service sanitized.
        """
        return self._enc.encode(rendered, allowed_special="all")

    def encode_rendered_batch(self, texts: Sequence[str]) -> list[list[int]]:
        """Batch form of :meth:`encode_rendered`."""
        return self._enc.encode_batch(list(texts), allowed_special="all")

    def apply_chat_template(
        self,
        messages: Sequence[Mapping[str, Any]],
        *,
        tools: Sequence[Mapping[str, Any]] | None = None,
        add_generation_prompt: bool = True,
        chat_template: str | None = None,
        **kwargs: Any,
    ) -> str:
        """Render messages with a sandboxed Jinja template.

        Raises:
            ValueError: If ``chat_template`` is missing (tiktoken vocabularies
                carry none) or the template calls ``raise_exception``.
        """
        if chat_template is None:
            raise ValueError("TiktokenTokenizer requires an explicit chat_template")
        template = _compile_template(chat_template)
        return template.render(
            messages=[dict(message) for message in messages],
            tools=tools,
            add_generation_prompt=add_generation_prompt,
            **kwargs,
        )

    # -- decoding ------------------------------------------------------------

    def decode(
        self,
        ids: Sequence[int] | int,
        *,
        skip_special_tokens: bool = False,
        spaces_between_special_tokens: bool = True,
    ) -> str:
        """Decode ids to text, replacing invalid UTF-8.

        ``spaces_between_special_tokens`` is accepted for protocol compatibility;
        tiktoken inserts no spaces around special tokens.
        """
        del spaces_between_special_tokens
        sequence = [ids] if isinstance(ids, int) else [int(i) for i in ids]
        if skip_special_tokens:
            sequence = [i for i in sequence if i not in self._special_ids]
        return self._enc.decode(sequence)

    def convert_ids_to_tokens(
        self, ids: Sequence[int], *, skip_special_tokens: bool = False
    ) -> list[str]:
        """Token strings by id, using reversible ``surrogateescape`` decoding."""
        out: list[str] = []
        for item in ids:
            ident = int(item)
            if skip_special_tokens and ident in self._special_ids:
                continue
            raw = self._tbl[ident] if 0 <= ident < len(self._tbl) else b""
            out.append(raw.decode(_UTF8, errors="surrogateescape"))
        return out

    def convert_tokens_to_string(self, tokens: Sequence[str]) -> str:
        """Merge ``surrogateescape`` token strings back into text."""
        data = b"".join(t.encode(_UTF8, errors="surrogateescape") for t in tokens)
        return data.decode(_UTF8, errors="replace")

    def new_decode_stream(self, *, skip_special_tokens: bool = False) -> DecodeStreamLike | None:
        """Always ``None``: tiktoken has no native stream; use the byte path."""
        del skip_special_tokens
        return None

    # -- vocabulary ----------------------------------------------------------

    def get_vocab(self) -> dict[str, int]:
        """A copy of the string-keyed vocabulary. O(vocab): not for hot paths."""
        if self._vocab_str is None:
            self._vocab_str = {
                raw.decode(_UTF8, errors="surrogateescape"): ident
                for ident, raw in enumerate(self._tbl)
                if raw
            }
        return dict(self._vocab_str)

    def get_added_vocab(self) -> dict[str, int]:
        """A copy of the special tokens."""
        return dict(self._special_str2id)

    def get_decoded_vocab(self) -> tuple[bytes, ...]:
        """Bytes of each token by id (empty for gaps); shared, read-only."""
        return self._tbl

    # -- byte table ----------------------------------------------------------

    def token_bytes_table(self) -> tuple[bytes, ...]:
        """Bytes of each token by id; exact by construction. Shared, read-only."""
        return self._tbl

    def verify_byte_path(self, samples: Sequence[str] = ()) -> bool:
        """Probe the byte table against ``decode`` and record the result."""
        self.byte_path_verified = verify_byte_table(self, samples or _PROBES)
        return self.byte_path_verified
