"""Compiling a constraint once per grammar, with llguidance, before the request is queued.

:class:`GrammarCompiler` turns :class:`~ayaka.core.constraints.StructuredParams` into a grammar
matcher in its start state and keeps it under a key, the ``constraint_id`` the request carries.
Every request with the same constraint and tokenizer starts from a copy of that template, so the
second identical request compiles nothing. The cache is bounded (least recently used goes first).

Compiling runs on a worker thread with a deadline: a grammar that takes longer, an invalid
schema, or a schema too large is refused with :class:`~ayaka.core.errors.InvalidRequestError`
before the request reaches the queue. Without the ``structured`` extra the request is refused with
:class:`~ayaka.core.errors.UnsupportedFeatureError` naming it, not an import traceback.

llguidance needs the bytes of every token. They come from the tokenizer's byte table when it has
one (byte-level BPE, tiktoken); otherwise each token is decoded alone, which is exact for most
vocabularies but not for SentencePiece's leading spaces.

This module imports llguidance on the first compile, never at import.
"""

from __future__ import annotations

import concurrent.futures
import hashlib
import threading
from collections import OrderedDict
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Final, cast

from ayaka.core.constraints import ConstraintKind
from ayaka.core.errors import CapabilityError, InvalidRequestError, UnsupportedFeatureError
from ayaka.sampling.structured.grammar import grammar_text
from ayaka.sampling.structured.matcher import ConstraintMatcher
from ayaka.utils.import_utils import require_module

if TYPE_CHECKING:
    from collections.abc import Collection, Sequence

    from ayaka.core.constraints import StructuredParams
    from ayaka.sampling.structured.matcher import GrammarState
    from ayaka.tokenizers.ports import TokenizerLike

__all__ = ["COMPILE_SECONDS", "MAX_COMPILED", "GrammarCompiler"]

#: Seconds a grammar may take to compile before the request is refused.
COMPILE_SECONDS: Final = 5.0

#: Compiled grammars kept for reuse.
MAX_COMPILED: Final = 64


@dataclass(frozen=True, slots=True)
class _Compiled:
    template: GrammarState
    params: StructuredParams


class _Vocabulary:
    """What llguidance's ``TokenizerWrapper`` reads: token bytes, specials, EOS, a tokenizer."""

    def __init__(
        self,
        tokenizer: TokenizerLike,
        vocab_size: int,
        eos: Collection[int],
        reserved: Collection[int] = (),
    ) -> None:
        self._tokenizer = tokenizer
        table: list[bytes] = list(_byte_table(tokenizer))
        # Marker tokens are declared special: llguidance then never lets them form ordinary text
        # (a closing marker inside a JSON string would end the call for the parser), and a
        # grammar can still demand one by id.
        specials = set(tokenizer.all_special_ids) | set(reserved)
        tokens = [b""] * vocab_size
        for token in range(vocab_size):
            if token < len(table) and table[token]:
                tokens[token] = table[token]
            elif token not in specials:
                tokens[token] = _decode_one(tokenizer, token)
            if token in specials and not tokens[token]:
                tokens[token] = _decode_one(tokenizer, token) or f"<[{token}]>".encode()
        self.tokens = tokens
        self.special_token_ids = sorted(token for token in specials if token < vocab_size)
        eos_ids = sorted(eos)
        self.eos_token_id = eos_ids[0] if eos_ids else 0
        self.bos_token_id = None

    def __call__(self, text: str | bytes) -> list[int]:
        """Token ids whose bytes are ``text``: llguidance trusts them for the bytes it forces.

        A SentencePiece tokenizer encodes ``'{"'`` as ``'▁{"'``, a token that writes a space
        first; llguidance would then allow only that token where the grammar forbids the space.
        Such a text is encoded after a newline and the newline's tokens are dropped, which
        leaves the tokens of the text alone.
        """
        if isinstance(text, bytes):
            text = text.decode("utf-8", errors="replace")
        ids = list(self._tokenizer.encode(text, add_special_tokens=False))
        wanted = text.encode()
        if self._spells(ids) == wanted:
            return ids
        anchor = list(self._tokenizer.encode("\n", add_special_tokens=False))
        anchored = list(self._tokenizer.encode("\n" + text, add_special_tokens=False))
        if anchored[: len(anchor)] == anchor and self._spells(anchored[len(anchor) :]) == wanted:
            return anchored[len(anchor) :]
        return ids

    def _spells(self, ids: Sequence[int]) -> bytes:
        return b"".join(self.tokens[token] for token in ids if 0 <= token < len(self.tokens))


def _openers(
    tokens: Sequence[bytes], specials: Collection[int]
) -> tuple[frozenset[int], frozenset[int]]:
    """The tokens whose text opens a JSON object after optional blanks, and those to skip.

    Skipped: the blank tokens, and the special tokens other than an end (``specials``), which
    decode to no text; Llama 3.1 writes ``<|python_tag|>`` before a call about a third of the time.
    """
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


def _reserved(params: StructuredParams) -> frozenset[int]:
    """The tool-call marker tokens, which only the grammar's own references may produce."""
    syntax = params.syntax
    if syntax is None:
        return frozenset()
    return frozenset(t for t in (syntax.open_token, syntax.close_token) if t is not None)


def _byte_table(tokenizer: TokenizerLike) -> tuple[bytes, ...]:
    """Each token's bytes in context: the adapter's grammar table, else its byte-level table.

    An empty result leaves ``decode([token])`` as the fallback, which is right only for
    tokenizers whose tokens decode alone as they do in context.
    """
    table = getattr(tokenizer, "grammar_token_bytes", None) or getattr(
        tokenizer, "token_bytes_table", None
    )
    if table is None:
        return ()
    try:
        return tuple(table())
    except (TypeError, ValueError):
        return ()


def _decode_one(tokenizer: TokenizerLike, token: int) -> bytes:
    try:
        return tokenizer.decode([token]).encode()
    except (ValueError, KeyError, IndexError):
        return b""


class GrammarCompiler:
    """Compiles constraints for one tokenizer and model vocabulary.

    Args:
        tokenizer: The engine's tokenizer, or ``None`` (then no constraint can be compiled).
        vocab_size: The model's vocabulary, the width of every mask.
        eos_token_ids: The model's end-of-sequence ids.
        timeout: Seconds a compile may take.
        max_entries: Compiled grammars kept.
    """

    def __init__(
        self,
        tokenizer: TokenizerLike | None,
        *,
        vocab_size: int,
        eos_token_ids: Collection[int],
        timeout: float = COMPILE_SECONDS,
        max_entries: int = MAX_COMPILED,
    ) -> None:
        self._tokenizer = tokenizer
        self._vocab = vocab_size
        self._eos = tuple(sorted(eos_token_ids))
        self._timeout = timeout
        self._max_entries = max_entries
        self._cache: OrderedDict[str, _Compiled] = OrderedDict()
        self._lock = threading.Lock()
        self._ll_tokenizers: dict[frozenset[int], Any] = {}
        # The tokens that open a JSON object and the ones that keep waiting for it (blank or
        # special), for optional calls without an opening marker; from the first vocabulary built.
        self._openers: tuple[frozenset[int], frozenset[int]] | None = None
        self._pool: concurrent.futures.ThreadPoolExecutor | None = None
        self.compiles = 0
        """How many grammars were compiled (cache misses)."""

    @property
    def vocab_size(self) -> int:
        """Bits of every mask."""
        return self._vocab

    def __len__(self) -> int:
        return len(self._cache)

    def compile(self, params: StructuredParams) -> str:
        """Compile a constraint, or find it compiled; return its ``constraint_id``.

        Raises:
            UnsupportedFeatureError: Without the ``structured`` extra or a tokenizer, or for a
                lazy tool call whose opening marker is not one token.
            InvalidRequestError: For an invalid grammar, or one that took too long to compile.
        """
        if (
            params.kind is ConstraintKind.TOOL_CALL
            and params.syntax is not None
            and not params.syntax.opening
            and not params.syntax.bare
        ):
            raise UnsupportedFeatureError(
                f"tool-call constraints need the tokens of the opening marker "
                f"{params.syntax.open!r}",
                context={"features": ("structured tool calls",)},
            )
        key = hashlib.blake2b(params.canonical().encode(), digest_size=16).hexdigest()
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                return key
        template = self._compile(grammar_text(params), _reserved(params))
        with self._lock:
            self._cache[key] = _Compiled(template, params)
            self.compiles += 1
            while len(self._cache) > self._max_entries:
                self._cache.popitem(last=False)
        return key

    def matcher(self, constraint_id: str, params: StructuredParams) -> ConstraintMatcher:
        """A running constraint for one request, from a compiled template.

        ``params`` are the request's own. The cache key covers only what changes the grammar, so
        requests with the same grammar share one template while their start differs:
        ``start_after`` and ``reasoning_open`` follow each request's thinking. Taken from the
        cached entry, they would start every request the way the first one to compile that
        grammar started.

        Raises:
            KeyError: If the constraint was never compiled or has been evicted.
            ValueError: If ``params`` do not compile to ``constraint_id``.
        """
        with self._lock:
            entry = self._cache[constraint_id]
        if params is not entry.params and params.canonical() != entry.params.canonical():
            raise ValueError("the parameters do not match the compiled constraint")
        syntax = params.syntax
        is_call = params.kind is ConstraintKind.TOOL_CALL
        brace: frozenset[int] | None = None
        skip: frozenset[int] = frozenset()
        if is_call and syntax is not None and syntax.bare and params.tool_choice == "auto":
            brace, skip = self._openers or (frozenset(), frozenset())
        return ConstraintMatcher(
            entry.template,
            vocab_size=self._vocab,
            eos_token_ids=self._eos,
            start_after=params.start_after,
            reasoning_open=params.reasoning_open,
            opening=syntax.opening if is_call and syntax is not None else (),
            forced=is_call and params.tool_choice != "auto",
            parallel=params.parallel,
            brace=brace,
            skip=skip,
        )

    def _compile(self, grammar: str, reserved: frozenset[int]) -> GrammarState:
        llguidance = self._module()
        tokenizer = self._llg_tokenizer(llguidance, reserved)
        if self._pool is None:
            self._pool = concurrent.futures.ThreadPoolExecutor(
                max_workers=1, thread_name_prefix="ayaka-grammar"
            )
        future = self._pool.submit(self._build, llguidance, tokenizer, grammar)
        try:
            return future.result(timeout=self._timeout)
        except concurrent.futures.TimeoutError:
            raise InvalidRequestError(
                f"the output constraint took longer than {self._timeout:g} s to compile; "
                "simplify the schema"
            ) from None

    @staticmethod
    def _build(llguidance: Any, tokenizer: Any, grammar: str) -> GrammarState:
        error = llguidance.LLMatcher.validate_grammar(grammar, tokenizer)
        if error:
            raise InvalidRequestError(f"the output constraint is not valid: {error}")
        matcher = llguidance.LLMatcher(tokenizer, grammar, log_level=0)
        if matcher.is_error():
            raise InvalidRequestError(f"the output constraint is not valid: {matcher.get_error()}")
        return cast("GrammarState", matcher)

    def _module(self) -> Any:
        try:
            return require_module(
                "llguidance",
                capability="structured",
                remedy="install the structured extra: uv sync --extra structured",
            )
        except CapabilityError as exc:
            raise UnsupportedFeatureError(
                "structured output needs the structured extra (llguidance), which is not installed",
                context={"features": ("structured",)},
            ) from exc

    def _llg_tokenizer(self, llguidance: Any, reserved: frozenset[int]) -> Any:
        """llguidance's view of the vocabulary, one per set of reserved marker tokens."""
        found = self._ll_tokenizers.get(reserved)
        if found is not None:
            return found
        if self._tokenizer is None:
            raise UnsupportedFeatureError(
                "structured output needs a tokenizer", context={"features": ("structured",)}
            )
        vocabulary = _Vocabulary(self._tokenizer, self._vocab, self._eos, reserved)
        if self._openers is None:
            specials = set(vocabulary.special_token_ids) - set(self._eos)
            self._openers = _openers(vocabulary.tokens, specials)
        wrapper = llguidance.TokenizerWrapper(vocabulary)
        found = llguidance.LLTokenizer(
            wrapper, n_vocab=self._vocab, eos_token=list(self._eos) or None
        )
        self._ll_tokenizers[reserved] = found
        return found

    def close(self) -> None:
        """Stop the compile thread."""
        if self._pool is not None:
            self._pool.shutdown(wait=False, cancel_futures=True)
            self._pool = None
