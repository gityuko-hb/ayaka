"""Default structured output backend built on ``llguidance``."""

from __future__ import annotations

import concurrent.futures
from typing import TYPE_CHECKING, Any, cast

import numpy as np

from ayaka.core.errors import CapabilityError, InvalidRequestError, UnsupportedFeatureError
from ayaka.sampling.structured.backend import (
    StructuredOutputBackendName,
    StructuredOutputGrammar,
    json_openers,
    vocabulary_bytes,
)
from ayaka.sampling.structured.grammar import grammar_text
from ayaka.utils.import_utils import require_module

if TYPE_CHECKING:
    from collections.abc import Collection, Sequence

    from numpy.typing import NDArray

    from ayaka.core.constraints import StructuredParams
    from ayaka.tokenizers.ports import TokenizerLike

__all__ = ["LLGuidanceBackend", "LLGuidanceGrammar"]


class _Vocabulary:
    """Adapter exposing token byte spellings and encoding helpers to ``llguidance``."""

    def __init__(
        self,
        tokenizer: TokenizerLike,
        vocab_size: int,
        eos: Collection[int],
        reserved: Collection[int] = (),
    ) -> None:
        self._tokenizer = tokenizer
        tokens, specials = vocabulary_bytes(tokenizer, vocab_size, reserved)
        self.tokens = tokens
        self.special_token_ids = specials
        eos_ids = sorted(eos)
        self.eos_token_id = eos_ids[0] if eos_ids else 0
        self.bos_token_id = None

    def __call__(self, text: str | bytes) -> list[int]:
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


class LLGuidanceGrammar:
    """``llguidance.LLMatcher`` adapter implementing :class:`StructuredOutputGrammar`."""

    def __init__(self, matcher: Any, vocab_size: int) -> None:
        self._matcher = matcher
        self._words = (vocab_size + 31) // 32
        self._vocab = vocab_size

    def compute_bitmask(self) -> bytes:
        return cast("bytes", self._matcher.compute_bitmask())

    def consume_token(self, sampled_token: int) -> bool:
        return bool(self._matcher.consume_token(sampled_token))

    def accept_tokens(self, tokens: Sequence[int]) -> bool:
        if not tokens:
            return True
        ok = bool(self._matcher.consume_tokens(list(tokens)))
        return ok and not bool(self._matcher.is_error())

    def validate_tokens(self, tokens: Sequence[int]) -> bool:
        if not tokens:
            return True
        return int(self._matcher.validate_tokens(list(tokens))) == len(tokens)

    def rollback(self, num_tokens: int) -> None:
        if num_tokens > 0:
            self._matcher.rollback(num_tokens)

    def fill_bitmask(self) -> NDArray[np.uint32]:
        words = np.frombuffer(self.compute_bitmask(), dtype="<u4")
        return words[: self._words].astype(np.uint32)

    def is_stopped(self) -> bool:
        return bool(self._matcher.is_stopped())

    def is_terminated(self) -> bool:
        return self.is_stopped()

    def is_error(self) -> bool:
        return bool(self._matcher.is_error())

    def get_error(self) -> str:
        return str(self._matcher.get_error())

    def reset(self) -> None:
        self._matcher.reset()

    def deep_copy(self) -> StructuredOutputGrammar:
        return LLGuidanceGrammar(self._matcher.deep_copy(), self._vocab)


class LLGuidanceBackend:
    """Compile :class:`~ayaka.core.constraints.StructuredParams` using ``llguidance``."""

    def __init__(
        self,
        tokenizer: TokenizerLike | None,
        *,
        vocab_size: int,
        eos_token_ids: Collection[int],
        timeout: float,
    ) -> None:
        self._tokenizer = tokenizer
        self._vocab = vocab_size
        self._eos = tuple(sorted(eos_token_ids))
        self._timeout = timeout
        self._ll_tokenizers: dict[frozenset[int], Any] = {}
        self._openers: tuple[frozenset[int], frozenset[int]] | None = None
        self._pool: concurrent.futures.ThreadPoolExecutor | None = None

    @property
    def name(self) -> StructuredOutputBackendName:
        return "llguidance"

    def compile(
        self,
        params: StructuredParams,
        *,
        reserved: frozenset[int] = frozenset(),
    ) -> StructuredOutputGrammar:
        llguidance = self._module()
        tokenizer = self._llg_tokenizer(llguidance, reserved)
        tok_iface = self._tokenizer
        special_map = (
            {
                tok: int(ids[0])
                for tok in tok_iface.all_special_tokens
                if tok
                and len(ids := tok_iface.encode(tok, add_special_tokens=False)) == 1
                and int(ids[0]) in tok_iface.all_special_ids
            }
            if tok_iface is not None
            else None
        )
        grammar = grammar_text(params, special_tokens=special_map)
        if self._pool is None:
            self._pool = concurrent.futures.ThreadPoolExecutor(
                max_workers=1, thread_name_prefix="ayaka-grammar"
            )
        future = self._pool.submit(self._build, llguidance, tokenizer, grammar, self._vocab)
        try:
            return future.result(timeout=self._timeout)
        except concurrent.futures.TimeoutError:
            raise InvalidRequestError(
                f"the output constraint took longer than {self._timeout:g} s to compile; "
                "simplify the schema"
            ) from None

    def openers(self) -> tuple[frozenset[int], frozenset[int]]:
        return self._openers or (frozenset(), frozenset())

    def close(self) -> None:
        if self._pool is not None:
            self._pool.shutdown(wait=False, cancel_futures=True)
            self._pool = None

    @staticmethod
    def _build(llguidance: Any, tokenizer: Any, grammar: str, vocab_size: int) -> LLGuidanceGrammar:
        error = llguidance.LLMatcher.validate_grammar(grammar, tokenizer)
        if error:
            raise InvalidRequestError(f"the output constraint is not valid: {error}")
        matcher = llguidance.LLMatcher(tokenizer, grammar, log_level=0)
        if matcher.is_error():
            raise InvalidRequestError(f"the output constraint is not valid: {matcher.get_error()}")
        return LLGuidanceGrammar(matcher, vocab_size)

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
            self._openers = json_openers(vocabulary.tokens, specials)
        wrapper = llguidance.TokenizerWrapper(vocabulary)
        found = llguidance.LLTokenizer(
            wrapper, n_vocab=self._vocab, eos_token=list(self._eos) or None
        )
        self._ll_tokenizers[reserved] = found
        return found
