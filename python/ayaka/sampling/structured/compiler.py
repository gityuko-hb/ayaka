"""Compiling structured output constraints per backend and caching compiled templates."""

from __future__ import annotations

import hashlib
import threading
from collections import OrderedDict
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

from ayaka.core.constraints import ConstraintKind
from ayaka.core.errors import ConfigError, UnsupportedFeatureError
from ayaka.sampling.structured.backend import (
    StructuredOutputBackend,
    StructuredOutputBackendName,
    StructuredOutputGrammar,
    reserved_marker_tokens,
)
from ayaka.sampling.structured.llguidance_backend import LLGuidanceBackend
from ayaka.sampling.structured.matcher import ConstraintMatcher
from ayaka.sampling.structured.xgrammar_backend import XGrammarBackend

if TYPE_CHECKING:
    from collections.abc import Collection

    from ayaka.core.constraints import StructuredParams
    from ayaka.tokenizers.ports import TokenizerLike

__all__ = ["COMPILE_SECONDS", "MAX_COMPILED", "GrammarCompiler"]

#: Seconds a grammar may take to compile before the request is refused.
COMPILE_SECONDS: Final = 5.0

#: Compiled grammars kept for reuse.
MAX_COMPILED: Final = 64


@dataclass(frozen=True, slots=True)
class _Compiled:
    template: StructuredOutputGrammar
    params: StructuredParams


def _make_backend(
    name: StructuredOutputBackendName,
    tokenizer: TokenizerLike | None,
    *,
    vocab_size: int,
    eos_token_ids: Collection[int],
    timeout: float,
) -> StructuredOutputBackend:
    if name == "llguidance":
        return LLGuidanceBackend(
            tokenizer, vocab_size=vocab_size, eos_token_ids=eos_token_ids, timeout=timeout
        )
    if name == "xgrammar":
        return XGrammarBackend(tokenizer, vocab_size=vocab_size, eos_token_ids=eos_token_ids)
    raise ConfigError(
        "engine.structured_backend",
        "INVALID_CHOICE",
        f"unknown structured backend {name!r}; expected 'llguidance' or 'xgrammar'",
    )


class GrammarCompiler:
    """Compiles constraints once per canonical key and builds per-request matchers."""

    def __init__(
        self,
        tokenizer: TokenizerLike | None,
        *,
        vocab_size: int,
        eos_token_ids: Collection[int],
        timeout: float = COMPILE_SECONDS,
        max_entries: int = MAX_COMPILED,
        backend: StructuredOutputBackendName = "llguidance",
    ) -> None:
        self._tokenizer = tokenizer
        self._vocab = vocab_size
        self._eos = tuple(sorted(eos_token_ids))
        self._max_entries = max_entries
        self._backend_name: StructuredOutputBackendName = backend
        self._backend = _make_backend(
            backend,
            tokenizer,
            vocab_size=vocab_size,
            eos_token_ids=eos_token_ids,
            timeout=timeout,
        )
        self._cache: OrderedDict[str, _Compiled] = OrderedDict()
        self._lock = threading.Lock()
        self.compiles = 0

    @property
    def backend_name(self) -> StructuredOutputBackendName:
        """Configured compiler backend name."""
        return self._backend_name

    @property
    def vocab_size(self) -> int:
        """Bits of every mask."""
        return self._vocab

    def __len__(self) -> int:
        return len(self._cache)

    def compile(self, params: StructuredParams) -> str:
        """Compile ``params`` or reuse cached template; return its ``constraint_id``."""
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
        key_payload = f"{self._backend_name}:{params.canonical()}"
        key = hashlib.blake2b(key_payload.encode(), digest_size=16).hexdigest()
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                return key
        template = self._backend.compile(params, reserved=reserved_marker_tokens(params))
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
            brace, skip = self._backend.openers()
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

    def close(self) -> None:
        """Release compiler backend resources."""
        self._backend.close()
