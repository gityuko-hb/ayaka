"""Optional ``xgrammar`` structured-output compiler backend.

Applies the production dual-compiler split:

1. ``json_compiler`` disables special tokens so JSON schemas and objects cannot emit control
   tokens that bypass string/structure rules.
2. ``full_vocab_compiler`` preserves special tokens so ``structural_tag`` and marker-delimited
   grammars can require or match model control tokens natively.
"""

from __future__ import annotations

import json
import logging
import threading
from typing import TYPE_CHECKING, Any

import numpy as np

from ayaka.core.constraints import ConstraintKind
from ayaka.core.errors import CapabilityError, InvalidRequestError, UnsupportedFeatureError
from ayaka.sampling.structured.backend import (
    StructuredOutputBackendName,
    StructuredOutputGrammar,
    json_openers,
    vocabulary_bytes,
)
from ayaka.sampling.structured.grammar import call_schema, grammar_text
from ayaka.utils.import_utils import require_module

if TYPE_CHECKING:
    from collections.abc import Collection, Mapping, Sequence

    from numpy.typing import NDArray

    from ayaka.core.constraints import StructuredParams
    from ayaka.tokenizers.ports import TokenizerLike

__all__ = ["XGrammarBackend", "XGrammarGrammar"]

_LOG = logging.getLogger(__name__)


def _json_str(value: Mapping[str, object]) -> str:
    return json.dumps(value, separators=(",", ":"), ensure_ascii=False)


class XGrammarGrammar:
    """Adapter around ``xgrammar.GrammarMatcher`` implementing :class:`StructuredOutputGrammar`."""

    def __init__(self, compiled: Any, xgrammar_mod: Any, vocab_size: int) -> None:
        self._compiled = compiled
        self._xgr = xgrammar_mod
        self._vocab = vocab_size
        self._words = (vocab_size + 31) // 32
        self._matcher = xgrammar_mod.GrammarMatcher(compiled)
        self._bitmask = np.empty((1, self._words), dtype=np.int32)
        self._error = ""

    def compute_bitmask(self) -> bytes:
        return self.fill_bitmask().astype("<u4").tobytes()

    def fill_bitmask(self) -> NDArray[np.uint32]:
        self._bitmask.fill(-1)
        self._matcher.fill_next_token_bitmask(self._bitmask)
        return self._bitmask[0].view(np.uint32).copy()

    def consume_token(self, sampled_token: int) -> bool:
        if self._error:
            return False
        ok = bool(self._matcher.accept_token(sampled_token))
        if not ok:
            self._error = f"token {sampled_token} rejected by xgrammar matcher"
        return ok

    def accept_tokens(self, tokens: Sequence[int]) -> bool:
        return all(self.consume_token(token) for token in tokens)

    def validate_tokens(self, tokens: Sequence[int]) -> bool:
        if not tokens:
            return True
        accepted = 0
        for token in tokens:
            if not self._matcher.accept_token(token):
                break
            accepted += 1
        if accepted:
            self._matcher.rollback(accepted)
        return accepted == len(tokens)

    def rollback(self, num_tokens: int) -> None:
        if num_tokens <= 0:
            return
        self._error = ""
        self._matcher.rollback(num_tokens)

    def is_stopped(self) -> bool:
        return bool(self._matcher.is_terminated())

    def is_terminated(self) -> bool:
        return self.is_stopped()

    def is_error(self) -> bool:
        return bool(self._error)

    def get_error(self) -> str:
        return self._error

    def reset(self) -> None:
        self._error = ""
        self._matcher.reset()

    def deep_copy(self) -> StructuredOutputGrammar:
        clone = XGrammarGrammar.__new__(XGrammarGrammar)
        clone._compiled = self._compiled
        clone._xgr = self._xgr
        clone._vocab = self._vocab
        clone._words = self._words
        clone._matcher = self._matcher.fork()
        clone._bitmask = np.empty((1, self._words), dtype=np.int32)
        clone._error = self._error
        return clone


def _build_tokenizer_info(
    xgr: Any,
    tokenizer: TokenizerLike,
    vocab_size: int,
    eos: Collection[int],
    *,
    include_specials: bool,
) -> tuple[Any, tuple[frozenset[int], frozenset[int]]]:
    tokens, special_ids = vocabulary_bytes(tokenizer, vocab_size)
    openers = json_openers(tokens, set(special_ids) - set(eos))
    eos_list = sorted(eos) or None
    hf_tok = getattr(tokenizer, "_tokenizer", None)
    if hf_tok is not None and include_specials:
        try:
            info = xgr.TokenizerInfo.from_huggingface(
                hf_tok, vocab_size=vocab_size, stop_token_ids=eos_list
            )
            return info, openers
        except (TypeError, ValueError, RuntimeError) as exc:
            _LOG.debug("TokenizerInfo.from_huggingface unavailable, using raw byte vocab: %s", exc)
    encoded_vocab = [t.decode("latin1") if t else "" for t in tokens]
    if not include_specials:
        # Mask non-EOS special tokens out of JSON-only grammars.
        for sid in special_ids:
            if sid not in eos and sid < len(encoded_vocab):
                encoded_vocab[sid] = ""
    info = xgr.TokenizerInfo(
        encoded_vocab,
        xgr.VocabType.RAW,
        vocab_size=vocab_size,
        stop_token_ids=eos_list,
    )
    return info, openers


class XGrammarBackend:
    """Compile constraints with ``xgrammar`` using dual JSON and full-vocab compilers."""

    def __init__(
        self,
        tokenizer: TokenizerLike | None,
        *,
        vocab_size: int,
        eos_token_ids: Collection[int],
    ) -> None:
        self._tokenizer = tokenizer
        self._vocab = vocab_size
        self._eos = tuple(sorted(eos_token_ids))
        self._lock = threading.Lock()
        self._json_compiler: Any = None
        self._full_compiler: Any = None
        self._openers: tuple[frozenset[int], frozenset[int]] | None = None

    @property
    def name(self) -> StructuredOutputBackendName:
        return "xgrammar"

    def compile(
        self,
        params: StructuredParams,
        *,
        reserved: frozenset[int] = frozenset(),
    ) -> StructuredOutputGrammar:
        _ = reserved
        xgr = self._module()
        json_compiler, full_compiler = self._compilers(xgr)
        try:
            compiled = self._compile_with(xgr, params, json_compiler, full_compiler)
        except Exception as exc:
            raise InvalidRequestError(f"the output constraint is not valid: {exc}") from exc
        return XGrammarGrammar(compiled, xgr, self._vocab)

    def openers(self) -> tuple[frozenset[int], frozenset[int]]:
        return self._openers or (frozenset(), frozenset())

    def close(self) -> None:
        with self._lock:
            if self._json_compiler is not None:
                self._json_compiler.clear_cache()
            if self._full_compiler is not None:
                self._full_compiler.clear_cache()

    def _module(self) -> Any:
        try:
            return require_module(
                "xgrammar",
                capability="xgrammar",
                remedy="install the xgrammar extra: uv sync --extra xgrammar",
            )
        except CapabilityError as exc:
            raise UnsupportedFeatureError(
                "the xgrammar structured backend needs the xgrammar extra, which is not installed",
                context={"features": ("xgrammar",)},
            ) from exc

    def _compilers(self, xgr: Any) -> tuple[Any, Any]:
        with self._lock:
            if self._json_compiler is not None and self._full_compiler is not None:
                return self._json_compiler, self._full_compiler
            if self._tokenizer is None:
                raise UnsupportedFeatureError(
                    "structured output needs a tokenizer", context={"features": ("xgrammar",)}
                )
            json_info, openers = _build_tokenizer_info(
                xgr, self._tokenizer, self._vocab, self._eos, include_specials=False
            )
            full_info, _ = _build_tokenizer_info(
                xgr, self._tokenizer, self._vocab, self._eos, include_specials=True
            )
            self._openers = openers
            self._json_compiler = xgr.GrammarCompiler(json_info)
            self._full_compiler = xgr.GrammarCompiler(full_info)
            return self._json_compiler, self._full_compiler

    def _compile_with(
        self, xgr: Any, params: StructuredParams, json_compiler: Any, full_compiler: Any
    ) -> Any:
        if params.kind is ConstraintKind.JSON_SCHEMA:
            assert params.schema is not None  # noqa: S101
            return json_compiler.compile_json_schema(
                _json_str(params.schema), any_whitespace=False, separators=(",", ":")
            )
        if params.kind is ConstraintKind.JSON_OBJECT:
            return json_compiler.compile_builtin_json_grammar()
        if params.kind is ConstraintKind.REGEX:
            assert params.pattern is not None  # noqa: S101
            return full_compiler.compile_regex(params.pattern)
        if params.kind is ConstraintKind.CHOICE:
            options = " | ".join(json.dumps(c, ensure_ascii=False) for c in params.choices)
            return full_compiler.compile_lark(f"start: {options}\n")
        if params.kind is ConstraintKind.GRAMMAR:
            assert params.grammar is not None  # noqa: S101
            if "::=" in params.grammar:
                return full_compiler.compile_grammar(params.grammar)
            return full_compiler.compile_lark(params.grammar)
        if params.kind is ConstraintKind.STRUCTURAL_TAG:
            assert params.structural_tag is not None  # noqa: S101
            st_mod = xgr.structural_tag
            spec = params.structural_tag
            tag_formats: list[Any] = []
            for item in spec.tags:
                if item.grammar is not None:
                    content = st_mod.GrammarFormat(grammar=item.grammar, syntax="lark")
                else:
                    schema = dict(item.schema) if item.schema is not None else {"type": "object"}
                    content = st_mod.JSONSchemaFormat(json_schema=schema)
                tag_formats.append(
                    st_mod.TagFormat(begin=item.begin, content=content, end=item.end)
                )
            st = st_mod.StructuralTag(
                format=st_mod.TriggeredTagsFormat(
                    triggers=list(spec.triggers),
                    tags=tag_formats,
                    at_least_one=spec.at_least_one,
                    stop_after_first=spec.stop_after_first,
                )
            )
            return full_compiler.compile_structural_tag(st)

        # ConstraintKind.TOOL_CALL
        syntax = params.syntax
        assert syntax is not None  # noqa: S101
        only = params.tool_choice if params.tool_choice not in ("auto", "required") else None
        schema = call_schema(
            params.tools,
            strict=params.strict or (syntax.bare and params.tool_choice == "auto"),
            only=only,
            arguments_key=syntax.arguments_key,
        )
        if syntax.bare and not (params.parallel and syntax.separator):
            return json_compiler.compile_json_schema(
                _json_str(schema), any_whitespace=False, separators=(",", ":")
            )
        if not syntax.bare:
            lark = (
                f"start: NEWLINE? call_body NEWLINE? {json.dumps(syntax.close)}\n"
                f"call_body: %json {_json_str(schema)}\n"
                'NEWLINE: "\\n"\n'
            )
            return full_compiler.compile_lark(lark)
        return full_compiler.compile_lark(grammar_text(params))
