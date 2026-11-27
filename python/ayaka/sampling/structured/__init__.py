"""Structured output: grammars from JSON schemas, regexes, choices, structural tags, and tools."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ayaka.utils.lazy_exports import build_lazy_exports

if TYPE_CHECKING:
    from ayaka.sampling.structured.backend import (
        StructuredOutputBackend,
        StructuredOutputBackendName,
        StructuredOutputGrammar,
    )
    from ayaka.sampling.structured.compiler import COMPILE_SECONDS, MAX_COMPILED, GrammarCompiler
    from ayaka.sampling.structured.grammar import call_schema, compact_schema, grammar_text
    from ayaka.sampling.structured.llguidance_backend import LLGuidanceBackend, LLGuidanceGrammar
    from ayaka.sampling.structured.matcher import ConstraintMatcher, GrammarState
    from ayaka.sampling.structured.runtime import ConstraintRuntime
    from ayaka.sampling.structured.xgrammar_backend import XGrammarBackend, XGrammarGrammar

__all__ = [
    "COMPILE_SECONDS",
    "MAX_COMPILED",
    "ConstraintMatcher",
    "ConstraintRuntime",
    "GrammarCompiler",
    "GrammarState",
    "LLGuidanceBackend",
    "LLGuidanceGrammar",
    "StructuredOutputBackend",
    "StructuredOutputBackendName",
    "StructuredOutputGrammar",
    "XGrammarBackend",
    "XGrammarGrammar",
    "call_schema",
    "compact_schema",
    "grammar_text",
]

_EXPORTS = {
    "backend": (
        "StructuredOutputBackend",
        "StructuredOutputBackendName",
        "StructuredOutputGrammar",
    ),
    "compiler": ("COMPILE_SECONDS", "MAX_COMPILED", "GrammarCompiler"),
    "grammar": ("call_schema", "compact_schema", "grammar_text"),
    "llguidance_backend": ("LLGuidanceBackend", "LLGuidanceGrammar"),
    "matcher": ("ConstraintMatcher", "GrammarState"),
    "runtime": ("ConstraintRuntime",),
    "xgrammar_backend": ("XGrammarBackend", "XGrammarGrammar"),
}

__getattr__, __dir__ = build_lazy_exports(__name__, _EXPORTS, globals())
