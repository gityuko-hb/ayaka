"""Structured output: grammars from JSON schemas and tool definitions, and their matchers.

Public names are resolved from the module that defines them on first access; llguidance is
imported only when a constraint is compiled, so importing this package never loads it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ayaka.utils.lazy_exports import build_lazy_exports

if TYPE_CHECKING:
    from ayaka.sampling.structured.compiler import COMPILE_SECONDS, MAX_COMPILED, GrammarCompiler
    from ayaka.sampling.structured.grammar import call_schema, grammar_text
    from ayaka.sampling.structured.matcher import ConstraintMatcher, GrammarState
    from ayaka.sampling.structured.runtime import ConstraintRuntime

__all__ = [
    "COMPILE_SECONDS",
    "MAX_COMPILED",
    "ConstraintMatcher",
    "ConstraintRuntime",
    "GrammarCompiler",
    "GrammarState",
    "call_schema",
    "grammar_text",
]

_EXPORTS = {
    "compiler": ("COMPILE_SECONDS", "MAX_COMPILED", "GrammarCompiler"),
    "grammar": ("call_schema", "grammar_text"),
    "matcher": ("ConstraintMatcher", "GrammarState"),
    "runtime": ("ConstraintRuntime",),
}

__getattr__, __dir__ = build_lazy_exports(__name__, _EXPORTS, globals())
