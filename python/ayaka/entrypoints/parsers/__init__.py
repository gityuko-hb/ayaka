"""Output parsers: reasoning blocks and function calls found in streamed text.

Each parser is an incremental state machine over ``(token ids, text)`` pieces. Markers that are
single tokens are matched by id; where the stream is cut never changes the events. The format of a
model is chosen by name when the server starts, never guessed from the output.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ayaka.utils.lazy_exports import build_lazy_exports

if TYPE_CHECKING:
    from ayaka.entrypoints.parsers.formats import (
        REASONING_FORMATS,
        TOOL_CALL_FORMATS,
        ReasoningDefault,
        ReasoningFormat,
        ToolCallFormat,
        reasoning_format,
        suggest_tool_call_parser,
        tool_call_format,
    )
    from ayaka.entrypoints.parsers.markers import Marker, MarkerScanner, resolve_marker
    from ayaka.entrypoints.parsers.output import OutputFormat, OutputParser
    from ayaka.entrypoints.parsers.output_grammar import (
        GrammarCoverage,
        OutputGrammarContext,
        ToolStrictLevel,
        build_tool_constraint,
    )
    from ayaka.entrypoints.parsers.parameters import ToolSchemas
    from ayaka.entrypoints.parsers.reasoning import ReasoningParser
    from ayaka.entrypoints.parsers.tool_calls.base import ToolCallParserLike
    from ayaka.entrypoints.parsers.tool_calls.factory import build_tool_call_parser
    from ayaka.entrypoints.parsers.tool_calls.json_parser import ToolCallParser
    from ayaka.entrypoints.parsers.tool_calls.xml_parser import XmlToolCallParser
    from ayaka.entrypoints.parsers.unified import (
        ChatResponseProcessor,
        CombinedParser,
        OutputInterpretation,
        UnifiedParser,
    )

__all__ = [
    "REASONING_FORMATS",
    "TOOL_CALL_FORMATS",
    "ChatResponseProcessor",
    "CombinedParser",
    "GrammarCoverage",
    "Marker",
    "MarkerScanner",
    "OutputFormat",
    "OutputGrammarContext",
    "OutputInterpretation",
    "OutputParser",
    "ReasoningDefault",
    "ReasoningFormat",
    "ReasoningParser",
    "ToolCallFormat",
    "ToolCallParser",
    "ToolCallParserLike",
    "ToolSchemas",
    "ToolStrictLevel",
    "UnifiedParser",
    "XmlToolCallParser",
    "build_tool_call_parser",
    "build_tool_constraint",
    "reasoning_format",
    "resolve_marker",
    "suggest_tool_call_parser",
    "tool_call_format",
]

_EXPORTS = {
    "markers": ("Marker", "MarkerScanner", "resolve_marker"),
    "output": ("OutputFormat", "OutputParser"),
    "output_grammar": (
        "GrammarCoverage",
        "OutputGrammarContext",
        "ToolStrictLevel",
        "build_tool_constraint",
    ),
    "parameters": ("ToolSchemas",),
    "formats": (
        "REASONING_FORMATS",
        "TOOL_CALL_FORMATS",
        "ReasoningDefault",
        "ReasoningFormat",
        "ToolCallFormat",
        "reasoning_format",
        "suggest_tool_call_parser",
        "tool_call_format",
    ),
    "reasoning": ("ReasoningParser",),
    "tool_calls.base": ("ToolCallParserLike",),
    "tool_calls.factory": ("build_tool_call_parser",),
    "tool_calls.json_parser": ("ToolCallParser",),
    "tool_calls.xml_parser": ("XmlToolCallParser",),
    "unified": (
        "ChatResponseProcessor",
        "CombinedParser",
        "OutputInterpretation",
        "UnifiedParser",
    ),
}

__getattr__, __dir__ = build_lazy_exports(__name__, _EXPORTS, globals())
