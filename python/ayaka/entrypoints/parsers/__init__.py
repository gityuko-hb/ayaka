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
        ReasoningFormat,
        ToolCallFormat,
        reasoning_format,
        suggest_tool_call_parser,
        tool_call_format,
    )
    from ayaka.entrypoints.parsers.markers import Marker, MarkerScanner, resolve_marker
    from ayaka.entrypoints.parsers.output import OutputFormat, OutputParser
    from ayaka.entrypoints.parsers.reasoning import ReasoningParser
    from ayaka.entrypoints.parsers.tool_calls import ToolCallParser
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
    "Marker",
    "MarkerScanner",
    "OutputFormat",
    "OutputInterpretation",
    "OutputParser",
    "ReasoningFormat",
    "ReasoningParser",
    "ToolCallFormat",
    "ToolCallParser",
    "UnifiedParser",
    "reasoning_format",
    "resolve_marker",
    "suggest_tool_call_parser",
    "tool_call_format",
]

_EXPORTS = {
    "markers": ("Marker", "MarkerScanner", "resolve_marker"),
    "output": ("OutputFormat", "OutputParser"),
    "formats": (
        "REASONING_FORMATS",
        "TOOL_CALL_FORMATS",
        "ReasoningFormat",
        "ToolCallFormat",
        "reasoning_format",
        "suggest_tool_call_parser",
        "tool_call_format",
    ),
    "reasoning": ("ReasoningParser",),
    "tool_calls": ("ToolCallParser",),
    "unified": (
        "ChatResponseProcessor",
        "CombinedParser",
        "OutputInterpretation",
        "UnifiedParser",
    ),
}

__getattr__, __dir__ = build_lazy_exports(__name__, _EXPORTS, globals())
