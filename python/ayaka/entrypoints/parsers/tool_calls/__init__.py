"""Incremental tool-call parsers for JSON and XML/tag model families."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ayaka.utils.lazy_exports import build_lazy_exports

if TYPE_CHECKING:
    from ayaka.entrypoints.parsers.tool_calls.base import RunBuffer, ToolCallParserLike
    from ayaka.entrypoints.parsers.tool_calls.factory import build_tool_call_parser
    from ayaka.entrypoints.parsers.tool_calls.json_parser import ToolCallParser
    from ayaka.entrypoints.parsers.tool_calls.xml_parser import XmlToolCallParser

__all__ = [
    "RunBuffer",
    "ToolCallParser",
    "ToolCallParserLike",
    "XmlToolCallParser",
    "build_tool_call_parser",
]

_EXPORTS = {
    "base": ("RunBuffer", "ToolCallParserLike"),
    "factory": ("build_tool_call_parser",),
    "json_parser": ("ToolCallParser",),
    "xml_parser": ("XmlToolCallParser",),
}

__getattr__, __dir__ = build_lazy_exports(__name__, _EXPORTS, globals())
