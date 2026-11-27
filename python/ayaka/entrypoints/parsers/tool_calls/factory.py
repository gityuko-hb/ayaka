"""Factory selecting the concrete tool-call parser for a :class:`ToolCallFormat`."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ayaka.entrypoints.parsers.tool_calls.json_parser import ToolCallParser
from ayaka.entrypoints.parsers.tool_calls.xml_parser import XmlToolCallParser

if TYPE_CHECKING:
    from ayaka.entrypoints.parsers.formats import ToolCallFormat
    from ayaka.entrypoints.parsers.markers import Marker
    from ayaka.entrypoints.parsers.parameters import ToolSchemas
    from ayaka.entrypoints.parsers.tool_calls.base import ToolCallParserLike

__all__ = ["build_tool_call_parser"]


def build_tool_call_parser(
    fmt: ToolCallFormat,
    *,
    call_id_prefix: str,
    markers: tuple[Marker, Marker] | None = None,
    schemas: ToolSchemas | None = None,
) -> ToolCallParserLike:
    """Create a JSON or XML/tag incremental tool-call parser matching ``fmt.style``."""
    if fmt.style == "xml":
        return XmlToolCallParser(
            fmt,
            call_id_prefix=call_id_prefix,
            markers=markers,
            schemas=schemas,
        )
    return ToolCallParser(
        fmt,
        call_id_prefix=call_id_prefix,
        markers=markers,
        schemas=schemas,
    )
