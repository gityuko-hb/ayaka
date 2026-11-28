"""JSON Schema-guided parameter lookup and string coercion for tool-call parsers.

XML and tag-based tool-call formats emit parameter values as plain text (``<parameter=days>5``
or ``<parameter=unit>celsius``). Without checking the tool's schema, a parser cannot know whether
``"5"`` should become ``5`` (``integer``) or stay ``"5"`` (``string``). :class:`ToolSchemas` indexes
each offered tool's parameter schema once per request so parsers coerce only non-string schema
fields and preserve declared strings verbatim.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, cast

if TYPE_CHECKING:
    from collections.abc import Sequence

__all__ = ["ToolSchemas"]


class _ToolLike(Protocol):
    @property
    def name(self) -> str: ...

    @property
    def parameters(self) -> Mapping[str, object]: ...


def _schema_types(prop: Mapping[str, object]) -> frozenset[str]:
    found: set[str] = set()
    raw_type = prop.get("type")
    if isinstance(raw_type, str):
        found.add(raw_type)
    elif isinstance(raw_type, list):
        for entry in cast("list[object]", raw_type):
            if isinstance(entry, str):
                found.add(entry)
    for combiner in ("anyOf", "oneOf"):
        variants = prop.get(combiner)
        if isinstance(variants, list):
            for variant in cast("list[object]", variants):
                if isinstance(variant, dict):
                    found |= _schema_types(cast("Mapping[str, object]", variant))
    return frozenset(found)


@dataclass(frozen=True, slots=True)
class ToolSchemas:
    """Immutable parameter-type index across all tools offered on one request."""

    _properties: Mapping[str, Mapping[str, Mapping[str, object]]]

    @classmethod
    def from_tools(cls, tools: Sequence[_ToolLike | Mapping[str, object]]) -> ToolSchemas:
        """Build from schema/spec objects or raw OpenAI/Anthropic tool mappings."""
        by_tool: dict[str, dict[str, Mapping[str, object]]] = {}
        for tool in tools:
            if isinstance(tool, Mapping):
                raw_fn = tool.get("function")
                if isinstance(raw_fn, Mapping):
                    fn_map = cast("Mapping[str, object]", raw_fn)
                    name_val = fn_map.get("name")
                    params_val = fn_map.get("parameters")
                else:
                    name_val = tool.get("name")
                    params_val = tool.get("parameters") or tool.get("input_schema")
                if not isinstance(name_val, str) or not name_val:
                    continue
                name = name_val
                schema_map: Mapping[str, object] = (
                    cast("Mapping[str, object]", params_val)
                    if isinstance(params_val, Mapping)
                    else {}
                )
            else:
                name = tool.name
                schema_map = tool.parameters
            props_raw = schema_map.get("properties")
            if isinstance(props_raw, Mapping):
                props_map = cast("Mapping[str, object]", props_raw)
                by_tool[name] = {
                    k: cast("Mapping[str, object]", v)
                    for k, v in props_map.items()
                    if isinstance(v, Mapping)
                }
            else:
                by_tool[name] = {}
        return cls(by_tool)

    def parameter_schema(self, tool_name: str, param_name: str) -> Mapping[str, object] | None:
        """Return the JSON schema mapping for ``tool_name.param_name``, if declared."""
        tool_props = self._properties.get(tool_name)
        if tool_props is None:
            return None
        return tool_props.get(param_name)

    def coerce_value(self, tool_name: str, param_name: str, raw: str) -> object:
        """Convert raw text ``raw`` according to the declared schema, keeping strings intact."""
        prop = self.parameter_schema(tool_name, param_name)
        if prop is None:
            return raw
        types = _schema_types(prop)
        if not types:
            return raw
        stripped = raw.strip()
        if "string" in types and len(types) == 1:
            return raw
        if "null" in types and stripped.lower() == "null":
            return None
        if "boolean" in types:
            lowered = stripped.lower()
            if lowered == "true":
                return True
            if lowered == "false":
                return False
        if "integer" in types:
            try:
                return int(stripped)
            except ValueError:
                pass
        if "number" in types:
            try:
                return float(stripped)
            except ValueError:
                pass
        if "array" in types or "object" in types:
            try:
                parsed: object = json.loads(stripped)
            except ValueError:
                parsed = None
            if ("array" in types and isinstance(parsed, list)) or (
                "object" in types and isinstance(parsed, dict)
            ):
                return cast("object", parsed)
        return raw

    def coerce_arguments(
        self, tool_name: str, arguments: Mapping[str, object]
    ) -> dict[str, object]:
        """Coerce string values inside ``arguments`` according to ``tool_name``'s schema."""
        return {
            key: self.coerce_value(tool_name, key, value) if isinstance(value, str) else value
            for key, value in arguments.items()
        }
