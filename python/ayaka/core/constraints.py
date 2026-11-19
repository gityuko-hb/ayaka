"""What a request asks its output to obey: a JSON schema, a JSON object, or a tool-call grammar.

:class:`StructuredParams` is the protocol-neutral description the OpenAI and Anthropic adapters
build from ``response_format``, ``text.format``, ``output_format``, ``tools``, ``tool_choice`` and
``strict``. It is plain data: the grammar compiler lives in :mod:`ayaka.sampling.structured`, so
core never imports a grammar library.

Two ideas shape it:

* A constraint may start late. ``start_after`` names a token id that switches it on (the end of a
  reasoning block, the marker that opens a tool call); until that token is emitted the output is
  free and costs nothing.
* A tool call is constrained **lazily** by default: text before a call is free, the call itself
  (name and arguments) follows the grammar, and what comes after depends on ``parallel``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING, Final, cast

from .errors import InvalidRequestError

if TYPE_CHECKING:
    from collections.abc import Mapping

__all__ = [
    "MAX_SCHEMA_BYTES",
    "ConstraintKind",
    "StructuredParams",
    "ToolCallSyntax",
    "ToolSchema",
    "as_schema",
]

#: Largest schema, in bytes of canonical JSON, a request may send (all tools together).
MAX_SCHEMA_BYTES: Final[int] = 64 * 1024


class ConstraintKind(StrEnum):
    """What the output must be."""

    JSON_SCHEMA = "json_schema"
    JSON_OBJECT = "json_object"
    TOOL_CALL = "tool_call"


@dataclass(frozen=True, slots=True)
class ToolSchema:
    """A function the model may call: its name and the JSON schema of its arguments."""

    name: str
    parameters: Mapping[str, object] = field(default_factory=dict[str, object])

    def __post_init__(self) -> None:
        if not self.name:
            raise InvalidRequestError("a tool needs a name")


@dataclass(frozen=True, slots=True)
class ToolCallSyntax:
    """How the loaded model writes a call: the markers around a JSON ``{name, arguments}`` body.

    Attributes:
        open: The text of the opening marker.
        close: The text of the closing marker.
        open_token: The opening marker's token id when it is one token, else ``None``. A lazy
            constraint needs it: it starts when the model emits that token.
        close_token: The closing marker's token id when it is one token. The grammar then
            demands that token, because the parser recognises a one-token marker by its id and
            the same characters spelled byte by byte would not close the call.
    """

    open: str
    close: str
    open_token: int | None = None
    close_token: int | None = None


@dataclass(frozen=True, slots=True, kw_only=True)
class StructuredParams:
    """A request's output constraint.

    Attributes:
        kind: JSON schema, JSON object or tool call.
        schema: The JSON schema for ``JSON_SCHEMA``.
        tools: The functions for ``TOOL_CALL``.
        syntax: How the model writes a call, for ``TOOL_CALL``.
        tool_choice: ``"auto"`` (a call is optional: lazy), ``"required"`` (the answer starts
            with a call to any tool) or the name of the one tool to call.
        strict: Constrain the arguments to each tool's schema; otherwise any JSON object.
        parallel: After a call, let the model write more (more calls); otherwise the answer
            ends after the first call.
        start_after: A token id that switches the constraint on, such as the end of a reasoning
            block; ``None`` to start at the first generated token (a lazy tool call starts at
            ``syntax.open_token`` instead).

    Raises:
        InvalidRequestError: If the fields do not fit the kind, or the schema is too large.
    """

    kind: ConstraintKind
    schema: Mapping[str, object] | None = None
    tools: tuple[ToolSchema, ...] = ()
    syntax: ToolCallSyntax | None = None
    tool_choice: str = "auto"
    strict: bool = False
    parallel: bool = True
    start_after: int | None = None

    def __post_init__(self) -> None:
        if self.kind is ConstraintKind.JSON_SCHEMA and self.schema is None:
            raise InvalidRequestError("a json_schema constraint needs a schema")
        if self.kind is ConstraintKind.TOOL_CALL:
            if not self.tools:
                raise InvalidRequestError("a tool-call constraint needs at least one tool")
            if self.syntax is None:
                raise InvalidRequestError("a tool-call constraint needs the model's call syntax")
            names = [tool.name for tool in self.tools]
            if self.tool_choice not in ("auto", "required") and self.tool_choice not in names:
                raise InvalidRequestError(
                    f"tool_choice names {self.tool_choice!r}, which is not among the tools",
                    context={"param": "tool_choice"},
                )
        size = len(self.canonical().encode())
        if size > MAX_SCHEMA_BYTES:
            raise InvalidRequestError(
                f"the schemas of this request take {size} bytes; at most {MAX_SCHEMA_BYTES} are "
                "accepted"
            )

    @property
    def lazy(self) -> bool:
        """Whether the constraint waits for the model to open a tool call."""
        return self.kind is ConstraintKind.TOOL_CALL and self.tool_choice == "auto"

    def canonical(self) -> str:
        """A stable text of everything that changes the grammar: the cache key of its compile."""
        body = {
            "kind": str(self.kind),
            "schema": self.schema,
            "tools": [{"name": tool.name, "parameters": tool.parameters} for tool in self.tools],
            "syntax": (
                [self.syntax.open, self.syntax.close, self.syntax.close_token]
                if self.syntax is not None
                else None
            ),
            "tool_choice": self.tool_choice,
            "strict": self.strict,
        }
        try:
            # Key order is kept: the order of a schema's properties is the order they are written.
            return json.dumps(body, separators=(",", ":"), ensure_ascii=False)
        except (TypeError, ValueError) as exc:
            raise InvalidRequestError(f"a schema is not JSON: {exc}") from exc


def as_schema(value: object, what: str) -> Mapping[str, object]:
    """A JSON object from a request, or an invalid-request error naming ``what``."""
    if not isinstance(value, dict):
        raise InvalidRequestError(f"{what} must be a JSON object", context={"param": what})
    return cast("Mapping[str, object]", value)
