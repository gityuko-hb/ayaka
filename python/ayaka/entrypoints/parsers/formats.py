"""Each output format, described once: what the parsers read and what the grammar enforces.

A model family writes its reasoning block and its function calls in one layout. That layout is a
record here, and every part of the server that depends on it reads the record: the reasoning and
tool-call parsers of this package, and the output constraint the server builds for a request
(:meth:`~ayaka.entrypoints.parsers.output.OutputFormat.call_syntax` and
:meth:`~ayaka.entrypoints.parsers.output.OutputFormat.constraint_start`), so the grammar starts
exactly where the parser starts reading a call.

* :class:`ReasoningFormat`: the two markers of the block, and who opens it: the model with its
  first token (``think``), or the chat template in the prompt (``deepseek_r1``).
* :class:`ToolCallFormat`: the markers around a call (or none), the keys of the call's JSON
  object, and the separator between calls written without markers.

Formats are chosen by name when the server starts (``--reasoning-parser``,
``--tool-call-parser``), never guessed from the output.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

from ayaka.core.errors import ConfigError
from ayaka.entrypoints.parsers.markers import Marker, resolve_marker

if TYPE_CHECKING:
    from ayaka.tokenizers.ports import TokenizerLike

__all__ = [
    "REASONING_FORMATS",
    "TOOL_CALL_FORMATS",
    "ReasoningFormat",
    "ToolCallFormat",
    "reasoning_format",
    "tool_call_format",
]


@dataclass(frozen=True, slots=True)
class ReasoningFormat:
    """How a model family marks its reasoning.

    Attributes:
        name: The name the server option uses.
        open: The marker that opens the block.
        close: The marker that closes it.
        starts_inside: Whether the chat template already opened the block in the prompt, so the
            output starts inside it. Otherwise the model opens it, and only with the first thing
            it writes: an opening marker after any answer text is text.
    """

    name: str
    open: str
    close: str
    starts_inside: bool = False

    def markers(self, tokenizer: TokenizerLike | None = None) -> tuple[Marker, Marker]:
        """The opening and closing markers as ``tokenizer`` spells them (by text without one).

        Raises:
            ConfigError: If a marker is a special token the detokenizer drops.
        """
        return resolve_marker(self.open, tokenizer), resolve_marker(self.close, tokenizer)


REASONING_FORMATS: Final[dict[str, ReasoningFormat]] = {
    # Qwen3, QwQ, SmolLM3 and others: the model writes both markers.
    "think": ReasoningFormat("think", "<think>", "</think>"),
    # DeepSeek-R1 and its distillations: the template ends the prompt with "<think>\n".
    "deepseek_r1": ReasoningFormat("deepseek_r1", "<think>", "</think>", starts_inside=True),
}


def reasoning_format(name: str) -> ReasoningFormat:
    """The format called ``name``.

    Raises:
        ConfigError: If there is no such format.
    """
    try:
        return REASONING_FORMATS[name]
    except KeyError:
        raise ConfigError(
            "server.reasoning_parser",
            "INVALID_CHOICE",
            f"unknown reasoning parser {name!r}; known: {sorted(REASONING_FORMATS)}",
        ) from None


@dataclass(frozen=True, slots=True)
class ToolCallFormat:
    """How a model family writes a function call.

    Attributes:
        name: The name the server option uses.
        open: The marker before the JSON body, or ``None`` for a call written without markers.
        close: The marker after it, or ``None`` with ``open``.
        arguments: The key of the arguments in the call's JSON object.
        typed_name: A second key that holds the function name when the object also says
            ``"type": "function"``; ``None`` if only ``name`` does.
        separator: Between two calls written without markers, the one character that leads to
            the next call; ``None`` with markers.
    """

    name: str
    open: str | None
    close: str | None
    arguments: str = "arguments"
    typed_name: str | None = None
    separator: str | None = None

    def __post_init__(self) -> None:
        if (self.open is None) != (self.close is None):
            raise ValueError("a call format has both markers or none")
        if self.separator is not None and (self.open is not None or len(self.separator) != 1):
            raise ValueError("a separator is one character, for calls written without markers")

    @property
    def name_key(self) -> str:
        """The key of the function name in the call's JSON object."""
        return "name"

    def markers(self, tokenizer: TokenizerLike | None = None) -> tuple[Marker, Marker] | None:
        """The markers as ``tokenizer`` spells them (by text without one); ``None`` without any.

        Raises:
            ConfigError: If a marker is a special token the detokenizer drops.
        """
        if self.open is None or self.close is None:
            return None
        return resolve_marker(self.open, tokenizer), resolve_marker(self.close, tokenizer)


TOOL_CALL_FORMATS: Final[dict[str, ToolCallFormat]] = {
    # Hermes 2/3, Qwen2.5, Qwen3, SmolLM3 and others.
    "hermes": ToolCallFormat("hermes", "<tool_call>", "</tool_call>"),
    # Llama 3.x: a bare JSON object, its arguments under "parameters". The chat templates of
    # Llama-3.1 and Llama-3.2 both teach this layout; Llama-3.2-1B also writes the function name
    # under "function" next to "type": "function" when the tools are shown wrapped that way.
    # Several calls are objects separated by ";" (measured on Llama-3.2-1B-Instruct).
    "llama3_json": ToolCallFormat(
        "llama3_json", None, None, arguments="parameters", typed_name="function", separator=";"
    ),
}


def tool_call_format(name: str) -> ToolCallFormat:
    """The format called ``name``.

    Raises:
        ConfigError: If there is no such format.
    """
    try:
        return TOOL_CALL_FORMATS[name]
    except KeyError:
        raise ConfigError(
            "server.tool_call_parser",
            "INVALID_CHOICE",
            f"unknown tool call parser {name!r}; known: {sorted(TOOL_CALL_FORMATS)}",
        ) from None
