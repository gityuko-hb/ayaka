"""Each output format, described once: what the parsers read and what the grammar enforces.

A model family writes its reasoning block and its function calls in one layout. That layout is a
record here, and every part of the server that depends on it reads the record: the reasoning and
tool-call parsers of this package, and the output constraint the server builds for a request
(:meth:`~ayaka.entrypoints.parsers.output.OutputFormat.call_syntax` and
:meth:`~ayaka.entrypoints.parsers.output.OutputFormat.constraint_start`), so the grammar starts
exactly where the parser starts reading a call.

Formats are chosen by name when the server starts (``--reasoning-parser``,
``--tool-call-parser``), never guessed from the output.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, Final, Literal

from ayaka.core.errors import ConfigError
from ayaka.entrypoints.parsers.markers import Marker, resolve_marker

if TYPE_CHECKING:
    from ayaka.tokenizers.ports import TokenizerLike

__all__ = [
    "REASONING_FORMATS",
    "TOOL_CALL_FORMATS",
    "ReasoningDefault",
    "ReasoningFormat",
    "ToolCallFormat",
    "reasoning_format",
    "suggest_tool_call_parser",
    "tool_call_format",
]


class ReasoningDefault(StrEnum):
    """Default template thinking behavior when the client sends no preference."""

    ON_UNLESS = "on_unless"
    OFF_UNLESS = "off_unless"


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
        default_mode: Default request thinking mode when unspecified.
        tool_start: Opening tool-call markers that recover an unclosed reasoning block when the
            model forgets to emit ``close`` before calling a tool.
        tool_start_at_line_start: Whether ``tool_start`` recovery requires a line boundary.
        force_nonempty_content: Keep trailing text available if a block never produces content.
    """

    name: str
    open: str
    close: str
    starts_inside: bool = False
    default_mode: ReasoningDefault = ReasoningDefault.OFF_UNLESS
    tool_start: tuple[str, ...] = ()
    tool_start_at_line_start: bool = False
    force_nonempty_content: bool = False

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
    "deepseek_r1": ReasoningFormat(
        "deepseek_r1",
        "<think>",
        "</think>",
        starts_inside=True,
        default_mode=ReasoningDefault.ON_UNLESS,
        tool_start=("<\uff5cDSML\uff5cfunction_calls>",),
    ),
    # DeepSeek-V3 / V3.1/3.2: optional thinking block with explicit opening marker or DSML call.
    "deepseek_v3": ReasoningFormat(
        "deepseek_v3",
        "<think>",
        "</think>",
        starts_inside=False,
        default_mode=ReasoningDefault.OFF_UNLESS,
        tool_start=("<tool_call>", "<\uff5cDSML\uff5cfunction_calls>"),
    ),
    # GLM-4.5 / GLM-4.6: think block with recovery before <tool_call>.
    "glm45": ReasoningFormat(
        "glm45",
        "<think>",
        "</think>",
        starts_inside=False,
        tool_start=("<tool_call>",),
        force_nonempty_content=True,
    ),
    # MiniMax M3: think block with line-start recovery on <minimax:tool_call> or <tool_call>.
    "minimax_m3": ReasoningFormat(
        "minimax_m3",
        "<think>",
        "</think>",
        starts_inside=False,
        tool_start=("<minimax:tool_call>", "<tool_call>"),
        tool_start_at_line_start=True,
    ),
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
        open: The marker before the call body, or ``None`` for a call written without markers.
        close: The marker after it, or ``None`` with ``open``.
        arguments: The key of the arguments in the JSON call object.
        typed_name: A second key holding the function name next to ``"type": "function"``.
        separator: Between two calls written without markers, the one separator character.
        style: Wire encoding of calls: ``"json"`` or ``"xml"``.
    """

    name: str
    open: str | None
    close: str | None
    arguments: str = "arguments"
    typed_name: str | None = None
    separator: str | None = None
    style: Literal["json", "xml"] = "json"

    def __post_init__(self) -> None:
        if (self.open is None) != (self.close is None):
            raise ValueError("a call format has both markers or none")
        if self.separator is not None and (self.open is not None or len(self.separator) != 1):
            raise ValueError("a separator is one character, for calls written without markers")

    def markers(self, tokenizer: TokenizerLike | None = None) -> tuple[Marker, Marker] | None:
        """The markers as ``tokenizer`` spells them (by text without one); ``None`` without any.

        Raises:
            ConfigError: If a marker is a special token the detokenizer drops.
        """
        if self.open is None or self.close is None:
            return None
        return resolve_marker(self.open, tokenizer), resolve_marker(self.close, tokenizer)


TOOL_CALL_FORMATS: Final[dict[str, ToolCallFormat]] = {
    # Hermes 2/3, Qwen2.5, Qwen3 (JSON), SmolLM3 and others.
    "hermes": ToolCallFormat("hermes", "<tool_call>", "</tool_call>"),
    "qwen3_json": ToolCallFormat("qwen3_json", "<tool_call>", "</tool_call>"),
    # DeepSeek V3/R1 JSON tool call markers.
    "deepseek_json": ToolCallFormat("deepseek_json", "<|tool_call_begin|>", "<|tool_call_end|>"),
    # Llama 3.x: a bare JSON object separated by ";".
    "llama3_json": ToolCallFormat(
        "llama3_json", None, None, arguments="parameters", typed_name="function", separator=";"
    ),
    # Qwen3-Coder: <tool_call><function=NAME><parameter=K>V</parameter></function></tool_call>.
    "qwen3_coder": ToolCallFormat("qwen3_coder", "<tool_call>", "</tool_call>", style="xml"),
    # MiMo XML: <tool_call><function=NAME><parameter=K>V</parameter></function></tool_call>.
    "mimo": ToolCallFormat("mimo", "<tool_call>", "</tool_call>", style="xml"),
    # GLM XML: <tool_call>NAME<arg_key>K</arg_key><arg_value>V</arg_value></tool_call>.
    "glm_xml": ToolCallFormat("glm_xml", "<tool_call>", "</tool_call>", style="xml"),
    # MiniMax M2 / M3 XML: <minimax:tool_call><invoke name="NAME">...</minimax:tool_call>.
    "minimax_m2": ToolCallFormat(
        "minimax_m2", "<minimax:tool_call>", "</minimax:tool_call>", style="xml"
    ),
    "minimax_m3": ToolCallFormat(
        "minimax_m3", "<minimax:tool_call>", "</minimax:tool_call>", style="xml"
    ),
    # DeepSeek DSML function_calls / invoke / parameter blocks.
    "deepseek_dsml": ToolCallFormat(
        "deepseek_dsml",
        "<\uff5cDSML\uff5cfunction_calls>",
        "</\uff5cDSML\uff5cfunction_calls>",
        style="xml",
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


def suggest_tool_call_parser(chat_template: str | None) -> str | None:
    """The ``--tool-call-parser`` name suggested by ``chat_template``, or ``None``.

    Reads only the model's declared chat template and returns ``None`` when the template is
    absent or does not clearly match one of :data:`TOOL_CALL_FORMATS`.
    """
    if not chat_template:
        return None
    if "<\uff5cDSML\uff5cfunction_calls>" in chat_template or "DSML\uff5cinvoke" in chat_template:
        return "deepseek_dsml"
    if "<minimax:tool_call>" in chat_template:
        if "reasoning_split" in chat_template or "minimax_m3" in chat_template.lower():
            return "minimax_m3"
        return "minimax_m2"
    if "<arg_key>" in chat_template and "<arg_value>" in chat_template:
        return "glm_xml"
    if "<function=" in chat_template and "<parameter=" in chat_template:
        if "mimo" in chat_template.lower():
            return "mimo"
        return "qwen3_coder"
    if "<|tool_call_begin|>" in chat_template:
        return "deepseek_json"
    hermes_open = TOOL_CALL_FORMATS["hermes"].open
    if hermes_open is not None and hermes_open in chat_template:
        return "hermes"
    if (
        "<|start_header_id|>" in chat_template
        and '"name"' in chat_template
        and '"parameters"' in chat_template
    ):
        return "llama3_json"
    return None
