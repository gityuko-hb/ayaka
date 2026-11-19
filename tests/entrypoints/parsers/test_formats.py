"""One record per output format, read by the parsers and by the output constraint alike.

The central property: fed the same tokens, the constraint matcher is on exactly when the parser is
somewhere the request constrained -- inside a call body, at the start of a forced call, or in a
JSON answer -- and off inside a reasoning block and in free text. The grammar here is a stand-in
that accepts every token (and, for a call, stops after the closing marker), so the property is
about where the constraint starts and stops, not about the grammar itself. Tokens are drawn at
random (seeded) from what the matcher's mask allows, as the sampler would.
"""

from __future__ import annotations

import random
from typing import TYPE_CHECKING

import numpy as np
import pytest

from ayaka.core.constraints import ConstraintKind, StructuredParams, ToolSchema
from ayaka.core.outputs import CompletionOutput, RequestOutput, Usage
from ayaka.core.types import RequestId
from ayaka.entrypoints.parsers.formats import (
    REASONING_FORMATS,
    TOOL_CALL_FORMATS,
    reasoning_format,
    suggest_tool_call_parser,
    tool_call_format,
)
from ayaka.entrypoints.parsers.markers import Marker
from ayaka.entrypoints.parsers.output import OutputFormat
from ayaka.sampling.structured.grammar import call_schema
from ayaka.sampling.structured.matcher import ConstraintMatcher
from tokenizer_fakes import ToolTokenizer

if TYPE_CHECKING:
    from ayaka.entrypoints.parsers.output import OutputParser

pytestmark = pytest.mark.unit

_THINK, _END_THINK, _CALL, _END_CALL = 1001, 1002, 1003, 1004
_TEXT = {ord(c): c for c in 'ab {}":'}
_MARKERS = {
    _THINK: "<think>",
    _END_THINK: "</think>",
    _CALL: "<tool_call>",
    _END_CALL: "</tool_call>",
}
_VOCAB = 1024


class _AcceptAll:
    """A grammar that allows every token; a call's grammar stops after its closing marker."""

    def __init__(self, stop_on: int | None) -> None:
        self._stop_on = stop_on
        self._stopped = False

    def compute_bitmask(self) -> bytes:
        return np.full(_VOCAB // 32, 0xFFFFFFFF, dtype="<u4").tobytes()

    def consume_token(self, sampled_token: int) -> bool:
        self._stopped = sampled_token == self._stop_on
        return True

    def is_stopped(self) -> bool:
        return self._stopped

    def is_error(self) -> bool:
        return False

    def get_error(self) -> str:
        return ""

    def deep_copy(self) -> _AcceptAll:
        return _AcceptAll(self._stop_on)


def _format(reasoning: str | None) -> OutputFormat:
    return OutputFormat(
        reasoning=REASONING_FORMATS[reasoning] if reasoning else None,
        reasoning_markers=(Marker("<think>", _THINK), Marker("</think>", _END_THINK))
        if reasoning
        else None,
        tool_calls=TOOL_CALL_FORMATS["hermes"],
        tool_markers=(Marker("<tool_call>", _CALL), Marker("</tool_call>", _END_CALL)),
    )


def _matcher(fmt: OutputFormat, kind: str) -> ConstraintMatcher:
    """The matcher the server builds for ``kind``, from the format's own description."""
    start_after, reasoning_open = fmt.constraint_start()
    syntax = fmt.call_syntax()
    assert syntax is not None
    if kind == "json":
        params = StructuredParams(
            kind=ConstraintKind.JSON_OBJECT, start_after=start_after, reasoning_open=reasoning_open
        )
    else:
        params = StructuredParams(
            kind=ConstraintKind.TOOL_CALL,
            tools=(ToolSchema("f"),),
            syntax=syntax,
            tool_choice="required" if kind == "forced" else "auto",
            start_after=start_after,
            reasoning_open=reasoning_open,
        )
    is_call = params.kind is ConstraintKind.TOOL_CALL
    return ConstraintMatcher(
        _AcceptAll(syntax.close_token if is_call else None),
        vocab_size=_VOCAB,
        eos_token_ids=(),
        start_after=params.start_after,
        reasoning_open=params.reasoning_open,
        opening=syntax.opening if is_call else (),
        forced=is_call and params.tool_choice != "auto",
    )


def _constrained(parser: OutputParser, kind: str) -> bool:
    """Where the parser is, as the constraint should see it."""
    if parser.in_reasoning:
        return False
    if kind == "json":
        return True
    if kind == "forced":
        return parser.in_call or parser.opened_calls == 0
    return parser.in_call


def _fragment(token: int) -> RequestOutput:
    text = _MARKERS.get(token) or _TEXT[token]
    return RequestOutput(
        request_id=RequestId("r"),
        outputs=(CompletionOutput(index=0, token_ids=(token,), text=text),),
        finished=False,
        usage=Usage(prompt_tokens=1, completion_tokens=0),
    )


def _allowed(mask: np.ndarray | None) -> list[int]:
    tokens = [*_TEXT, *_MARKERS]
    if mask is None:
        return tokens
    return [token for token in tokens if mask[token // 32] >> (token % 32) & 1]


class TestRegistry:
    @pytest.mark.parametrize("name", sorted(REASONING_FORMATS))
    def test_every_reasoning_format_resolves(self, name: str) -> None:
        fmt = reasoning_format(name)
        opening, closing = fmt.markers(ToolTokenizer())

        assert (opening.text, closing.text) == (fmt.open, fmt.close)

    @pytest.mark.parametrize("name", sorted(TOOL_CALL_FORMATS))
    def test_every_call_format_resolves_and_gives_the_grammar_its_arguments_key(
        self, name: str
    ) -> None:
        fmt = tool_call_format(name)
        output = OutputFormat.resolve(tool_calls=name, tokenizer=ToolTokenizer())
        syntax = output.call_syntax()

        assert syntax is not None
        assert syntax.bare == (fmt.open is None)  # no marker: only a forced call has a grammar
        schema = call_schema((ToolSchema("f"),), strict=False, arguments_key=syntax.arguments_key)
        assert syntax.arguments_key == fmt.arguments
        assert schema["required"] == ["name", fmt.arguments]

    def test_the_constraint_starts_where_the_parser_does(self) -> None:
        assert _format(None).constraint_start() == (None, None)
        assert _format("think").constraint_start() == (_END_THINK, _THINK)
        assert _format("deepseek_r1").constraint_start() == (_END_THINK, None)

    def test_suggest_tool_call_parser_reads_the_chat_template(self) -> None:
        assert suggest_tool_call_parser(None) is None
        assert suggest_tool_call_parser("") is None
        assert suggest_tool_call_parser("<|im_start|>assistant\n<tool_call>\n") == "hermes"
        llama_template = (
            "<|start_header_id|>user<|end_header_id|>\n"
            'Respond in the format {"name": function name, "parameters": dictionary}.\n'
            "<|eot_id|>"
        )
        assert suggest_tool_call_parser(llama_template) == "llama3_json"
        # A Llama-style template without function calling does not suggest a parser.
        assert (
            suggest_tool_call_parser("<|start_header_id|>user<|end_header_id|><|eot_id|>") is None
        )


@pytest.mark.parametrize("seed", range(12))
@pytest.mark.parametrize("kind", ["json", "lazy", "forced"])
@pytest.mark.parametrize("reasoning", [None, "think", "deepseek_r1"])
def test_the_matcher_is_on_exactly_when_the_parser_is_constrained(
    reasoning: str | None, kind: str, seed: int
) -> None:
    rng = random.Random(seed)  # noqa: S311  # a seeded property test, not security
    fmt = _format(reasoning)
    parser = fmt.parser(call_id_prefix="c", tools=True)
    matcher = _matcher(fmt, kind)
    history: list[int] = []

    assert matcher.active == _constrained(parser, kind), "before the first token"
    for _ in range(40):
        token = rng.choice(_allowed(matcher.mask()))
        history.append(token)
        parser.feed(_fragment(token))
        matcher.accept([token])
        assert matcher.active == _constrained(parser, kind), (history, matcher.state)


def test_a_model_that_does_not_reason_is_constrained_from_its_first_token() -> None:
    """The bug this record fixed: with ``think``, the constraint used to wait for a ``</think>``
    a model that answers at once never writes, so it never switched on."""
    matcher = _matcher(_format("think"), "json")
    first = ord("{")

    mask = matcher.mask()
    assert mask is not None
    assert set(_allowed(mask)) >= {first, _THINK}  # the answer or the block
    matcher.accept([first])

    assert matcher.active
    assert matcher.state == "ON"


def test_a_block_opened_later_is_text_for_both() -> None:
    fmt = _format("think")
    parser = fmt.parser(call_id_prefix="c", tools=True)
    matcher = _matcher(fmt, "lazy")
    for token in (ord("a"), _THINK, ord("b"), _CALL):
        parser.feed(_fragment(token))
        matcher.accept([token])

    assert not parser.in_reasoning
    assert parser.in_call
    assert matcher.active
