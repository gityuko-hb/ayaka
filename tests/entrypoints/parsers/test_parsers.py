"""The incremental output parsers: markers, reasoning blocks and function calls.

The central property is that a parser's events do not depend on where the stream is cut. A model
output is written here as a list of ``(token id, text)`` pairs: ordinary characters are one token
each (id = code point), and a marker that is one token of the vocabulary has an id above 1000. The
same output is fed whole, one token at a time and cut at every token boundary, and the merged
events must always be the same.
"""

from __future__ import annotations

from itertools import pairwise
from typing import TYPE_CHECKING

import pytest

from ayaka.core.errors import ConfigError, ErrorInfo
from ayaka.core.outputs import CompletionOutput, RequestOutput, Usage
from ayaka.core.types import FinishReason, RequestId
from ayaka.entrypoints.conversation import (
    ErrorEvent,
    FinishEvent,
    ReasoningDelta,
    TextDelta,
    ToolCallDelta,
    ToolCallEnd,
    ToolCallStart,
    UsageEvent,
    merge_deltas,
)
from ayaka.entrypoints.parsers.markers import Marker, MarkerScanner, resolve_marker
from ayaka.entrypoints.parsers.output import OutputFormat
from ayaka.entrypoints.parsers.reasoning import REASONING_FORMATS, reasoning_format
from ayaka.entrypoints.parsers.tool_calls import TOOL_CALL_FORMATS, tool_call_format
from tokenizer_fakes import FakeTokenizer

if TYPE_CHECKING:
    from collections.abc import Sequence

    from ayaka.entrypoints.conversation import OutputEvent
    from ayaka.entrypoints.parsers.output import OutputParser

pytestmark = pytest.mark.unit

_MARKER_IDS = {"<think>": 1001, "</think>": 1002, "<tool_call>": 1003, "</tool_call>": 1004}
_BY_ID = {
    "think": (Marker("<think>", 1001), Marker("</think>", 1002)),
    "hermes": (Marker("<tool_call>", 1003), Marker("</tool_call>", 1004)),
}

type Token = tuple[int, str]


def tokens(*parts: str) -> list[Token]:
    """An output: each part is a marker token if it is a marker, otherwise one token per char."""
    out: list[Token] = []
    for part in parts:
        if part in _MARKER_IDS:
            out.append((_MARKER_IDS[part], part))
        else:
            out.extend((ord(char), char) for char in part)
    return out


def _format(*, reasoning: str | None, tools: str | None, by_id: bool = True) -> OutputFormat:
    return OutputFormat(
        reasoning=REASONING_FORMATS[reasoning] if reasoning else None,
        reasoning_markers=_BY_ID["think"] if by_id and reasoning else None,
        tool_calls=TOOL_CALL_FORMATS[tools] if tools else None,
        tool_markers=_BY_ID[tools] if by_id and tools else None,
    )


def _fragment(chunk: Sequence[Token], *, finished: bool, **end: object) -> RequestOutput:
    completion = CompletionOutput(
        index=0,
        token_ids=tuple(token for token, _ in chunk),
        text="".join(text for _, text in chunk),
        finish_reason=end.get("reason", FinishReason.STOP) if finished else None,  # type: ignore[arg-type]
        stop_reason=end.get("stop") if finished else None,  # type: ignore[arg-type]
    )
    return RequestOutput(
        request_id=RequestId("r"),
        outputs=(completion,),
        finished=finished,
        usage=Usage(prompt_tokens=5, completion_tokens=int(end.get("count", 0))),  # type: ignore[call-overload]
        error=end.get("error"),  # type: ignore[arg-type]
    )


def run(
    parser: OutputParser, output: Sequence[Token], cuts: Sequence[int], **end: object
) -> list[OutputEvent]:
    """Feed ``output`` cut at ``cuts`` (token indexes) and return the merged events."""
    bounds = [0, *cuts, len(output)]
    events: list[OutputEvent] = []
    for i, (start, stop) in enumerate(pairwise(bounds)):
        last = i == len(bounds) - 2
        events += parser.feed(_fragment(output[start:stop], finished=last, **end))
    return merge_deltas(events)


def assert_cut_invariant(
    fmt: OutputFormat, output: Sequence[Token], *, tools: bool = True
) -> list[OutputEvent]:
    """The merged events are the same however the output is cut; return them."""
    whole = run(fmt.parser(call_id_prefix="call_x", tools=tools), output, [])
    each = run(fmt.parser(call_id_prefix="call_x", tools=tools), output, range(1, len(output)))
    assert each == whole
    for cut in range(1, len(output)):
        parser = fmt.parser(call_id_prefix="call_x", tools=tools)
        assert run(parser, output, [cut]) == whole, f"cut at {cut}"
    return whole


class TestMarkerScanner:
    def test_a_marker_split_over_two_pieces_is_found(self) -> None:
        scanner = MarkerScanner()
        marker = Marker("<tool_call>")

        assert scanner.scan("hello <tool", [marker]) == ("hello ", None)
        assert scanner.held == "<tool"
        assert scanner.scan("_call>rest", [marker]) == ("", marker)
        assert scanner.scan("", [marker]) == ("rest", None)

    def test_text_that_stops_looking_like_a_marker_is_released(self) -> None:
        scanner = MarkerScanner()
        marker = Marker("<think>")

        assert scanner.scan("a <th", [marker]) == ("a ", None)
        assert scanner.scan("ere", [marker]) == ("<there", None)
        assert scanner.flush() == ""

    def test_a_marker_matched_by_id_needs_its_token(self) -> None:
        scanner = MarkerScanner()
        marker = Marker("<think>", 7)

        assert scanner.scan("say <think> please", [marker]) == ("say <think> please", None)
        scanner.note_tokens([7])
        assert scanner.scan("<think>", [marker]) == ("", marker)

    def test_flush_releases_the_held_text(self) -> None:
        scanner = MarkerScanner()
        scanner.scan("end <", [Marker("<x>")])

        assert scanner.flush() == "<"
        assert scanner.held == ""


class TestResolveMarker:
    def test_a_multi_token_marker_is_matched_by_text(self) -> None:
        assert resolve_marker("<think>", FakeTokenizer()) == Marker("<think>")

    def test_a_single_token_marker_is_matched_by_id(self) -> None:
        class OneToken(FakeTokenizer):
            def encode(self, text: str, *, add_special_tokens: bool = True) -> list[int]:
                return [1001] if text == "<think>" else super().encode(text)

        assert resolve_marker("<think>", OneToken()) == Marker("<think>", 1001)

    def test_a_special_token_marker_is_refused(self) -> None:
        # <|im_end|> is a special token of the fake tokenizer, which the detokenizer drops.
        class Special(FakeTokenizer):
            def encode(self, text: str, *, add_special_tokens: bool = True) -> list[int]:
                return [259] if text == "<|im_end|>" else super().encode(text)

        with pytest.raises(ConfigError, match="special token"):
            resolve_marker("<|im_end|>", Special())

    def test_unknown_format_names_are_config_errors(self) -> None:
        with pytest.raises(ConfigError, match="unknown reasoning parser"):
            reasoning_format("nope")
        with pytest.raises(ConfigError, match="unknown tool call parser"):
            tool_call_format("nope")


class TestReasoning:
    def test_reasoning_and_answer_separate_and_edge_whitespace_is_dropped(self) -> None:
        fmt = _format(reasoning="think", tools=None)
        output = tokens("<think>", "\nlet me see\n", "</think>", "\n\nThe answer.")

        events = assert_cut_invariant(fmt, output)

        assert events == [
            ReasoningDelta("let me see"),
            TextDelta("The answer."),
            UsageEvent(5, 0),
            FinishEvent(FinishReason.STOP),
        ]

    def test_a_template_that_opens_the_block_starts_inside_it(self) -> None:
        fmt = OutputFormat(
            reasoning=REASONING_FORMATS["deepseek_r1"], reasoning_markers=_BY_ID["think"]
        )
        output = tokens("hmm ", "</think>", " ok")

        assert assert_cut_invariant(fmt, output)[:2] == [ReasoningDelta("hmm"), TextDelta("ok")]

    def test_markers_spelled_with_ordinary_tokens_are_text(self) -> None:
        fmt = _format(reasoning="think", tools=None)
        output = [(ord(c), c) for c in "use <think> tags"]

        assert assert_cut_invariant(fmt, output)[0] == TextDelta("use <think> tags")

    def test_without_ids_markers_match_by_text(self) -> None:
        fmt = _format(reasoning="think", tools=None, by_id=False)
        output = [(ord(c), c) for c in "<think>x</think>y"]

        assert assert_cut_invariant(fmt, output)[:2] == [ReasoningDelta("x"), TextDelta("y")]

    def test_an_unclosed_block_keeps_its_text(self) -> None:
        fmt = _format(reasoning="think", tools=None)
        output = tokens("<think>", "cut off  ")

        assert assert_cut_invariant(fmt, output)[0] == ReasoningDelta("cut off  ")

    def test_reasoning_tokens_are_counted_by_step(self) -> None:
        fmt = _format(reasoning="think", tools=None)
        output = tokens("<think>", "abc", "</think>", "de")

        parser = fmt.parser(call_id_prefix="c", tools=False)
        events = run(parser, output, range(1, len(output)), count=len(output))

        usage = next(event for event in events if isinstance(event, UsageEvent))
        assert usage.reasoning_tokens == 4  # "a", "b", "c" and the closing marker


_CALL = '{"name": "get_weather", "arguments": {"city": "Paris", "days": [1, 2]}}'


class TestToolCalls:
    def test_a_call_streams_its_name_then_the_raw_arguments(self) -> None:
        fmt = _format(reasoning=None, tools="hermes")
        output = tokens("Checking.\n", "<tool_call>", "\n" + _CALL + "\n", "</tool_call>")

        events = assert_cut_invariant(fmt, output)

        assert events == [
            TextDelta("Checking."),
            ToolCallStart(0, "call_x_0", "get_weather"),
            ToolCallDelta(0, '{"city": "Paris", "days": [1, 2]}'),
            ToolCallEnd(0, valid=True),
            UsageEvent(5, 0),
            FinishEvent(FinishReason.STOP, tool_calls=True),
        ]

    def test_the_call_starts_before_its_arguments_are_complete(self) -> None:
        fmt = _format(reasoning=None, tools="hermes")
        parser = fmt.parser(call_id_prefix="c", tools=True)
        head = tokens("<tool_call>", '{"name": "f", "arguments": {"a": ')

        events = parser.feed(_fragment(head, finished=False))

        assert events == [ToolCallStart(0, "c_0", "f"), ToolCallDelta(0, '{"a": ')]

    def test_two_calls_and_the_whitespace_between_them(self) -> None:
        fmt = _format(reasoning=None, tools="hermes")
        first = '{"name": "a", "arguments": {}}'
        second = '{"arguments": {"x": "}"}, "name": "b"}'
        output = tokens("<tool_call>", first, "</tool_call>", "\n\n", "<tool_call>", second)
        output += tokens("</tool_call>", "\n")

        events = assert_cut_invariant(fmt, output)

        assert events[:6] == [
            ToolCallStart(0, "call_x_0", "a"),
            ToolCallDelta(0, "{}"),
            ToolCallEnd(0, valid=True),
            ToolCallStart(1, "call_x_1", "b"),
            ToolCallDelta(1, '{"x": "}"}'),
            ToolCallEnd(1, valid=True),
        ]
        assert not any(isinstance(event, TextDelta) for event in events)

    def test_broken_arguments_are_passed_on_as_written_and_marked_invalid(self) -> None:
        fmt = _format(reasoning=None, tools="hermes")
        output = tokens("<tool_call>", '{"name": "f", "arguments": {"a": 1,}}', "</tool_call>")

        events = assert_cut_invariant(fmt, output)

        assert events[:3] == [
            ToolCallStart(0, "call_x_0", "f"),
            ToolCallDelta(0, '{"a": 1,}'),
            ToolCallEnd(0, valid=False),
        ]

    def test_a_body_without_a_name_is_shown_as_text(self) -> None:
        fmt = _format(reasoning=None, tools="hermes")
        output = tokens("<tool_call>", "not json at all", "</tool_call>", " after")

        events = assert_cut_invariant(fmt, output)

        assert events[0] == TextDelta("<tool_call>not json at all</tool_call> after")
        assert isinstance(events[-1], FinishEvent)
        assert not events[-1].tool_calls

    def test_indexes_count_only_reported_calls(self) -> None:
        fmt = _format(reasoning=None, tools="hermes")
        output = tokens("<tool_call>", "junk", "</tool_call>", "<tool_call>", _CALL, "</tool_call>")

        events = assert_cut_invariant(fmt, output)

        assert ToolCallStart(0, "call_x_0", "get_weather") in events

    def test_a_call_cut_by_the_token_limit_ends_invalid(self) -> None:
        fmt = _format(reasoning=None, tools="hermes")
        output = tokens("<tool_call>", '{"name": "f", "arguments": {"a": "unfini')

        events = assert_cut_invariant(fmt, output)

        assert events[:3] == [
            ToolCallStart(0, "call_x_0", "f"),
            ToolCallDelta(0, '{"a": "unfini'),
            ToolCallEnd(0, valid=False),
        ]

    def test_without_tools_in_the_request_the_markers_are_text(self) -> None:
        fmt = _format(reasoning=None, tools="hermes")
        output = tokens("<tool_call>", _CALL, "</tool_call>")

        events = assert_cut_invariant(fmt, output, tools=False)

        assert events[0] == TextDelta("<tool_call>" + _CALL + "</tool_call>")

    def test_text_after_a_call_keeps_its_leading_whitespace(self) -> None:
        fmt = _format(reasoning=None, tools="hermes")
        output = tokens("<tool_call>", '{"name":"f","arguments":{}}', "</tool_call>", "\nDone")

        events = assert_cut_invariant(fmt, output)

        assert TextDelta("\nDone") in events

    def test_reasoning_then_a_call(self) -> None:
        fmt = _format(reasoning="think", tools="hermes")
        output = tokens("<think>", "need weather", "</think>", "\n\n", "<tool_call>", _CALL)
        output += tokens("</tool_call>")

        events = assert_cut_invariant(fmt, output)

        assert events[:2] == [
            ReasoningDelta("need weather"),
            ToolCallStart(0, "call_x_0", "get_weather"),
        ]

    def test_long_arguments_are_parsed_in_linear_time(self) -> None:
        fmt = _format(reasoning=None, tools="hermes")
        parser = fmt.parser(call_id_prefix="c", tools=True)
        body = '{"name": "write", "arguments": {"text": "' + "x" * 200_000 + '"}}'
        output = tokens("<tool_call>", body, "</tool_call>")

        events = run(parser, output, range(1, len(output), 7))

        assert ToolCallEnd(0, valid=True) in events


class TestEnds:
    def test_a_stop_string_is_reported_with_the_finish(self) -> None:
        fmt = OutputFormat()

        events = run(fmt.parser(call_id_prefix="c", tools=False), tokens("ab"), [1], stop="END")

        assert events[-1] == FinishEvent(FinishReason.STOP, stop_sequence="END")

    def test_length_is_a_normal_finish(self) -> None:
        fmt = OutputFormat()
        parser = fmt.parser(call_id_prefix="c", tools=False)

        events = run(parser, tokens("ab"), [], reason=FinishReason.LENGTH, count=2)

        assert events[-2:] == [UsageEvent(5, 2), FinishEvent(FinishReason.LENGTH)]

    @pytest.mark.parametrize(
        ("reason", "error", "expected"),
        [
            (FinishReason.CANCELLED, None, ErrorEvent("cancelled", "the request was cancelled")),
            (
                FinishReason.TIMEOUT,
                None,
                ErrorEvent("request_timeout", "the request ran out of time", retryable=True),
            ),
            (
                FinishReason.ERROR,
                ErrorInfo("backend_error", "kernel failed"),
                ErrorEvent("backend_error", "kernel failed"),
            ),
        ],
    )
    def test_other_ends_are_one_error_event_with_no_usage(
        self, reason: FinishReason, error: ErrorInfo | None, expected: ErrorEvent
    ) -> None:
        fmt = OutputFormat()
        parser = fmt.parser(call_id_prefix="c", tools=False)

        events = run(parser, tokens("ab"), [], reason=reason, error=error)

        assert events == [TextDelta("ab"), expected]

    def test_nothing_is_reported_after_the_end(self) -> None:
        parser = OutputFormat().parser(call_id_prefix="c", tools=False)
        parser.feed(_fragment(tokens("a"), finished=True))

        assert parser.feed(_fragment(tokens("b"), finished=True)) == []
