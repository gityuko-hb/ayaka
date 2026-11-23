"""The incremental output parsers: markers, reasoning blocks and function calls.

The central property is that a parser's events do not depend on where the stream is cut. A model
output is written here as a list of ``(token id, text)`` pairs: ordinary characters are one token
each (id = code point), and a marker that is one token of the vocabulary has an id above 1000. The
same output is fed whole, one token at a time and cut at every token boundary, and the merged
events must always be the same.
"""

from __future__ import annotations

import json
import random
import re
from itertools import pairwise
from pathlib import Path
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
from ayaka.entrypoints.parsers.formats import (
    REASONING_FORMATS,
    TOOL_CALL_FORMATS,
    reasoning_format,
    tool_call_format,
)
from ayaka.entrypoints.parsers.markers import Marker, MarkerScanner, resolve_marker
from ayaka.entrypoints.parsers.output import ChoicesParser, OutputFormat
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
        tool_markers=_BY_ID.get(tools) if by_id and tools else None,  # llama3_json has none
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

    def test_a_repeated_first_character_is_held_only_from_the_last_possible_start(self) -> None:
        scanner = MarkerScanner()
        marker = Marker("<tool_call>")

        assert scanner.scan("a<<<", [marker]) == ("a<<", None)
        assert scanner.held == "<"
        assert scanner.scan("tool_call>b", [marker]) == ("", marker)

    def test_a_marker_matched_by_id_needs_its_token(self) -> None:
        scanner = MarkerScanner()
        marker = Marker("<think>", 7)

        assert scanner.scan("say <think> please", [marker]) == ("say <think> please", None)
        scanner.note_tokens([7])
        assert scanner.scan("<think>", [marker]) == ("", marker)

    def test_a_credit_is_dropped_once_its_fragment_is_scanned(self) -> None:
        """The real token came with text that had no active marker: its credit does not wait."""
        scanner = MarkerScanner()
        marker = Marker("<think>", 7)

        scanner.note_tokens([7])
        assert scanner.scan("<think>", [Marker("</think>", 8)]) == ("<think>", None)
        assert scanner.scan("later <think>", [marker]) == ("later <think>", None)

    def test_a_credit_lasts_while_its_fragment_is_scanned_in_several_states(self) -> None:
        scanner = MarkerScanner()
        opening, closing = Marker("<a>", 1), Marker("</a>", 2)

        scanner.note_tokens([1, 2])
        assert scanner.scan("x<a>y</a>z", [opening]) == ("x", opening)
        assert scanner.scan("", [closing]) == ("y", closing)
        assert scanner.scan("", [opening]) == ("z", None)

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

    def test_with_two_choices_the_reasoning_tokens_of_both_are_counted(self) -> None:
        fmt = _format(reasoning="think", tools=None)
        first = tokens("<think>", "abc", "</think>", "de")  # four reasoning tokens, as above
        second = tokens("<think>", "x", "</think>", "yz")  # two
        parser = ChoicesParser([fmt.parser(call_id_prefix=f"c{i}", tools=False) for i in (0, 1)])
        pieces = [
            CompletionOutput(
                index=index,
                token_ids=tuple(token for token, _ in output[i : i + 1]),
                text="".join(text for _, text in output[i : i + 1]),
                finish_reason=FinishReason.STOP if i == len(output) - 1 else None,
            )
            for index, output in enumerate((first, second))
            for i in range(len(output))
        ]
        count = len(first) + len(second)
        events: list[OutputEvent] = []
        for k, piece in enumerate(pieces):
            last = k == len(pieces) - 1
            events += parser.feed(
                RequestOutput(
                    request_id=RequestId("r"),
                    outputs=(piece,),
                    finished=last,
                    usage=Usage(prompt_tokens=5, completion_tokens=count if last else 0),
                )
            )

        [usage] = [event for event in events if isinstance(event, UsageEvent)]
        assert usage.completion_tokens == count
        assert usage.reasoning_tokens == 4 + 2  # 0 before: the choices' counts were dropped


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

    def test_a_marker_token_inside_reasoning_leaves_no_credit_for_the_answer(self) -> None:
        """A real ``<tool_call>`` token in the reasoning, then the same string spelled out.

        The answer's spelled markers are text however the output is cut: the reasoning's token
        was never the tool-call parser's to spend.
        """
        fmt = _format(reasoning="think", tools="hermes")
        output = tokens("<think>", "maybe ", "<tool_call>", " no", "</think>", "use <tool_call>")
        output += tokens(_CALL, "</tool_call>")

        events = assert_cut_invariant(fmt, output)

        assert events[0] == ReasoningDelta("maybe <tool_call> no")
        assert all(not isinstance(event, ToolCallStart) for event in events)
        assert events[1] == TextDelta(f"use <tool_call>{_CALL}</tool_call>")

    def test_a_marker_token_inside_the_block_does_not_open_a_later_one(self) -> None:
        fmt = _format(reasoning="think", tools=None)
        output = tokens("<think>", "a ", "<think>", " b", "</think>", "ans <think> c")

        events = assert_cut_invariant(fmt, output, tools=False)

        assert events[:2] == [ReasoningDelta("a <think> b"), TextDelta("ans <think> c")]

    def test_a_stray_closing_token_does_not_close_a_later_spelled_call(self) -> None:
        """A real ``</tool_call>`` in the text, then a call whose end is spelled: it stays open."""
        fmt = _format(reasoning=None, tools="hermes")
        output = tokens("a ", "</tool_call>", " b ", "<tool_call>", _CALL)
        output += [(ord(char), char) for char in "</tool_call>"]  # spelled, not the token

        events = assert_cut_invariant(fmt, output)

        assert events[0] == TextDelta("a </tool_call> b")
        assert events[1] == ToolCallStart(0, "call_x_0", "get_weather")
        assert ToolCallEnd(0, valid=False) in events  # cut by the end, not closed

    def test_a_repeated_first_character_of_the_marker_is_text_up_to_the_last_one(self) -> None:
        """``<<tool_call>``: the first ``<`` is text, the marker starts at the second."""
        fmt = _format(reasoning=None, tools="hermes", by_id=False)

        events = assert_cut_invariant(fmt, tokens("a<<tool_call>", _CALL, "</tool_call>"))

        assert events[:2] == [TextDelta("a<"), ToolCallStart(0, "call_x_0", "get_weather")]

    def test_an_opening_marker_inside_a_call_is_part_of_its_body(self) -> None:
        fmt = _format(reasoning=None, tools="hermes")
        output = tokens("<tool_call>", '{"name": "f", "arguments": {"s": "', "<tool_call>")
        output += tokens('"}}', "</tool_call>")

        events = assert_cut_invariant(fmt, output)

        assert events[:3] == [
            ToolCallStart(0, "call_x_0", "f"),
            ToolCallDelta(0, '{"s": "<tool_call>"}'),
            ToolCallEnd(0, valid=True),
        ]

    def test_a_key_written_with_an_escape_is_the_key_it_decodes_to(self) -> None:
        fmt = _format(reasoning=None, tools="hermes")
        body = '{"\\u006eame": "f", "argu\\u006dents": {"a": 1}}'

        events = assert_cut_invariant(fmt, tokens("<tool_call>", body, "</tool_call>"))

        assert events[:3] == [
            ToolCallStart(0, "call_x_0", "f"),
            ToolCallDelta(0, '{"a": 1}'),
            ToolCallEnd(0, valid=True),
        ]

    def test_long_arguments_are_parsed_in_linear_time(self) -> None:
        fmt = _format(reasoning=None, tools="hermes")
        parser = fmt.parser(call_id_prefix="c", tools=True)
        body = '{"name": "write", "arguments": {"text": "' + "x" * 200_000 + '"}}'
        output = tokens("<tool_call>", body, "</tool_call>")

        events = run(parser, output, range(1, len(output), 7))

        assert ToolCallEnd(0, valid=True) in events
        arguments = "".join(e.arguments for e in events if isinstance(e, ToolCallDelta))
        assert arguments == '{"text": "' + "x" * 200_000 + '"}'

    @pytest.mark.parametrize(
        "arguments",
        [
            r'{"q": "say \"hi\" \\ "}',  # escaped quotes and a backslash
            '{"\\u006eame": "x", "s": "}{]"}',  # an escaped key; braces inside a string
            '{"a": [1, {"b": "}"}, [2]], "c": null, "d": -1.5e3, "e": true}',
            '{"emoji": "\U0001f600 \u00e9", "nl": "a\\nb"}',
        ],
    )
    def test_awkward_arguments_pass_through_at_every_cut(self, arguments: str) -> None:
        fmt = _format(reasoning=None, tools="hermes")
        body = '{"name": "f", "arguments": ' + arguments + "}"

        events = assert_cut_invariant(fmt, tokens("<tool_call>", body, "</tool_call>"))

        sent = "".join(e.arguments for e in events if isinstance(e, ToolCallDelta))
        assert sent == arguments
        assert ToolCallEnd(0, valid=True) in events

    def test_a_long_whitespace_run_is_kept_whole(self) -> None:
        """A small model writing 100K newlines: kept, cut at every 13th token, in linear time."""
        fmt = _format(reasoning="think", tools="hermes")
        output = tokens(
            "<think>", "a" + "\n" * 100_000 + "b", "</think>", "c" + " " * 100_000 + "d"
        )
        parser = fmt.parser(call_id_prefix="c", tools=True)

        events = run(parser, output, range(1, len(output), 13))

        assert events[0] == ReasoningDelta("a" + "\n" * 100_000 + "b")
        assert events[1] == TextDelta("c" + " " * 100_000 + "d")


class TestToolBodyProperty:
    """Random calls cut at random places, with a fixed seed: cutting changes nothing."""

    @staticmethod
    def _value(rng: random.Random, depth: int = 0) -> object:
        kind = rng.randrange(6 if depth < 3 else 4)
        match kind:
            case 0:
                return rng.choice([True, False, None, 0, -12, 3.5])
            case 1 | 2:
                return "".join(
                    rng.choice('ab"\\{}[]:, \n\u00e9\U0001f600') for _ in range(rng.randrange(8))
                )
            case 3:
                return rng.randrange(1000)
            case 4:
                items = rng.randrange(3)
                return [TestToolBodyProperty._value(rng, depth + 1) for _ in range(items)]
            case _:
                items = rng.randrange(3)
                return {f"k{i}": TestToolBodyProperty._value(rng, depth + 1) for i in range(items)}

    @classmethod
    def _arguments(cls, rng: random.Random) -> str:
        return json.dumps(
            {f"p{i}": cls._value(rng) for i in range(rng.randrange(1, 4))},
            ensure_ascii=rng.random() < 0.5,
            separators=rng.choice([(",", ":"), (", ", ": ")]),
        )

    @staticmethod
    def _same_at_random_cuts(rng: random.Random, output: list[Token]) -> list[OutputEvent]:
        """The events of ``output`` cut nowhere; ten random cuttings must give the same."""
        fmt = _format(reasoning=None, tools="hermes")
        whole = run(fmt.parser(call_id_prefix="c", tools=True), output, [])
        for _ in range(10):
            count = rng.randrange(1, min(20, len(output)))
            cuts = sorted(rng.sample(range(1, len(output)), count))
            assert run(fmt.parser(call_id_prefix="c", tools=True), output, cuts) == whole
        return whole

    @pytest.mark.parametrize("seed", range(30))
    def test_random_cuts_equal_no_cut_and_the_pieces_are_the_model_text(self, seed: int) -> None:
        rng = random.Random(seed)  # noqa: S311  # a seeded property test, not security
        arguments = self._arguments(rng)
        body = '{"name": "tool", "arguments": ' + arguments + "}"

        whole = self._same_at_random_cuts(rng, tokens("<tool_call>", body, "</tool_call>"))

        sent = "".join(e.arguments for e in whole if isinstance(e, ToolCallDelta))
        assert sent == arguments
        assert ToolCallEnd(0, valid=True) in whole

    @pytest.mark.parametrize("seed", range(30))
    def test_a_damaged_body_is_cut_the_same_and_ends_valid_only_if_it_parses(
        self, seed: int
    ) -> None:
        rng = random.Random(1000 + seed)  # noqa: S311  # a seeded property test, not security
        body = '{"name": "tool", "arguments": ' + self._arguments(rng) + "}"
        at = rng.randrange(len(body))
        damage = rng.choice(["", '"', "\\", "{", "}", "]", ","])
        body = body[:at] + damage + body[at + rng.randrange(2) :]

        whole = self._same_at_random_cuts(rng, tokens("<tool_call>", body, "</tool_call>"))

        ends = [event for event in whole if isinstance(event, ToolCallEnd)]
        assert all(end.valid == _is_call(body) for end in ends)


def _is_call(body: str) -> bool:
    """Whether ``body`` is a call as the parser means it: a JSON object with name and arguments."""
    try:
        value = json.loads(body)
    except ValueError:
        return False
    return (
        isinstance(value, dict)
        and isinstance(value.get("name"), str)
        and isinstance(value.get("arguments"), dict)
    )


_LLAMA_CALL = '{"name": "get_weather", "parameters": {"city": "Paris", "days": [1, 2]}}'


class TestToolCallsWithoutMarkers:
    """``llama3_json``: a bare JSON object whose arguments are called ``parameters``."""

    def test_the_format_has_no_markers_and_the_other_key(self) -> None:
        fmt = tool_call_format("llama3_json")

        assert fmt.markers() is None
        assert fmt.arguments == "parameters"
        assert OutputFormat.resolve(tool_calls="llama3_json").tool_markers is None
        assert tool_call_format("hermes").arguments == "arguments"

    def test_a_call_streams_its_name_then_the_raw_parameters(self) -> None:
        fmt = _format(reasoning=None, tools="llama3_json")

        events = assert_cut_invariant(fmt, tokens("\n", _LLAMA_CALL))

        assert events == [
            ToolCallStart(0, "call_x_0", "get_weather"),
            ToolCallDelta(0, '{"city": "Paris", "days": [1, 2]}'),
            ToolCallEnd(0, valid=True),
            UsageEvent(5, 0),
            FinishEvent(FinishReason.STOP, tool_calls=True),
        ]

    def test_the_call_starts_before_its_parameters_are_complete(self) -> None:
        fmt = _format(reasoning=None, tools="llama3_json")
        parser = fmt.parser(call_id_prefix="c", tools=True)
        head = tokens('{"name": "f", "parameters": {"a": ')

        events = parser.feed(_fragment(head, finished=False))

        assert events == [ToolCallStart(0, "c_0", "f"), ToolCallDelta(0, '{"a": ')]

    def test_a_name_that_comes_after_the_parameters_still_makes_a_call(self) -> None:
        fmt = _format(reasoning=None, tools="llama3_json")

        events = assert_cut_invariant(fmt, tokens('{"parameters": {"x": "}"}, "name": "b"}'))

        assert events[:3] == [
            ToolCallStart(0, "call_x_0", "b"),
            ToolCallDelta(0, '{"x": "}"}'),
            ToolCallEnd(0, valid=True),
        ]

    def test_the_hermes_key_is_not_the_arguments_of_this_format(self) -> None:
        fmt = _format(reasoning=None, tools="llama3_json")

        events = assert_cut_invariant(fmt, tokens('{"name": "f", "arguments": {"a": 1}}'))

        assert events[:2] == [ToolCallStart(0, "call_x_0", "f"), ToolCallEnd(0, valid=False)]

    def test_parameters_that_are_not_an_object_end_invalid(self) -> None:
        fmt = _format(reasoning=None, tools="llama3_json")

        events = assert_cut_invariant(fmt, tokens('{"name": "f", "parameters": [1]}'))

        assert events[:3] == [
            ToolCallStart(0, "call_x_0", "f"),
            ToolCallDelta(0, "[1]"),
            ToolCallEnd(0, valid=False),
        ]

    @pytest.mark.parametrize(
        "answer",
        [
            "It is sunny in Paris.",
            "\n  Leading whitespace is part of a text answer.",
            "Let me check {the weather}.",
            "[1, 2]",
        ],
    )
    def test_an_answer_that_does_not_start_with_a_brace_is_text(self, answer: str) -> None:
        fmt = _format(reasoning=None, tools="llama3_json")

        events = assert_cut_invariant(fmt, tokens(answer))

        assert events[0] == TextDelta(answer)
        assert events[-1] == FinishEvent(FinishReason.STOP)

    def test_text_is_released_as_it_comes(self) -> None:
        fmt = _format(reasoning=None, tools="llama3_json")
        parser = fmt.parser(call_id_prefix="c", tools=True)

        events = parser.feed(_fragment(tokens("Hello wor"), finished=False))

        assert events == [TextDelta("Hello wor")]

    @pytest.mark.parametrize(
        "answer",
        [
            '{"city": "Paris"}',  # an object with no name: a JSON answer, not a call
            '  {"city": "Paris"} and more',
            '{"name": 7, "parameters": {}}',  # a name that is not a string
            '{"na',  # cut before a name: held back, then released as text
            "{",
            "{oops",
            '{"parameters": {"a": 1}}',
        ],
    )
    def test_an_object_that_is_not_a_call_stays_the_text_it_was(self, answer: str) -> None:
        fmt = _format(reasoning=None, tools="llama3_json")

        events = assert_cut_invariant(fmt, tokens(answer))

        assert events[0] == TextDelta(answer)
        assert events[-1] == FinishEvent(FinishReason.STOP)

    def test_a_call_cut_by_the_token_limit_ends_invalid(self) -> None:
        fmt = _format(reasoning=None, tools="llama3_json")

        events = assert_cut_invariant(fmt, tokens('{"name": "f", "parameters": {"a": "unfini'))

        assert events[:3] == [
            ToolCallStart(0, "call_x_0", "f"),
            ToolCallDelta(0, '{"a": "unfini'),
            ToolCallEnd(0, valid=False),
        ]

    def test_a_broken_call_swallows_the_rest_of_the_answer(self) -> None:
        fmt = _format(reasoning=None, tools="llama3_json")

        events = assert_cut_invariant(fmt, tokens('{"name": "f", oops "parameters": {}} after'))

        assert events[:2] == [ToolCallStart(0, "call_x_0", "f"), ToolCallEnd(0, valid=False)]
        assert not any(isinstance(event, TextDelta) for event in events)

    def test_whitespace_after_the_call_is_layout_and_later_text_is_kept(self) -> None:
        fmt = _format(reasoning=None, tools="llama3_json")

        quiet = assert_cut_invariant(fmt, tokens(_LLAMA_CALL, " \n"))
        said = assert_cut_invariant(fmt, tokens(_LLAMA_CALL, "\nDone"))

        assert not any(isinstance(event, TextDelta) for event in quiet)
        assert TextDelta("\nDone") in said

    def test_calls_separated_by_a_semicolon_are_several_calls(self) -> None:
        fmt = _format(reasoning=None, tools="llama3_json")
        second = '{"name": "b", "parameters": {"x": 1}}'

        events = assert_cut_invariant(fmt, tokens(_LLAMA_CALL, "; ", second, " \n"))

        assert events[:6] == [
            ToolCallStart(0, "call_x_0", "get_weather"),
            ToolCallDelta(0, '{"city": "Paris", "days": [1, 2]}'),
            ToolCallEnd(0, valid=True),
            ToolCallStart(1, "call_x_1", "b"),
            ToolCallDelta(1, '{"x": 1}'),
            ToolCallEnd(1, valid=True),
        ]
        assert not any(isinstance(event, TextDelta) for event in events)

    @pytest.mark.parametrize(
        ("after", "text"),
        [
            ('\n{"name": "b", "parameters": {}}', '\n{"name": "b", "parameters": {}}'),  # no ';'
            ('{"name": "b", "parameters": {}}', '{"name": "b", "parameters": {}}'),
            (", " + '{"name": "b", "parameters": {}}', ', {"name": "b", "parameters": {}}'),
            ("; and then", "; and then"),
            (";; {", ";; {"),
            (";", ";"),  # a separator with no call after it is shown, not dropped
            ("; {oops", "; {oops"),
            ('; {"a": 1}', '; {"a": 1}'),  # an object that is not a call
        ],
    )
    def test_only_a_semicolon_leads_to_a_next_call_anything_else_is_text(
        self, after: str, text: str
    ) -> None:
        fmt = _format(reasoning=None, tools="llama3_json")

        events = assert_cut_invariant(fmt, tokens(_LLAMA_CALL, after))

        assert events[:3] == [
            ToolCallStart(0, "call_x_0", "get_weather"),
            ToolCallDelta(0, '{"city": "Paris", "days": [1, 2]}'),
            ToolCallEnd(0, valid=True),
        ]
        assert events[3] == TextDelta(text)
        assert sum(isinstance(event, ToolCallStart) for event in events) == 1

    def test_the_second_call_may_be_cut_by_the_token_limit(self) -> None:
        fmt = _format(reasoning=None, tools="llama3_json")

        events = assert_cut_invariant(fmt, tokens(_LLAMA_CALL, '; {"name": "b", "parameters": {"x'))

        assert events[3:6] == [
            ToolCallStart(1, "call_x_1", "b"),
            ToolCallDelta(1, '{"x'),
            ToolCallEnd(1, valid=False),
        ]

    def test_a_single_call_request_ends_after_the_first_call(self) -> None:
        fmt = _format(reasoning=None, tools="llama3_json")
        parser = fmt.parser(call_id_prefix="c", tools=True, parallel_tool_calls=False)
        output = tokens(_LLAMA_CALL, "; ", '{"name": "b", "parameters": {}}')

        events = run(parser, output, [len(_LLAMA_CALL), len(_LLAMA_CALL) + 3])

        assert parser.ended_early
        assert sum(isinstance(event, ToolCallStart) for event in events) == 1
        assert events[-1] == FinishEvent(FinishReason.STOP, tool_calls=True)

    def test_a_function_key_next_to_the_function_type_is_the_name(self) -> None:
        fmt = _format(reasoning=None, tools="llama3_json")
        call = '{"type": "function", "function": "get_weather", "parameters": {"city": "Paris"}}'

        events = assert_cut_invariant(fmt, tokens(call))

        assert events[:3] == [
            ToolCallStart(0, "call_x_0", "get_weather"),
            ToolCallDelta(0, '{"city": "Paris"}'),
            ToolCallEnd(0, valid=True),
        ]

    def test_the_type_may_come_after_the_function(self) -> None:
        fmt = _format(reasoning=None, tools="llama3_json")
        call = '{"function": "f", "parameters": {}, "type": "function"}'

        events = assert_cut_invariant(fmt, tokens(call))

        assert events[:3] == [
            ToolCallStart(0, "call_x_0", "f"),
            ToolCallDelta(0, "{}"),
            ToolCallEnd(0, valid=True),
        ]

    @pytest.mark.parametrize(
        "answer",
        [
            '{"function": "f", "parameters": {}}',  # no type
            '{"type": "tool", "function": "f", "parameters": {}}',  # another type
            '{"type": "function", "function": {"name": "f"}, "parameters": {}}',  # not a string
        ],
    )
    def test_a_function_key_alone_is_not_a_name(self, answer: str) -> None:
        fmt = _format(reasoning=None, tools="llama3_json")

        events = assert_cut_invariant(fmt, tokens(answer))

        assert events[0] == TextDelta(answer)

    def test_the_hermes_format_does_not_read_a_function_key(self) -> None:
        fmt = _format(reasoning=None, tools="hermes")
        call = '{"type": "function", "function": "f", "arguments": {}}'

        events = assert_cut_invariant(fmt, tokens("<tool_call>", call, "</tool_call>"))

        assert events[0] == TextDelta("<tool_call>" + call + "</tool_call>")

    def test_without_tools_in_the_request_the_object_is_text(self) -> None:
        fmt = _format(reasoning=None, tools="llama3_json")

        events = assert_cut_invariant(fmt, tokens(_LLAMA_CALL), tools=False)

        assert events[0] == TextDelta(_LLAMA_CALL)

    def test_reasoning_then_a_call(self) -> None:
        fmt = _format(reasoning="think", tools="llama3_json")
        output = tokens("<think>", "need weather", "</think>", "\n\n", _LLAMA_CALL)

        events = assert_cut_invariant(fmt, output)

        assert events[:2] == [
            ReasoningDelta("need weather"),
            ToolCallStart(0, "call_x_0", "get_weather"),
        ]

    def test_a_marker_written_in_the_answer_is_not_special(self) -> None:
        fmt = _format(reasoning=None, tools="llama3_json")
        output = tokens("<tool_call>", _CALL, "</tool_call>")

        events = assert_cut_invariant(fmt, output)

        assert events[0] == TextDelta("<tool_call>" + _CALL + "</tool_call>")

    def test_long_parameters_are_parsed_in_linear_time(self) -> None:
        fmt = _format(reasoning=None, tools="llama3_json")
        parser = fmt.parser(call_id_prefix="c", tools=True)
        body = '{"name": "write", "parameters": {"text": "' + "x" * 200_000 + '"}}'
        output = tokens(body)

        events = run(parser, output, range(1, len(output), 7))

        assert ToolCallEnd(0, valid=True) in events


_LLAMA_ANSWERS = json.loads(
    (Path(__file__).parent / "fixtures" / "llama-3.2-1b-answers.json").read_text(encoding="utf-8")
)["answers"]
_PARIS = ("get_weather", '{"city": "Paris"}')
_TOKYO_WEATHER = ("get_weather", '{"city": "Tokyo"}')
_TOKYO_TIME = ("get_time", '{"city": "Tokyo"}')
# What each recorded answer holds, read off the text by hand: the calls as (name, raw parameters).
_RECORDED_CALLS = {
    "typed-single": [_PARIS],
    "typed-two-same-tool": [_PARIS, _TOKYO_WEATHER],
    "typed-two-tools": [_PARIS, _TOKYO_TIME],
    "named-after-tool-result": [("get_time", '{"city": "Paris"}')],
    "named-single": [_PARIS],
    "named-two-same-tool": [_PARIS, _TOKYO_WEATHER],
    "named-two-tools-sampled": [_PARIS, _TOKYO_TIME],
}
_RECORDED_TEXT = ("python-dict-is-text", "code-is-text")


class TestRecordedLlamaAnswers:
    """Answers of Llama-3.2-1B-Instruct, as written (see ``fixtures/llama-3.2-1b-answers.json``)."""

    def test_the_fixture_holds_what_the_tests_expect(self) -> None:
        assert {a["id"] for a in _LLAMA_ANSWERS} == {*_RECORDED_CALLS, *_RECORDED_TEXT}
        for answer in _LLAMA_ANSWERS:
            dropped = answer["raw"]
            for special in ("<|python_tag|>", "<|eom_id|>", "<|eot_id|>"):
                dropped = dropped.replace(special, "")
            assert dropped == answer["text"], answer["id"]

    @pytest.mark.parametrize("answer", _RECORDED_CALLS)
    def test_a_recorded_answer_with_calls_parses_into_those_calls(self, answer: str) -> None:
        fmt = _format(reasoning=None, tools="llama3_json")
        text = next(a["text"] for a in _LLAMA_ANSWERS if a["id"] == answer)

        events = assert_cut_invariant(fmt, tokens(text))

        expected: list[OutputEvent] = []
        for index, (name, parameters) in enumerate(_RECORDED_CALLS[answer]):
            expected += [
                ToolCallStart(index, f"call_x_{index}", name),
                ToolCallDelta(index, parameters),
                ToolCallEnd(index, valid=True),
            ]
        assert events == [
            *expected,
            UsageEvent(5, 0),
            FinishEvent(FinishReason.STOP, tool_calls=True),
        ]

    @pytest.mark.parametrize("answer", _RECORDED_TEXT)
    def test_a_recorded_answer_that_is_not_a_call_stays_text(self, answer: str) -> None:
        fmt = _format(reasoning=None, tools="llama3_json")
        text = next(a["text"] for a in _LLAMA_ANSWERS if a["id"] == answer)

        events = assert_cut_invariant(fmt, tokens(text))

        assert events == [
            TextDelta(text),
            UsageEvent(5, 0),
            FinishEvent(FinishReason.STOP),
        ]


_QWEN3_ANSWERS = json.loads(
    (Path(__file__).parent / "fixtures" / "qwen3-1.7b-answers.json").read_text(encoding="utf-8")
)["answers"]
_QWEN3_MARKERS = re.compile(r"(<think>|</think>|<tool_call>|</tool_call>)")
# What each recorded answer holds, read off the text by hand: the calls as (name, raw arguments),
# and the answer text after the reasoning.
_QWEN3_CALLS = {
    "think-then-call": [_PARIS],
    "think-then-two-calls": [_PARIS, _TOKYO_TIME],
    "no-think-call": [_PARIS],
    "think-then-text": [],
    "after-tool-result": [],
}
_QWEN3_TEXT = {
    "think-then-text": "The capital of Japan is Tokyo.",
    "after-tool-result": "The current weather in Paris is 18°C with clouds.",
}


def _qwen3_tokens(text: str) -> list[Token]:
    """The recorded text as Qwen3 generated it: each marker one token, as the fixture records."""
    return tokens(*(part for part in _QWEN3_MARKERS.split(text) if part))


class TestRecordedQwen3Answers:
    """Answers of Qwen3-1.7B, generated by Ayaka (see ``fixtures/qwen3-1.7b-answers.json``)."""

    def test_the_fixture_holds_what_the_tests_expect(self) -> None:
        assert {a["id"] for a in _QWEN3_ANSWERS} == set(_QWEN3_CALLS)
        for answer in _QWEN3_ANSWERS:
            assert answer["raw"].removesuffix("<|im_end|>") == answer["text"], answer["id"]
            assert answer["ended_by"] == "stop", answer["id"]

    @pytest.mark.parametrize("answer", _QWEN3_CALLS)
    def test_reasoning_calls_and_text_are_split_as_written(self, answer: str) -> None:
        fmt = _format(reasoning="think", tools="hermes")
        recorded = next(a for a in _QWEN3_ANSWERS if a["id"] == answer)
        text = recorded["text"]

        events = merge_deltas(assert_cut_invariant(fmt, _qwen3_tokens(text)))

        reasoning = [e for e in events if isinstance(e, ReasoningDelta)]
        if recorded["enable_thinking"]:
            inside = text.split("<think>", 1)[1].split("</think>", 1)[0]
            assert reasoning == [
                ReasoningDelta(inside.strip("\n"))
            ]  # the parser trims the newlines
        else:
            assert reasoning == []
        calls = [
            (start.name, delta.arguments)
            for start, delta in pairwise(events)
            if isinstance(start, ToolCallStart) and isinstance(delta, ToolCallDelta)
        ]
        assert calls == _QWEN3_CALLS[answer]
        assert all(e.valid for e in events if isinstance(e, ToolCallEnd))
        answered = "".join(e.text for e in events if isinstance(e, TextDelta)).strip()
        assert answered == _QWEN3_TEXT.get(answer, "")
        assert events[-1] == FinishEvent(FinishReason.STOP, tool_calls=bool(calls))


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
