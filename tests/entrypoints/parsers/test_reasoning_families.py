"""Tests for enhanced reasoning parser families and unclosed-block tool recovery."""

from __future__ import annotations

import random
from dataclasses import replace

import pytest

from ayaka.entrypoints.parsers.formats import REASONING_FORMATS, ReasoningDefault
from ayaka.entrypoints.parsers.markers import Marker
from ayaka.entrypoints.parsers.reasoning import ReasoningParser

pytestmark = pytest.mark.unit


def _collect_pieces(
    fmt_name: str,
    chunks: list[str],
    *,
    starts_inside: bool | None = None,
    tool_start: tuple[str, ...] | None = None,
) -> tuple[str, str]:
    fmt = REASONING_FORMATS[fmt_name]
    if tool_start is not None:
        fmt = replace(fmt, tool_start=tool_start)
    parser = ReasoningParser(
        fmt,
        markers=(Marker(fmt.open, None), Marker(fmt.close, None)),
        starts_inside_override=starts_inside,
    )
    reasoning: list[str] = []
    answer: list[str] = []
    for chunk in chunks:
        for is_reason, text in parser.feed(chunk):
            (reasoning if is_reason else answer).append(text)
    for is_reason, text in parser.finish():
        (reasoning if is_reason else answer).append(text)
    return "".join(reasoning), "".join(answer)


def test_reasoning_format_defaults_and_families_exist() -> None:
    assert set(REASONING_FORMATS) >= {"think", "deepseek_r1", "deepseek_v3", "glm45", "minimax_m3"}
    assert REASONING_FORMATS["deepseek_r1"].default_mode is ReasoningDefault.ON_UNLESS
    assert REASONING_FORMATS["deepseek_v3"].default_mode is ReasoningDefault.OFF_UNLESS
    assert REASONING_FORMATS["glm45"].force_nonempty_content is True
    assert REASONING_FORMATS["minimax_m3"].tool_start_at_line_start is True


@pytest.mark.parametrize(
    ("fmt_name", "tool_start", "sample", "expected_reason", "expected_answer"),
    [
        (
            "think",
            None,
            "<think>step 1</think>answer",
            "step 1",
            "answer",
        ),
        (
            "think",
            ("<tool_call>",),
            '<think>forgot to close <tool_call>{"name":"f","arguments":{}}</tool_call>',
            "forgot to close",
            '<tool_call>{"name":"f","arguments":{}}</tool_call>',
        ),
        (
            "deepseek_r1",
            None,
            "already thinking\n<\uff5cDSML\uff5cfunction_calls>"
            '<\uff5cDSML\uff5cinvoke name="x"/></\uff5cDSML\uff5cfunction_calls>',
            "already thinking",
            "<\uff5cDSML\uff5cfunction_calls>"
            '<\uff5cDSML\uff5cinvoke name="x"/></\uff5cDSML\uff5cfunction_calls>',
        ),
        (
            "deepseek_v3",
            None,
            "<think>plan</think><\uff5cDSML\uff5cfunction_calls></\uff5cDSML\uff5cfunction_calls>",
            "plan",
            "<\uff5cDSML\uff5cfunction_calls></\uff5cDSML\uff5cfunction_calls>",
        ),
        (
            "glm45",
            None,
            "<think>plan \u00e9\u2603 <tool_call>fn</tool_call>",
            "plan \u00e9\u2603",
            "<tool_call>fn</tool_call>",
        ),
    ],
)
def test_reasoning_split_invariance_and_recovery(
    fmt_name: str,
    tool_start: tuple[str, ...] | None,
    sample: str,
    expected_reason: str,
    expected_answer: str,
) -> None:
    one_shot = _collect_pieces(fmt_name, [sample], tool_start=tool_start)
    assert one_shot == (expected_reason, expected_answer)

    char_by_char = _collect_pieces(fmt_name, list(sample), tool_start=tool_start)
    assert char_by_char == one_shot

    rng = random.Random(42)  # noqa: S311  # deterministic chunk-boundary fuzzer in test
    chunks: list[str] = []
    idx = 0
    while idx < len(sample):
        step = rng.randint(1, 5)
        chunks.append(sample[idx : idx + step])
        idx += step
    assert _collect_pieces(fmt_name, chunks, tool_start=tool_start) == one_shot


def test_minimax_m3_requires_line_start_for_tool_recovery() -> None:
    mid_line = (
        "<think>mention <minimax:tool_call> inline\n"
        '<minimax:tool_call><invoke name="ok"/></minimax:tool_call>'
    )
    reason_full, answer_full = _collect_pieces("minimax_m3", [mid_line])
    assert reason_full == "mention <minimax:tool_call> inline"
    assert answer_full == '<minimax:tool_call><invoke name="ok"/></minimax:tool_call>'

    reason_chars, answer_chars = _collect_pieces("minimax_m3", list(mid_line))
    assert (reason_chars, answer_chars) == (reason_full, answer_full)
