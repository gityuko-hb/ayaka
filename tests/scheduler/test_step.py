"""``StepInput`` and ``StepOutput``: consistency checks, several rows per request, immutability."""

from __future__ import annotations

import dataclasses
from typing import Any

import pytest

from ayaka.core.errors import EngineError, ErrorInfo
from ayaka.core.params import SamplingParams
from ayaka.core.types import ForwardMode
from ayaka.scheduler.step import StepInput, StepOutput
from fake_runner import rid

pytestmark = pytest.mark.unit

_EMPTY: dict[str, Any] = {
    "request_ids": (),
    "batch_rows": (),
    "rows_per_request": (),
    "start_positions": (),
    "tokens": (),
    "need_logits": (),
    "sampling_params": (),
}


def _input(**overrides: Any) -> StepInput:
    fields: dict[str, Any] = {
        "step_id": 1,
        "mode": ForwardMode.PREFILL,
        "request_ids": (rid("a"), rid("b")),
        "batch_rows": (0, 1),
        "rows_per_request": (3, 2),
        "start_positions": (0, 4),
        "tokens": ((5, 6, 7), (8, 9)),
        "need_logits": (False, True),
        "sampling_params": (SamplingParams(), SamplingParams()),
    }
    fields.update(overrides)
    return StepInput(**fields)


def test_a_prefill_step_reads_back_what_it_was_built_with() -> None:
    step = _input()

    assert (step.num_requests, step.num_rows) == (2, 5)
    assert step.draft_slots == (0, 0)
    assert step.table_updates == ()


def test_a_decode_request_may_take_several_rows_and_the_runner_fills_the_drafts() -> None:
    step = _input(
        mode=ForwardMode.VERIFY,
        rows_per_request=(3, 1),
        start_positions=(7, 9),
        tokens=((42,), (43,)),
        need_logits=(True, True),
    )

    assert step.num_rows == 4
    assert step.draft_slots == (2, 0)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"batch_rows": (0,)}, "differ in length"),
        ({"tokens": ((5, 6, 7),)}, "differ in length"),
        (_EMPTY, "no requests"),
        ({"request_ids": (rid("a"), rid("a"))}, "twice"),
        ({"batch_rows": (1, 1)}, "one batch row"),
        ({"mode": ForwardMode.IDLE}, "cannot run in mode"),
        ({"mode": ForwardMode.MIXED}, "cannot run in mode"),
        ({"rows_per_request": (0, 2), "tokens": ((), (8, 9))}, "at least one row"),
        ({"start_positions": (-1, 4)}, "position >= 0"),
        ({"tokens": ((5, 6), (8, 9))}, "knows all its tokens"),
        (
            {"mode": ForwardMode.DECODE, "rows_per_request": (2, 1), "tokens": ((5,), (8,))},
            "one row per request",
        ),
        (
            {"mode": ForwardMode.DECODE, "rows_per_request": (1, 1), "tokens": ((), (8,))},
            "known tokens",
        ),
    ],
)
def test_an_inconsistent_step_input_is_refused(overrides: dict[str, Any], message: str) -> None:
    with pytest.raises(EngineError, match=message):
        _input(**overrides)


def test_a_step_input_cannot_be_changed() -> None:
    step = _input()

    with pytest.raises(dataclasses.FrozenInstanceError):
        step.step_id = 2  # pyright: ignore[reportAttributeAccessIssue]
    with pytest.raises((AttributeError, TypeError)):  # frozen slots: no new attribute either
        step.extra = 1  # pyright: ignore[reportAttributeAccessIssue]


def test_a_step_input_is_hashable_so_a_fingerprint_is_cheap() -> None:
    assert hash(_input()) == hash(_input())
    assert hash(_input()) != hash(_input(start_positions=(0, 5)))


def test_a_step_output_needs_one_token_list_per_request() -> None:
    output = StepOutput(step_id=1, request_ids=(rid("a"), rid("b")), tokens=((), (3,)))

    assert output.releasable is None
    assert output.accepted == ()
    with pytest.raises(EngineError, match="token lists"):
        StepOutput(step_id=1, request_ids=(rid("a"), rid("b")), tokens=((3,),))
    with pytest.raises(EngineError, match="token lists"):
        StepOutput(step_id=1, request_ids=(rid("a"),))


def test_a_failed_step_needs_no_tokens() -> None:
    output = StepOutput(
        step_id=1, request_ids=(rid("a"),), error=ErrorInfo(code="backend_error", message="boom")
    )

    assert output.tokens == ()


def test_per_request_extras_must_match_the_request_count() -> None:
    with pytest.raises(EngineError, match="accepted"):
        StepOutput(step_id=1, request_ids=(rid("a"),), tokens=((1,),), accepted=(0, 0))
    with pytest.raises(EngineError, match="logprobs"):
        StepOutput(step_id=1, request_ids=(rid("a"),), tokens=((1,),), logprobs=(None, None))


def test_a_step_output_cannot_be_changed() -> None:
    output = StepOutput(step_id=1, request_ids=(rid("a"),), tokens=((1,),))

    with pytest.raises(dataclasses.FrozenInstanceError):
        output.tokens = ()  # pyright: ignore[reportAttributeAccessIssue]
