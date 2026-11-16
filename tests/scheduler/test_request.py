"""``Request``: live state from a tokenized request, and the table of legal status moves."""

from __future__ import annotations

import itertools

import pytest

from ayaka.core.errors import EngineError
from ayaka.core.types import FinishReason
from ayaka.scheduler.request import LEGAL_TRANSITIONS, Request, RequestEvent, RequestStatus
from fake_runner import make_request, rid

pytestmark = pytest.mark.unit

_ILLEGAL_MOVES = sorted(
    set(itertools.product(RequestStatus, RequestStatus)) - set(LEGAL_TRANSITIONS), key=str
)


def _request(prompt_length: int = 6) -> Request:
    return Request.from_tokenized(make_request("a", prompt_length, priority=3), arrival=1.5)


def test_from_tokenized_copies_the_envelope() -> None:
    tokenized = make_request("a", 6, priority=3)

    request = Request.from_tokenized(tokenized, arrival=1.5, deadline=9.0)

    assert request.request_id == rid("a")
    assert request.prompt == tokenized.prompt_token_ids
    assert request.params is tokenized.sampling
    assert (request.priority, request.arrival, request.deadline) == (3, 1.5, 9.0)
    assert request.status is RequestStatus.WAITING
    assert (request.row, request.computed, request.pending, request.output) == (-1, 0, None, [])


def test_prefilling_is_derived_from_computed_tokens() -> None:
    request = _request(6)

    assert request.is_prefilling
    request.computed = 6
    assert not request.is_prefilling


def test_the_outputs_of_two_requests_are_separate_lists() -> None:
    first, second = _request(), _request()

    first.output.append(1)

    assert second.output == []


@pytest.mark.parametrize(("source", "target"), sorted(LEGAL_TRANSITIONS, key=str))
def test_a_legal_move_changes_the_status(source: RequestStatus, target: RequestStatus) -> None:
    request = _request()
    request._status = source  # pyright: ignore[reportPrivateUsage]

    request.move_to(target)

    assert request.status is target


@pytest.mark.parametrize(("source", "target"), _ILLEGAL_MOVES)
def test_an_illegal_move_raises_and_changes_nothing(
    source: RequestStatus, target: RequestStatus
) -> None:
    request = _request()
    request._status = source  # pyright: ignore[reportPrivateUsage]

    with pytest.raises(EngineError, match="cannot move"):
        request.move_to(target)

    assert request.status is source


def test_nothing_leaves_finished() -> None:
    assert not [move for move in LEGAL_TRANSITIONS if move[0] is RequestStatus.FINISHED]


def test_abort_requested_follows_the_reason() -> None:
    request = _request()

    assert not request.abort_requested
    request.abort_reason = FinishReason.CANCELLED
    assert request.abort_requested


def test_an_event_is_finished_only_with_a_reason() -> None:
    assert not RequestEvent(request_id=rid("a"), token_ids=(1,)).finished
    assert RequestEvent(request_id=rid("a"), finish_reason=FinishReason.STOP).finished
