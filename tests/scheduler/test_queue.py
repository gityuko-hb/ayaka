"""``RequestQueue`` and the policies: service order against a plain reference."""

from __future__ import annotations

import random

import pytest

from ayaka.core.errors import EngineError
from ayaka.scheduler.queue import RequestQueue, by_priority, fcfs, resolve_policy
from ayaka.scheduler.request import Request
from fake_runner import make_request, rid

pytestmark = pytest.mark.unit


def _request(name: str | int, priority: int = 0) -> Request:
    return Request.from_tokenized(make_request(name, 3, priority=priority), arrival=0.0)


def _names(queue: RequestQueue) -> list[str]:
    return [str(r.request_id) for r in queue]


def test_fcfs_serves_in_arrival_order_whatever_the_priority() -> None:
    queue = RequestQueue(fcfs)

    for name, priority in (("a", 0), ("b", 9), ("c", -3)):
        queue.push(_request(name, priority))

    assert _names(queue) == ["a", "b", "c"]


def test_priority_serves_higher_first_and_ties_in_arrival_order() -> None:
    queue = RequestQueue(by_priority)

    for name, priority in (("a", 0), ("b", 2), ("c", 0), ("d", 2), ("e", 1)):
        queue.push(_request(name, priority))

    assert _names(queue) == ["b", "d", "e", "a", "c"]


@pytest.mark.parametrize(("name", "policy"), [("fcfs", fcfs), ("priority", by_priority)])
def test_resolve_policy_names_the_functions(name: str, policy: object) -> None:
    assert resolve_policy(name) is policy  # pyright: ignore[reportArgumentType]


def test_peek_does_not_remove_and_pop_does() -> None:
    queue = RequestQueue()
    queue.push(_request("a"))
    queue.push(_request("b"))

    peeked = queue.peek()

    assert peeked is not None
    assert peeked.request_id == rid("a")
    assert len(queue) == 2
    assert queue.pop().request_id == rid("a")
    assert queue.pop().request_id == rid("b")
    assert queue.peek() is None
    with pytest.raises(IndexError):
        queue.pop()


def test_remove_takes_a_request_out_of_the_middle() -> None:
    queue = RequestQueue()
    for name in "abc":
        queue.push(_request(name))

    removed = queue.remove(rid("b"))

    assert removed.request_id == rid("b")
    assert rid("b") not in queue
    assert _names(queue) == ["a", "c"]
    with pytest.raises(KeyError):
        queue.remove(rid("b"))


def test_removing_the_head_moves_peek_on() -> None:
    queue = RequestQueue()
    queue.push(_request("a"))
    queue.push(_request("b"))

    queue.remove(rid("a"))

    head = queue.peek()
    assert head is not None
    assert head.request_id == rid("b")


def test_pushing_an_id_that_is_queued_is_refused() -> None:
    queue = RequestQueue()
    queue.push(_request("a"))

    with pytest.raises(EngineError, match="already queued"):
        queue.push(_request("a"))

    assert len(queue) == 1


def test_an_id_can_be_queued_again_after_it_left() -> None:
    queue = RequestQueue()
    queue.push(_request("a"))
    queue.remove(rid("a"))

    queue.push(_request("a"))

    assert _names(queue) == ["a"]
    assert queue.pop().request_id == rid("a")
    assert queue.peek() is None


@pytest.mark.parametrize("seed", range(5))
@pytest.mark.parametrize("policy", [fcfs, by_priority], ids=["fcfs", "priority"])
def test_random_operations_match_a_sorted_reference(seed: int, policy: object) -> None:
    rng = random.Random(seed)  # noqa: S311  # a reproducible workload, not a secret
    queue = RequestQueue(policy)  # pyright: ignore[reportArgumentType]
    reference: list[tuple[tuple[int, ...], int, Request]] = []
    counter = 0

    for _ in range(300):
        action = rng.choice(("push", "push", "pop", "remove"))
        if action == "push":
            counter += 1
            request = _request(counter, rng.randint(-2, 2))
            queue.push(request)
            reference.append((policy(request), counter, request))  # pyright: ignore[reportCallIssue,reportUnknownMemberType]
        elif action == "pop" and reference:
            reference.sort(key=lambda entry: entry[:2])
            assert queue.pop() is reference.pop(0)[2]
        elif action == "remove" and reference:
            victim = rng.choice(reference)
            reference.remove(victim)
            assert queue.remove(victim[2].request_id) is victim[2]
        reference.sort(key=lambda entry: entry[:2])
        assert [r.request_id for r in queue] == [e[2].request_id for e in reference]
        assert len(queue) == len(reference)
