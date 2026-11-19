from __future__ import annotations

import random

import pytest

from ayaka.tokenizers.decoding.stop import StopChecker


def run(checker: StopChecker, *deltas: str) -> tuple[str, str | None]:
    emitted, matched = "", None
    for delta in deltas:
        text, matched = checker.feed(delta)
        emitted += text
        if matched is not None:
            break
    return emitted, matched


@pytest.mark.unit
class TestStopChecker:
    def test_without_stops_everything_passes_through(self) -> None:
        checker = StopChecker((), False)

        assert not checker.has_stops
        assert checker.feed("abc") == ("abc", None)

    def test_stop_is_cut_out_by_default(self) -> None:
        assert run(StopChecker(("END",), False), "abcEND tail") == ("abc", "END")

    def test_include_keeps_the_stop_string(self) -> None:
        assert run(StopChecker(("END",), True), "abcEND tail") == ("abcEND", "END")

    def test_stop_split_across_deltas_is_found(self) -> None:
        assert run(StopChecker(("END",), False), "abE", "N", "Dxyz") == ("ab", "END")

    def test_withholds_only_the_possible_prefix(self) -> None:
        checker = StopChecker(("END",), False)

        assert checker.feed("abcEN") == ("abc", None)
        assert checker.pending == "EN"

    def test_text_that_cannot_start_a_stop_is_released_at_once(self) -> None:
        checker = StopChecker(("END",), False)

        assert checker.feed("abcde") == ("abcde", None)
        assert checker.pending == ""

    def test_a_repeated_first_character_holds_the_longest_possible_start(self) -> None:
        checker = StopChecker(("<<end>>",), False)

        assert checker.feed("a<<<e") == ("a<", None)
        assert checker.pending == "<<e"
        assert checker.feed("nd>>z") == ("", "<<end>>")

    def test_earliest_match_wins_across_stops(self) -> None:
        assert run(StopChecker(("zz", "bc"), False), "abczz") == ("a", "bc")

    def test_flush_releases_held_text_once(self) -> None:
        checker = StopChecker(("END",), False)
        checker.feed("abE")

        assert checker.flush() == "E"
        assert checker.flush() == ""

    def test_the_match_that_ends_first_wins(self) -> None:
        """``b`` ends before ``abc``; the old rule (starts first) depended on the split."""
        assert run(StopChecker(("abc", "b"), False), "abc") == ("a", "b")
        assert run(StopChecker(("abc", "b"), False), "ab", "c") == ("a", "b")

    def test_of_stops_that_end_together_the_longest_is_reported(self) -> None:
        assert run(StopChecker(("c", "abc"), True), "xabc") == ("xabc", "abc")
        assert run(StopChecker(("c", "abc"), False), "xa", "bc") == ("x", "abc")

    def test_feed_after_done_is_ignored(self) -> None:
        checker = StopChecker(("x",), False)
        checker.feed("ax")

        assert checker.feed("more") == ("", None)

    def test_single_character_stops_need_no_lookback(self) -> None:
        assert run(StopChecker(("\n",), False), "ab\ncd") == ("ab", "\n")

    def test_empty_strings_are_dropped(self) -> None:
        checker = StopChecker(("",), False)

        assert not checker.has_stops
        assert checker.feed("a") == ("a", None)

    def test_empty_delta_is_a_noop(self) -> None:
        assert StopChecker(("END",), False).feed("") == ("", None)


# -- splitting does not change the result -----------------------------------------------------

_ALPHABET = "ab<\u00e9"  # small, so that stops overlap and text often holds the start of one


def _random_case(rng: random.Random) -> tuple[tuple[str, ...], str]:
    stops = [
        "".join(rng.choice(_ALPHABET) for _ in range(rng.randint(1, 5)))
        for _ in range(rng.randint(1, 3))
    ]
    for _ in range(rng.randint(0, 2)):  # a stop that sits inside another: ("abc", "b")
        whole = rng.choice(stops)
        first = rng.randrange(len(whole))
        stops.append(whole[first : rng.randint(first + 1, len(whole))])
    text = "".join(rng.choice(_ALPHABET) for _ in range(rng.randint(0, 40)))
    if rng.random() < 0.5:
        cut = rng.randint(0, len(text))
        text = text[:cut] + rng.choice(stops) + text[cut:]
    return tuple(stops), text


def _split(rng: random.Random, text: str) -> list[str]:
    if len(text) < 2:
        return [text]
    cuts = sorted(rng.sample(range(1, len(text)), rng.randint(0, len(text) - 1)))
    return [text[a:b] for a, b in zip([0, *cuts], [*cuts, len(text)], strict=True)]


def _outcome(stops: tuple[str, ...], include: bool, deltas: list[str]) -> tuple[str, str | None]:
    checker = StopChecker(stops, include)
    emitted, matched = "", None
    for delta in deltas:
        text, matched = checker.feed(delta)
        emitted += text
        if matched is not None:
            break
    if matched is None:
        emitted += checker.flush()
    return emitted, matched


@pytest.mark.unit
@pytest.mark.parametrize("seed", range(40))
def test_every_split_gives_the_result_of_one_feed(seed: int) -> None:
    """A property test with a fixed seed: 25 random splits of each of 40 random cases."""
    rng = random.Random(seed)  # noqa: S311  # a seeded property test, not security
    stops, text = _random_case(rng)
    for include in (False, True):
        whole = _outcome(stops, include, [text])
        for _ in range(25):
            assert _outcome(stops, include, _split(rng, text)) == whole, (stops, text)


@pytest.mark.unit
@pytest.mark.parametrize("seed", range(20))
def test_held_text_is_always_a_possible_start_of_a_stop(seed: int) -> None:
    rng = random.Random(1000 + seed)  # noqa: S311  # a seeded property test, not security
    stops, text = _random_case(rng)
    checker = StopChecker(stops, False)
    for delta in _split(rng, text):
        _, matched = checker.feed(delta)
        if matched is not None:
            break
        held = checker.pending
        assert not held or any(stop.startswith(held) and stop != held for stop in stops)
