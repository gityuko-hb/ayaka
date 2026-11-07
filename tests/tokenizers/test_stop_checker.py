from __future__ import annotations

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

        assert checker.feed("abcde") == ("abc", None)
        assert checker.pending == "de"

    def test_earliest_match_wins_across_stops(self) -> None:
        assert run(StopChecker(("zz", "bc"), False), "abczz") == ("a", "bc")

    def test_flush_releases_held_text_once(self) -> None:
        checker = StopChecker(("END",), False)
        checker.feed("abE")

        assert checker.flush() == "bE"
        assert checker.flush() == ""

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
