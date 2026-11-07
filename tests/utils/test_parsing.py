from __future__ import annotations

import pytest

from ayaka.utils.parsing import format_human_int, parse_bool, parse_human_int

pytestmark = pytest.mark.unit


class TestParseBool:
    @pytest.mark.parametrize("text", ["1", "true", "TRUE", " yes ", "y", "on"])
    def test_accepts_true_spellings(self, text: str) -> None:
        assert parse_bool(text)

    @pytest.mark.parametrize("text", ["0", "false", "FALSE", " no ", "n", "off"])
    def test_accepts_false_spellings(self, text: str) -> None:
        assert not parse_bool(text)

    def test_error_names_the_value_and_the_options(self) -> None:
        with pytest.raises(ValueError, match="cannot parse 'maybe' as a boolean"):
            parse_bool("maybe", what="flag")


class TestParseHumanInt:
    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("0", 0),
            ("4096", 4096),
            ("1k", 1000),
            ("1M", 10**6),
            ("1.5M", 1_500_000),
            ("25.6k", 25_600),
            ("1Ki", 2**10),
            ("1Mi", 2**20),
            ("1Gi", 2**30),
            (" 2k ", 2000),
        ],
    )
    def test_parses_plain_si_and_iec_forms(self, text: str, expected: int) -> None:
        assert parse_human_int(text) == expected

    @pytest.mark.parametrize(
        "text",
        ["", "   ", "1.0", "1.5Gi", "1KB", "-1", "1e3", "one", "1.2.3k"],
    )
    def test_rejects_ambiguous_or_invalid_forms(self, text: str) -> None:
        with pytest.raises(
            ValueError,
            match=r"cannot parse|empty string|not an integer|decimals are not allowed",
        ):
            parse_human_int(text)


class TestFormatHumanInt:
    @pytest.mark.parametrize(
        ("value", "binary", "expected"),
        [
            (0, True, "0"),
            (1000, True, "1000"),
            (1000, False, "1k"),
            (1024, True, "1Ki"),
            (2**20, True, "1Mi"),
            (2_000_000, False, "2M"),
            (25_600, False, "25600"),
            (-1024, True, "-1Ki"),
        ],
    )
    def test_formats_only_exact_multiples(self, value: int, binary: bool, expected: str) -> None:
        assert format_human_int(value, binary=binary) == expected

    @pytest.mark.parametrize("value", [0, 1, 1000, 1024, 2_000_000, 2**20])
    def test_round_trips_through_parser(self, value: int) -> None:
        assert parse_human_int(format_human_int(value, binary=False)) == value
        assert parse_human_int(format_human_int(value, binary=True)) == value
