from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import pytest

from ayaka.utils.hf_utils import optional_file, parse_eos_token_ids, read_json

if TYPE_CHECKING:
    from pathlib import Path


@pytest.mark.unit
class TestOptionalFile:
    def test_finds_a_file_in_a_local_directory(self, tmp_path: Path) -> None:
        (tmp_path / "tokenizer.json").write_text("{}", encoding="utf-8")

        found = optional_file(
            str(tmp_path), "tokenizer.json", revision=None, cache_dir=None, offline=True
        )

        assert found == tmp_path / "tokenizer.json"

    def test_missing_file_is_none_not_an_error(self, tmp_path: Path) -> None:
        found = optional_file(
            str(tmp_path), "chat_template.json", revision=None, cache_dir=None, offline=True
        )

        assert found is None

    def test_a_directory_with_that_name_is_not_a_file(self, tmp_path: Path) -> None:
        (tmp_path / "tokenizer.json").mkdir()

        assert (
            optional_file(
                str(tmp_path), "tokenizer.json", revision=None, cache_dir=None, offline=True
            )
            is None
        )


@pytest.mark.unit
class TestReadJson:
    def test_reads_valid_json(self, tmp_path: Path) -> None:
        path = tmp_path / "c.json"
        path.write_text('{"a": 1}', encoding="utf-8")

        assert read_json(path) == {"a": 1}

    def test_none_path_gives_none(self) -> None:
        assert read_json(None) is None

    def test_invalid_json_warns_and_gives_none(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        path = tmp_path / "bad.json"
        path.write_text("{not json", encoding="utf-8")

        with caplog.at_level(logging.WARNING, logger="ayaka.utils.hf_utils"):
            assert read_json(path) is None

        assert "unreadable" in caplog.text

    def test_missing_file_warns_and_gives_none(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level(logging.WARNING, logger="ayaka.utils.hf_utils"):
            assert read_json(tmp_path / "absent.json") is None

        assert caplog.records


@pytest.mark.unit
class TestParseEosTokenIds:
    @pytest.mark.parametrize(
        ("config", "expected"),
        [
            ({"eos_token_id": 2}, {2}),
            ({"eos_token_id": [2, 151645]}, {2, 151645}),
            ({"eos_token_id": [2, True, "x", 3.5, None]}, {2}),
            ({"eos_token_id": True}, set()),
            ({"eos_token_id": "2"}, set()),
            ({}, set()),
            (None, set()),
        ],
    )
    def test_extracts_only_integer_ids(
        self, config: dict[str, Any] | None, expected: set[int]
    ) -> None:
        assert parse_eos_token_ids(config) == frozenset(expected)
