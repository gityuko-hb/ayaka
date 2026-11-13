from __future__ import annotations

import logging
import sys
import threading
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from ayaka.core.errors import CapabilityError
from ayaka.utils.hf_utils import hub_offline, optional_file, parse_eos_token_ids, read_json


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


class FakeHub:
    """Stands in for ``huggingface_hub`` so these tests need no package.

    ``flag_seen`` records ``constants.HF_HUB_OFFLINE`` at each download, which is what the
    real library consults while it builds its headers.
    """

    def __init__(self, monkeypatch: pytest.MonkeyPatch, *, offline: bool = False) -> None:
        self.constants = SimpleNamespace(HF_HUB_OFFLINE=offline)
        self.flag_seen: list[bool] = []
        self.result: Any = "cached/file.json"

        class EntryNotFoundError(Exception):
            pass

        class LocalEntryNotFoundError(EntryNotFoundError):
            pass

        self.errors = SimpleNamespace(
            EntryNotFoundError=EntryNotFoundError, LocalEntryNotFoundError=LocalEntryNotFoundError
        )
        hub = SimpleNamespace(hf_hub_download=self._download)
        monkeypatch.setitem(sys.modules, "huggingface_hub", hub)
        monkeypatch.setitem(sys.modules, "huggingface_hub.constants", self.constants)
        monkeypatch.setitem(sys.modules, "huggingface_hub.errors", self.errors)

    def _download(self, source: str, filename: str, **kwargs: Any) -> str:
        del source, filename, kwargs
        self.flag_seen.append(self.constants.HF_HUB_OFFLINE)
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


@pytest.fixture
def hub(monkeypatch: pytest.MonkeyPatch) -> FakeHub:
    return FakeHub(monkeypatch)


@pytest.mark.unit
class TestHubOffline:
    def test_flips_the_hub_flag_for_the_block_and_restores_it(self, hub: FakeHub) -> None:
        with hub_offline():
            assert hub.constants.HF_HUB_OFFLINE is True

        assert hub.constants.HF_HUB_OFFLINE is False

    def test_restores_the_flag_when_the_block_raises(self, hub: FakeHub) -> None:
        with pytest.raises(RuntimeError, match="boom"), hub_offline():
            raise RuntimeError("boom")

        assert hub.constants.HF_HUB_OFFLINE is False

    def test_an_already_offline_hub_stays_offline(self, monkeypatch: pytest.MonkeyPatch) -> None:
        hub = FakeHub(monkeypatch, offline=True)

        with hub_offline():
            pass

        assert hub.constants.HF_HUB_OFFLINE is True

    def test_nested_blocks_restore_only_when_the_outermost_ends(self, hub: FakeHub) -> None:
        with hub_offline():
            with hub_offline():
                pass
            assert hub.constants.HF_HUB_OFFLINE is True

        assert hub.constants.HF_HUB_OFFLINE is False

    def test_overlapping_blocks_in_two_threads_restore_once(self, hub: FakeHub) -> None:
        a_in, b_in, a_may_leave = threading.Event(), threading.Event(), threading.Event()
        b_may_leave, a_out = threading.Event(), threading.Event()

        def a() -> None:
            with hub_offline():
                a_in.set()
                b_in.wait(5)
                a_may_leave.wait(5)
            a_out.set()

        def b() -> None:
            a_in.wait(5)
            with hub_offline():
                b_in.set()
                b_may_leave.wait(5)

        threads = [threading.Thread(target=a), threading.Thread(target=b)]
        for thread in threads:
            thread.start()
        assert b_in.wait(5)
        a_may_leave.set()
        assert a_out.wait(5)
        still_offline = hub.constants.HF_HUB_OFFLINE  # B is still inside its block
        b_may_leave.set()
        for thread in threads:
            thread.join(5)

        assert still_offline is True
        assert hub.constants.HF_HUB_OFFLINE is False

    def test_disabled_runs_unchanged_and_never_imports_the_hub(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setitem(sys.modules, "huggingface_hub.constants", None)

        with hub_offline(False):
            ran = True

        assert ran

    def test_enabled_without_the_package_names_the_fix(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setitem(sys.modules, "huggingface_hub.constants", None)

        with pytest.raises(CapabilityError) as info, hub_offline():
            pass

        assert info.value.remedy == "pip install huggingface_hub"


@pytest.mark.unit
class TestOptionalFileOnTheHub:
    def lookup(self, *, offline: bool) -> Path | None:
        return optional_file(
            "org/model", "tokenizer.json", revision=None, cache_dir=None, offline=offline
        )

    def test_an_offline_lookup_runs_with_the_hub_switched_offline(self, hub: FakeHub) -> None:
        found = self.lookup(offline=True)

        assert found == Path("cached/file.json")
        assert hub.flag_seen == [True]
        assert hub.constants.HF_HUB_OFFLINE is False  # and it is put back afterwards

    def test_an_online_lookup_leaves_the_hub_alone(self, hub: FakeHub) -> None:
        self.lookup(offline=False)

        assert hub.flag_seen == [False]

    @pytest.mark.parametrize("error", ["EntryNotFoundError", "LocalEntryNotFoundError"])
    def test_a_missing_file_is_none(self, hub: FakeHub, error: str) -> None:
        hub.result = getattr(hub.errors, error)("missing")

        assert self.lookup(offline=True) is None
