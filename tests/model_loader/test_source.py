"""Resolving a model name to a directory: a local path first, the Hub cache or network after.

``huggingface_hub`` is replaced by a stand-in module, so these tests need neither the ``hf``
extra nor the network.
"""

from __future__ import annotations

import sys
from types import ModuleType
from typing import TYPE_CHECKING, Any

import pytest

from ayaka.configs.model_source import ModelSourceConfig
from ayaka.core.errors import CapabilityError, ModelLoadError
from ayaka.model_loader.source import resolve_source

if TYPE_CHECKING:
    from pathlib import Path

pytestmark = pytest.mark.unit


class FakeHub:
    """A ``huggingface_hub`` that records its calls and the offline flag seen during them."""

    def __init__(self, snapshot: Path, error: BaseException | None = None) -> None:
        self.snapshot = snapshot
        self.error = error
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.offline_during_call: list[bool] = []
        self.constants = ModuleType("huggingface_hub.constants")
        self.constants.HF_HUB_OFFLINE = False  # type: ignore[attr-defined]  # a stand-in module

    def snapshot_download(self, repo_id: str, **kwargs: Any) -> str:
        self.calls.append((repo_id, kwargs))
        self.offline_during_call.append(bool(self.constants.HF_HUB_OFFLINE))  # type: ignore[attr-defined]
        if self.error is not None:
            raise self.error
        return str(self.snapshot)


@pytest.fixture
def hub(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> FakeHub:
    snapshot = tmp_path / "snapshots" / "abc123"
    snapshot.mkdir(parents=True)
    fake = FakeHub(snapshot)
    module = ModuleType("huggingface_hub")
    module.snapshot_download = fake.snapshot_download  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "huggingface_hub", module)
    monkeypatch.setitem(sys.modules, "huggingface_hub.constants", fake.constants)
    return fake


def test_a_local_directory_is_returned_as_it_is(tmp_path: Path) -> None:
    assert resolve_source(ModelSourceConfig(str(tmp_path))) == tmp_path


def test_a_local_directory_never_touches_the_hub(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setitem(sys.modules, "huggingface_hub", None)  # importing it would fail

    assert resolve_source(ModelSourceConfig(str(tmp_path), offline=True)) == tmp_path


def test_a_directory_named_like_a_repo_id_wins_over_the_hub(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, hub: FakeHub
) -> None:
    (tmp_path / "org" / "name").mkdir(parents=True)
    monkeypatch.chdir(tmp_path)

    resolved = resolve_source(ModelSourceConfig("org/name"))

    assert resolved.resolve() == (tmp_path / "org" / "name").resolve()
    assert hub.calls == []


def test_a_repo_id_is_downloaded_with_only_the_files_the_loader_reads(hub: FakeHub) -> None:
    resolved = resolve_source(ModelSourceConfig("org/name"))

    assert resolved == hub.snapshot
    [(repo, kwargs)] = hub.calls
    assert repo == "org/name"
    assert kwargs == {
        "revision": None,
        "cache_dir": None,
        "local_files_only": False,
        "allow_patterns": ["config.json", "*.safetensors", "model.safetensors.index.json"],
    }


def test_revision_and_cache_directory_are_passed_on(hub: FakeHub) -> None:
    resolve_source(ModelSourceConfig("org/name", revision="v1.0", cache_dir="/cache"))

    [(_, kwargs)] = hub.calls
    assert kwargs["revision"] == "v1.0"
    assert kwargs["cache_dir"] == "/cache"


def test_offline_uses_the_cache_only_and_the_hub_is_offline_during_the_call(hub: FakeHub) -> None:
    resolve_source(ModelSourceConfig("org/name", offline=True))

    [(_, kwargs)] = hub.calls
    assert kwargs["local_files_only"] is True
    assert hub.offline_during_call == [True]
    assert hub.constants.HF_HUB_OFFLINE is False  # type: ignore[attr-defined]  # restored


def test_online_leaves_the_hub_switch_alone(hub: FakeHub) -> None:
    resolve_source(ModelSourceConfig("org/name"))

    assert hub.offline_during_call == [False]


@pytest.mark.parametrize(
    ("error", "offline", "fragment"),
    [
        (OSError("404 repository not found"), False, "from the Hugging Face Hub: 404"),
        (FileNotFoundError("not in the cache"), True, r"the local Hub cache \(offline\): not in"),
        (ValueError("repo id is malformed"), False, "malformed"),
    ],
)
def test_a_hub_failure_is_a_model_load_error_that_says_where_it_looked(
    hub: FakeHub, error: BaseException, offline: bool, fragment: str
) -> None:
    hub.error = error

    with pytest.raises(ModelLoadError, match=fragment) as raised:
        resolve_source(ModelSourceConfig("org/name", offline=offline, revision="main"))

    assert raised.value.context == {"model": "org/name", "revision": "main", "offline": offline}
    assert raised.value.__cause__ is error


def test_a_hub_failure_that_is_not_an_os_or_value_error_is_not_swallowed(hub: FakeHub) -> None:
    hub.error = KeyboardInterrupt()

    with pytest.raises(KeyboardInterrupt):
        resolve_source(ModelSourceConfig("org/name"))


@pytest.mark.parametrize("name", ["no-such-directory", "./missing/dir", "a/b/c", "/no/such/place"])
def test_something_that_is_neither_a_directory_nor_a_repo_id_is_refused(
    hub: FakeHub, name: str
) -> None:
    with pytest.raises(ModelLoadError, match="not a directory, and it is not a Hugging"):
        resolve_source(ModelSourceConfig(name))

    assert hub.calls == []


def test_a_repo_id_without_the_hf_extra_names_the_extra(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "huggingface_hub", None)

    with pytest.raises(CapabilityError, match=r"uv sync --extra hf"):
        resolve_source(ModelSourceConfig("org/name"))
