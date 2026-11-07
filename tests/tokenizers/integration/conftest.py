"""Fixtures for tests that load real HuggingFace and tiktoken tokenizers.

These need the ``hf`` and ``tiktoken`` extras and run in the CI job ``adapters``. They
build their own tiny tokenizers (see ``tests/_support/real_tokenizers.py``) and must
never reach the network, which an autouse fixture enforces.
"""

from __future__ import annotations

import socket
from importlib import import_module
from typing import TYPE_CHECKING, Any

import pytest

from ayaka.tokenizers.factory import DefaultTokenizerFactory
from real_tokenizers import build_hf_dir, build_tiktoken_dir, hf_config, tiktoken_config

if TYPE_CHECKING:
    from pathlib import Path

    from ayaka.tokenizers.contracts import LoadedTokenizer

LOCAL_HOSTS = {"127.0.0.1", "::1", "localhost", ""}


@pytest.fixture(autouse=True)
def network_attempts(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Fail on any lookup or connection to a non-local host and record the attempt.

    Loopback and Unix sockets stay allowed because asyncio and multiprocessing use them.
    The list lets a test assert that nothing even tried, in case a library swallows the
    error this raises.
    """
    attempts: list[str] = []
    real_connect = socket.socket.connect
    real_getaddrinfo = socket.getaddrinfo

    def connect(self: socket.socket, address: Any) -> None:
        if isinstance(address, tuple) and str(address[0]) not in LOCAL_HOSTS:  # pyright: ignore[reportUnknownArgumentType]
            attempts.append(f"connect {address[0]}")  # pyright: ignore[reportUnknownMemberType]
            raise AssertionError(f"network connection attempted: {address[0]}")  # pyright: ignore[reportUnknownMemberType]
        real_connect(self, address)

    def getaddrinfo(host: Any, *args: Any, **kwargs: Any) -> Any:
        if host is not None and str(host) not in LOCAL_HOSTS:
            attempts.append(f"resolve {host}")
            raise AssertionError(f"network lookup attempted: {host}")
        return real_getaddrinfo(host, *args, **kwargs)

    monkeypatch.setattr(socket.socket, "connect", connect)
    monkeypatch.setattr(socket, "getaddrinfo", getaddrinfo)
    return attempts


@pytest.fixture
def cold_hub_cache(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Make huggingface_hub behave as if it had never run on this machine.

    It caches an "agent registry" fetched from the Hub for a day on disk and for the life
    of the process. On a developer machine that cache is usually warm, so an offline load
    that still tried to reach the Hub passed unnoticed there and failed only on fresh CI
    runners. Tests about network use must start from a cold cache to mean anything.
    """
    constants = import_module("huggingface_hub.constants")
    detect_agent = import_module("huggingface_hub.utils._detect_agent")
    monkeypatch.setattr(
        constants, "AGENT_HARNESSES_PATH", str(tmp_path / "agents.json"), raising=False
    )
    monkeypatch.setattr(detect_agent, "_registry", None, raising=False)


@pytest.fixture(scope="session")
def hf_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return build_hf_dir(tmp_path_factory.mktemp("hf") / "model")


@pytest.fixture(scope="session")
def hf_dir_template_file(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """No template in the tokenizer config; it only exists in ``chat_template.json``."""
    root = tmp_path_factory.mktemp("hf_json") / "model"
    return build_hf_dir(root, chat_template=None, template_json=True)


@pytest.fixture(scope="session")
def hf_dir_bare(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """No template anywhere, so only the ChatML guess can render chat."""
    return build_hf_dir(tmp_path_factory.mktemp("hf_bare") / "model", chat_template=None)


@pytest.fixture(scope="session")
def tiktoken_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return build_tiktoken_dir(tmp_path_factory.mktemp("tiktoken"))


@pytest.fixture(scope="session")
def hf_loaded(hf_dir: Path) -> LoadedTokenizer:
    return DefaultTokenizerFactory.from_config(hf_config(hf_dir))


@pytest.fixture(scope="session")
def tiktoken_loaded(tiktoken_dir: Path) -> LoadedTokenizer:
    return DefaultTokenizerFactory.from_config(tiktoken_config(tiktoken_dir))
