"""A missing optional package must fail loudly, name the capability and say how to fix it.

``sys.modules[name] = None`` makes ``import name`` raise, whether or not the package is
installed, so these tests behave the same in every environment.
"""

from __future__ import annotations

import base64
import sys
from typing import TYPE_CHECKING

import pytest

from ayaka.configs.tokenizer import TokenizerConfig
from ayaka.tokenizers.adapters.hf import HfTokenizer
from ayaka.tokenizers.adapters.tiktoken import TiktokenTokenizer
from ayaka.tokenizers.factory import DefaultTokenizerFactory
from ayaka.utils.hf_utils import optional_file
from ayaka.utils.import_utils import CapabilityError

if TYPE_CHECKING:
    from pathlib import Path


def block(monkeypatch: pytest.MonkeyPatch, *names: str) -> None:
    for name in names:
        monkeypatch.setitem(sys.modules, name, None)


def vocab_file(tmp_path: Path) -> Path:
    path = tmp_path / "tiktoken.model"
    lines = [f"{base64.b64encode(bytes([i])).decode()} {i}" for i in range(256)]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


@pytest.mark.unit
class TestMissingPackages:
    def test_hf_adapter_names_the_capability_and_the_fix(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        block(monkeypatch, "transformers")

        with pytest.raises(CapabilityError) as info:
            HfTokenizer.from_config(TokenizerConfig("org/model", mode="hf"))

        assert info.value.capability == "tokenizer.hf"
        assert info.value.remedy == "pip install transformers"
        assert "transformers" in str(info.value)

    def test_the_factory_does_not_hide_the_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        block(monkeypatch, "transformers")

        with pytest.raises(CapabilityError, match="transformers"):
            DefaultTokenizerFactory.from_config(TokenizerConfig("org/model", mode="hf"))

    def test_tiktoken_adapter_names_the_capability_and_the_fix(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        block(monkeypatch, "tiktoken")

        with pytest.raises(CapabilityError) as info:
            TiktokenTokenizer(mergeable_ranks={b"a": 0}, special_tokens={})

        assert info.value.capability == "tokenizer.tiktoken"
        assert info.value.remedy == "pip install tiktoken"

    def test_a_valid_vocabulary_file_still_reports_the_missing_package(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        block(monkeypatch, "tiktoken")
        config = TokenizerConfig(str(vocab_file(tmp_path)), mode="tiktoken")

        with pytest.raises(CapabilityError, match="tiktoken"):
            DefaultTokenizerFactory.from_config(config)

    def test_hub_lookup_needs_huggingface_hub(self, monkeypatch: pytest.MonkeyPatch) -> None:
        block(monkeypatch, "huggingface_hub")

        with pytest.raises(CapabilityError) as info:
            optional_file(
                "org/model", "tokenizer.json", revision=None, cache_dir=None, offline=True
            )

        assert info.value.remedy == "pip install huggingface_hub"

    def test_a_local_directory_needs_no_hub_package(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        block(monkeypatch, "huggingface_hub")

        found = optional_file(
            str(tmp_path), "tokenizer.json", revision=None, cache_dir=None, offline=True
        )

        assert found is None
