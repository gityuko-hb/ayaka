"""``HfTokenizer`` logic against a stand-in for a HuggingFace tokenizer.

These tests need no ``transformers``; loading a real checkpoint (``from_config``) is
exercised separately by hand because it needs the package and model files.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from ayaka.tokenizers.adapters.hf import HfTokenizer


class ByteLevel:
    """Class name the adapter looks for when detecting byte-level BPE."""


class Sequence:
    def __init__(self, *steps: object) -> None:
        self._steps = steps

    def __iter__(self) -> Any:
        return iter(self._steps)


class StubRust:
    def __init__(self, pre_tokenizer: object | None) -> None:
        self.pre_tokenizer = pre_tokenizer
        self.calls: list[str] = []

    def no_truncation(self) -> None:
        self.calls.append("no_truncation")

    def no_padding(self) -> None:
        self.calls.append("no_padding")

    def to_str(self) -> str:
        return '{"stub": true}'


class StubHf:
    """Just enough of ``PreTrainedTokenizer`` for the adapter's constructor."""

    name_or_path = "stub/model"
    is_fast = True
    vocab_size = 4
    eos_token_id = 3
    bos_token_id = 2
    pad_token_id = None
    chat_template = "tmpl"
    all_special_ids = (2, 3)
    all_special_tokens = ("<s>", "</s>")

    def __init__(self, rust: StubRust | None = None) -> None:
        self._tokenizer = rust
        self.added_tokens_decoder = {
            2: SimpleNamespace(content="<s>", special=True),
            3: SimpleNamespace(content="</s>", special=True),
            9: SimpleNamespace(content="<|im_start|>", special=True),
            10: SimpleNamespace(content="<plain>", special=False),
        }

    def get_vocab(self) -> dict[str, int]:
        return {"a": 0, "Ġ": 1, "<s>": 2, "</s>": 3, "<|im_start|>": 9, "<plain>": 10}

    def get_added_vocab(self) -> dict[str, int]:
        return {"<s>": 2, "</s>": 3, "<|im_start|>": 9, "<plain>": 10}

    def convert_tokens_to_string(self, tokens: list[str]) -> str:
        return "".join(" " if t == "Ġ" else t for t in tokens)


def adapter(pre_tokenizer: object | None = None, **kwargs: Any) -> HfTokenizer:
    return HfTokenizer(StubHf(StubRust(pre_tokenizer)), **kwargs)


@pytest.mark.unit
class TestMetadata:
    def test_basic_properties_are_snapshotted(self) -> None:
        tok = adapter()

        assert tok.name_or_path == "stub/model"
        assert tok.eos_token_id == 3
        assert tok.vocab_size == 4
        assert tok.max_token_id == 10
        assert tok.valid_token_ids == frozenset({0, 1, 2, 3, 9, 10})
        assert tok.chat_template == "tmpl"
        assert tok.is_fast
        assert not tok.supports_chunked_encode

    def test_special_ids_include_added_tokens_flagged_special(self) -> None:
        tok = adapter()

        assert tok.all_special_ids == frozenset({2, 3, 9})
        assert "<|im_start|>" in tok.all_special_tokens
        assert "<plain>" not in tok.all_special_tokens
        assert tok.all_special_tokens[:2] == ("<s>", "</s>")

    def test_extra_eos_ids_merge_with_the_primary_one(self) -> None:
        assert adapter(eos_token_ids=[1, 5]).eos_token_ids == frozenset({1, 3, 5})
        assert adapter().eos_token_ids == frozenset({3})

    def test_vocab_accessors_return_copies(self) -> None:
        tok = adapter()
        tok.get_vocab()["zzz"] = 99
        tok.get_added_vocab().clear()

        assert "zzz" not in tok.get_vocab()
        assert tok.get_added_vocab()

    def test_heavy_views_are_shared_not_rebuilt(self) -> None:
        tok = adapter(ByteLevel())

        assert tok.valid_token_ids is tok.valid_token_ids
        assert tok.token_bytes_table() is tok.token_bytes_table()

    def test_built_in_truncation_and_padding_are_switched_off(self) -> None:
        rust = StubRust(None)
        HfTokenizer(StubHf(rust))

        assert rust.calls == ["no_truncation", "no_padding"]


@pytest.mark.unit
class TestByteTable:
    def test_byte_level_vocab_is_mapped_back_to_bytes(self) -> None:
        table = adapter(ByteLevel()).token_bytes_table()

        assert len(table) == 11
        assert table[0] == b"a"
        assert table[1] == b" "  # U+0120 stands for the space byte in GPT-2's alphabet
        assert table[2] == b"<s>"  # added tokens are literal
        assert table[5] == b""  # gaps stay empty

    def test_sequence_pre_tokenizer_containing_byte_level_counts(self) -> None:
        assert adapter(Sequence(object(), ByteLevel())).token_bytes_table()

    @pytest.mark.parametrize("pre_tokenizer", [None, object(), Sequence(object())])
    def test_other_vocabularies_have_no_byte_table(self, pre_tokenizer: object | None) -> None:
        tok = adapter(pre_tokenizer)

        assert tok.token_bytes_table() == ()
        assert not tok.verify_byte_path()
        assert not tok.byte_path_verified

    def test_slow_tokenizers_have_no_rust_backend(self) -> None:
        tok = HfTokenizer(StubHf(None))

        assert tok.token_bytes_table() == ()
        assert tok.new_decode_stream() is None


@pytest.mark.unit
class TestFingerprint:
    def test_is_stable_and_hex(self) -> None:
        first, second = adapter(), adapter()

        assert first.fingerprint() == second.fingerprint()
        assert len(first.fingerprint()) == 32

    def test_changes_with_the_chat_template(self) -> None:
        other = StubHf(StubRust(None))
        other.chat_template = "different"

        assert HfTokenizer(other).fingerprint() != adapter().fingerprint()

    def test_changes_with_the_special_tokens(self) -> None:
        other = StubHf(StubRust(None))
        del other.added_tokens_decoder[9]

        assert HfTokenizer(other).fingerprint() != adapter().fingerprint()

    def test_slow_tokenizer_hashes_the_vocabulary(self) -> None:
        slow, other = StubHf(None), StubHf(None)
        other.get_vocab = lambda: {"a": 0, "b": 1}  # type: ignore[method-assign]
        other.get_added_vocab = lambda: {}  # type: ignore[method-assign]

        assert HfTokenizer(slow).fingerprint() != HfTokenizer(other).fingerprint()


@pytest.mark.unit
class TestFromConfig:
    def test_token_id_only_mode_is_not_loaded_here(self) -> None:
        from ayaka.configs.tokenizer import TokenizerConfig

        with pytest.raises(ValueError, match="DefaultTokenizerFactory"):
            HfTokenizer.from_config(TokenizerConfig("x", skip_tokenizer_init=True))

    @pytest.mark.parametrize("mode", ["tiktoken", "mistral"])
    def test_other_modes_belong_to_other_adapters(self, mode: Any) -> None:
        from ayaka.configs.tokenizer import TokenizerConfig

        with pytest.raises(ValueError, match="unsupported tokenizer mode"):
            HfTokenizer.from_config(TokenizerConfig("x", mode=mode))
