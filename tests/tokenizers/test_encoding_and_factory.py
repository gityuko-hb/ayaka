from __future__ import annotations

import base64
import json
from collections.abc import Callable
from typing import TYPE_CHECKING

import pytest

from ayaka.configs.tokenizer import TokenizerConfig
from ayaka.core.errors import ConfigError
from ayaka.tokenizers.adapters.tiktoken import TiktokenTokenizer, load_tiktoken_vocab
from ayaka.tokenizers.contracts import ChatInput, ChatMessage, EncodeOptions
from ayaka.tokenizers.encoding.chunked import encode_chunked, safe_split_points, split_into_chunks
from ayaka.tokenizers.encoding.media import encode_media_chat
from ayaka.tokenizers.errors import TokenizerOverloaded
from ayaka.tokenizers.factory import DefaultTokenizerFactory, validate_loading_config
from ayaka.tokenizers.service import TokenizerService
from tokenizer_fakes import IM_END, IM_START, FakeTokenizer, RenderedFake

if TYPE_CHECKING:
    from pathlib import Path

Make = Callable[..., TokenizerService]
OPTIONS = EncodeOptions(max_output_tokens=8)


def chat(content: str) -> ChatInput:
    return ChatInput((ChatMessage("user", content),))


@pytest.mark.unit
class TestChunking:
    TEXT = "".join(f"line {i} of the prompt\n" for i in range(40))

    @pytest.mark.parametrize("target", [1, 30, 100, 500])
    def test_chunks_always_rejoin_to_the_original(self, target: int) -> None:
        assert "".join(split_into_chunks(self.TEXT, target)) == self.TEXT

    def test_cuts_follow_a_newline_before_non_space(self) -> None:
        for cut in safe_split_points(self.TEXT, 50):
            assert self.TEXT[cut - 1] == "\n"
            assert not self.TEXT[cut].isspace()

    def test_short_text_and_bad_target_give_no_cuts(self) -> None:
        assert safe_split_points("short", 100) == []
        assert safe_split_points(self.TEXT, 0) == []

    def test_no_newline_means_no_cut(self) -> None:
        assert safe_split_points("x" * 1000, 10) == []

    def test_whitespace_after_newline_blocks_the_cut(self) -> None:
        # The newline at 10 is followed by spaces, so the first usable cut is after the one at 21.
        assert safe_split_points("a" * 10 + "\n  indented\n" + "b" * 10, 5) == [22]

    def test_chunks_are_encoded_independently_when_the_adapter_allows_it(self) -> None:
        tokenizer = FakeTokenizer(chunked=True)

        ids, chunks = encode_chunked(
            tokenizer, self.TEXT, chunk_chars=100, add_special_tokens=False
        )

        assert chunks > 1
        assert ids == tokenizer.encode(self.TEXT, add_special_tokens=False)

    @pytest.mark.parametrize(
        ("chunked", "special"), [(False, False), (True, True)], ids=["unsupported", "special-on"]
    )
    def test_whole_prompt_path_when_chunking_is_not_safe(
        self, chunked: bool, special: bool
    ) -> None:
        _, chunks = encode_chunked(
            FakeTokenizer(chunked=chunked), self.TEXT, chunk_chars=100, add_special_tokens=special
        )

        assert chunks == 1


@pytest.mark.unit
class TestMediaChat:
    def test_markers_become_placeholder_runs(self, make_service: Make) -> None:
        service = make_service(RenderedFake())

        result, starts = encode_media_chat(
            service, chat("look [IMG] now"), OPTIONS, (("[IMG]", 3),), placeholder=0
        )

        ids = result.token_ids
        assert ids[0] == IM_START
        assert IM_END in ids
        assert ids[starts[0] : starts[0] + 3] == (0, 0, 0)
        assert len(starts) == 1

    def test_text_around_markers_is_encoded_without_straddling(self, make_service: Make) -> None:
        service = make_service(RenderedFake())

        result, (start,) = encode_media_chat(
            service, chat("A[M]B"), OPTIONS, (("[M]", 2),), placeholder=1
        )

        assert result.token_ids[start - 1] == ord("A")
        assert result.token_ids[start + 2] == ord("B")

    @pytest.mark.parametrize(
        "content", ["no marker here", "twice [M] and [M]"], ids=["missing", "duplicated"]
    )
    def test_template_must_keep_each_marker_exactly_once(
        self, make_service: Make, content: str
    ) -> None:
        with pytest.raises(ValueError, match="exactly once"):
            encode_media_chat(
                make_service(RenderedFake()), chat(content), OPTIONS, (("[M]", 1),), placeholder=0
            )

    def test_reordered_markers_are_rejected(self, make_service: Make) -> None:
        with pytest.raises(ValueError, match="reordered"):
            encode_media_chat(
                make_service(RenderedFake()),
                chat("[A] then [B]"),
                OPTIONS,
                (("[B]", 1), ("[A]", 1)),
                placeholder=0,
            )

    def test_truncation_is_refused(self, make_service: Make) -> None:
        with pytest.raises(ValueError, match="truncate"):
            encode_media_chat(
                make_service(RenderedFake()),
                chat("[M]"),
                EncodeOptions(max_output_tokens=8, truncate=True),
                (("[M]", 1),),
                placeholder=0,
            )

    @pytest.mark.parametrize("placeholder", [-1, 300])
    def test_placeholder_must_be_a_valid_id(self, make_service: Make, placeholder: int) -> None:
        with pytest.raises(ValueError, match="placeholder"):
            encode_media_chat(
                make_service(RenderedFake()), chat("[M]"), OPTIONS, (("[M]", 1),), placeholder
            )

    def test_oversized_chat_is_rejected_before_rendering(self, make_service: Make) -> None:
        service = make_service(RenderedFake(), encode_max_pending_bytes=8)

        with pytest.raises(TokenizerOverloaded):
            encode_media_chat(service, chat("x" * 50), OPTIONS, (), placeholder=0)


@pytest.mark.unit
class TestLoadingConfig:
    @pytest.mark.parametrize(
        ("kwargs", "path"),
        [
            ({"mode": "mistral"}, "tokenizer.mode"),
            ({"mode": "slow", "detokenize_backend": "stream"}, "tokenizer.detokenize_backend"),
            ({"mode": "tiktoken", "detokenize_backend": "stream"}, "tokenizer.detokenize_backend"),
            ({"mode": "slow", "detokenize_backend": "bytes"}, "tokenizer.detokenize_backend"),
        ],
    )
    def test_impossible_combinations_are_rejected_early(
        self, kwargs: dict[str, str], path: str
    ) -> None:
        with pytest.raises(ConfigError) as info:
            validate_loading_config(TokenizerConfig("t", **kwargs))  # type: ignore[arg-type]

        assert info.value.code == "UNSUPPORTED_BACKEND"
        assert info.value.path == path

    @pytest.mark.parametrize(
        "kwargs",
        [
            {},
            {"mode": "hf", "detokenize_backend": "stream"},
            {"mode": "hf", "detokenize_backend": "bytes"},
            {"mode": "tiktoken", "detokenize_backend": "bytes"},
            {"mode": "slow", "detokenize_backend": "window"},
        ],
    )
    def test_possible_combinations_pass(self, kwargs: dict[str, str]) -> None:
        validate_loading_config(TokenizerConfig("t", **kwargs))  # type: ignore[arg-type]

    def test_token_id_only_mode_skips_every_check(self) -> None:
        validate_loading_config(TokenizerConfig("t", mode="mistral", skip_tokenizer_init=True))

    def test_token_id_only_mode_loads_nothing(self) -> None:
        loaded = DefaultTokenizerFactory.from_config(TokenizerConfig("t", skip_tokenizer_init=True))

        assert loaded.tokenizer is None
        assert loaded.capabilities.backend == "disabled"

    def test_unsupported_mode_fails_before_loading_anything(self) -> None:
        with pytest.raises(ConfigError, match="mistral"):
            DefaultTokenizerFactory.from_config(TokenizerConfig("t", mode="mistral"))


def b64(raw: bytes) -> str:
    return base64.b64encode(raw).decode()


@pytest.mark.unit
class TestTiktokenVocabFile:
    def test_line_format(self, tmp_path: Path) -> None:
        path = tmp_path / "v.tiktoken"
        path.write_text(f"{b64(b'a')} 0\n\n{b64(b'ab')} 1\n", encoding="utf-8")

        ranks, special, pat, n_vocab = load_tiktoken_vocab(path)

        assert ranks == {b"a": 0, b"ab": 1}
        assert (special, pat, n_vocab) == ({}, None, None)

    def test_bom_is_ignored(self, tmp_path: Path) -> None:
        path = tmp_path / "v.tiktoken"
        path.write_bytes(b"\xef\xbb\xbf" + f"{b64(b'a')} 0\n".encode())

        assert load_tiktoken_vocab(path)[0] == {b"a": 0}

    def test_flat_json_mapping(self, tmp_path: Path) -> None:
        path = tmp_path / "v.json"
        path.write_text(json.dumps({b64(b"a"): 0, b64(b"b"): 1}), encoding="utf-8")

        assert load_tiktoken_vocab(path)[0] == {b"a": 0, b"b": 1}

    def test_xtok_json_carries_specials_pattern_and_size(self, tmp_path: Path) -> None:
        path = tmp_path / "v.json"
        document = {
            "regular_tokens": [{"bytes": list(b"a"), "token": 0}],
            "special_tokens": [{"bytes": list(b"<|end|>"), "token": 5}],
            "pat_str": r"\w+",
            "vocab_size": 6,
        }
        path.write_text(json.dumps(document), encoding="utf-8")

        ranks, special, pat, n_vocab = load_tiktoken_vocab(path)

        assert ranks == {b"a": 0}
        assert special == {"<|end|>": 5}
        assert (pat, n_vocab) == (r"\w+", 6)

    def test_malformed_line_names_the_line(self, tmp_path: Path) -> None:
        path = tmp_path / "v.tiktoken"
        path.write_text(f"{b64(b'a')} 0\nbroken\n", encoding="utf-8")

        with pytest.raises(ValueError, match=r":2: expected"):
            load_tiktoken_vocab(path)

    def test_empty_file_is_rejected(self, tmp_path: Path) -> None:
        path = tmp_path / "v.tiktoken"
        path.write_text("\n\n", encoding="utf-8")

        with pytest.raises(ValueError, match="empty"):
            load_tiktoken_vocab(path)

    def test_binary_sentencepiece_file_gets_a_helpful_error(self, tmp_path: Path) -> None:
        path = tmp_path / "tokenizer.model"
        path.write_bytes(b"\x0a\xff\xfe\x80binary")

        with pytest.raises(ValueError, match="SentencePiece"):
            load_tiktoken_vocab(path)

    def test_hub_ids_are_refused_with_a_clear_message(self) -> None:
        with pytest.raises(ValueError, match="local files only"):
            TiktokenTokenizer.from_config(TokenizerConfig("org/model", mode="tiktoken"))

    def test_directory_without_a_vocabulary_is_reported(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="no tiktoken vocabulary"):
            TiktokenTokenizer.from_config(TokenizerConfig(str(tmp_path), mode="tiktoken"))
