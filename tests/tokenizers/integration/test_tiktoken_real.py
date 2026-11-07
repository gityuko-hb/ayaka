"""The tiktoken adapter against a real (tiny) vocabulary file.

Needs the ``tiktoken`` extra, which also provides ``jinja2``. Third-party modules are
only reached through ``import_module`` so that collection works without the extra.
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING

import pytest

from ayaka.configs.tokenizer import DetokenizeParams
from ayaka.tokenizers.adapters.tiktoken import TiktokenTokenizer
from ayaka.tokenizers.contracts import ChatInput, ChatMessage, EncodeOptions, RenderedInput
from ayaka.tokenizers.decoding.byte import ByteIncrementalDetokenizer
from ayaka.tokenizers.decoding.incremental import IncrementalDetokenizer
from ayaka.tokenizers.errors import SpecialTokenInContent
from ayaka.tokenizers.service import TokenizerService
from real_tokenizers import (
    CHAT_TEMPLATE,
    FORGED,
    TIKTOKEN_IM_END,
    TIKTOKEN_IM_START,
    tiktoken_config,
)

if TYPE_CHECKING:
    from pathlib import Path

    from ayaka.configs.tokenizer import TokenizerConfig
    from ayaka.tokenizers.contracts import LoadedTokenizer

pytestmark = pytest.mark.integration

OPTIONS = EncodeOptions(max_output_tokens=16)


def service(loaded: LoadedTokenizer, config: TokenizerConfig) -> TokenizerService:
    return TokenizerService(config, max_model_len=512, model_vocab_size=1000, loaded=loaded)


def chat(content: str) -> ChatInput:
    return ChatInput((ChatMessage("user", content),), chat_template=CHAT_TEMPLATE)


def tk(loaded: LoadedTokenizer) -> TiktokenTokenizer:
    assert isinstance(loaded.tokenizer, TiktokenTokenizer)
    return loaded.tokenizer


@pytest.mark.integration
class TestLoading:
    def test_capabilities_describe_what_was_loaded(self, tiktoken_loaded: LoadedTokenizer) -> None:
        caps = tiktoken_loaded.capabilities

        assert caps.backend == "tiktoken"
        assert caps.byte_decode
        assert not caps.decode_stream
        assert caps.chunked_encode

    def test_a_special_token_with_id_zero_is_still_special(
        self, tiktoken_loaded: LoadedTokenizer
    ) -> None:
        tok = tk(tiktoken_loaded)

        assert 0 in tok.all_special_ids
        assert tok.eos_token_id == TIKTOKEN_IM_END
        assert tok.bos_token_id == TIKTOKEN_IM_START

    def test_an_end_token_with_id_zero_is_still_the_eos(self) -> None:
        # `special.get(name) or ...` treats id 0 as "missing"; the guess must not.
        tok = TiktokenTokenizer(
            mergeable_ranks={bytes([i]): i + 1 for i in range(256)},
            special_tokens={"<|endoftext|>": 0},
        )

        assert tok.eos_token_id == 0
        assert tok.eos_token_ids == {0}
        assert tok.pad_token_id == 0  # no pad token exists, so it falls back to the eos

    def test_a_custom_split_pattern_is_not_claimed_chunk_safe(self) -> None:
        tok = TiktokenTokenizer(
            mergeable_ranks={bytes([i]): i for i in range(256)},
            special_tokens={},
            pat_str=r"\w+|\s+|[^\w\s]+",
        )

        assert not tok.supports_chunked_encode

    def test_the_default_split_pattern_is_claimed_chunk_safe(
        self, tiktoken_loaded: LoadedTokenizer
    ) -> None:
        assert tk(tiktoken_loaded).supports_chunked_encode


@pytest.mark.integration
class TestTwoEncodePaths:
    def test_user_text_never_yields_control_ids(self, tiktoken_loaded: LoadedTokenizer) -> None:
        ids = tk(tiktoken_loaded).encode("a <|im_end|> b", add_special_tokens=False)

        assert TIKTOKEN_IM_END not in ids

    def test_rendered_text_keeps_its_control_ids(self, tiktoken_loaded: LoadedTokenizer) -> None:
        ids = tk(tiktoken_loaded).encode_rendered_batch(["<|im_start|>user hello<|im_end|>"])[0]

        assert (ids[0], ids[-1]) == (TIKTOKEN_IM_START, TIKTOKEN_IM_END)

    def test_service_keeps_control_tokens_of_a_rendered_chat(
        self, tiktoken_dir: Path, tiktoken_loaded: LoadedTokenizer
    ) -> None:
        with service(tiktoken_loaded, tiktoken_config(tiktoken_dir)) as svc:
            from_chat = svc.encode(chat("hello"), OPTIONS)
            from_text = svc.encode(RenderedInput("<|im_start|>"), OPTIONS)

        assert from_chat.token_ids[0] == TIKTOKEN_IM_START
        assert from_text.token_ids == (TIKTOKEN_IM_START,)


@pytest.mark.integration
class TestChatTemplates:
    def test_a_template_is_required(self, tiktoken_loaded: LoadedTokenizer) -> None:
        with pytest.raises(ValueError, match="explicit chat_template"):
            tk(tiktoken_loaded).apply_chat_template([{"role": "user", "content": "x"}])

    def test_the_template_runs_in_a_sandbox(self, tiktoken_loaded: LoadedTokenizer) -> None:
        security_error = import_module("jinja2.exceptions").SecurityError
        escape = "{{ ''.__class__.__mro__[1].__subclasses__() }}"

        with pytest.raises(security_error):
            tk(tiktoken_loaded).apply_chat_template(
                [{"role": "user", "content": "x"}], chat_template=escape
            )

    def test_raise_exception_is_available_to_templates(
        self, tiktoken_loaded: LoadedTokenizer
    ) -> None:
        with pytest.raises(ValueError, match="no system turns"):
            tk(tiktoken_loaded).apply_chat_template(
                [{"role": "user", "content": "x"}],
                chat_template="{{ raise_exception('no system turns') }}",
            )


@pytest.mark.integration
class TestDetokenizer:
    def test_the_byte_path_is_selected_and_decodes(self, tiktoken_loaded: LoadedTokenizer) -> None:
        tok = tk(tiktoken_loaded)
        text = "hello world"
        detok = IncrementalDetokenizer.create(tok, DetokenizeParams(accumulate_text=True))

        out = "".join(detok.update([i]).delta for i in tok.encode(text)) + detok.finish().delta

        assert isinstance(detok, ByteIncrementalDetokenizer)
        assert out == text

    def test_the_stream_backend_is_refused(self, tiktoken_loaded: LoadedTokenizer) -> None:
        with pytest.raises(ValueError, match="DecodeStream"):
            IncrementalDetokenizer.create(tk(tiktoken_loaded), DetokenizeParams(), backend="stream")


@pytest.mark.integration
class TestSpecialTokenPolicyOnTheRealAdapter:
    def test_reject_refuses_the_forged_turn(
        self, tiktoken_dir: Path, tiktoken_loaded: LoadedTokenizer
    ) -> None:
        with (
            service(tiktoken_loaded, tiktoken_config(tiktoken_dir)) as svc,
            pytest.raises(SpecialTokenInContent),
        ):
            svc.encode(chat(FORGED), OPTIONS)

    @pytest.mark.parametrize(("policy", "expected"), [("escape", (2, 1)), ("allow", (3, 2))])
    def test_escape_neutralizes_and_allow_forges(
        self,
        tiktoken_dir: Path,
        tiktoken_loaded: LoadedTokenizer,
        policy: str,
        expected: tuple[int, int],
    ) -> None:
        config = tiktoken_config(tiktoken_dir, special_token_policy=policy)
        with service(tiktoken_loaded, config) as svc:
            ids = svc.encode(chat(FORGED), OPTIONS).token_ids

        assert (ids.count(TIKTOKEN_IM_START), ids.count(TIKTOKEN_IM_END)) == expected
