"""``ProcessTokenizerPool``: real spawned worker processes around the fake tokenizer.

The workers build their service with a module-level factory from ``tokenizer_fakes``
(spawn imports it by name), so no third-party package is needed. A pool starts its
processes lazily, so the tests that never submit work cost nothing.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

import pytest

from ayaka.configs.tokenizer import DetokenizeParams, TokenizerConfig
from ayaka.tokenizers.contracts import (
    ChatInput,
    ChatMessage,
    EncodeOptions,
    RenderedInput,
    TextInput,
    TokenIdsInput,
)
from ayaka.tokenizers.encoding.media import encode_media_chat
from ayaka.tokenizers.errors import TokenizerOverloaded
from ayaka.tokenizers.process_pool import ProcessTokenizerPool
from tokenizer_fakes import MODEL_VOCAB, make_rendered_service, make_service

if TYPE_CHECKING:
    from collections.abc import Iterator

    from ayaka.tokenizers.service import TokenizerService

pytestmark = pytest.mark.integration

CONFIG = TokenizerConfig("fake", mode="hf")
OPTIONS = EncodeOptions(max_output_tokens=8)
CHAT = ChatInput((ChatMessage("user", "hi [IMG] there"),))
INPUTS = [
    TextInput("hello world"),
    TextInput("héllo", add_special_tokens=False),
    RenderedInput("<|im_start|>x"),
    ChatInput((ChatMessage("user", "hi"),)),
    TokenIdsInput((1, 2, 3)),
]


def new_pool(
    config: TokenizerConfig = CONFIG, *, workers: int = 1, max_pending: int = 64
) -> ProcessTokenizerPool:
    return ProcessTokenizerPool(
        config,
        max_model_len=64,
        model_vocab_size=MODEL_VOCAB,
        workers=workers,
        max_pending=max_pending,
        factory=make_service,
    )


@pytest.fixture(scope="module")
def local() -> Iterator[TokenizerService]:
    service = make_service()
    yield service
    service.close()


@pytest.fixture(scope="module")
def pool() -> Iterator[ProcessTokenizerPool]:
    pool = new_pool(workers=2)
    yield pool
    pool.close()


@pytest.mark.integration
class TestEncode:
    @pytest.mark.parametrize("value", INPUTS, ids=lambda v: type(v).__name__)
    def test_matches_the_in_process_result(
        self, pool: ProcessTokenizerPool, local: TokenizerService, value: object
    ) -> None:
        assert pool.encode(value, OPTIONS) == local.encode(value, OPTIONS)  # type: ignore[arg-type]

    def test_submit_returns_a_future(self, pool: ProcessTokenizerPool) -> None:
        assert pool.submit(TextInput("hi"), OPTIONS).result(timeout=60).token_ids[-1] == ord("i")

    def test_encode_async_awaits_the_child(self, pool: ProcessTokenizerPool) -> None:
        result = asyncio.run(pool.encode_async(TextInput("hi"), OPTIONS))

        assert result.token_ids[-1] == ord("i")

    def test_errors_in_the_child_reach_the_caller(self, pool: ProcessTokenizerPool) -> None:
        with pytest.raises(ValueError, match="not supported by model/tokenizer"):
            pool.encode(TokenIdsInput((299,)), OPTIONS)


@pytest.mark.integration
class TestDetokenizer:
    PARAMS = DetokenizeParams(stop=("wör",))
    TEXT = "héllo wörld 🚀"

    def test_matches_an_in_process_detokenizer_step_by_step(
        self, pool: ProcessTokenizerPool, local: TokenizerService
    ) -> None:
        remote = pool.new_detokenizer(self.PARAMS)
        reference = local.new_detokenizer(self.PARAMS)

        for token in self.TEXT.encode():
            got, want = remote.update([token]), reference.update([token])
            assert got == want
            if want.stop_matched:
                break
        else:
            pytest.fail("the stop string was never matched")

        assert remote.finish() == reference.finish()

    def test_consecutive_decoders_are_spread_over_the_workers(
        self, pool: ProcessTokenizerPool
    ) -> None:
        first, second = pool.new_detokenizer(self.PARAMS), pool.new_detokenizer(self.PARAMS)

        assert first._worker != second._worker
        first.close()
        second.close()

    def test_a_finished_decoder_is_terminal(self, pool: ProcessTokenizerPool) -> None:
        remote = pool.new_detokenizer(DetokenizeParams())
        remote.update(list(b"abc"))
        remote.finish()

        with pytest.raises(RuntimeError, match="terminal"):
            remote.update([1])
        assert remote.finish().delta == ""

    def test_a_closed_decoder_is_terminal(self, pool: ProcessTokenizerPool) -> None:
        remote = pool.new_detokenizer(DetokenizeParams())
        remote.close()

        with pytest.raises(RuntimeError, match="terminal"):
            remote.update([1])


@pytest.mark.integration
class TestMedia:
    def test_matches_the_in_process_result(self) -> None:
        markers = (("[IMG]", 3),)
        pool = ProcessTokenizerPool(
            CONFIG,
            max_model_len=64,
            model_vocab_size=MODEL_VOCAB,
            workers=1,
            factory=make_rendered_service,
        )
        local = make_rendered_service()
        try:
            got = pool.encode_multimodal(CHAT, OPTIONS, markers, 0)
            want = encode_media_chat(local, CHAT, OPTIONS, markers, 0)
        finally:
            pool.close()
            local.close()

        assert got == want


@pytest.mark.integration
class TestLimitsAndLifecycle:
    def test_the_byte_budget_is_enforced_before_any_work_starts(self) -> None:
        pool = new_pool(TokenizerConfig("fake", mode="hf", encode_max_pending_bytes=10), workers=1)
        try:
            with pytest.raises(TokenizerOverloaded, match="byte capacity"):
                pool.submit(TextInput("x" * 20), OPTIONS)
        finally:
            pool.close()

    def test_the_job_limit_is_enforced(self) -> None:
        pool = new_pool(workers=1, max_pending=1)
        pool._capacity.acquire()
        try:
            with pytest.raises(TokenizerOverloaded, match="capacity exhausted"):
                pool.submit(TextInput("hi"), OPTIONS)
        finally:
            pool._capacity.release()
            pool.close()

    def test_a_closed_pool_refuses_work(self) -> None:
        pool = new_pool(workers=1)
        pool.close()

        with pytest.raises(RuntimeError, match="closed"):
            pool.submit(TextInput("hi"), OPTIONS)

    def test_children_must_run_their_tokenizer_inline(self) -> None:
        with pytest.raises(ValueError, match="inline child tokenizers"):
            new_pool(TokenizerConfig("fake", mode="hf", encode_pool_workers=2))

    @pytest.mark.parametrize(("workers", "max_pending"), [(0, 1), (1, 0)])
    def test_sizes_must_be_positive(self, workers: int, max_pending: int) -> None:
        with pytest.raises(ValueError, match=">= 1"):
            new_pool(workers=workers, max_pending=max_pending)
