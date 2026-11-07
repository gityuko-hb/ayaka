"""Fixtures for the tokenizer tests; the fakes themselves live in ``tests/_support``."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

from ayaka.configs.tokenizer import TokenizerConfig
from ayaka.tokenizers.service import TokenizerService
from tokenizer_fakes import MODEL_VOCAB, FakeTokenizer, loaded_for

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

    from ayaka.tokenizers.contracts import LoadedTokenizer


@pytest.fixture
def fake() -> FakeTokenizer:
    return FakeTokenizer()


@pytest.fixture
def verified_fake() -> FakeTokenizer:
    tokenizer = FakeTokenizer()
    assert tokenizer.verify_byte_path()
    return tokenizer


@pytest.fixture
def make_service() -> Iterator[Callable[..., TokenizerService]]:
    """Build a service over a fake; closes everything it made at teardown."""
    made: list[TokenizerService] = []

    def factory(
        tokenizer: FakeTokenizer | None = None,
        *,
        loaded: LoadedTokenizer | None = None,
        max_model_len: int = 64,
        model_vocab_size: int = MODEL_VOCAB,
        **config: Any,
    ) -> TokenizerService:
        chosen = loaded if loaded is not None else loaded_for(tokenizer or FakeTokenizer())
        service = TokenizerService(
            TokenizerConfig("fake", mode="hf", **config),
            max_model_len=max_model_len,
            model_vocab_size=model_vocab_size,
            loaded=chosen,
        )
        made.append(service)
        return service

    try:
        yield factory
    finally:
        for service in made:
            service.close()
