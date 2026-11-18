"""The prefix cache end to end on the tiny GPT-2: a warm cache gives the tokens of a cold one.

The oracle is ``greedy_reference``, GPT-2's formulas rerun over the whole sequence with no cache.
A request that reuses cached blocks must produce exactly those tokens and report the reused
prompt tokens in ``Usage.cached_tokens``.
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, Any

import pytest

from ayaka.core.params import SamplingParams
from ayaka.core.request import GenerateRequest
from ayaka.core.types import RequestId

torch: Any = pytest.importorskip("torch")

from ayaka.engine.llm import LLM  # noqa: E402
from tiny_model import (  # noqa: E402  # needs the importorskip above
    TINY_BLOCK_SIZE,
    TinyCheckpoint,
    assert_cache_released,
    greedy_reference,
    prompt_ids,
    tiny_engine_config,
    tiny_tokenizer,
    write_tiny_gpt2,
)

if TYPE_CHECKING:
    from collections.abc import Iterator

    from ayaka.core.outputs import RequestOutput

pytestmark = pytest.mark.unit

_PROMPT = "a prompt that is longer than several blocks of the cache"


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory: pytest.TempPathFactory) -> TinyCheckpoint:
    return write_tiny_gpt2(tmp_path_factory.mktemp("e2e-prefix"))


def _build(checkpoint: TinyCheckpoint, *, prefix_cache: bool = True) -> LLM:
    config = tiny_engine_config(checkpoint.directory)
    config = dataclasses.replace(
        config, cache=dataclasses.replace(config.cache, prefix_cache=prefix_cache)
    )
    return LLM.from_config(config, tokenizer=tiny_tokenizer())


@pytest.fixture
def llm(checkpoint: TinyCheckpoint) -> Iterator[LLM]:
    with _build(checkpoint) as engine:
        yield engine


def _one(result: RequestOutput | list[RequestOutput]) -> RequestOutput:
    assert not isinstance(result, list)
    return result


def _greedy(count: int) -> SamplingParams:
    return SamplingParams(temperature=0.0, max_tokens=count)


def test_a_warm_request_gives_the_tokens_of_a_cold_one(
    llm: LLM, checkpoint: TinyCheckpoint
) -> None:
    expected = greedy_reference(checkpoint.state, checkpoint.config, prompt_ids(_PROMPT), 12)

    cold = _one(llm.generate(_PROMPT, _greedy(12)))
    warm = _one(llm.generate(_PROMPT, _greedy(12)))

    prompt_length = len(prompt_ids(_PROMPT))
    assert list(cold.outputs[0].token_ids) == expected
    assert list(warm.outputs[0].token_ids) == expected
    assert cold.usage is not None
    assert warm.usage is not None
    assert cold.usage.cached_tokens == 0
    assert warm.usage.cached_tokens == (prompt_length // TINY_BLOCK_SIZE) * TINY_BLOCK_SIZE
    assert warm.usage.cached_tokens > 0
    assert_cache_released(llm.engine)


def test_a_prompt_of_whole_blocks_reuses_all_but_the_last_block(
    llm: LLM, checkpoint: TinyCheckpoint
) -> None:
    tokens = tuple(prompt_ids("x" * (3 * TINY_BLOCK_SIZE - 1)))
    assert len(tokens) % TINY_BLOCK_SIZE == 0
    expected = greedy_reference(checkpoint.state, checkpoint.config, tokens, 6)

    _one(llm.generate(tokens, _greedy(6)))
    warm = _one(llm.generate(tokens, _greedy(6)))

    assert list(warm.outputs[0].token_ids) == expected
    assert warm.usage is not None
    assert warm.usage.cached_tokens == len(tokens) - TINY_BLOCK_SIZE


def test_a_second_turn_reuses_the_generated_blocks_of_the_first(
    llm: LLM, checkpoint: TinyCheckpoint
) -> None:
    first_prompt = tuple(prompt_ids(_PROMPT))
    first = _one(llm.generate(first_prompt, _greedy(20)))
    follow_up = (*first_prompt, *first.outputs[0].token_ids, *b" and then?")
    expected = greedy_reference(checkpoint.state, checkpoint.config, follow_up, 8)

    second = _one(llm.generate(follow_up, _greedy(8)))

    assert list(second.outputs[0].token_ids) == expected
    assert second.usage is not None
    # Everything the first turn committed (prompt plus all but its last token), in whole blocks.
    committed = len(first_prompt) + len(first.outputs[0].token_ids) - 1
    assert second.usage.cached_tokens == (committed // TINY_BLOCK_SIZE) * TINY_BLOCK_SIZE


def test_two_requests_in_one_batch_and_a_disabled_cache_agree(
    llm: LLM, checkpoint: TinyCheckpoint
) -> None:
    expected = greedy_reference(checkpoint.state, checkpoint.config, prompt_ids(_PROMPT), 10)

    batch = llm.generate([_PROMPT, _PROMPT, _PROMPT], _greedy(10))
    assert isinstance(batch, list)
    with _build(checkpoint, prefix_cache=False) as cold_engine:
        cold = _one(cold_engine.generate(_PROMPT, _greedy(10)))
        warm_again = _one(cold_engine.generate(_PROMPT, _greedy(10)))

    assert all(list(out.outputs[0].token_ids) == expected for out in batch)
    assert list(cold.outputs[0].token_ids) == expected
    assert warm_again.usage is not None
    assert warm_again.usage.cached_tokens == 0
    assert_cache_released(llm.engine)


def test_a_request_can_opt_out_and_a_salt_isolates(llm: LLM) -> None:
    _one(llm.generate(_PROMPT, _greedy(4)))

    opted_out = _one(
        llm.generate(
            GenerateRequest(
                request_id=RequestId("opt-out"),
                prompt=_PROMPT,
                sampling=_greedy(4),
                prefix_cache=False,
            )
        )
    )
    salted = _one(
        llm.generate(
            GenerateRequest(
                request_id=RequestId("salted"),
                prompt=_PROMPT,
                sampling=_greedy(4),
                cache_salt="tenant-b",
            )
        )
    )

    assert opted_out.usage is not None
    assert salted.usage is not None
    assert opted_out.usage.cached_tokens == 0
    assert salted.usage.cached_tokens == 0
    assert_cache_released(llm.engine)
