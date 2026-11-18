"""Penalties, logit bias, allowed ids, ``min_tokens`` and logprobs, end to end on the tiny GPT-2.

The oracle reruns GPT-2's formulas over the whole sequence with no cache (``reference_hidden``),
then applies the bias and the penalty formulas of the OpenAI and vLLM documentation written out
here, and takes the argmax. Nothing is a hand-written string.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

from ayaka.core.params import SamplingParams
from ayaka.core.request import GenerateRequest
from ayaka.core.types import FinishReason, RequestId

torch: Any = pytest.importorskip("torch")

from ayaka.engine.llm import LLM  # noqa: E402
from tiny_model import (  # noqa: E402  # needs the importorskip above
    TinyCheckpoint,
    assert_cache_released,
    prompt_ids,
    reference_hidden,
    tiny_engine_config,
    tiny_tokenizer,
    write_tiny_gpt2,
)

if TYPE_CHECKING:
    from collections.abc import Iterator

    from ayaka.core.outputs import RequestOutput

pytestmark = pytest.mark.unit

_PROMPT = "the cat sat on the mat and the cat"


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory: pytest.TempPathFactory) -> TinyCheckpoint:
    return write_tiny_gpt2(tmp_path_factory.mktemp("e2e-controls"))


@pytest.fixture
def llm(checkpoint: TinyCheckpoint) -> Iterator[LLM]:
    with LLM.from_config(
        tiny_engine_config(checkpoint.directory), tokenizer=tiny_tokenizer()
    ) as engine:
        yield engine


def _one(result: RequestOutput | list[RequestOutput]) -> RequestOutput:
    assert not isinstance(result, list)
    return result


def _reference(
    checkpoint: TinyCheckpoint,
    prompt: list[int],
    count: int,
    *,
    repetition: float = 1.0,
    frequency: float = 0.0,
    presence: float = 0.0,
    bias: dict[int, float] | None = None,
) -> tuple[list[int], list[float]]:
    """Greedy tokens under the formulas, and the raw logprob of each chosen token."""
    state, config = checkpoint.state, checkpoint.config
    table = state["transformer.wte.weight"]
    tokens = list(prompt)
    out: list[int] = []
    raw_logprobs: list[float] = []
    for _ in range(count):
        x = (reference_hidden(state, config, torch.tensor(tokens))[-1] @ table.T).double()
        for token, value in (bias or {}).items():
            x[token] += value
        raw = torch.log_softmax(x, dim=-1)
        for token in range(x.shape[0]):
            seen = tokens.count(token)
            emitted = out.count(token)
            if seen and repetition != 1.0:
                x[token] = x[token] / repetition if x[token] > 0 else x[token] * repetition
            x[token] -= frequency * emitted
            if emitted:
                x[token] -= presence
        chosen = int(torch.nonzero(x == x.max())[0])
        raw_logprobs.append(float(raw[chosen]))
        tokens.append(chosen)
        out.append(chosen)
    return out, raw_logprobs


@pytest.mark.parametrize(
    ("repetition", "frequency", "presence"),
    [(1.3, 0.0, 0.0), (1.0, 0.4, 0.0), (1.0, 0.0, 0.6), (1.2, 0.2, 0.3)],
    ids=["repetition", "frequency", "presence", "all"],
)
def test_greedy_penalties_follow_the_formulas(
    llm: LLM,
    checkpoint: TinyCheckpoint,
    repetition: float,
    frequency: float,
    presence: float,
) -> None:
    expected, _ = _reference(
        checkpoint,
        prompt_ids(_PROMPT),
        10,
        repetition=repetition,
        frequency=frequency,
        presence=presence,
    )
    sampling = SamplingParams(
        temperature=0.0,
        max_tokens=10,
        repetition_penalty=repetition,
        frequency_penalty=frequency,
        presence_penalty=presence,
    )

    out = _one(llm.generate(_PROMPT, sampling))

    assert list(out.outputs[0].token_ids) == expected
    assert len(llm.engine.runner.penalties) == 0  # dropped at the end
    assert_cache_released(llm.engine)


def test_a_logit_bias_moves_the_choice(llm: LLM, checkpoint: TinyCheckpoint) -> None:
    bias = {ord("q"): 30.0}
    expected, _ = _reference(checkpoint, prompt_ids(_PROMPT), 6, bias=bias)

    out = _one(
        llm.generate(_PROMPT, SamplingParams(temperature=0.0, max_tokens=6, logit_bias=bias))
    )

    assert list(out.outputs[0].token_ids) == expected
    assert expected.count(ord("q")) > 0


def test_a_stop_id_cannot_end_the_answer_before_min_tokens(llm: LLM) -> None:
    sampling = SamplingParams(
        temperature=0.0,
        max_tokens=10,
        min_tokens=3,
        logit_bias={ord("z"): 100.0},
        stop_token_ids=frozenset({ord("z")}),
    )

    out = _one(llm.generate(_PROMPT, sampling))

    tokens = list(out.outputs[0].token_ids)
    assert len(tokens) == 4
    assert ord("z") not in tokens[:3]
    assert tokens[3] == ord("z")
    assert out.outputs[0].finish_reason is FinishReason.STOP


def test_an_allowed_set_of_one_id_always_yields_it(llm: LLM) -> None:
    sampling = SamplingParams(
        temperature=1.0, seed=3, max_tokens=5, allowed_token_ids=frozenset({ord("k")})
    )

    out = _one(llm.generate(_PROMPT, sampling))

    assert list(out.outputs[0].token_ids) == [ord("k")] * 5


def test_raw_logprobs_of_greedy_tokens_match_the_reference(
    llm: LLM, checkpoint: TinyCheckpoint
) -> None:
    expected, raw = _reference(checkpoint, prompt_ids(_PROMPT), 5)

    out = _one(llm.generate(_PROMPT, SamplingParams(temperature=0.0, max_tokens=5, logprobs=3)))

    completion = out.outputs[0]
    assert completion.logprobs is not None
    assert [entry.token_id for entry in completion.logprobs] == expected
    for entry, reference in zip(completion.logprobs, raw, strict=True):
        assert entry.logprob == pytest.approx(reference, abs=1e-4)
        assert entry.rank == 1
        assert len(entry.top) == 3
        assert entry.top[0][0] == entry.token_id
        values = [value for _, value in entry.top]
        assert values == sorted(values, reverse=True)
        assert entry.decoded == chr(entry.token_id)
    assert completion.cumulative_logprob == pytest.approx(sum(raw), abs=1e-3)


def test_streamed_and_final_logprobs_agree(llm: LLM) -> None:
    sampling = SamplingParams(temperature=0.8, seed=4, top_k=4, max_tokens=6, logprobs=2)
    whole = _one(llm.generate(_PROMPT, sampling))

    pieces = []
    for fragment in llm.stream(_PROMPT, sampling):
        pieces += list(fragment.outputs[0].logprobs or ())

    assert whole.outputs[0].logprobs is not None
    # Two requests: the second attaches the first's prompt from the prefix cache, which may move
    # the last float bits of a logprob, never a token.
    assert [p.token_id for p in pieces] == [p.token_id for p in whole.outputs[0].logprobs]
    for got, want in zip(pieces, whole.outputs[0].logprobs, strict=True):
        assert got.logprob == pytest.approx(want.logprob, abs=1e-5)
        assert [t for t, _ in got.top] == [t for t, _ in want.top]


def test_a_penalised_sampled_request_draws_the_same_alone_or_in_a_batch(llm: LLM) -> None:
    sampling = SamplingParams(
        temperature=0.9, top_k=4, seed=8, max_tokens=8, presence_penalty=0.5, logit_bias={66: 1.0}
    )
    alone = _one(llm.generate(_PROMPT, sampling))

    together = llm.generate(
        [_PROMPT, "another prompt", _PROMPT],
        [sampling, SamplingParams(temperature=0.0, max_tokens=8), sampling],
    )

    assert isinstance(together, list)
    assert together[0].outputs[0].token_ids == alone.outputs[0].token_ids
    assert together[2].outputs[0].token_ids == alone.outputs[0].token_ids


def test_only_a_penalised_request_holds_counts(llm: LLM) -> None:
    engine = llm.engine
    engine.add_request(
        GenerateRequest(
            request_id=RequestId("plain"),
            prompt=_PROMPT,
            sampling=SamplingParams(temperature=0.0, max_tokens=3),
        )
    )
    assert len(engine.runner.penalties) == 0
    engine.add_request(
        GenerateRequest(
            request_id=RequestId("pen"),
            prompt=_PROMPT,
            sampling=SamplingParams(temperature=0.0, max_tokens=3, frequency_penalty=0.5),
        )
    )
    assert len(engine.runner.penalties) == 1
    while engine.has_unfinished:
        engine.step()

    assert len(engine.runner.penalties) == 0
