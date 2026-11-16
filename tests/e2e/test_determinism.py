"""Same request, same tokens: nothing around a request may change what it generates.

The tiny GPT-2 runs on the CPU in float32 with weights whose greedy choices are well separated,
and sampling draws its random numbers from the request's seed and the position of the token, not
from a shared stream. So the tokens of a request must be the same:

* run after run, and engine after engine, for the same seed;
* alone, or in a batch of any size, in any order, or joined by other requests half way;
* whatever the largest step (how a prompt is cut into chunks) and the cache block size.

Different seeds must give different text, or the seed would not be doing anything.
"""

from __future__ import annotations

import itertools
from typing import TYPE_CHECKING, Any

import pytest

from ayaka.core.params import SamplingParams
from ayaka.core.request import GenerateRequest
from ayaka.core.types import RequestId

torch: Any = pytest.importorskip("torch")

from ayaka.engine.llm import LLM  # noqa: E402
from tiny_model import (  # noqa: E402  # needs the importorskip above
    MAX_POSITIONS,
    TinyCheckpoint,
    assert_cache_released,
    greedy_reference,
    prompt_ids,
    tiny_engine_config,
    tiny_tokenizer,
    write_tiny_gpt2,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

pytestmark = pytest.mark.unit

TOKENS = 20


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory: pytest.TempPathFactory) -> TinyCheckpoint:
    return write_tiny_gpt2(tmp_path_factory.mktemp("e2e-determinism"))


def _build(checkpoint: TinyCheckpoint, *, block_size: int = 4, **options: Any) -> LLM:
    # A cache of 128 tokens of room for each of eight requests, whatever the block size.
    blocks = 1 + 8 * -(-MAX_POSITIONS // block_size)
    config = tiny_engine_config(
        checkpoint.directory, block_size=block_size, num_blocks=blocks, **options
    )
    return LLM.from_config(config, tokenizer=tiny_tokenizer())


def _sampled(seed: int, *, top_k: int = 16, **more: Any) -> SamplingParams:
    """Sampling at temperature 1, cut to the 16 likeliest tokens.

    The cut keeps the draw among the tokens the tiny model favours, the printable ASCII ones; an
    uncut draw can land on an id the byte tokenizer has no text for and fail that request.
    """
    return SamplingParams(temperature=1.0, max_tokens=TOKENS, seed=seed, top_k=top_k, **more)


def _request(name: str, prompt: str, sampling: SamplingParams) -> GenerateRequest:
    return GenerateRequest(request_id=RequestId(name), prompt=prompt, sampling=sampling)


#: Four requests that differ in prompt, length, seed and sampling mode.
REQUESTS = (
    _request("a", "the first request", _sampled(101)),
    _request("b", "second", _sampled(202, top_k=8)),
    _request(
        "c",
        "a third request with a prompt that is clearly the longest of the four",
        _sampled(303, top_p=0.9),
    ),
    _request("d", "fourth, greedy", SamplingParams(temperature=0.0, max_tokens=TOKENS)),
)


def _run(llm: LLM, requests: Sequence[GenerateRequest]) -> dict[str, tuple[int, ...]]:
    outs = llm.generate(list(requests))
    assert isinstance(outs, list)
    assert all(out.error is None for out in outs)
    return {str(out.request_id): out.outputs[0].token_ids for out in outs}


@pytest.fixture(scope="module")
def alone(checkpoint: TinyCheckpoint) -> dict[str, tuple[int, ...]]:
    """What each request generates when it has the engine to itself."""
    result: dict[str, tuple[int, ...]] = {}
    for request in REQUESTS:
        with _build(checkpoint, max_running=1) as llm:
            result |= _run(llm, [request])
    return result


class TestSameSeedSameTokens:
    def test_repeated_runs_on_one_engine(self, checkpoint: TinyCheckpoint) -> None:
        with _build(checkpoint) as llm:
            runs = [_run(llm, [REQUESTS[0]]) for _ in range(3)]

            assert runs[0] == runs[1] == runs[2]
            assert len(runs[0]["a"]) == TOKENS
            assert_cache_released(llm.engine)

    def test_a_fresh_engine_each_time(self, checkpoint: TinyCheckpoint) -> None:
        runs: list[dict[str, tuple[int, ...]]] = []
        for _ in range(3):
            with _build(checkpoint) as llm:
                runs.append(_run(llm, REQUESTS))

        assert runs[0] == runs[1] == runs[2]

    def test_a_request_id_does_not_matter_only_the_seed_does(
        self, checkpoint: TinyCheckpoint
    ) -> None:
        sampling = _sampled(55)
        with _build(checkpoint) as llm:
            first = _run(llm, [_request("one-name", "same prompt", sampling)])
            second = _run(llm, [_request("another-name", "same prompt", sampling)])

        assert first["one-name"] == second["another-name"]

    def test_greedy_needs_no_seed_and_matches_the_reference(
        self, checkpoint: TinyCheckpoint
    ) -> None:
        greedy = SamplingParams(temperature=0.0, max_tokens=TOKENS)
        with _build(checkpoint) as llm:
            runs = [llm.generate("greedy", greedy) for _ in range(2)]

        tokens = tuple(
            greedy_reference(checkpoint.state, checkpoint.config, prompt_ids("greedy"), TOKENS)
        )
        for run in runs:
            assert not isinstance(run, list)
            assert run.outputs[0].token_ids == tokens


class TestSeedsDiffer:
    def test_different_seeds_give_different_tokens(self, checkpoint: TinyCheckpoint) -> None:
        seeds = (1, 2, 3, 4, 5, 6)
        with _build(checkpoint) as llm:
            runs = [_run(llm, [_request("s", "seeded", _sampled(seed))])["s"] for seed in seeds]

        assert len(set(runs)) == len(seeds)

    def test_sampling_is_not_greedy(self, checkpoint: TinyCheckpoint) -> None:
        greedy = greedy_reference(checkpoint.state, checkpoint.config, prompt_ids("seeded"), TOKENS)
        with _build(checkpoint) as llm:
            runs = [_run(llm, [_request("s", "seeded", _sampled(seed))])["s"] for seed in range(6)]

        assert any(run != tuple(greedy) for run in runs)


class TestNotDependentOnTheBatch:
    @pytest.mark.parametrize(
        "order", list(itertools.permutations(range(len(REQUESTS)), 3)), ids=str
    )
    def test_any_three_of_the_four_in_any_order(
        self, checkpoint: TinyCheckpoint, alone: dict[str, tuple[int, ...]], order: tuple[int, ...]
    ) -> None:
        requests = [REQUESTS[n] for n in order]
        with _build(checkpoint, max_running=4) as llm:
            together = _run(llm, requests)

            assert_cache_released(llm.engine)
        assert together == {str(r.request_id): alone[str(r.request_id)] for r in requests}

    @pytest.mark.parametrize("max_running", [1, 2, 3, 4])
    def test_all_four_whatever_number_run_at_once(
        self, checkpoint: TinyCheckpoint, alone: dict[str, tuple[int, ...]], max_running: int
    ) -> None:
        with _build(checkpoint, max_running=max_running) as llm:
            together = _run(llm, REQUESTS)

        assert together == alone

    def test_the_same_request_twice_in_one_batch_gives_the_same_tokens(
        self, checkpoint: TinyCheckpoint, alone: dict[str, tuple[int, ...]]
    ) -> None:
        twin = _request("twin", "the first request", _sampled(101))
        with _build(checkpoint) as llm:
            together = _run(llm, [REQUESTS[0], twin, REQUESTS[1]])

        assert together["a"] == together["twin"] == alone["a"]

    def test_companions_that_arrive_while_it_runs_change_nothing(
        self, checkpoint: TinyCheckpoint, alone: dict[str, tuple[int, ...]]
    ) -> None:
        with _build(checkpoint, max_running=4) as llm:
            engine = llm.engine
            engine.add_request(REQUESTS[0])
            for late in REQUESTS[1:]:
                engine.step()
                engine.step()
                engine.add_request(late)
            finished: dict[str, tuple[int, ...]] = {}
            while engine.has_unfinished:
                for out in engine.step():
                    finished[str(out.request_id)] = out.outputs[0].token_ids

            assert finished == alone
            assert_cache_released(engine)

    def test_a_request_that_ends_early_beside_it_changes_nothing(
        self, checkpoint: TinyCheckpoint, alone: dict[str, tuple[int, ...]]
    ) -> None:
        short = _request(
            "short", "ends soon", SamplingParams(temperature=1.0, max_tokens=2, seed=9, top_k=16)
        )
        with _build(checkpoint) as llm:
            together = _run(llm, [short, REQUESTS[0], REQUESTS[2]])

        assert together["a"] == alone["a"]
        assert together["c"] == alone["c"]
        assert len(together["short"]) == 2


class TestNotDependentOnHowTheWorkIsCut:
    @pytest.mark.parametrize("budget", [4, 5, 9, 16, 33, 128], ids=lambda n: f"step-{n}")
    def test_the_largest_step_does_not_change_any_request(
        self, checkpoint: TinyCheckpoint, alone: dict[str, tuple[int, ...]], budget: int
    ) -> None:
        with _build(checkpoint, max_tokens_per_step=budget) as llm:
            together = _run(llm, REQUESTS)

            assert_cache_released(llm.engine)
        assert together == alone

    @pytest.mark.parametrize("block_size", [1, 2, 4, 8, 16], ids=lambda n: f"block-{n}")
    def test_the_cache_block_size_does_not_change_any_request(
        self, checkpoint: TinyCheckpoint, alone: dict[str, tuple[int, ...]], block_size: int
    ) -> None:
        with _build(checkpoint, block_size=block_size, max_tokens_per_step=16) as llm:
            together = _run(llm, REQUESTS)

            assert_cache_released(llm.engine)
        assert together == alone

    @pytest.mark.parametrize(("block_size", "budget"), [(2, 2), (8, 8), (16, 16), (4, 7)])
    def test_block_size_and_step_size_together(
        self,
        checkpoint: TinyCheckpoint,
        alone: dict[str, tuple[int, ...]],
        block_size: int,
        budget: int,
    ) -> None:
        with _build(
            checkpoint, block_size=block_size, max_tokens_per_step=budget, max_running=2
        ) as llm:
            together = _run(llm, list(reversed(REQUESTS)))

        assert together == alone

    def test_greedy_tokens_are_the_reference_under_every_cut(
        self, checkpoint: TinyCheckpoint
    ) -> None:
        prompt = "greedy under every cut of the work"
        expected = tuple(
            greedy_reference(checkpoint.state, checkpoint.config, prompt_ids(prompt), TOKENS)
        )
        greedy = SamplingParams(temperature=0.0, max_tokens=TOKENS)
        for block_size, budget in itertools.product((1, 4, 16), (16, 40)):
            with _build(checkpoint, block_size=block_size, max_tokens_per_step=budget) as llm:
                out = llm.generate(prompt, greedy)

            assert not isinstance(out, list)
            assert out.outputs[0].token_ids == expected, (block_size, budget)
