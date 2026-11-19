"""Sampling controls: which rows get which transform, the penalty counts and the logprobs."""

from __future__ import annotations

import math
from typing import Any

import pytest

from ayaka.core.params import LogprobMode, SamplingParams
from ayaka.core.types import RequestId

torch: Any = pytest.importorskip("torch")

from ayaka.ops.backends.torch import sampling as torch_sampling  # noqa: E402
from ayaka.sampling.batch import SamplingBatch, SamplingRow  # noqa: E402
from ayaka.sampling.logits import (  # noqa: E402
    LogitRow,
    LogprobRequest,
    build_controls,
    needs_controls,
)
from ayaka.sampling.logprobs import UNLIKELY_LOGPROB, token_logprobs  # noqa: E402
from ayaka.sampling.penalties import PenaltyStates  # noqa: E402
from ayaka.sampling.sampler import Sampler  # noqa: E402

pytestmark = pytest.mark.unit

VOCAB = 40
EOS = 39


def _row(name: str = "a", produced: int = 0, **params: Any) -> LogitRow:
    return LogitRow(RequestId(name), SamplingParams(**params), produced)


def _controls(rows: list[LogitRow], states: PenaltyStates | None = None) -> Any:
    return build_controls(
        rows, vocab_size=VOCAB, eos_token_ids={EOS}, penalties=states, device="cpu"
    )


def _bits(mask: Any, row: int) -> set[int]:
    words = [int(w) & 0xFFFFFFFF for w in mask[row].tolist()]
    return {t for t in range(VOCAB) if words[t // 32] >> (t % 32) & 1}


class TestBuild:
    def test_plain_rows_need_nothing(self) -> None:
        rows = [_row(temperature=0.7, top_k=5), _row("b", temperature=0.0)]

        assert not any(needs_controls(row.params) for row in rows)
        assert _controls(rows) is None

    def test_min_tokens_needs_a_mask_only_while_it_holds_something_back(self) -> None:
        params = SamplingParams(min_tokens=2)

        assert needs_controls(params)
        assert needs_controls(params, 1)
        assert not needs_controls(params, 2)
        assert not needs_controls(params, 0, has_eos=False)
        assert not needs_controls(SamplingParams(min_tokens=2, ignore_eos=True), 0)
        assert needs_controls(
            SamplingParams(min_tokens=2, ignore_eos=True, stop_token_ids=frozenset({5})), 0
        )

    def test_allowed_ids_and_min_tokens_become_bits(self) -> None:
        rows = [
            _row(allowed_token_ids=frozenset({1, 33})),
            _row("b", produced=1, min_tokens=2, stop_token_ids=frozenset({5})),
            _row("c", produced=2, min_tokens=2, stop_token_ids=frozenset({5})),
            _row("d", min_tokens=2, ignore_eos=True),
            _row("e", temperature=0.0),
        ]

        controls = _controls(rows)

        mask = controls.bitmask
        assert _bits(mask, 0) == {1, 33}
        assert _bits(mask, 1) == set(range(VOCAB)) - {5, EOS}
        assert _bits(mask, 2) == set(range(VOCAB))  # min_tokens reached
        assert _bits(mask, 3) == set(range(VOCAB))  # ignore_eos and no stop ids
        assert _bits(mask, 4) == set(range(VOCAB))
        assert controls.bias is None
        assert controls.penalties is None

    def test_bias_pairs_name_their_rows(self) -> None:
        controls = _controls([_row(temperature=0.0), _row("b", logit_bias={3: 1.5, 7: -2.0})])

        rows, tokens, biases = controls.bias
        assert rows.tolist() == [1, 1]
        assert tokens.tolist() == [3, 7]
        assert biases.tolist() == [1.5, -2.0]

    def test_logprob_requests_follow_the_parameters(self) -> None:
        controls = _controls(
            [
                _row(logprobs=3),
                _row("b", logprobs=2, logprob_token_ids=(4, 9), logprob_mode="sampling"),
                _row("c"),
            ]
        )

        assert controls.logprobs == (
            LogprobRequest(3, None, LogprobMode.RAW),
            LogprobRequest(2, (4, 9), LogprobMode.SAMPLING),
            None,
        )
        assert controls.any_raw_logprobs
        assert controls.any_sampling_logprobs


class TestPenaltyStates:
    def test_only_a_penalised_request_gets_counts(self) -> None:
        states = PenaltyStates(VOCAB, "cpu")

        states.admit(RequestId("plain"), (1, 2), SamplingParams())
        states.admit(RequestId("p"), (1, 1, 2), SamplingParams(presence_penalty=0.5))

        assert len(states) == 1
        assert RequestId("p") in states

    def test_counts_start_from_the_prompt_and_grow_with_commits(self) -> None:
        states = PenaltyStates(VOCAB, "cpu")
        rid = RequestId("p")
        states.admit(rid, (1, 1, 2), SamplingParams(repetition_penalty=1.2))

        states.commit(rid, (2, 3))
        seen, output = states.rows([rid, None])

        assert seen[0, :4].tolist() == [0, 2, 2, 1]
        assert output[0, :4].tolist() == [0, 0, 1, 1]
        assert int(seen[1].sum()) == 0
        states.drop(rid)
        assert len(states) == 0

    def test_counts_of_two_requests_do_not_leak(self) -> None:
        states = PenaltyStates(VOCAB, "cpu")
        params = SamplingParams(frequency_penalty=0.3)
        states.admit(RequestId("a"), (1,), params)
        states.admit(RequestId("b"), (2,), params)

        states.commit(RequestId("a"), (5,))
        seen, output = states.rows([RequestId("a"), RequestId("b")])

        assert output[1].sum() == 0
        assert seen[1, 5] == 0


class TestTokenLogprobs:
    def test_rank_top_and_ties_by_the_smaller_id(self) -> None:
        lp = torch.log_softmax(torch.tensor([[1.0, 3.0, 3.0, 0.0, 2.0]]), dim=-1)
        found = token_logprobs(
            [LogprobRequest(3, None, LogprobMode.RAW)],
            torch.tensor([2]),
            raw=lp,
            drawn=None,
        )

        [entry] = found
        assert entry is not None
        assert entry.token_id == 2
        assert entry.rank == 2  # token 1 ties with it and has the smaller id
        assert [token for token, _ in entry.top] == [1, 2, 4]
        assert entry.logprob == pytest.approx(float(lp[0, 2]))

    def test_explicit_ids_and_a_token_without_probability(self) -> None:
        lp = torch.tensor([[-math.inf, -0.1, -2.5]])
        found = token_logprobs(
            [LogprobRequest(2, (0, 2), LogprobMode.SAMPLING), None],
            torch.tensor([1, 0]),
            raw=None,
            drawn=torch.cat([lp, lp]),
        )

        entry, nothing = found
        assert nothing is None
        assert entry is not None
        assert entry.top == ((0, UNLIKELY_LOGPROB), (2, pytest.approx(-2.5)))


def _batch(rows: int, **params: Any) -> Any:
    return SamplingBatch.build(
        [
            SamplingRow(params=SamplingParams(**params), seed=5, position=10 + i)
            for i in range(rows)
        ],
        device="cpu",
    )


class TestSamplerPipeline:
    def test_raw_logprobs_ignore_temperature_and_filters_and_sampling_logprobs_do_not(
        self,
    ) -> None:
        sampler = Sampler(torch_sampling)  # pyright: ignore[reportArgumentType]
        logits = torch.tensor([[2.0, 1.0, 0.5, 0.0] + [-3.0] * (VOCAB - 4)])
        found: dict[str, Any] = {}
        for mode in ("raw", "sampling"):
            for temperature in (0.5, 1.5):
                params = {"temperature": temperature, "top_k": 2, "logprobs": 1}
                rows = [_row(logprob_mode=mode, **params)]
                out = torch.empty(1, dtype=torch.int64)
                [entry] = sampler.sample(logits, _batch(1, **params), out, _controls(rows))
                assert entry is not None
                found[f"{mode}-{temperature}"] = entry.top[0][1]

        assert found["raw-0.5"] == found["raw-1.5"]
        assert found["sampling-0.5"] != found["sampling-1.5"]

    def test_a_greedy_row_with_a_penalty_applies_it(self) -> None:
        sampler = Sampler(torch_sampling)  # pyright: ignore[reportArgumentType]
        states = PenaltyStates(VOCAB, "cpu")
        rid = RequestId("a")
        states.admit(rid, (0,), SamplingParams(temperature=0.0, presence_penalty=2.0))
        states.commit(rid, (0,))
        logits = torch.tensor([[1.0, 0.5] + [-3.0] * (VOCAB - 2)])
        out = torch.empty(1, dtype=torch.int64)
        rows = [LogitRow(rid, SamplingParams(temperature=0.0, presence_penalty=2.0), 1)]

        sampler.sample(logits, _batch(1, temperature=0.0), out, _controls(rows, states))

        assert int(out[0]) == 1  # token 0 lost 2.0 and fell below token 1

    def test_an_allowed_set_of_one_always_yields_it(self) -> None:
        sampler = Sampler(torch_sampling)  # pyright: ignore[reportArgumentType]
        params = {"temperature": 1.0, "allowed_token_ids": frozenset({17})}
        out = torch.empty(4, dtype=torch.int64)

        sampler.sample(
            torch.randn(4, VOCAB),
            _batch(4, **params),
            out,
            _controls([_row(str(i), **params) for i in range(4)]),
        )

        assert out.tolist() == [17] * 4

    def test_the_model_logits_are_not_changed(self) -> None:
        sampler = Sampler(torch_sampling)  # pyright: ignore[reportArgumentType]
        logits = torch.randn(1, VOCAB).half()
        before = logits.clone()
        params = {"temperature": 0.0, "logit_bias": {3: 50.0}}

        sampler.sample(
            logits,
            _batch(1, **params),
            torch.empty(1, dtype=torch.int64),
            _controls([_row(**params)]),
        )

        assert torch.equal(logits, before)
