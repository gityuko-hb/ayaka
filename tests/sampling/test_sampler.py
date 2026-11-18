"""SamplingBatch and Sampler: the per-step snapshot and the determinism the engine relies on.

The central property: the token of a row is a function of its logits, its parameters, its seed and
its position, and of nothing else. Tests build the same row in batches of different size and order,
with padding, and in a window next to other positions, and compare the tokens.
"""

from __future__ import annotations

import dataclasses
import math
from typing import TYPE_CHECKING, Any

import pytest

from ayaka.core.errors import BackendError
from ayaka.core.params import SamplingParams
from ayaka.ops.backend import load_backend
from ayaka.ops.sampling import DrawStream
from ayaka.sampling.rng import MAX_SEED, draw_uniform
from ayaka.utils.dtypes import DType

torch: Any = pytest.importorskip("torch")

from ayaka.sampling.batch import SamplingBatch, SamplingRow  # noqa: E402
from ayaka.sampling.sampler import Sampler  # noqa: E402

if TYPE_CHECKING:
    from ayaka.ops.sampling import SamplingOps

pytestmark = pytest.mark.unit

VOCAB = 64
GREEDY = SamplingParams(temperature=0.0)
RANDOM = SamplingParams(temperature=0.8, top_k=20, top_p=0.9, min_p=0.01)


def _ops() -> SamplingOps:
    return load_backend(["torch"], device="cpu", dtype=DType.FP32).sampling


def _row(
    params: SamplingParams = RANDOM, seed: int = 1, position: int = 10, sequence: int = 0
) -> SamplingRow:
    return SamplingRow(params=params, seed=seed, position=position, sequence=sequence)


def _run(rows: list[SamplingRow], logits: Any, *, size: int | None = None) -> list[int]:
    batch = SamplingBatch.build(rows, device="cpu", size=size)
    out = torch.empty(batch.size, dtype=torch.int64)
    Sampler(_ops()).sample(logits, batch, out)
    return out.tolist()


def _logits_of(key: int) -> Any:
    """The logits of a request: fixed by the key, as a model's output would be."""
    generator = torch.Generator().manual_seed(key)
    return torch.randn(VOCAB, generator=generator) * 2.0


class TestSamplingRow:
    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("seed", -1),
            ("seed", MAX_SEED + 1),
            ("position", -1),
            ("sequence", -1),
            ("sequence", 2**32),
        ],
    )
    def test_refuses_out_of_range_values(self, field: str, value: int) -> None:
        fields = {"params": RANDOM, "seed": 1, "position": 1, "sequence": 0, field: value}

        with pytest.raises(ValueError, match=f"{field} must be in"):
            SamplingRow(**fields)


class TestSamplingBatchBuild:
    def test_tensors_hold_the_rows_in_order(self) -> None:
        rows = [
            _row(
                SamplingParams(temperature=0.5, top_k=7, top_p=0.8, min_p=0.1), seed=3, position=9
            ),
            _row(SamplingParams(temperature=1.0), seed=4, position=2, sequence=1),
        ]

        batch = SamplingBatch.build(rows, device="cpu")

        assert batch.size == 2
        assert batch.seeds.tolist() == [3, 4]
        assert batch.seeds.dtype == torch.int64
        assert batch.positions.tolist() == [9, 2]
        assert batch.sequences.tolist() == [0, 1]
        assert batch.sequences.dtype == torch.int32
        assert batch.temperature.tolist() == pytest.approx([0.5, 1.0])
        assert batch.top_k.tolist() == [7, 0]  # the disabled sentinel -1 becomes 0
        assert batch.top_k.dtype == torch.int32
        assert batch.top_p.tolist() == pytest.approx([0.8, 1.0])
        assert batch.min_p.tolist() == pytest.approx([0.1, 0.0])
        assert not batch.all_greedy

    def test_greedy_rows_are_stored_neutral(self) -> None:
        by_temperature = SamplingParams(temperature=0.0, top_p=0.5, min_p=0.2)
        by_top_k = SamplingParams(temperature=1.0, top_k=1, top_p=0.5)

        batch = SamplingBatch.build([_row(by_temperature), _row(by_top_k)], device="cpu")

        assert batch.temperature.tolist() == [0.0, 0.0]
        assert batch.top_k.tolist() == [0, 0]
        assert batch.top_p.tolist() == [1.0, 1.0]
        assert batch.min_p.tolist() == [0.0, 0.0]
        assert batch.all_greedy

    def test_all_greedy_is_false_as_soon_as_one_row_samples(self) -> None:
        assert not SamplingBatch.build([_row(GREEDY), _row(RANDOM)], device="cpu").all_greedy

    def test_padding_adds_greedy_rows_at_position_zero(self) -> None:
        batch = SamplingBatch.build([_row(RANDOM, seed=5, position=8)], device="cpu", size=4)

        assert batch.size == 4
        assert batch.seeds.tolist() == [5, 0, 0, 0]
        assert batch.positions.tolist() == [8, 0, 0, 0]
        assert batch.temperature[1:].tolist() == [0.0, 0.0, 0.0]
        assert batch.temperature[0].item() == pytest.approx(0.8)

    def test_padding_alone_does_not_stop_the_greedy_path(self) -> None:
        batch = SamplingBatch.build([_row(GREEDY)], device="cpu", size=4)

        assert batch.all_greedy

    def test_a_size_below_the_rows_is_refused(self) -> None:
        with pytest.raises(ValueError, match="size 1 is smaller than the 2 rows"):
            SamplingBatch.build([_row(), _row()], device="cpu", size=1)

    def test_an_empty_batch_is_valid(self) -> None:
        batch = SamplingBatch.build([], device="cpu")

        assert batch.size == 0
        assert batch.all_greedy

    def test_the_batch_is_immutable(self) -> None:
        batch = SamplingBatch.build([_row()], device="cpu")

        with pytest.raises(dataclasses.FrozenInstanceError):
            batch.all_greedy = True  # pyright: ignore[reportAttributeAccessIssue]

    def test_device_is_that_of_the_tensors(self) -> None:
        assert SamplingBatch.build([_row()], device="cpu").device == torch.device("cpu")


class TestSampler:
    def test_greedy_batch_takes_the_argmax_path_without_drawing(self) -> None:
        calls: list[str] = []
        real = _ops()

        class Spy:
            def argmax(self, logits: Any, out: Any) -> None:
                calls.append("argmax")
                real.argmax(logits, out)

            def uniform(self, *args: Any, **kwargs: Any) -> None:
                calls.append("uniform")

            def sample(self, *args: Any, **kwargs: Any) -> None:
                calls.append("sample")

        logits = torch.tensor([[0.0, 3.0, 1.0], [2.0, 2.0, 0.0]])
        batch = SamplingBatch.build([_row(GREEDY), _row(SamplingParams(top_k=1))], device="cpu")
        out = torch.empty(2, dtype=torch.int64)

        Sampler(Spy()).sample(logits, batch, out)  # pyright: ignore[reportArgumentType]

        assert calls == ["argmax"]
        assert out.tolist() == [1, 0]

    def test_sampled_batch_draws_from_the_pick_stream_at_the_row_positions(self) -> None:
        seen: dict[str, Any] = {}
        real = _ops()

        class Spy:
            def argmax(self, logits: Any, out: Any) -> None:
                raise AssertionError("not the greedy path")

            def uniform(
                self, seeds: Any, positions: Any, sequences: Any, out: Any, *, stream: int
            ) -> None:
                seen["stream"] = stream
                real.uniform(seeds, positions, sequences, out, stream=stream)
                seen["u"] = out.clone()

            def sample(
                self,
                logits: Any,
                temperature: Any,
                top_k: Any,
                top_p: Any,
                min_p: Any,
                u: Any,
                out: Any,
                *,
                kept_logprobs: Any = None,
            ) -> None:
                real.sample(logits, temperature, top_k, top_p, min_p, u, out)

        rows = [_row(seed=7, position=30), _row(seed=8, position=31, sequence=2)]
        batch = SamplingBatch.build(rows, device="cpu")

        Sampler(Spy()).sample(torch.randn(2, VOCAB), batch, torch.empty(2, dtype=torch.int64))  # pyright: ignore[reportArgumentType]

        assert seen["stream"] == DrawStream.PICK
        assert seen["u"].tolist() == [draw_uniform(7, 30), draw_uniform(8, 31, sequence=2)]

    def test_a_sampled_token_is_the_inverse_cdf_of_the_drawn_number(self) -> None:
        logits = torch.log(torch.tensor([[0.5, 0.3, 0.15, 0.05]] * 40))
        rows = [_row(SamplingParams(temperature=1.0), seed=21, position=p) for p in range(40)]

        tokens = _run(rows, logits)

        for position, token in enumerate(tokens):
            u = draw_uniform(21, position)
            if min(abs(u - edge) for edge in (0.5, 0.8, 0.95)) < 1e-5:
                continue
            assert token == next(i for i, edge in enumerate((0.5, 0.8, 0.95, 1.0)) if u < edge)

    def test_the_same_row_gives_the_same_token_in_any_batch(self) -> None:
        requests = [(seed, position) for seed in (1, 2, 3) for position in (5, 6, 90)]
        table = {request: _logits_of(request[0] * 1000 + request[1]) for request in requests}

        def tokens_of(
            order: list[tuple[int, int]], size: int | None = None
        ) -> dict[tuple[int, int], int]:
            rows = [_row(RANDOM, seed=s, position=p) for s, p in order]
            logits = torch.stack([table[request] for request in order])
            if size is not None:  # a fixed-shape step carries logits for the padding rows too
                logits = torch.cat([logits, torch.zeros(size - len(order), VOCAB)])
            result = _run(rows, logits, size=size)
            return dict(zip(order, result, strict=False))  # the padding rows' tokens are dropped

        reference = tokens_of(requests)
        assert tokens_of(list(reversed(requests))) == reference
        assert tokens_of(requests[3:7]) == {k: reference[k] for k in requests[3:7]}
        assert tokens_of(requests[:1]) == {requests[0]: reference[requests[0]]}
        padded = tokens_of(requests, size=16)
        assert padded == reference

    def test_a_row_in_a_speculative_window_draws_what_plain_decoding_would(self) -> None:
        # Plain decoding of position 12 is one row; a verify window has rows for 10, 11, 12, 13.
        logits_12 = _logits_of(12)
        alone = _run([_row(RANDOM, seed=9, position=12)], logits_12[None])

        window_logits = torch.stack([_logits_of(10), _logits_of(11), logits_12, _logits_of(13)])
        window = _run([_row(RANDOM, seed=9, position=p) for p in (10, 11, 12, 13)], window_logits)

        assert window[2] == alone[0]

    def test_rows_of_one_request_use_their_sequence_index(self) -> None:
        logits = torch.zeros(1, VOCAB)
        params = SamplingParams(temperature=1.0)
        tokens = {
            _run([_row(params, seed=4, position=3, sequence=index)], logits)[0]
            for index in range(24)
        }

        assert len(tokens) > 4  # n sequences of one request are not copies of each other

    def test_a_run_repeats_exactly(self) -> None:
        rows = [_row(RANDOM, seed=s, position=s * 3) for s in range(8)]
        logits = torch.stack([_logits_of(s) for s in range(8)])

        assert _run(rows, logits) == _run(rows, logits)

    def test_mixed_batch_greedy_rows_ignore_the_draw(self) -> None:
        logits = torch.stack([_logits_of(1), _logits_of(2)])
        rows = [_row(GREEDY, seed=1), _row(RANDOM, seed=2)]

        tokens = _run(rows, logits)

        assert tokens[0] == int(logits[0].argmax())

    def test_a_row_count_that_disagrees_with_the_batch_is_refused(self) -> None:
        batch = SamplingBatch.build([_row(), _row()], device="cpu")

        with pytest.raises(ValueError, match=r"logits must be \[2, vocab\]"):
            Sampler(_ops()).sample(torch.zeros(3, VOCAB), batch, torch.empty(2, dtype=torch.int64))

    def test_unorderable_logits_reach_the_caller_as_a_backend_error(self) -> None:
        logits = torch.full((1, VOCAB), math.nan)

        with pytest.raises(BackendError, match="needs a finite value"):
            _run([_row()], logits)
