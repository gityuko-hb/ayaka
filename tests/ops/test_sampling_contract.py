"""Sampling ops on the torch backend: Philox on tensors, argmax and the inverse-CDF sample.

The oracle for ``sample`` is :func:`_naive_sample`, a literal float64 transcription of the
algorithm in ``ayaka.ops.sampling`` over Python lists: sort by ``(p descending, id ascending)``,
cut, walk the cumulative sum. The op computes in fp32, so its probabilities differ from the
oracle's by about ``1e-7`` relative; a draw whose ``u`` is within ``BOUNDARY`` of a CDF boundary
could legitimately go either way and is not compared.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

import pytest

from ayaka.core.errors import BackendError
from ayaka.ops.backend import load_backend
from ayaka.ops.sampling import DrawStream
from ayaka.sampling.rng import draw_uniform, philox4x32_10
from ayaka.utils.dtypes import DType

torch: Any = pytest.importorskip("torch")

from ayaka.ops.backends.torch.sampling import philox_rounds  # noqa: E402

if TYPE_CHECKING:
    from ayaka.ops.sampling import SamplingOps

pytestmark = pytest.mark.unit

BOUNDARY = 1e-5

#: (counter, key, expected output): the ``philox4x32 10`` lines of ``tests/kat_vectors`` in
#: Random123 (https://github.com/DEShawResearch/random123); also in tests/sampling/test_rng.py.
RANDOM123_VECTORS = [
    (
        (0x00000000, 0x00000000, 0x00000000, 0x00000000),
        (0x00000000, 0x00000000),
        (0x6627E8D5, 0xE169C58D, 0xBC57AC4C, 0x9B00DBD8),
    ),
    (
        (0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF),
        (0xFFFFFFFF, 0xFFFFFFFF),
        (0x408F276D, 0x41C83B0E, 0xA20BC7C6, 0x6D5451FD),
    ),
    (
        (0x243F6A88, 0x85A308D3, 0x13198A2E, 0x03707344),
        (0xA4093822, 0x299F31D0),
        (0xD16CFE09, 0x94FDCCEB, 0x5001E420, 0x24126EA1),
    ),
]


def _ops() -> SamplingOps:
    return load_backend(["torch"], device="cpu", dtype=DType.FP32).sampling


def _tensor(values: Any, dtype: Any) -> Any:
    return torch.tensor(values, dtype=dtype)


def _uniform(
    seeds: list[int],
    positions: list[int],
    sequences: list[int] | None = None,
    *,
    stream: int = DrawStream.PICK,
) -> Any:
    out = torch.empty(len(seeds), dtype=torch.float32)
    _ops().uniform(
        _tensor(seeds, torch.int64),
        _tensor(positions, torch.int64),
        _tensor(sequences or [0] * len(seeds), torch.int64),
        out,
        stream=stream,
    )
    return out


def _sample(
    logits: Any,
    u: list[float],
    *,
    temperature: list[float] | float = 1.0,
    top_k: list[int] | int = 0,
    top_p: list[float] | float = 1.0,
    min_p: list[float] | float = 0.0,
) -> list[int]:
    rows = logits.shape[0]

    def column(value: Any, dtype: Any) -> Any:
        return _tensor(value if isinstance(value, list) else [value] * rows, dtype)

    out = torch.empty(rows, dtype=torch.int64)
    _ops().sample(
        logits,
        column(temperature, torch.float32),
        column(top_k, torch.int32),
        column(top_p, torch.float32),
        column(min_p, torch.float32),
        _tensor(u, torch.float32),
        out,
    )
    return out.tolist()


def _argmax(logits: Any) -> list[int]:
    out = torch.empty(logits.shape[0], dtype=torch.int64)
    _ops().argmax(logits, out)
    return out.tolist()


def _naive_cdf(
    row: list[float], temperature: float, top_k: int, top_p: float, min_p: float
) -> list[tuple[int, float, float]]:
    """Return ``(token id, cdf before, cdf after)`` in canonical order, as fractions."""
    scaled = [value / temperature for value in row]
    by_logit = sorted(range(len(row)), key=lambda token: (-scaled[token], token))
    kept = by_logit[:top_k] if top_k else by_logit
    peak = max(scaled[token] for token in kept)
    weights = {token: math.exp(scaled[token] - peak) for token in kept}
    total = sum(weights.values())
    probability = {token: weight / total for token, weight in weights.items()}
    order = sorted(kept, key=lambda token: (-probability[token], token))
    chosen: list[int] = []
    running = 0.0
    for token in order:
        if running < top_p or not chosen:
            chosen.append(token)
        running += probability[token]
    largest = probability[order[0]]
    chosen = [token for token in chosen if probability[token] >= min_p * largest]
    total = sum(probability[token] for token in chosen)
    cdf: list[tuple[int, float, float]] = []
    running = 0.0
    for token in chosen:
        cdf.append((token, running / total, (running + probability[token]) / total))
        running += probability[token]
    return cdf


def _naive_sample(
    row: list[float], u: float, temperature: float, top_k: int, top_p: float, min_p: float
) -> int | None:
    """The oracle's token, or ``None`` when ``u`` is too close to a boundary to compare."""
    cdf = _naive_cdf(row, temperature, top_k, top_p, min_p)
    if any(abs(u - upper) < BOUNDARY for _, _, upper in cdf):
        return None
    return next(token for token, _, upper in cdf if u < upper)


class TestPhiloxOnTensors:
    @pytest.mark.parametrize(("counter", "key", "expected"), RANDOM123_VECTORS)
    def test_matches_the_random123_known_answers(
        self, counter: tuple[int, ...], key: tuple[int, ...], expected: tuple[int, ...]
    ) -> None:
        words = philox_rounds(
            tuple(_tensor([word], torch.int64) for word in counter),  # pyright: ignore[reportArgumentType]
            tuple(_tensor([word], torch.int64) for word in key),  # pyright: ignore[reportArgumentType]
        )

        assert tuple(int(word[0]) for word in words) == expected

    def test_matches_the_pure_python_implementation_on_random_words(self) -> None:
        generator = torch.Generator().manual_seed(7)
        words = torch.randint(0, 2**32, (6, 512), generator=generator, dtype=torch.int64)

        actual = philox_rounds(tuple(words[:4]), tuple(words[4:]))  # pyright: ignore[reportArgumentType]

        for column in range(512):
            counter = tuple(int(words[row, column]) for row in range(4))
            key = tuple(int(words[row, column]) for row in (4, 5))
            expected = philox4x32_10(counter, key)  # pyright: ignore[reportArgumentType]
            assert tuple(int(word[column]) for word in actual) == expected


class TestUniform:
    def test_matches_the_host_reference(self) -> None:
        seeds = [0, 1, 2**40 + 5, 2**63 - 1]
        positions = [0, 17, 2**35 + 3, 2**63 - 1]
        sequences = [0, 1, 2, 2**32 - 1]

        actual = _uniform(seeds, positions, sequences)

        expected = [
            draw_uniform(seed, position, sequence=sequence)
            for seed, position, sequence in zip(seeds, positions, sequences, strict=True)
        ]
        assert actual.tolist() == expected

    @pytest.mark.parametrize("stream", list(DrawStream))
    def test_every_stream_matches_the_host_reference(self, stream: DrawStream) -> None:
        actual = _uniform([9, 9], [4, 5], stream=stream)

        assert actual.tolist() == [
            draw_uniform(9, 4, stream=stream),
            draw_uniform(9, 5, stream=stream),
        ]

    def test_a_row_does_not_depend_on_the_batch_around_it(self) -> None:
        seeds = [11, 12, 13, 14, 15]
        positions = [100, 101, 102, 103, 104]
        together = _uniform(seeds, positions)

        for index in range(5):
            assert _uniform([seeds[index]], [positions[index]]).item() == together[index].item()
        order = [3, 0, 4, 1, 2]
        shuffled = _uniform([seeds[i] for i in order], [positions[i] for i in order])
        assert shuffled.tolist() == [together[i].item() for i in order]

    def test_values_are_in_the_unit_interval(self) -> None:
        values = _uniform([3] * 1000, list(range(1000)))

        assert float(values.min()) >= 0.0
        assert float(values.max()) < 1.0

    @pytest.mark.parametrize("argument", ["seeds", "positions", "sequences"])
    def test_refuses_a_negative_argument(self, argument: str) -> None:
        arguments: dict[str, Any] = {
            "seeds": _tensor([1], torch.int64),
            "positions": _tensor([1], torch.int64),
            "sequences": _tensor([1], torch.int32),
        }
        arguments[argument] = arguments[argument] * -1

        with pytest.raises(BackendError, match=f"{argument} must not be negative"):
            _ops().uniform(**arguments, out=torch.empty(1, dtype=torch.float32), stream=0)

    def test_refuses_a_sequence_that_does_not_fit_a_word(self) -> None:
        with pytest.raises(BackendError, match="sequences must be below 2"):
            _ops().uniform(
                _tensor([1], torch.int64),
                _tensor([1], torch.int64),
                _tensor([2**32], torch.int64),
                torch.empty(1, dtype=torch.float32),
                stream=0,
            )

    def test_refuses_an_unknown_stream(self) -> None:
        with pytest.raises(BackendError, match="stream must be a DrawStream"):
            _uniform([1], [1], stream=7)

    def test_refuses_the_wrong_dtypes_and_shapes(self) -> None:
        with pytest.raises(BackendError, match="out must be float32"):
            _ops().uniform(
                _tensor([1], torch.int64),
                _tensor([1], torch.int64),
                _tensor([0], torch.int32),
                torch.empty(1, dtype=torch.float64),
                stream=0,
            )
        with pytest.raises(BackendError, match="seeds must be int64"):
            _ops().uniform(
                _tensor([1], torch.int32),
                _tensor([1], torch.int64),
                _tensor([0], torch.int32),
                torch.empty(1, dtype=torch.float32),
                stream=0,
            )


class TestArgmax:
    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
    def test_picks_the_largest_logit(self, dtype: Any) -> None:
        logits = _tensor([[0.0, 3.0, 1.0], [-1.0, -2.0, -0.5]], dtype)

        assert _argmax(logits) == [1, 2]

    def test_a_tie_goes_to_the_smallest_id(self) -> None:
        logits = _tensor(
            [[1.0, 5.0, 5.0, 5.0], [7.0, 7.0, 0.0, 7.0], [2.0, 2.0, 2.0, 2.0]], torch.float32
        )

        assert _argmax(logits) == [1, 0, 0]

    def test_masked_tokens_are_never_chosen(self) -> None:
        logits = _tensor([[-math.inf, -3.0, -math.inf]], torch.float32)

        assert _argmax(logits) == [1]

    @pytest.mark.parametrize(
        "row",
        [[math.nan, 1.0], [math.inf, 1.0], [-math.inf, -math.inf]],
        ids=["nan", "inf", "no-finite"],
    )
    def test_refuses_logits_it_cannot_order(self, row: list[float]) -> None:
        with pytest.raises(BackendError, match="needs a finite value"):
            _argmax(_tensor([row], torch.float32))

    def test_refuses_logits_that_are_not_two_dimensional_or_not_floating(self) -> None:
        with pytest.raises(BackendError, match=r"logits must be \[rows, vocab\]"):
            _argmax(_tensor([1.0, 2.0], torch.float32))
        with pytest.raises(BackendError, match="logits is"):
            _argmax(_tensor([[1, 2]], torch.int64))


class TestSample:
    def test_a_greedy_row_returns_the_argmax_whatever_u_is(self) -> None:
        logits = _tensor([[1.0, 4.0, 4.0, 0.0]] * 3, torch.float32)

        assert _sample(logits, [0.0, 0.5, 0.99], temperature=0.0, top_p=0.3) == [1, 1, 1]

    def test_equal_logits_are_ordered_by_id_so_u_picks_by_position(self) -> None:
        vocab = 8
        logits = torch.zeros(vocab, vocab)
        u = [(index + 0.5) / vocab for index in range(vocab)]

        assert _sample(logits, u) == list(range(vocab))

    def test_top_k_keeps_exactly_k_and_prefers_smaller_ids_among_ties(self) -> None:
        logits = torch.zeros(3, 6)

        assert _sample(logits, [0.0, 0.5, 0.999], top_k=3) == [0, 1, 2]

    def test_top_k_keeps_the_largest(self) -> None:
        logits = _tensor([[0.0, 5.0, 1.0, 4.0, 2.0]] * 2, torch.float32)

        # Kept: ids 1 and 3, with p ~ 0.73 and 0.27.
        assert _sample(logits, [0.1, 0.99], top_k=2) == [1, 3]

    def test_top_p_keeps_the_shortest_prefix_reaching_the_mass(self) -> None:
        logits = torch.log(_tensor([[0.5, 0.3, 0.15, 0.05]], torch.float32))

        assert _sample(logits, [0.9]) == [2]  # without top-p the cdf is .5, .8, .95
        assert _sample(logits, [0.9], top_p=0.79) == [1]  # kept {0, 1}
        assert _sample(logits, [0.9], top_p=0.81) == [2]  # kept {0, 1, 2}
        assert _sample(logits, [0.99], top_p=0.01) == [0]  # always at least one element

    def test_min_p_drops_tokens_below_a_fraction_of_the_largest(self) -> None:
        logits = torch.log(_tensor([[0.5, 0.3, 0.15, 0.05]], torch.float32))

        assert _sample(logits, [0.99], min_p=0.09) == [3]  # threshold .045: all kept
        assert _sample(logits, [0.99], min_p=0.2) == [2]  # threshold .1: {0, 1, 2}
        assert _sample(logits, [0.99], min_p=0.4) == [1]  # threshold .2: {0, 1}

    def test_the_order_of_the_filters_is_top_k_then_top_p_then_min_p(self) -> None:
        logits = torch.log(_tensor([[0.4, 0.3, 0.2, 0.1]], torch.float32))

        # top_k=3 leaves {.4,.3,.2}/.9 = {.444,.333,.222}; top_p=.7 then keeps {0, 1}.
        assert _sample(logits, [0.99], top_k=3, top_p=0.7) == [1]
        # min_p=.6 against the largest (.4): threshold .24 keeps {0, 1}.
        assert _sample(logits, [0.99], min_p=0.6) == [1]

    def test_temperature_scales_the_logits(self) -> None:
        logits = _tensor([[0.0, 1.0]], torch.float32)

        # Token 1 comes first in the canonical order; its mass is 0.731 at T=1 and 0.525 at T=10.
        assert _sample(logits, [0.6], temperature=1.0) == [1]
        assert _sample(logits, [0.6], temperature=10.0) == [0]

    def test_mixed_greedy_and_sampled_rows(self) -> None:
        logits = _tensor([[0.0, 2.0, 2.0]] * 2, torch.float32)

        result = _sample(logits, [0.9, 0.9], temperature=[0.0, 1.0])

        assert result[0] == 1
        assert result[1] == 2

    def test_a_row_does_not_depend_on_the_rows_around_it(self) -> None:
        generator = torch.Generator().manual_seed(3)
        logits = torch.randn(6, 40, generator=generator)
        u = [0.05, 0.3, 0.55, 0.7, 0.85, 0.97]
        together = _sample(logits, u, top_k=10, top_p=0.9, min_p=0.01)

        alone = [
            _sample(logits[i : i + 1], [u[i]], top_k=10, top_p=0.9, min_p=0.01)[0] for i in range(6)
        ]
        assert alone == together

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_accepts_half_precision_logits(self, dtype: Any) -> None:
        logits = _tensor([[0.0, 2.0, 1.0]], dtype)

        assert _sample(logits, [0.5], temperature=0.0) == [1]
        assert _sample(logits, [0.5]) in ([0], [1], [2])

    def test_u_near_one_never_leaves_the_kept_set(self) -> None:
        logits = _tensor([[3.0, 2.0, 1.0, 0.0, -50.0]], torch.float32)
        top = math.nextafter(1.0, 0.0)

        assert _sample(logits, [float(torch.tensor(top, dtype=torch.float32))], top_k=3) == [2]

    def test_masked_tokens_are_never_drawn(self) -> None:
        logits = _tensor([[-math.inf, 0.0, -math.inf, 0.0]], torch.float32)

        assert _sample(logits, [0.0]) == [1]
        assert _sample(logits, [0.99]) == [3]

    def test_matches_the_naive_float64_oracle(self) -> None:
        generator = torch.Generator().manual_seed(2024)
        settings = [
            (1.0, 0, 1.0, 0.0),
            (0.7, 0, 1.0, 0.0),
            (1.3, 5, 1.0, 0.0),
            (1.0, 0, 0.8, 0.0),
            (1.0, 0, 1.0, 0.05),
            (0.9, 12, 0.9, 0.02),
        ]
        compared = 0
        for temperature, top_k, top_p, min_p in settings:
            logits = torch.randn(64, 50, generator=generator) * 2.0
            u = torch.rand(64, generator=generator).tolist()
            actual = _sample(
                logits,
                u,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                min_p=min_p,
            )
            rows = logits.double().tolist()
            for row, draw, token in zip(rows, u, actual, strict=True):
                # ``u`` is rounded to fp32 on the way in; the oracle sees the same value.
                rounded = float(torch.tensor(draw, dtype=torch.float32))
                expected = _naive_sample(row, rounded, temperature, top_k, top_p, min_p)
                if expected is not None:
                    compared += 1
                    assert token == expected
        assert compared > 350  # of 384, the rest sat on a boundary

    def test_frequencies_follow_the_distribution(self) -> None:
        """Draws at 20 000 positions of a fixed seed, against the exact probabilities.

        Deterministic (fixed seed); the bound is five standard deviations of a frequency,
        ``sqrt(p (1 - p) / n) <= 0.5 / sqrt(n)``, so a correct sampler fails with probability
        below one in a million for any seed, and a wrong CDF (an off-by-one in the order, a
        missing renormalisation) is off by far more.
        """
        count = 20_000
        probabilities = [0.4, 0.3, 0.2, 0.1]
        logits = torch.log(_tensor([probabilities], torch.float32)).expand(count, 4).contiguous()
        u = _uniform([77] * count, list(range(count)))

        tokens = _sample(logits, u.tolist())

        for token, probability in enumerate(probabilities):
            frequency = tokens.count(token) / count
            bound = 5 * math.sqrt(probability * (1 - probability) / count)
            assert abs(frequency - probability) < bound

    def test_frequencies_follow_the_filtered_distribution(self) -> None:
        count = 20_000
        logits = torch.log(_tensor([[0.4, 0.3, 0.2, 0.1]], torch.float32))
        logits = logits.expand(count, 4).contiguous()
        u = _uniform([5] * count, list(range(count)))

        tokens = _sample(logits, u.tolist(), top_p=0.65)  # kept {0, 1}: 4/7 and 3/7

        assert set(tokens) == {0, 1}
        for token, probability in ((0, 4 / 7), (1, 3 / 7)):
            bound = 5 * math.sqrt(probability * (1 - probability) / count)
            assert abs(tokens.count(token) / count - probability) < bound

    def test_refuses_unorderable_logits_and_wrong_parameter_types(self) -> None:
        with pytest.raises(BackendError, match="needs a finite value"):
            _sample(_tensor([[math.nan, 0.0]], torch.float32), [0.5])
        with pytest.raises(BackendError, match="top_k must be int32"):
            _ops().sample(
                torch.zeros(1, 3),
                _tensor([1.0], torch.float32),
                _tensor([0], torch.int64),
                _tensor([1.0], torch.float32),
                _tensor([0.0], torch.float32),
                _tensor([0.5], torch.float32),
                torch.empty(1, dtype=torch.int64),
            )
        with pytest.raises(BackendError, match="out must be int64"):
            _ops().sample(
                torch.zeros(1, 3),
                _tensor([1.0], torch.float32),
                _tensor([0], torch.int32),
                _tensor([1.0], torch.float32),
                _tensor([0.0], torch.float32),
                _tensor([0.5], torch.float32),
                torch.empty(1, dtype=torch.int32),
            )
