"""The counter-based random numbers: published Philox vectors and what the engine relies on.

Known-answer vectors: the ``philox4x32 10`` lines of ``tests/kat_vectors`` in the Random123 library
(D. E. Shaw Research, https://github.com/DEShawResearch/random123), each line being
``counter(4 words) key(2 words) expected output(4 words)``, in hexadecimal.

The distribution test is deterministic: the seed is fixed, so it passes or fails identically on
every run. The tolerance is there so that a legitimate change to the draw layout is judged by
statistics, not by pinned digits. For ``n`` independent uniform draws the mean has standard
deviation ``1 / sqrt(12 n)`` and the frequency of one of ``b`` equal bins ``sqrt(p (1 - p) / n)``
with ``p = 1 / b``; the bounds below are five of those deviations, whose chance of being exceeded
by a correct generator is below one in a million per statistic.
"""

from __future__ import annotations

import math

import pytest

from ayaka.ops.sampling import DrawStream
from ayaka.sampling.rng import MAX_SEED, MAX_SEQUENCE, draw_uniform, philox4x32_10, resolve_seed

pytestmark = pytest.mark.unit

#: (counter, key, expected output) from Random123 ``tests/kat_vectors``.
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


class TestPhilox:
    @pytest.mark.parametrize(("counter", "key", "expected"), RANDOM123_VECTORS)
    def test_matches_the_random123_known_answers(
        self,
        counter: tuple[int, int, int, int],
        key: tuple[int, int],
        expected: tuple[int, int, int, int],
    ) -> None:
        assert philox4x32_10(counter, key) == expected

    @pytest.mark.parametrize("word", [-1, 1 << 32])
    def test_refuses_a_word_outside_32_bits(self, word: int) -> None:
        with pytest.raises(ValueError, match="32"):
            philox4x32_10((word, 0, 0, 0), (0, 0))


class TestDrawUniform:
    def test_is_the_top_24_bits_of_the_first_output_word(self) -> None:
        # Counter (0, 0, 0, 0) and key (0, 0) is the first Random123 vector: 0x6627E8D5.
        assert draw_uniform(0, 0) == 0x6627E8 / 2**24

    def test_position_and_seed_use_both_words(self) -> None:
        position = (7 << 32) | 5
        seed = (3 << 32) | 9
        words = philox4x32_10((5, 7, int(DrawStream.PICK), 0), (9, 3))

        assert draw_uniform(seed, position) == (words[0] >> 8) / 2**24

    def test_stream_and_sequence_select_the_counter_words(self) -> None:
        words = philox4x32_10((12, 0, int(DrawStream.RESID), 4), (99, 0))

        assert draw_uniform(99, 12, stream=DrawStream.RESID, sequence=4) == (words[0] >> 8) / 2**24

    def test_is_a_pure_function(self) -> None:
        assert draw_uniform(42, 1000) == draw_uniform(42, 1000)

    def test_stream_values_are_fixed(self) -> None:
        # Recorded outputs of seeded requests depend on these values; changing one is a break.
        assert [int(stream) for stream in DrawStream] == [0, 1, 2, 3]

    def test_values_are_in_the_unit_interval_with_24_bits(self) -> None:
        for position in range(500):
            value = draw_uniform(1, position)
            assert 0.0 <= value < 1.0
            assert value * 2**24 == int(value * 2**24)

    def test_streams_sequences_positions_and_seeds_give_different_numbers(self) -> None:
        base = draw_uniform(5, 100)

        assert draw_uniform(5, 100, stream=DrawStream.ACCEPT) != base
        assert draw_uniform(5, 100, stream=DrawStream.RESID) != base
        assert draw_uniform(5, 100, stream=DrawStream.DRAFT) != base
        assert draw_uniform(5, 100, sequence=1) != base
        assert draw_uniform(5, 101) != base
        assert draw_uniform(6, 100) != base

    @pytest.mark.parametrize(
        ("seed", "position", "sequence"),
        [
            (-1, 0, 0),
            (MAX_SEED + 1, 0, 0),
            (0, -1, 0),
            (0, MAX_SEED + 1, 0),
            (0, 0, -1),
            (0, 0, MAX_SEQUENCE + 1),
        ],
    )
    def test_refuses_out_of_range_arguments(self, seed: int, position: int, sequence: int) -> None:
        with pytest.raises(ValueError, match="must be in"):
            draw_uniform(seed, position, sequence=sequence)

    def test_the_largest_arguments_are_accepted(self) -> None:
        assert 0.0 <= draw_uniform(MAX_SEED, MAX_SEED, sequence=MAX_SEQUENCE) < 1.0

    def test_mean_and_bins_of_a_fixed_seed_look_uniform(self) -> None:
        count, bins = 20_000, 8
        draws = [draw_uniform(2024, position) for position in range(count)]

        mean_bound = 5 / math.sqrt(12 * count)
        bin_bound = 5 * math.sqrt((1 / bins) * (1 - 1 / bins) / count)
        assert abs(sum(draws) / count - 0.5) < mean_bound
        for index in range(bins):
            share = sum(index / bins <= value < (index + 1) / bins for value in draws) / count
            assert abs(share - 1 / bins) < bin_bound

    def test_nearby_seeds_are_uncorrelated_on_the_same_position(self) -> None:
        count = 20_000
        first = [draw_uniform(1, position) for position in range(count)]
        second = [draw_uniform(2, position) for position in range(count)]
        mean_first, mean_second = sum(first) / count, sum(second) / count
        covariance = sum(
            (a - mean_first) * (b - mean_second) for a, b in zip(first, second, strict=True)
        ) / (count * (1 / 12))

        # The correlation of independent uniforms has standard deviation 1 / sqrt(n).
        assert abs(covariance) < 5 / math.sqrt(count)


class TestResolveSeed:
    def test_a_given_seed_is_used_as_it_is(self) -> None:
        assert resolve_seed("req-1", 1234) == 1234
        assert resolve_seed("req-1", 0) == 0

    def test_without_a_seed_it_is_derived_from_the_request_id(self) -> None:
        assert resolve_seed("req-1", None) == resolve_seed("req-1", None)
        assert resolve_seed("req-1", None) != resolve_seed("req-2", None)

    def test_the_derived_seed_is_stable_across_processes(self) -> None:
        # Pinned: it is reported to the client so the request can be repeated, so it must not
        # depend on ``hash`` randomisation or on the Python version.
        # The value is blake2b(digest_size=8) of the UTF-8 id, as a big-endian integer, masked
        # to 63 bits; it was computed once by running the function.
        assert resolve_seed("req-1", None) == 0x044754DE574A870B

    def test_a_seed_out_of_range_is_refused(self) -> None:
        with pytest.raises(ValueError, match="seed must be in"):
            resolve_seed("req-1", MAX_SEED + 1)
