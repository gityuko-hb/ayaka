"""The logit transforms of the sampling contract against formulas written out element by element."""

from __future__ import annotations

import math
from typing import Any

import pytest

from ayaka.core.errors import BackendError

torch: Any = pytest.importorskip("torch")

from ayaka.ops.backends.torch import sampling as ops  # noqa: E402

pytestmark = pytest.mark.unit

VOCAB = 70  # not a multiple of 32, so the last mask word is partial


def _words(allowed: list[set[int]]) -> Any:
    rows = []
    for ids in allowed:
        words = [0] * ((VOCAB + 31) // 32)
        for token in ids:
            words[token // 32] |= 1 << (token % 32)
        rows.append([w - (1 << 32) if w >= 1 << 31 else w for w in words])
    return torch.tensor(rows, dtype=torch.int32)


class TestBitmask:
    def test_cleared_bits_become_minus_infinity_and_set_bits_stay(self) -> None:
        logits = torch.randn(2, VOCAB)
        before = logits.clone()
        allowed = [{0, 31, 32, 63, 69}, set(range(VOCAB))]

        ops.apply_token_bitmask(logits, _words(allowed))

        for row, ids in enumerate(allowed):
            for token in range(VOCAB):
                if token in ids:
                    assert logits[row, token] == before[row, token]
                else:
                    assert logits[row, token] == -math.inf

    @pytest.mark.parametrize("vocab", [1, 31, 32, 33, 70, 256, 1000, 4096])
    def test_random_masks_keep_exactly_their_set_bits(self, vocab: int) -> None:
        """Token ``t`` is bit ``t % 32`` of word ``t // 32``, the sign bit included."""
        generator = torch.Generator().manual_seed(vocab)
        words = (vocab + 31) // 32
        rows = 5
        bitmask = torch.randint(
            -(2**31), 2**31 - 1, (rows, words), generator=generator, dtype=torch.int32
        )
        bitmask[0] = -1  # every token allowed, bit 31 of every word set
        bitmask[1] = 0  # none allowed
        bitmask[2, 0] = -(2**31)  # only bit 31 of the first word
        logits = torch.randn(rows, vocab, generator=generator)
        before = logits.clone()

        ops.apply_token_bitmask(logits, bitmask)

        for row in range(rows):
            for token in range(vocab):
                bit = (int(bitmask[row, token // 32]) >> (token % 32)) & 1  # an arithmetic shift
                if bit:
                    assert logits[row, token] == before[row, token]
                else:
                    assert logits[row, token] == -math.inf

    def test_a_mask_that_is_a_slice_of_a_larger_buffer_is_read_row_by_row(self) -> None:
        """The graph's buffer is one tensor read by every bucket as its first rows."""
        buffer = torch.full((4, 3), -1, dtype=torch.int32)
        buffer[1] = torch.tensor([1, 0, 0], dtype=torch.int32)
        logits = torch.zeros(2, 96)

        ops.apply_token_bitmask(logits, buffer[:2])

        assert bool(torch.isfinite(logits[0]).all())
        assert int(torch.isfinite(logits[1]).sum()) == 1
        assert logits[1, 0] == 0

    def test_the_workspace_must_be_float32(self) -> None:
        with pytest.raises(BackendError, match="float32 workspace"):
            ops.apply_token_bitmask(torch.zeros(1, VOCAB, dtype=torch.float16), _words([{1}]))


class TestBias:
    def test_biases_add_to_their_cells_only_and_a_mask_survives(self) -> None:
        logits = torch.zeros(2, VOCAB)
        logits[1, 5] = -math.inf

        ops.apply_logit_bias(
            logits,
            torch.tensor([0, 1, 1]),
            torch.tensor([3, 5, 7]),
            torch.tensor([2.5, 100.0, -1.0]),
        )

        assert logits[0, 3] == 2.5
        assert logits[1, 5] == -math.inf
        assert logits[1, 7] == -1.0
        assert int((logits != 0).sum()) == 3


def _penalised(x: float, seen: int, out: int, r: float, f: float, s: float) -> float:
    if seen > 0:
        x = x / r if x > 0 else x * r
    x -= f * out
    if out > 0:
        x -= s
    return x


class TestPenalties:
    def test_each_cell_follows_the_three_formulas(self) -> None:
        generator = torch.Generator().manual_seed(3)
        logits = torch.randn(3, VOCAB, generator=generator) * 4
        logits[2, 0] = -math.inf
        seen = torch.randint(0, 3, (3, VOCAB), generator=generator, dtype=torch.int32)
        out = torch.minimum(seen, torch.randint(0, 2, (3, VOCAB), generator=generator).int())
        r = torch.tensor([1.3, 1.0, 0.8])
        f = torch.tensor([0.0, 0.5, 0.2])
        s = torch.tensor([0.4, 0.0, 1.0])
        expected = [
            [
                _penalised(
                    float(logits[i, t]),
                    int(seen[i, t]),
                    int(out[i, t]),
                    float(r[i]),
                    float(f[i]),
                    float(s[i]),
                )
                for t in range(VOCAB)
            ]
            for i in range(3)
        ]

        ops.apply_penalties(logits, seen, out, r, f, s)

        torch.testing.assert_close(logits, torch.tensor(expected), rtol=1e-6, atol=1e-6)
        assert logits[2, 0] == -math.inf

    def test_a_neutral_row_is_unchanged(self) -> None:
        logits = torch.randn(1, VOCAB)
        before = logits.clone()
        counts = torch.ones(1, VOCAB, dtype=torch.int32)

        ops.apply_penalties(
            logits, counts, counts, torch.tensor([1.0]), torch.tensor([0.0]), torch.tensor([0.0])
        )

        assert torch.equal(logits, before)


class TestLogprobs:
    def test_log_softmax_matches_torch_and_keeps_masks(self) -> None:
        logits = torch.randn(2, VOCAB)
        logits[0, 4] = -math.inf
        out = torch.empty_like(logits)

        ops.log_softmax(logits, out)

        assert torch.allclose(out, torch.log_softmax(logits, dim=-1))
        assert out[0, 4] == -math.inf

    def test_kept_logprobs_describe_the_distribution_the_token_came_from(self) -> None:
        logits = torch.tensor([[3.0, 2.0, 1.0, 0.0, -1.0] + [-5.0] * (VOCAB - 5)])
        kept = torch.empty(1, VOCAB)
        token = torch.empty(1, dtype=torch.int64)

        ops.sample(
            logits,
            torch.tensor([0.5]),
            torch.tensor([3], dtype=torch.int32),
            torch.tensor([1.0]),
            torch.tensor([0.0]),
            torch.tensor([0.25]),
            token,
            kept_logprobs=kept,
        )

        finite = torch.isfinite(kept[0])
        assert finite.nonzero().flatten().tolist() == [0, 1, 2]
        reference = torch.log_softmax(logits[0, :3] / 0.5, dim=-1)
        assert torch.allclose(kept[0, :3], reference)
        assert int(token) in (0, 1, 2)

    def test_the_drawn_token_does_not_change_when_logprobs_are_asked(self) -> None:
        generator = torch.Generator().manual_seed(9)
        logits = torch.randn(4, VOCAB, generator=generator)
        args = (
            torch.tensor([0.0, 0.7, 1.0, 1.3]),
            torch.tensor([0, 5, 0, 10], dtype=torch.int32),
            torch.tensor([1.0, 0.9, 0.5, 1.0]),
            torch.tensor([0.0, 0.0, 0.1, 0.0]),
            torch.rand(4, generator=generator),
        )
        plain = torch.empty(4, dtype=torch.int64)
        with_lp = torch.empty(4, dtype=torch.int64)

        ops.sample(logits, *args, plain)
        ops.sample(logits, *args, with_lp, kept_logprobs=torch.empty(4, VOCAB))

        assert torch.equal(plain, with_lp)
