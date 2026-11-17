"""Rotary embedding ops on the torch backend: the cache and the rotation, in both pairings.

The oracle rotates each pair ``(x, y)`` by ``p * inv_freq[i]`` in float64 with explicit loops over
the pairs, for the two ways of pairing elements (halves and interleaved) that
:mod:`ayaka.ops.rope` defines. The ops compute in fp32 and round once, so the error against it is
that rounding: ``4 * eps * max|y|``.
"""

from __future__ import annotations

from typing import Any

import pytest

from ayaka.core.errors import BackendError
from ayaka.ops.backend import load_backend
from ayaka.utils.dtypes import DType

torch: Any = pytest.importorskip("torch")

pytestmark = pytest.mark.unit

DTYPES = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}


def _ops() -> Any:
    return load_backend(["torch"], device="cpu", dtype=DType.FP32).rope


def _inv_freq(rotary: int, theta: float = 10000.0) -> Any:
    return torch.tensor(
        [theta ** (-2 * i / rotary) for i in range(rotary // 2)], dtype=torch.float32
    )


def _cache(rotary: int, positions: int, factor: float = 1.0) -> Any:
    out = torch.empty(positions, rotary, dtype=torch.float32)
    _ops().build_cache(_inv_freq(rotary), out, attention_factor=factor)
    return out


def _oracle(x: Any, positions: Any, rotary: int, *, interleaved: bool, factor: float = 1.0) -> Any:
    out = x.double().clone()
    inv = _inv_freq(rotary).double()
    for row, position in enumerate(positions.tolist()):
        for i in range(rotary // 2):
            angle = position * float(inv[i])
            cos, sin = (
                torch.cos(torch.tensor(angle)) * factor,
                torch.sin(torch.tensor(angle)) * factor,
            )
            a, b = (2 * i, 2 * i + 1) if interleaved else (i, i + rotary // 2)
            first, second = x[row, :, a].double(), x[row, :, b].double()
            out[row, :, a] = first * cos - second * sin
            out[row, :, b] = second * cos + first * sin
    return out


class TestCache:
    def test_cosines_then_sines_of_position_times_frequency(self) -> None:
        cache = _cache(8, 5)

        angles = torch.arange(5, dtype=torch.float64)[:, None] * _inv_freq(8).double()[None, :]
        torch.testing.assert_close(cache[:, :4].double(), angles.cos(), rtol=0, atol=1e-6)
        torch.testing.assert_close(cache[:, 4:].double(), angles.sin(), rtol=0, atol=1e-6)

    def test_the_attention_factor_scales_both_halves(self) -> None:
        torch.testing.assert_close(_cache(8, 5, factor=1.5), _cache(8, 5) * 1.5)

    def test_out_must_be_float32_of_the_right_width(self) -> None:
        with pytest.raises(BackendError, match=r"out must be float32 \[3, 8\]"):
            _ops().build_cache(_inv_freq(8), torch.empty(3, 6), attention_factor=1.0)


@pytest.mark.parametrize("interleaved", [False, True], ids=["halves", "interleaved"])
@pytest.mark.parametrize("dtype", list(DTYPES.values()), ids=list(DTYPES))
class TestApply:
    def test_rotation_matches_the_oracle(self, dtype: Any, interleaved: bool) -> None:
        generator = torch.Generator().manual_seed(0)
        positions = torch.tensor([0, 1, 7, 31, 2], dtype=torch.int64)
        q = torch.randn(5, 4, 16, generator=generator).to(dtype)
        k = torch.randn(5, 2, 16, generator=generator).to(dtype)
        expected_q = _oracle(q, positions, 16, interleaved=interleaved)
        expected_k = _oracle(k, positions, 16, interleaved=interleaved)

        _ops().apply(positions, q, k, _cache(16, 32), interleaved=interleaved)

        for got, expected in ((q, expected_q), (k, expected_k)):
            atol = 4 * torch.finfo(dtype).eps * float(expected.abs().max())
            torch.testing.assert_close(got.double(), expected, rtol=0, atol=atol)

    def test_a_partial_rotation_leaves_the_tail_alone(self, dtype: Any, interleaved: bool) -> None:
        generator = torch.Generator().manual_seed(1)
        positions = torch.tensor([3, 9], dtype=torch.int32)
        q = torch.randn(2, 2, 12, generator=generator).to(dtype)
        k = torch.randn(2, 1, 12, generator=generator).to(dtype)
        tail_q, tail_k = q[..., 8:].clone(), k[..., 8:].clone()
        expected = _oracle(q, positions, 8, interleaved=interleaved)

        _ops().apply(positions, q, k, _cache(8, 16), interleaved=interleaved)

        assert torch.equal(q[..., 8:], tail_q)
        assert torch.equal(k[..., 8:], tail_k)
        atol = 4 * torch.finfo(dtype).eps * float(expected.abs().max())
        torch.testing.assert_close(q.double(), expected, rtol=0, atol=atol)


class TestApplyRejections:
    def test_position_zero_is_the_identity(self) -> None:
        q, k = torch.randn(1, 2, 8), torch.randn(1, 1, 8)
        q0, k0 = q.clone(), k.clone()

        _ops().apply(torch.zeros(1, dtype=torch.int64), q, k, _cache(8, 4), interleaved=False)

        torch.testing.assert_close(q, q0)
        torch.testing.assert_close(k, k0)

    def test_an_odd_or_too_wide_rotation_is_refused(self) -> None:
        q, k = torch.zeros(1, 2, 8), torch.zeros(1, 1, 8)

        with pytest.raises(BackendError, match="must be even, positive and at most head_dim 8"):
            _ops().apply(
                torch.zeros(1, dtype=torch.int64), q, k, torch.zeros(4, 10), interleaved=False
            )

    def test_k_must_match_q(self) -> None:
        with pytest.raises(BackendError, match="k must be"):
            _ops().apply(
                torch.zeros(1, dtype=torch.int64),
                torch.zeros(1, 2, 8),
                torch.zeros(1, 1, 8, dtype=torch.float16),
                _cache(8, 4),
                interleaved=False,
            )

    def test_positions_must_be_integers(self) -> None:
        with pytest.raises(BackendError, match="positions must be an int32 or int64"):
            _ops().apply(
                torch.zeros(1),
                torch.zeros(1, 2, 8),
                torch.zeros(1, 1, 8),
                _cache(8, 4),
                interleaved=False,
            )
