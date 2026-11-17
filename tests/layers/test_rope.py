"""The rotary embedding layer: the inverse frequencies of each scaling, and the table it keeps.

The frequencies are compared with copies of Hugging Face's ``modeling_rope_utils`` formulas written
out again here in float64 (default, linear, llama3, yarn). With ``transformers`` installed, the
integration test also compares them with the library's own ``ROPE_INIT_FUNCTIONS``.
"""

from __future__ import annotations

import math
from typing import Any

import pytest

from ayaka.configs.model import RopeScaling

torch: Any = pytest.importorskip("torch")

from ayaka.layers.rope import RotaryEmbedding, rope_inverse_frequencies  # noqa: E402
from ayaka.ops.backend import load_backend  # noqa: E402
from ayaka.utils.dtypes import DType  # noqa: E402


def _base(dim: int, theta: float) -> list[float]:
    return [1.0 / theta ** (i / dim) for i in range(0, dim, 2)]


def _yarn(dim: int, theta: float, factor: float, trained: int) -> tuple[list[float], float]:
    def correction(rotations: float) -> float:
        return (dim * math.log(trained / (rotations * 2 * math.pi))) / (2 * math.log(theta))

    low = max(math.floor(correction(32)), 0)
    high = min(math.ceil(correction(1)), dim - 1)
    if low == high:
        high += 0.001
    out = []
    for i, freq in enumerate(_base(dim, theta)):
        ramp = min(max((i - low) / (high - low), 0.0), 1.0)
        extrapolation = 1 - ramp
        out.append((freq / factor) * (1 - extrapolation) + freq * extrapolation)
    return out, 0.1 * math.log(factor) + 1


@pytest.mark.unit
class TestInverseFrequencies:
    def test_default(self) -> None:
        frequencies, factor = rope_inverse_frequencies(64, 500000.0, None)

        assert frequencies == pytest.approx(_base(64, 500000.0), rel=1e-12)
        assert factor == 1.0

    def test_linear_divides_by_the_factor(self) -> None:
        frequencies, _ = rope_inverse_frequencies(
            32, 10000.0, RopeScaling(kind="linear", factor=4.0)
        )

        assert frequencies == pytest.approx([f / 4 for f in _base(32, 10000.0)], rel=1e-12)

    def test_llama3_keeps_high_frequencies_and_divides_low_ones(self) -> None:
        scaling = RopeScaling(
            kind="llama3",
            factor=8.0,
            original_max_position_embeddings=8192,
            low_freq_factor=1.0,
            high_freq_factor=4.0,
        )
        base = _base(128, 500000.0)

        frequencies, _ = rope_inverse_frequencies(128, 500000.0, scaling)

        for freq, original in zip(frequencies, base, strict=True):
            wavelength = 2 * math.pi / original
            if wavelength < 8192 / 4:
                assert freq == pytest.approx(original, rel=1e-12)
            elif wavelength > 8192:
                assert freq == pytest.approx(original / 8)
            else:
                assert original / 8 <= freq <= original

    def test_yarn(self) -> None:
        scaling = RopeScaling(kind="yarn", factor=4.0, original_max_position_embeddings=4096)

        frequencies, factor = rope_inverse_frequencies(64, 10000.0, scaling)

        expected, expected_factor = _yarn(64, 10000.0, 4.0, 4096)
        assert frequencies == pytest.approx(expected, rel=1e-12)
        assert factor == pytest.approx(expected_factor)

    def test_an_odd_width_is_refused(self) -> None:
        with pytest.raises(ValueError, match="even and positive"):
            rope_inverse_frequencies(7, 10000.0, None)


@pytest.mark.unit
class TestRotaryEmbedding:
    def test_the_table_is_built_on_construction(self) -> None:
        ops = load_backend(["torch"], device="cpu", dtype=DType.FP32)

        rotary = RotaryEmbedding(8, max_positions=6, theta=10000.0, ops=ops)

        angles = torch.arange(6, dtype=torch.float64)[:, None] * torch.tensor(_base(8, 10000.0))
        torch.testing.assert_close(rotary.cache[:, :4].double(), angles.cos(), rtol=0, atol=1e-6)
        assert rotary.cache.dtype == torch.float32

    def test_on_the_meta_device_the_table_waits_for_reset(self) -> None:
        ops = load_backend(["torch"], device="cpu", dtype=DType.FP32)
        rotary = RotaryEmbedding(8, max_positions=6, theta=10000.0, ops=ops, device="meta")

        rotary.to_empty(device="cpu")
        rotary.reset_cache()

        reference = RotaryEmbedding(8, max_positions=6, theta=10000.0, ops=ops)
        assert torch.equal(rotary.cache, reference.cache)

    def test_the_table_is_not_a_parameter_nor_saved(self) -> None:
        ops = load_backend(["torch"], device="cpu", dtype=DType.FP32)
        rotary = RotaryEmbedding(8, max_positions=6, theta=10000.0, ops=ops)

        assert list(rotary.parameters()) == []
        assert rotary.state_dict() == {}


@pytest.mark.integration
@pytest.mark.parametrize(
    ("raw", "dim"),
    [
        ({"rope_type": "default"}, 64),
        ({"rope_type": "linear", "factor": 2.0}, 64),
        (
            {
                "rope_type": "llama3",
                "factor": 8.0,
                "original_max_position_embeddings": 8192,
                "low_freq_factor": 1.0,
                "high_freq_factor": 4.0,
            },
            128,
        ),
        ({"rope_type": "yarn", "factor": 4.0, "original_max_position_embeddings": 2048}, 64),
    ],
    ids=["default", "linear", "llama3", "yarn"],
)
def test_frequencies_match_transformers(raw: dict[str, object], dim: int) -> None:
    transformers: Any = pytest.importorskip("transformers")
    rope_utils: Any = pytest.importorskip("transformers.modeling_rope_utils")
    theta = 500000.0
    config = transformers.LlamaConfig(
        hidden_size=dim * 4,
        num_attention_heads=4,
        head_dim=dim,
        max_position_embeddings=8192,
        rope_parameters={"rope_theta": theta, **raw},
    )
    kind = str(raw["rope_type"])
    if kind == "default":  # the default lives in each model's rotary class, not in the table
        expected = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.int64).float() / dim))
        factor = 1.0
    else:
        expected, factor = rope_utils.ROPE_INIT_FUNCTIONS[kind](config, "cpu")

    ours, our_factor = rope_inverse_frequencies(
        dim, theta, RopeScaling.from_hf(raw, max_position_embeddings=8192)
    )

    torch.testing.assert_close(torch.tensor(ours, dtype=torch.float32), expected, rtol=1e-6, atol=0)
    assert our_factor == pytest.approx(float(factor))
