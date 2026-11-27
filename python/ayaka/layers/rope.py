"""The rotary position embedding and its scalings.

:class:`RotaryEmbedding` owns the fp32 cosine/sine table of
:mod:`ayaka.ops.rope` and rotates the queries and keys of a step with it. The
table is computed from the inverse frequencies, which is where the scalings
differ. With ``dim = rotary_dim`` and ``i`` in ``[0, dim / 2)``::

    default   inv_freq[i] = theta ** (-2i / dim)
    linear    inv_freq[i] / factor
    llama3    per band of wavelength 2*pi / inv_freq[i], against the trained context L:
                  wavelength < L / high_freq_factor     unchanged
                  wavelength > L / low_freq_factor      inv_freq[i] / factor
                  otherwise                             a linear blend of the two, by
                      s = (L / wavelength - low_freq_factor) / (high_freq_factor - low_freq_factor)
    yarn      a blend of the interpolated (inv_freq / factor) and the original frequencies
              over the bands between the corrections of beta_fast and beta_slow rotations,
              and cos and sin multiplied by attention_factor (0.1 * ln(factor) + 1 by default)

These are the formulas of Hugging Face's ``modeling_rope_utils`` for the same
names; ``tests/layers`` checks the frequencies against independent copies of
them and, with ``transformers`` installed, against the library itself.

The table is a buffer, not a parameter: no checkpoint stores it. A model built
on the meta device has no table until :meth:`RotaryEmbedding.reset_cache` runs,
which ``CausalLM.initialize_buffers`` does after the loader has given the model
its memory.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch
from torch import nn

if TYPE_CHECKING:
    from ayaka.configs.model import RopeScaling
    from ayaka.ops.backend import OpsBackend

__all__ = ["RotaryEmbedding", "rope_inverse_frequencies"]


def rope_inverse_frequencies(
    rotary_dim: int, theta: float, scaling: RopeScaling | None
) -> tuple[list[float], float]:
    """The ``rotary_dim / 2`` inverse frequencies and the attention factor of a rotary embedding.

    Pure Python on floats, so the table it leads to is the same on every device.

    Raises:
        ValueError: If ``rotary_dim`` is not even and positive.
    """
    if rotary_dim < 2 or rotary_dim % 2:
        raise ValueError(f"rotary_dim must be even and positive, got {rotary_dim}")
    half = rotary_dim // 2
    base = [theta ** -(2 * i / rotary_dim) for i in range(half)]
    if scaling is None:
        return base, 1.0
    factor = scaling.factor
    match scaling.kind:
        case "linear":
            return [freq / factor for freq in base], 1.0
        case "llama3":
            return _llama3(base, scaling), 1.0
        case "yarn":
            return _yarn(rotary_dim, theta, scaling)
        case "proportional":
            return _proportional(rotary_dim, theta, scaling), 1.0


def _proportional(rotary_dim: int, theta: float, scaling: RopeScaling) -> list[float]:
    half = rotary_dim // 2
    angles = int(scaling.partial_rotary_factor * rotary_dim // 2)
    freqs = [(theta ** -(2 * i / rotary_dim)) / scaling.factor for i in range(angles)]
    if angles < half:
        freqs.extend([0.0] * (half - angles))
    return freqs


def _trained_context(scaling: RopeScaling) -> int:
    trained = scaling.original_max_position_embeddings
    if trained is None:  # RopeScaling refuses this for llama3 and yarn
        raise ValueError(f"{scaling.kind} scaling needs original_max_position_embeddings")
    return trained


def _llama3(base: list[float], scaling: RopeScaling) -> list[float]:
    trained = _trained_context(scaling)
    low_wavelength = trained / scaling.low_freq_factor
    high_wavelength = trained / scaling.high_freq_factor
    scaled: list[float] = []
    for freq in base:
        wavelength = 2 * math.pi / freq
        if wavelength < high_wavelength:
            scaled.append(freq)
        elif wavelength > low_wavelength:
            scaled.append(freq / scaling.factor)
        else:
            smooth = (trained / wavelength - scaling.low_freq_factor) / (
                scaling.high_freq_factor - scaling.low_freq_factor
            )
            scaled.append((1 - smooth) * freq / scaling.factor + smooth * freq)
    return scaled


def _yarn(rotary_dim: int, theta: float, scaling: RopeScaling) -> tuple[list[float], float]:
    trained = _trained_context(scaling)
    factor = scaling.factor
    attention_factor = scaling.attention_factor
    if attention_factor is None:
        attention_factor = 0.1 * math.log(factor) + 1.0 if factor > 1 else 1.0

    def correction(rotations: float) -> float:
        # The dimension whose wavelength fits ``rotations`` times into the trained context.
        return (rotary_dim * math.log(trained / (rotations * 2 * math.pi))) / (2 * math.log(theta))

    raw_low = correction(scaling.beta_fast)
    raw_high = correction(scaling.beta_slow)
    low_val = math.floor(raw_low) if scaling.truncate else raw_low
    high_val = math.ceil(raw_high) if scaling.truncate else raw_high
    low = max(low_val, 0)
    high = min(high_val, rotary_dim - 1)
    high_f = high + 0.001 if low == high else float(high)
    half = rotary_dim // 2
    frequencies: list[float] = []
    for i in range(half):
        position_freq = theta ** (2 * i / rotary_dim)
        extrapolated = 1.0 / position_freq
        interpolated = 1.0 / (factor * position_freq)
        ramp = min(max((i - low) / (high_f - low), 0.0), 1.0)
        keep = 1.0 - ramp  # the share of the original frequency
        frequencies.append(interpolated * (1 - keep) + extrapolated * keep)
    return frequencies, attention_factor


class RotaryEmbedding(nn.Module):
    """Rotates queries and keys by their positions; shared by every attention layer of a model.

    Args:
        rotary_dim: Elements of a head that are rotated; even, at most the head width.
        max_positions: Rows of the table: positions ``0`` to ``max_positions - 1``.
        theta: Base of the frequencies (``rope_theta``).
        scaling: The context-extension scaling, or ``None``.
        interleaved: Pair elements ``2i, 2i + 1`` (GPT-J style) instead of ``i, i + half``.
        ops: The backend whose ``rope`` ops build the table and rotate.
        device: Where the table lives; ``"meta"`` leaves it unallocated.

    Raises:
        ValueError: If ``rotary_dim`` is not even and positive, or ``max_positions`` is not
            positive.
    """

    cache: torch.Tensor

    def __init__(
        self,
        rotary_dim: int,
        *,
        max_positions: int,
        theta: float,
        scaling: RopeScaling | None = None,
        interleaved: bool = False,
        ops: OpsBackend,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__()
        if max_positions < 1:
            raise ValueError(f"max_positions must be positive, got {max_positions}")
        inv_freq, attention_factor = rope_inverse_frequencies(rotary_dim, theta, scaling)
        self.rotary_dim = rotary_dim
        self.max_positions = max_positions
        self.interleaved = interleaved
        self.attention_factor = attention_factor
        self._inv_freq = inv_freq
        self._ops = ops.rope
        self.register_buffer(
            "cache",
            torch.empty(max_positions, rotary_dim, dtype=torch.float32, device=device),
            persistent=False,
        )
        if self.cache.device.type != "meta":
            self.reset_cache()

    def reset_cache(self) -> None:
        """Compute the table into the buffer, on whatever device it now lives."""
        inv_freq = torch.tensor(self._inv_freq, dtype=torch.float32, device=self.cache.device)
        self._ops.build_cache(inv_freq, self.cache, attention_factor=self.attention_factor)

    def forward(self, positions: torch.Tensor, q: torch.Tensor, k: torch.Tensor) -> None:
        """Rotate ``q`` ``[N, heads, head_dim]`` and ``k`` ``[N, kv_heads, head_dim]`` in place.

        ``positions`` is the ``[N]`` position of each row; every one must be below
        ``max_positions`` (the op does not check, a check would read the device).
        """
        self._ops.apply(positions, q, k, self.cache, interleaved=self.interleaved)
