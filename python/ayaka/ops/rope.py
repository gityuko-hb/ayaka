"""Rotary position embedding ops: the contract a backend implements for ``ayaka.layers.rope``.

The rotation
------------

A rotary embedding turns pairs of elements of every query and key head by an
angle that grows with the position. For a head of ``head_dim`` elements, the
first ``rotary_dim`` (even, at most ``head_dim``) take part; the rest are left
as they are. With ``half = rotary_dim // 2`` there are ``half`` frequencies
``inv_freq[i]``, and position ``p`` turns pair ``i`` by ``p * inv_freq[i]``.

The *cache* holds the cosines and sines once for every position, as fp32::

    cache[p, i]        = cos(p * inv_freq[i]) * attention_factor       i < half
    cache[p, half + i] = sin(p * inv_freq[i]) * attention_factor

``attention_factor`` is 1 except for YaRN. Which elements form pair ``i`` is the
*style*:

* ``interleaved=False`` (GPT-NeoX, Llama in the Hugging Face layout): elements
  ``i`` and ``i + half``.
* ``interleaved=True`` (GPT-J, Llama in Meta's original layout): elements
  ``2 * i`` and ``2 * i + 1``.

A pair ``(x, y)`` becomes ``(x * cos - y * sin, y * cos + x * sin)``.

Importing this module does not import torch.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from torch import Tensor

__all__ = ["RopeOps"]


class RopeOps(Protocol):
    """Building the cosine/sine cache and rotating queries and keys with it."""

    def build_cache(self, inv_freq: Tensor, out: Tensor, *, attention_factor: float) -> None:
        """Write the cache of the module docstring into ``out``.

        ``inv_freq`` is ``[half]`` float32 and ``out`` is ``[max_positions, 2 * half]``
        float32, on one device. Angles are computed in fp32 from the integer position.
        """
        ...

    def apply(
        self, positions: Tensor, q: Tensor, k: Tensor, cache: Tensor, *, interleaved: bool
    ) -> None:
        """Rotate the first ``rotary_dim`` elements of every head of ``q`` and ``k``, in place.

        ``positions`` is ``[N]`` int32 or int64, ``q`` is ``[N, num_heads, head_dim]``
        and ``k`` is ``[N, num_kv_heads, head_dim]``, both contiguous and in one
        compute dtype; ``cache`` is the fp32 ``[max_positions, rotary_dim]`` table
        :meth:`build_cache` wrote. Row ``n`` is rotated by position
        ``positions[n]``, which must be below ``max_positions``: the op does not
        check it, because a check would read the device. The rotation is computed
        in fp32 and rounded to the dtype of ``q`` once. ``q`` and ``k`` must not
        overlap.
        """
        ...
