"""Counter-based random numbers for sampling, as pure functions of the position.

The draw for the token at position ``pos`` of a request is
``philox4x32_10(key = seed, counter = (pos, stream, sequence))``; nothing in the
engine advances a generator. A re-run, a preempt followed by a resume, a different
batch and a speculative window all draw the same number at the same position. The exact
definition, which every backend of :class:`ayaka.ops.sampling.SamplingOps` must reproduce,
is in :mod:`ayaka.ops.sampling`; this module is its host-side reference in plain Python.
It is used for the known-answer tests of the tensor implementation, for tests that want
the number a row should have drawn, and to turn a request id into a seed.

Importing this module does not import torch.
"""

from __future__ import annotations

import hashlib
from typing import Final

from ayaka.ops.sampling import DrawStream

__all__ = ["MAX_SEED", "MAX_SEQUENCE", "draw_uniform", "philox4x32_10", "resolve_seed"]

#: Largest seed: the 63 bits that ``ayaka.core.params.SamplingParams`` accepts.
MAX_SEED: Final = (1 << 63) - 1
#: Largest sequence index: it occupies one 32-bit word of the counter.
MAX_SEQUENCE: Final = (1 << 32) - 1

_WORD: Final = 0xFFFFFFFF
# Philox-4x32 multipliers and Weyl key increments (Random123, philox.h). The tensor
# implementation in the torch backend repeats them; both are checked against the same vectors.
_MULTIPLIER_0: Final = 0xD2511F53
_MULTIPLIER_1: Final = 0xCD9E8D57
_WEYL_0: Final = 0x9E3779B9
_WEYL_1: Final = 0xBB67AE85
_ROUNDS: Final = 10
_UNIFORM_SCALE: Final = float(1 << 24)


def philox4x32_10(
    counter: tuple[int, int, int, int], key: tuple[int, int]
) -> tuple[int, int, int, int]:
    """Return the four output words of Philox-4x32 with 10 rounds.

    Args:
        counter: Four 32-bit words.
        key: Two 32-bit words.

    Raises:
        ValueError: If a word is outside ``[0, 2**32)``.
    """
    for word in (*counter, *key):
        if not 0 <= word <= _WORD:
            raise ValueError(f"Philox words must be in [0, 2**32), got {word}")
    c0, c1, c2, c3 = counter
    k0, k1 = key
    for round_index in range(_ROUNDS):
        if round_index:
            k0 = (k0 + _WEYL_0) & _WORD
            k1 = (k1 + _WEYL_1) & _WORD
        product0 = _MULTIPLIER_0 * c0
        product1 = _MULTIPLIER_1 * c2
        c0, c1, c2, c3 = (
            (product1 >> 32) ^ c1 ^ k0,
            product1 & _WORD,
            (product0 >> 32) ^ c3 ^ k1,
            product0 & _WORD,
        )
    return c0, c1, c2, c3


def draw_uniform(
    seed: int, position: int, *, stream: DrawStream = DrawStream.PICK, sequence: int = 0
) -> float:
    """Return the random number in ``[0, 1)`` that ``position`` draws on ``stream``.

    It has 24 random bits: the high 24 bits of the first Philox output word, divided
    by ``2**24``.

    Args:
        seed: The request seed, in ``[0, MAX_SEED]``.
        position: Index in the sequence (prompt and output) of the token the draw produces.
        stream: Which independent number of that position to draw.
        sequence: Index of the output sequence of a request with ``n > 1``, else 0.

    Raises:
        ValueError: If ``seed``, ``position`` or ``sequence`` is out of range.
    """
    if not 0 <= seed <= MAX_SEED:
        raise ValueError(f"seed must be in [0, {MAX_SEED}], got {seed}")
    if not 0 <= position <= MAX_SEED:
        raise ValueError(f"position must be in [0, {MAX_SEED}], got {position}")
    if not 0 <= sequence <= MAX_SEQUENCE:
        raise ValueError(f"sequence must be in [0, {MAX_SEQUENCE}], got {sequence}")
    words = philox4x32_10(
        (position & _WORD, position >> 32, int(DrawStream(stream)), sequence),
        (seed & _WORD, seed >> 32),
    )
    return (words[0] >> 8) / _UNIFORM_SCALE


def resolve_seed(request_id: str, seed: int | None) -> int:
    """Return the seed a request samples with.

    A seed the client gave is used as it is. Otherwise one is derived from the request id,
    the same way every time (not with ``hash``, which differs between processes), so the
    value can be reported with the result and the request repeated.

    Raises:
        ValueError: If ``seed`` is given and outside ``[0, MAX_SEED]``.
    """
    if seed is not None:
        if not 0 <= seed <= MAX_SEED:
            raise ValueError(f"seed must be in [0, {MAX_SEED}], got {seed}")
        return seed
    digest = hashlib.blake2b(request_id.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") & MAX_SEED
