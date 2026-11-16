"""Sampling ops: the contract a backend implements to turn logits into tokens.

Three ops cover the M0 path: :meth:`SamplingOps.uniform` produces the random
numbers, :meth:`SamplingOps.argmax` is the greedy path and
:meth:`SamplingOps.sample` draws from the filtered distribution. The order of
the steps and the tie-breaking rules are part of the contract, because every
backend must return the same token for the same logits and the same random
number (up to the last bit of the softmax, see below).

The random number
-----------------

A draw is a pure function of ``(seed, position, stream, sequence)``: nothing
advances, so a re-run, a request that is preempted and resumed, a different batch
composition or a speculative window all draw the same number at the same
position. The function is Philox-4x32-10 (Salmon et al., "Parallel random
numbers: as easy as 1, 2, 3", SC11; the reference implementation is Random123)::

    key     = (seed & 0xFFFFFFFF, seed >> 32)
    counter = (position & 0xFFFFFFFF, position >> 32, stream, sequence)
    u       = (philox4x32_10(counter, key)[0] >> 8) / 2**24      # in [0, 1)

``seed`` is a 63-bit non-negative integer, ``position`` a 63-bit non-negative
one: the index, in the sequence of prompt and output, of the token the draw
produces. ``stream`` is a :class:`DrawStream` and ``sequence`` the index of the
output sequence of a request with ``n > 1`` (0 otherwise). The result has 24
random bits, so it is exactly representable in fp32.

The token
---------

For one row of logits ``x`` and its parameters::

    x = x / temperature                 # temperature 0: return the largest x, smallest id on a tie
    top-k   keep the k largest x; ties at the threshold keep the smaller ids, so exactly k stay
    p = softmax(x over the kept set)
    order the kept set by (p descending, id ascending)               # the canonical order
    top-p   keep the shortest prefix of that order whose sum is >= top_p (at least one element)
    min-p   drop elements with p < min_p * (largest p)
    return the first element of the canonical order whose cumulative probability exceeds
    u * (the total kept probability)                                 # inverse CDF

Because the choice is the inverse CDF over a fixed order, two distributions that are close
give the same token for the same ``u`` far more often than two independent draws would: the
property that coupled drafting relies on. Gumbel-max and stateful ``multinomial`` do not have it.

Softmax in two backends may differ in the last bit, so a ``u`` that falls exactly on a
boundary of the CDF can give two different tokens in two backends. The determinism contract
holds inside one backend; across backends the distributions are compared, not the tokens.

Importing this module does not import torch.
"""

from __future__ import annotations

from enum import IntEnum
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from torch import Tensor

__all__ = ["DrawStream", "SamplingOps"]


class DrawStream(IntEnum):
    """Which independent random number a position draws.

    The value is the third word of the Philox counter, so the streams of one
    position are disjoint and fixed for good: changing a value changes every
    recorded output of a seeded request.

    Attributes:
        PICK: The ordinary sample of a row: plain decoding, the one-point rule and the bonus
            token of a speculative window. The same number whichever window the row is in.
        ACCEPT: The accept-or-reject decision of the ``p/q`` rule.
        RESID: The draw from the residual distribution after a rejection.
        DRAFT: The draw of a drafter that samples; it does not affect the output.
    """

    PICK = 0
    ACCEPT = 1
    RESID = 2
    DRAFT = 3


class SamplingOps(Protocol):
    """Random numbers and token selection over a batch of rows.

    Every tensor is on one device and has one entry per row (``B`` rows),
    except ``logits``, which is ``[B, vocab]``. ``logits`` may be fp16, bf16 or
    fp32; the op computes in fp32. A row of ``logits`` holds finite values and
    ``-inf`` (a masked token) and at least one finite value; a backend may
    refuse anything else, and the reference backend does. The caller allocates
    ``out``.
    """

    def uniform(
        self, seeds: Tensor, positions: Tensor, sequences: Tensor, out: Tensor, *, stream: int
    ) -> None:
        """Write the random number of every row into ``out``, as defined in the module docstring.

        ``seeds`` and ``positions`` are int64 and non-negative, ``sequences``
        int32 or int64 and non-negative, ``out`` float32. ``stream`` is a
        :class:`DrawStream`. The call is a pure function of its arguments and
        does not depend on ``B`` or on the order of the rows.
        """
        ...

    def argmax(self, logits: Tensor, out: Tensor) -> None:
        """Write the index of the largest logit of every row into ``out`` (int64).

        A tie goes to the smallest token id. This is the whole of greedy
        decoding: no softmax and no sort.
        """
        ...

    def sample(
        self,
        logits: Tensor,
        temperature: Tensor,
        top_k: Tensor,
        top_p: Tensor,
        min_p: Tensor,
        u: Tensor,
        out: Tensor,
    ) -> None:
        """Write one token id per row into ``out`` (int64), as the module docstring defines.

        ``temperature``, ``top_p``, ``min_p`` and ``u`` are float32; ``top_k``
        is int32. ``temperature`` 0 marks a greedy row, which returns what
        :meth:`argmax` would. ``top_k`` 0 disables top-k, ``top_p`` 1 disables
        top-p and ``min_p`` 0 disables min-p. ``u`` is in ``[0, 1)``. The
        parameters have been validated by ``ayaka.core.params.SamplingParams``;
        the op does not repair them.
        """
        ...
