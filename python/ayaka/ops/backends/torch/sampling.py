"""Reference sampling ops; implements :class:`ayaka.ops.sampling.SamplingOps`.

Philox-4x32-10 runs on int64 tensors holding 32-bit words, so it needs no
unsigned dtype and gives the same bits on CPU and CUDA. The 32 by 32 bit product
of a round is built from 16-bit limbs, so no intermediate leaves the positive
int64 range. The token is chosen by the algorithm in :mod:`ayaka.ops.sampling`,
written for clarity: one full-vocabulary stable sort per call and one pass per
filter. Eager calls check their logits and indices, which costs a
device-to-host synchronisation; do not measure speed with them. A check cannot
read the device while a CUDA graph is being captured, so during a capture the
checks are skipped and the captured graph refuses nothing: NaN logits then give
an unspecified token instead of a :class:`~ayaka.core.errors.BackendError`.
"""

from __future__ import annotations

import torch

from ayaka.core.errors import BackendError
from ayaka.ops.backends.torch._checks import capturing, require_compute_dtype, require_like
from ayaka.ops.sampling import DrawStream

__all__ = ["argmax", "philox_rounds", "sample", "uniform"]

_WORD = 0xFFFFFFFF
# Philox-4x32 multipliers and Weyl key increments (Random123, philox.h).
_MULTIPLIER_0 = 0xD2511F53
_MULTIPLIER_1 = 0xCD9E8D57
_WEYL_0 = 0x9E3779B9
_WEYL_1 = 0xBB67AE85
_ROUNDS = 10
_UNIFORM_SCALE = float(1 << 24)

type _Words = tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]


def philox_rounds(counter: _Words, key: tuple[torch.Tensor, torch.Tensor]) -> _Words:
    """Run Philox-4x32-10 on int64 tensors whose entries are 32-bit words.

    ``counter`` has four words and ``key`` two, all of the same shape. Returns the four output
    words. Exposed for the known-answer tests of the contract; the ops below are the interface.
    """
    c0, c1, c2, c3 = counter
    k0, k1 = key
    for round_index in range(_ROUNDS):
        if round_index:
            k0 = (k0 + _WEYL_0) & _WORD
            k1 = (k1 + _WEYL_1) & _WORD
        hi0, lo0 = _mulhilo(_MULTIPLIER_0, c0)
        hi1, lo1 = _mulhilo(_MULTIPLIER_1, c2)
        c0, c1, c2, c3 = hi1 ^ c1 ^ k0, lo1, hi0 ^ c3 ^ k1, lo0
    return c0, c1, c2, c3


def _mulhilo(multiplier: int, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """High and low 32 bits of ``multiplier * x`` for ``x`` below 2**32."""
    low_part = x * (multiplier & 0xFFFF)  # below 2**48
    high_part = x * (multiplier >> 16)  # below 2**48
    middle = low_part + ((high_part & 0xFFFF) << 16)
    return (high_part >> 16) + (middle >> 32), middle & _WORD


def uniform(
    seeds: torch.Tensor,
    positions: torch.Tensor,
    sequences: torch.Tensor,
    out: torch.Tensor,
    *,
    stream: int,
) -> None:
    op = "sampling.uniform"
    rows = out.shape[0] if out.ndim == 1 else -1
    require_like(op, "out", out, (rows,), torch.float32)
    require_like(op, "seeds", seeds, (rows,), torch.int64)
    require_like(op, "positions", positions, (rows,), torch.int64)
    if sequences.shape != (rows,) or sequences.dtype not in (torch.int32, torch.int64):
        raise BackendError(
            f"{op}: sequences must be an int32 or int64 tensor of shape [{rows}], "
            f"got {sequences.dtype} {list(sequences.shape)}",
            context={"op": op, "argument": "sequences"},
        )
    try:
        stream_word = int(DrawStream(stream))
    except ValueError:
        raise BackendError(
            f"{op}: stream must be a DrawStream, got {stream}",
            context={"op": op, "argument": "stream"},
        ) from None
    checked = not capturing(out)
    for name, tensor in (("seeds", seeds), ("positions", positions), ("sequences", sequences)):
        if checked and bool((tensor < 0).any()):
            raise BackendError(
                f"{op}: {name} must not be negative", context={"op": op, "argument": name}
            )
    wide_sequences = sequences.to(torch.int64)
    if checked and bool((wide_sequences > _WORD).any()):
        raise BackendError(
            f"{op}: sequences must be below 2**32", context={"op": op, "argument": "sequences"}
        )
    counter = (
        positions & _WORD,
        positions >> 32,
        torch.full_like(positions, stream_word),
        wide_sequences,
    )
    words = philox_rounds(counter, (seeds & _WORD, seeds >> 32))
    out.copy_((words[0] >> 8).to(torch.float32) / _UNIFORM_SCALE)


def argmax(logits: torch.Tensor, out: torch.Tensor) -> None:
    op = "sampling.argmax"
    _check_logits(op, logits)
    require_like(op, "out", out, logits.shape[:1], torch.int64)
    wide = logits.float()
    ids = torch.arange(wide.shape[-1], device=wide.device)
    # The smallest id among the maxima; torch.argmax does not promise which one it returns.
    best = wide.amax(dim=-1, keepdim=True)
    out.copy_(torch.where(wide == best, ids, wide.shape[-1]).amin(dim=-1))


def sample(
    logits: torch.Tensor,
    temperature: torch.Tensor,
    top_k: torch.Tensor,
    top_p: torch.Tensor,
    min_p: torch.Tensor,
    u: torch.Tensor,
    out: torch.Tensor,
) -> None:
    op = "sampling.sample"
    _check_logits(op, logits)
    rows, vocab = logits.shape
    for name, tensor in (
        ("temperature", temperature),
        ("top_p", top_p),
        ("min_p", min_p),
        ("u", u),
    ):
        require_like(op, name, tensor, (rows,), torch.float32)
    require_like(op, "top_k", top_k, (rows,), torch.int32)
    require_like(op, "out", out, (rows,), torch.int64)

    greedy = temperature <= 0
    x = logits.float() / torch.where(greedy, 1.0, temperature)[:, None]
    # One stable descending sort gives the canonical order: ties keep the smaller id first.
    # Softmax is monotone, so it is also the order of p except among probabilities that tie
    # after rounding, which are all too small to matter to the inverse CDF.
    sorted_x, order = torch.sort(x, dim=-1, descending=True, stable=True)
    rank = torch.arange(vocab, device=x.device)[None, :]

    keep = (top_k[:, None] == 0) | (rank < top_k[:, None])
    p = torch.softmax(torch.where(keep, sorted_x, -torch.inf), dim=-1)

    cumulative = torch.cumsum(p, dim=-1)
    before = torch.zeros_like(cumulative)
    before[:, 1:] = cumulative[:, :-1]
    keep &= (before < top_p[:, None]) | (rank == 0)
    keep &= p >= min_p[:, None] * p[:, :1]

    kept = torch.where(keep, p, 0.0)
    cdf = torch.cumsum(kept, dim=-1)
    # The first element whose cumulative probability exceeds u * total. The index is clamped to
    # the last element with probability, so rounding at the top of the CDF cannot walk off it.
    first = (cdf <= u[:, None] * cdf[:, -1:]).sum(dim=-1)
    last = ((kept > 0).sum(dim=-1) - 1).clamp(min=0)
    chosen = order.gather(1, torch.minimum(first, last)[:, None])[:, 0]
    out.copy_(torch.where(greedy, order[:, 0], chosen))


def _check_logits(op: str, logits: torch.Tensor) -> None:
    require_compute_dtype(op, "logits", logits)
    if logits.ndim != 2 or logits.shape[-1] == 0:
        raise BackendError(
            f"{op}: logits must be [rows, vocab], got {list(logits.shape)}",
            context={"op": op, "argument": "logits"},
        )
    if capturing(logits):
        return
    wide = logits.float()
    unusable = torch.isnan(wide) | (wide == torch.inf) | ~torch.isfinite(wide).any(dim=-1)[:, None]
    if bool(unusable.any()):
        raise BackendError(
            f"{op}: every row of logits needs a finite value, and none may be NaN or +inf",
            context={"op": op, "argument": "logits"},
        )
