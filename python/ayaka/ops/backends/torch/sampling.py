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

__all__ = [
    "apply_logit_bias",
    "apply_penalties",
    "apply_token_bitmask",
    "argmax",
    "log_softmax",
    "philox_rounds",
    "sample",
    "uniform",
]

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
    *,
    kept_logprobs: torch.Tensor | None = None,
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
    if kept_logprobs is not None:
        require_like(op, "kept_logprobs", kept_logprobs, (rows, vocab), torch.float32)

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
    if kept_logprobs is not None:
        # The kept probabilities, renormalised, scattered back from the sorted order.
        logp = torch.log(kept / cdf[:, -1:])
        kept_logprobs.fill_(-torch.inf)
        kept_logprobs.scatter_(1, order, logp)


def apply_token_bitmask(logits: torch.Tensor, bitmask: torch.Tensor) -> None:
    op = "sampling.apply_token_bitmask"
    rows, vocab = _check_workspace(op, logits)
    words = (vocab + 31) // 32
    require_like(op, "bitmask", bitmask, (rows, words), torch.int32)
    # Token t is bit t % 8 of byte t // 8 of the row: the words are little endian in memory, on the
    # CPU and the GPU alike. One byte per token and no int64, so a batch of 32 rows at a 128k
    # vocabulary needs about 8 MiB of scratch (the int64 form needed about 67 MiB, which every
    # captured graph that applies a mask kept in its pool), and the call is a few kernels without
    # a host read, so it can be captured.
    shifts = torch.arange(8, device=bitmask.device, dtype=torch.uint8)
    bits = (bitmask.contiguous().view(torch.uint8).unsqueeze(-1) >> shifts).bitwise_and_(1)
    logits.masked_fill_(bits.reshape(rows, words * 32)[:, :vocab] == 0, -torch.inf)


def apply_logit_bias(
    logits: torch.Tensor, rows: torch.Tensor, tokens: torch.Tensor, biases: torch.Tensor
) -> None:
    op = "sampling.apply_logit_bias"
    _check_workspace(op, logits)
    count = rows.shape[0] if rows.ndim == 1 else -1
    require_like(op, "rows", rows, (count,), torch.int64)
    require_like(op, "tokens", tokens, (count,), torch.int64)
    require_like(op, "biases", biases, (count,), torch.float32)
    # A masked logit stays masked: -inf plus a finite bias is -inf.
    logits.index_put_((rows, tokens), biases, accumulate=True)


def apply_penalties(
    logits: torch.Tensor,
    prompt_and_output: torch.Tensor,
    output: torch.Tensor,
    repetition: torch.Tensor,
    frequency: torch.Tensor,
    presence: torch.Tensor,
) -> None:
    op = "sampling.apply_penalties"
    rows, vocab = _check_workspace(op, logits)
    require_like(op, "prompt_and_output", prompt_and_output, (rows, vocab), torch.int32)
    require_like(op, "output", output, (rows, vocab), torch.int32)
    for name, tensor in (
        ("repetition", repetition),
        ("frequency", frequency),
        ("presence", presence),
    ):
        require_like(op, name, tensor, (rows,), torch.float32)
    r = repetition[:, None]
    seen = prompt_and_output > 0
    scaled = torch.where(logits > 0, logits / r, logits * r)
    logits.copy_(torch.where(seen, scaled, logits))
    logits.sub_(frequency[:, None] * output.to(torch.float32))
    logits.sub_(presence[:, None] * (output > 0).to(torch.float32))


def log_softmax(logits: torch.Tensor, out: torch.Tensor) -> None:
    op = "sampling.log_softmax"
    rows, vocab = _check_workspace(op, logits)
    require_like(op, "out", out, (rows, vocab), torch.float32)
    torch.log_softmax(logits, dim=-1, out=out)


def _check_workspace(op: str, logits: torch.Tensor) -> tuple[int, int]:
    """A ``[rows, vocab]`` float32 workspace the transforms change in place."""
    if logits.ndim != 2 or logits.dtype != torch.float32:
        raise BackendError(
            f"{op}: logits must be a [rows, vocab] float32 workspace, "
            f"got {logits.dtype} {list(logits.shape)}",
            context={"op": op, "argument": "logits"},
        )
    return int(logits.shape[0]), int(logits.shape[1])


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
