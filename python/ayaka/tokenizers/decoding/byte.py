"""Fast detokenization from a verified token-to-bytes table.

Each token id maps to the exact bytes it stands for, so decoding is a table lookup
plus an incremental UTF-8 decoder, with no per-token call into the tokenizer::

    adapter.verify_byte_path()          one-time probe at load (verify_byte_table)
          |  passes
          v
    ByteIncrementalDetokenizer          token id --> table[id] --> UTF-8 decoder --> text
          |  shares the base class: stop matching, finish(), locking
          v
    decoding.incremental.IncrementalDetokenizer

The probe is a capability check, not a proof for arbitrary decoder pipelines: until
it passes, :meth:`IncrementalDetokenizer.create` never selects this path.
"""

from __future__ import annotations

import codecs
import random
from typing import TYPE_CHECKING

from ayaka.tokenizers.decoding.incremental import IncrementalDetokenizer
from ayaka.utils.validation import require_int

if TYPE_CHECKING:
    from collections.abc import Sequence
    from threading import RLock

    from ayaka.configs.tokenizer import DetokenizeParams
    from ayaka.tokenizers.ports import ByteTokenizerLike

__all__ = ["ByteIncrementalDetokenizer", "verify_byte_table"]

_MAX_PROBE_LENGTH = 24


def verify_byte_table(
    tokenizer: ByteTokenizerLike, samples: Sequence[str], *, n_random: int = 512
) -> bool:
    """Check that concatenated table bytes equal ``decode`` under both skip policies.

    Covers the given sample texts, every special token on its own and ``n_random``
    seeded random id sequences, so the result is reproducible. Only valid ids are
    sampled (added and special tokens included).

    Args:
        tokenizer: The adapter whose table is being checked.
        samples: Texts to encode and compare, typically multilingual probes.
        n_random: How many random id sequences to compare.

    Returns:
        ``True`` if every comparison matched; ``False`` for an empty or
        inconsistent table.
    """
    table = tokenizer.token_bytes_table()
    if not table:
        return False
    valid = tuple(sorted(tokenizer.valid_token_ids))
    if not valid or valid[0] < 0 or valid[-1] >= len(table):
        return False
    specials = tokenizer.all_special_ids

    def matches(ids: Sequence[int]) -> bool:
        for skip in (False, True):
            kept = [token for token in ids if not (skip and token in specials)]
            data = b"".join(table[token] for token in kept)
            expected = tokenizer.decode(ids, skip_special_tokens=skip)
            if data.decode("utf-8", errors="replace") != expected:
                return False
        return True

    for sample in samples:
        if not matches(tokenizer.encode(sample, add_special_tokens=False)):
            return False
    if not all(matches([token]) for token in specials):
        return False
    rng = random.Random(0)  # noqa: S311  # deterministic probe, not security
    return all(
        matches(rng.choices(valid, k=rng.randint(1, _MAX_PROBE_LENGTH))) for _ in range(n_random)
    )


class ByteIncrementalDetokenizer(IncrementalDetokenizer):
    """Table-driven detokenizer; requires ``skip_special_tokens=True``.

    Special tokens contribute no text. A prompt that ends in the middle of a UTF-8
    character is rejected, because the output could not continue it cleanly.

    Attributes:
        _tbl: Shared read-only bytes-per-id table; never copied per request.
        _valid_ids: Legal token ids, shared with the adapter.
        _special: Ids that decode to nothing.
        _dec: Incremental UTF-8 decoder holding any partial character.
    """

    def __init__(
        self,
        tokenizer: ByteTokenizerLike,
        params: DetokenizeParams,
        *,
        prompt_token_ids: Sequence[int] = (),
        lock: RLock | None = None,
    ) -> None:
        super().__init__(params, lock=lock)
        if not params.skip_special_tokens:
            raise ValueError("byte backend requires skip_special_tokens=True")
        self._tbl = tokenizer.token_bytes_table()
        if not self._tbl:
            raise ValueError("empty byte table")
        self._valid_ids = tokenizer.valid_token_ids
        self._special = tokenizer.all_special_ids
        self._dec = codecs.getincrementaldecoder("utf-8")("replace")
        for token in prompt_token_ids:
            require_int(token, "prompt token id")
            self._validate_token(token)
            self._decode_next(token)
        if self._dec.getstate()[0]:
            raise ValueError("prompt ends inside a UTF-8 character")

    def _validate_token(self, token_id: int) -> None:
        if token_id not in self._valid_ids or token_id >= len(self._tbl):
            raise ValueError(f"unknown token id: {token_id}")

    def _decode_next(self, token_id: int) -> str:
        if token_id in self._special:
            return ""
        return self._dec.decode(self._tbl[token_id])

    def _decode_tail(self) -> str:
        return self._dec.decode(b"", final=True)
