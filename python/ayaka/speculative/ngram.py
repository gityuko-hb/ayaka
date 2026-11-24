"""Model-free n-gram lookup index and proposer (prompt + committed output + predicted output).

Every trigram of the history maps to a linked chain of its recent end positions in a compact
open-addressing hash table backed by ``array`` and ``numpy`` arrays (~28-40 bytes per history
token at 100K tokens, well below the 128-byte threshold in #127). Matching extends backwards up
to ``max_match`` tokens, inspects up to ``max_candidates`` recent occurrences, and returns the
tokens immediately following the longest match (most recent on ties).

Only committed tokens enter the index; draft tokens never do.
"""

from __future__ import annotations

from array import array
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, ClassVar, Final

import numpy as np

from ayaka.speculative.proposer import EMPTY_PROPOSAL, Proposal

if TYPE_CHECKING:
    from collections.abc import Collection, Sequence

    from numpy.typing import NDArray

    from ayaka.configs.model import ModelConfig
    from ayaka.core.types import RequestId, TokenId

__all__ = ["LookupProposer", "NgramIndex", "naive_ngram_match"]

_EMPTY_KEY: Final[int] = 0xFFFFFFFFFFFFFFFF
_U64_MASK: Final[int] = 0xFFFFFFFFFFFFFFFF


def _pack_trigram(t0: int, t1: int, t2: int) -> int:
    """Pack three non-negative token ids into a non-sentinel 64-bit integer key."""
    # Mix across 64 bits so arbitrarily large vocabularies work while keeping _EMPTY_KEY reserved.
    h = ((t0 & 0x1FFFFF) << 42) | ((t1 & 0x1FFFFF) << 21) | (t2 & 0x1FFFFF)
    h ^= ((t0 >> 21) * 0x9E3779B97F4A7C15) & _U64_MASK
    h ^= ((t1 >> 21) * 0xBF58476D1CE4E5B9) & _U64_MASK
    h ^= ((t2 >> 21) * 0x94D049BB133111EB) & _U64_MASK
    return 0 if h == _EMPTY_KEY else h


class NgramIndex:
    """Compact open-addressing trigram index over a single token sequence.

    Args:
        initial_tokens: Initial sequence to index (e.g., the request prompt).
        initial_capacity: Initial power-of-two slot count for the hash table.
    """

    def __init__(
        self, initial_tokens: Sequence[TokenId] = (), *, initial_capacity: int = 64
    ) -> None:
        cap = 16
        while cap < initial_capacity:
            cap *= 2
        self._history: array[int] = array("I")
        self._prev: array[int] = array("i")
        self._keys: NDArray[np.uint64] = np.full(cap, _EMPTY_KEY, dtype=np.uint64)
        self._heads: NDArray[np.int32] = np.full(cap, -1, dtype=np.int32)
        self._mask: int = cap - 1
        self._used: int = 0
        if initial_tokens:
            self.extend(initial_tokens)

    def __len__(self) -> int:
        return len(self._history)

    def memory_bytes(self) -> int:
        """Approximate memory occupied by the history, linked list, and hash table in bytes."""
        return (
            len(self._history) * self._history.itemsize
            + len(self._prev) * self._prev.itemsize
            + int(self._keys.nbytes)
            + int(self._heads.nbytes)
        )

    def _grow(self) -> None:
        old_keys = self._keys
        old_heads = self._heads
        new_cap = len(old_keys) * 2
        new_mask = new_cap - 1
        new_keys = np.full(new_cap, _EMPTY_KEY, dtype=np.uint64)
        new_heads = np.full(new_cap, -1, dtype=np.int32)
        keys_list = old_keys.tolist()
        heads_list = old_heads.tolist()
        for idx, key in enumerate(keys_list):
            if key == _EMPTY_KEY:
                continue
            slot = (key ^ (key >> 33)) & new_mask
            while new_keys[slot] != _EMPTY_KEY:
                slot = (slot + 1) & new_mask
            new_keys[slot] = key
            new_heads[slot] = heads_list[idx]
        self._keys = new_keys
        self._heads = new_heads
        self._mask = new_mask

    def _insert(self, pos: int) -> None:
        history = self._history
        key = _pack_trigram(history[pos - 2], history[pos - 1], history[pos])
        if (self._used + 1) * 2 > len(self._keys):
            self._grow()
        mask = self._mask
        keys = self._keys
        slot = (key ^ (key >> 33)) & mask
        while True:
            cur = int(keys[slot])
            if cur == _EMPTY_KEY:
                keys[slot] = key
                self._heads[slot] = pos
                self._used += 1
                return
            if cur == key:
                self._prev[pos] = int(self._heads[slot])
                self._heads[slot] = pos
                return
            slot = (slot + 1) & mask

    def _lookup_head(self, key: int) -> int:
        mask = self._mask
        keys = self._keys
        slot = (key ^ (key >> 33)) & mask
        while True:
            cur = int(keys[slot])
            if cur == key:
                return int(self._heads[slot])
            if cur == _EMPTY_KEY:
                return -1
            slot = (slot + 1) & mask

    def extend(self, tokens: Sequence[TokenId]) -> None:
        """Append committed tokens and index every new trigram end position."""
        if not tokens:
            return
        start = len(self._history)
        self._history.extend(tokens)
        self._prev.extend((-1,) * len(tokens))
        first_pos = max(2, start)
        for pos in range(first_pos, len(self._history)):
            self._insert(pos)

    def match(
        self,
        suffix: Sequence[TokenId],
        max_k: int,
        *,
        min_match: int = 3,
        max_match: int = 32,
        max_candidates: int = 4,
    ) -> tuple[tuple[TokenId, ...], int]:
        """Find the longest suffix match and return up to ``max_k`` following tokens.

        When ``min_match < 3``, falls back to scanning the history directly so short n-gram
        configurations still work cleanly; for ``min_match >= 3``, uses the O(1) trigram table.
        """
        if max_k <= 0 or len(suffix) < min_match or len(self._history) <= min_match:
            return (), 0
        if min_match < 3:
            return naive_ngram_match(
                self._history,
                suffix,
                max_k,
                min_match=min_match,
                max_match=max_match,
                max_candidates=max_candidates,
            )

        history = self._history
        prev = self._prev
        hist_len = len(history)
        suf_len = len(suffix)
        cap = min(max_match, suf_len)
        key = _pack_trigram(suffix[-3], suffix[-2], suffix[-1])
        pos = self._lookup_head(key)
        if pos < 0:
            return (), 0

        best_end = -1
        best_len = 0
        seen = 0
        s_last = suffix[-1]
        s_m2 = suffix[-2]
        s_m3 = suffix[-3]
        while pos >= 2:
            if pos + 1 < hist_len:
                if history[pos] == s_last and history[pos - 1] == s_m2 and history[pos - 2] == s_m3:
                    limit = min(cap, pos + 1)
                    matched = 3
                    while (
                        matched < limit and history[pos - matched] == suffix[suf_len - 1 - matched]
                    ):
                        matched += 1
                    if matched > best_len:
                        best_len = matched
                        best_end = pos
                        if best_len == cap:
                            break
                seen += 1
                if max_candidates > 0 and seen >= max_candidates:
                    break
            pos = prev[pos]

        if best_end < 0 or best_len < min_match:
            return (), 0
        end = min(hist_len, best_end + 1 + max_k)
        return tuple(history[best_end + 1 : end]), best_len


def naive_ngram_match(
    history: Sequence[TokenId],
    suffix: Sequence[TokenId],
    max_k: int,
    *,
    min_match: int = 3,
    max_match: int = 32,
    max_candidates: int = 4,
) -> tuple[tuple[TokenId, ...], int]:
    """Reference linear scan over ``history`` matching ``NgramIndex.match`` semantics.

    Scans backwards from the most recent end position, inspects at most ``max_candidates``
    trigram matches (0 = unlimited), and returns ``(tokens, best_len)``.
    """
    hist_len = len(history)
    suf_len = len(suffix)
    if max_k <= 0 or suf_len < min_match or hist_len <= min_match:
        return (), 0
    cap = min(max_match, suf_len)
    best_end = -1
    best_len = 0
    seen = 0
    for pos in range(hist_len - 2, min_match - 2, -1):
        # Check the min_match-length suffix ending at pos
        ok = True
        for d in range(min_match):
            if history[pos - d] != suffix[suf_len - 1 - d]:
                ok = False
                break
        if not ok:
            continue
        limit = min(cap, pos + 1)
        matched = min_match
        while matched < limit and history[pos - matched] == suffix[suf_len - 1 - matched]:
            matched += 1
        if matched > best_len:
            best_len = matched
            best_end = pos
            if best_len == cap:
                break
        seen += 1
        if max_candidates > 0 and seen >= max_candidates:
            break
    if best_end < 0 or best_len < min_match:
        return (), 0
    end = min(hist_len, best_end + 1 + max_k)
    return tuple(history[best_end + 1 : end]), best_len


@dataclass(slots=True)
class _RequestLookup:
    history: NgramIndex
    tail: list[int]
    prediction: tuple[TokenId, ...] = ()
    prediction_index: NgramIndex | None = None
    banned: frozenset[TokenId] = field(default_factory=frozenset[int])
    allowed: frozenset[TokenId] | None = None
    has_output: bool = False


class LookupProposer:
    """Per-request n-gram lookup drafter over committed history and optional predicted outputs."""

    name: ClassVar[str] = "lookup"

    def __init__(
        self,
        *,
        min_match: int = 3,
        max_match: int = 32,
        candidates: int = 4,
        accept_prediction: bool = True,
    ) -> None:
        self._min_match = min_match
        self._max_match = max_match
        self._candidates = candidates
        self._accept_prediction = accept_prediction
        self._requests: dict[RequestId, _RequestLookup] = {}

    def supports(self, model_config: ModelConfig) -> str | None:
        del model_config
        return None

    def on_admit(
        self,
        request_id: RequestId,
        prompt: Sequence[TokenId],
        *,
        prediction: Sequence[TokenId] = (),
        banned_tokens: Collection[TokenId] = (),
        allowed_tokens: Collection[TokenId] | None = None,
    ) -> None:
        index = NgramIndex(prompt)
        pred_tuple = tuple(prediction) if self._accept_prediction and prediction else ()
        pred_index = NgramIndex(pred_tuple) if pred_tuple else None
        tail = list(prompt[-self._max_match :])
        self._requests[request_id] = _RequestLookup(
            history=index,
            tail=tail,
            prediction=pred_tuple,
            prediction_index=pred_index,
            banned=frozenset(banned_tokens),
            allowed=frozenset(allowed_tokens) if allowed_tokens is not None else None,
        )

    def propose(self, request_id: RequestId, max_k: int) -> Proposal:
        state = self._requests.get(request_id)
        if state is None or max_k <= 0:
            return EMPTY_PROPOSAL

        tokens, conf = state.history.match(
            state.tail,
            max_k,
            min_match=self._min_match,
            max_match=self._max_match,
            max_candidates=self._candidates,
        )
        source = "lookup" if tokens else ""

        if state.prediction_index is not None:
            if not state.has_output and state.prediction:
                # Before any output exists, seed with the beginning of the prediction if history
                # does not already have a stronger match.
                pred_tokens = state.prediction[:max_k]
                pred_conf = self._min_match
            else:
                pred_tokens, pred_conf = state.prediction_index.match(
                    state.tail,
                    max_k,
                    min_match=self._min_match,
                    max_match=self._max_match,
                    max_candidates=self._candidates,
                )
            if pred_tokens and pred_conf >= conf:
                tokens = pred_tokens
                conf = pred_conf
                source = "prediction"

        if not tokens:
            return EMPTY_PROPOSAL
        filtered = _filter_tokens(tokens, state.banned, state.allowed)
        if not filtered:
            return EMPTY_PROPOSAL
        return Proposal(tokens=filtered, confidence=conf, source=source)

    def on_commit(self, request_id: RequestId, tokens: Sequence[TokenId]) -> None:
        state = self._requests.get(request_id)
        if state is None or not tokens:
            return
        state.history.extend(tokens)
        state.tail.extend(tokens)
        if len(state.tail) > self._max_match:
            del state.tail[: len(state.tail) - self._max_match]
        state.has_output = True

    def release(self, request_id: RequestId) -> None:
        self._requests.pop(request_id, None)

    def index_memory_bytes(self, request_id: RequestId) -> int:
        """Bytes held by the history index of ``request_id``, or 0 if unknown."""
        state = self._requests.get(request_id)
        return state.history.memory_bytes() if state is not None else 0


def _filter_tokens(
    tokens: tuple[TokenId, ...],
    banned: frozenset[TokenId],
    allowed: frozenset[TokenId] | None,
) -> tuple[TokenId, ...]:
    if not banned and allowed is None:
        return tokens
    kept: list[TokenId] = []
    for token in tokens:
        if token in banned:
            break
        if allowed is not None and token not in allowed:
            break
        kept.append(token)
    return tuple(kept)
