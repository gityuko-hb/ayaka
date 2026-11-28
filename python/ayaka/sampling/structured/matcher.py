"""One request's running output constraint with token masking, validation, and rollback."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum, auto
from typing import TYPE_CHECKING, Any, Protocol

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Collection, Iterable, Sequence

    from numpy.typing import NDArray

__all__ = ["ConstraintMatcher", "GrammarState"]

_LOG = logging.getLogger(__name__)


class GrammarState(Protocol):
    """Minimal compiled grammar interface supported by both backends and test doubles."""

    def compute_bitmask(self) -> bytes: ...

    def consume_token(self, sampled_token: int) -> bool: ...

    def is_stopped(self) -> bool: ...

    def is_error(self) -> bool: ...

    def get_error(self) -> str: ...

    def deep_copy(self) -> Any: ...


class _State(Enum):
    WAIT_START = auto()
    WAIT_OPEN = auto()
    WAIT_BRACE = auto()
    FREE = auto()
    FORCE = auto()
    ON = auto()
    END = auto()


@dataclass(frozen=True, slots=True)
class _Step:
    state: _State
    reasoning_open: int | None
    seen: tuple[int, ...]
    grammar: Any
    grammar_consumed: bool
    grammar_snapshot: Any = None


class ConstraintMatcher:
    """Running constraint state machine for one request, supporting rollback and draft windows."""

    def __init__(
        self,
        template: GrammarState,
        *,
        vocab_size: int,
        eos_token_ids: Collection[int],
        start_after: int | None = None,
        reasoning_open: int | None = None,
        opening: tuple[int, ...] = (),
        forced: bool = False,
        parallel: bool = True,
        brace: Collection[int] | None = None,
        skip: Collection[int] = (),
    ) -> None:
        self._template = template
        self._words = (vocab_size + 31) // 32
        self._eos = tuple(token for token in eos_token_ids if token < vocab_size)
        self._start_after = start_after
        self._opening = opening
        self._seen: list[int] = []
        self._forced = forced
        self._parallel = parallel
        self._brace = brace
        self._skip = skip
        self._grammar: Any = None
        self._state = _State.WAIT_START
        self._reasoning_open = reasoning_open if start_after is not None else None
        self._history: list[_Step] = []
        if start_after is None or self._reasoning_open is not None:
            self._begin()

    @property
    def active(self) -> bool:
        """Whether the next draw is constrained."""
        if self._reasoning_open is not None and self._state is _State.WAIT_OPEN:
            return False
        if self._state is _State.END:
            return bool(self._eos)
        return self._state in (_State.FORCE, _State.ON)

    @property
    def state(self) -> str:
        """Current state name for diagnostics and tests."""
        return "MAYBE" if self._reasoning_open is not None else self._state.name

    def _begin(self) -> None:
        if self._brace is not None:
            self._state = _State.WAIT_BRACE
        elif not self._opening:
            self._start_grammar()
        else:
            self._seen.clear()
            self._state = _State.FORCE if self._forced else _State.WAIT_OPEN

    def _start_grammar(self) -> None:
        self._grammar = self._template.deep_copy()
        self._state = _State.ON

    def _wait_brace(self, token: int) -> None:
        if token in self._skip:
            return
        grammar = self._template.deep_copy()
        brace = self._brace
        if (
            brace is not None
            and token in brace
            and grammar.consume_token(token)
            and not grammar.is_error()
        ):
            self._grammar = grammar
            self._state = _State.ON
            return
        self._state = _State.FREE

    def mask(self) -> NDArray[np.uint32] | None:
        """Allowed tokens for the next draw as ``uint32`` words, or ``None`` when free."""
        words = self._mask()
        opening = self._reasoning_open
        if words is not None and opening is not None and opening // 32 < self._words:
            words = words.copy()
            words[opening // 32] |= np.uint32(1 << (opening % 32))
        return words

    def _mask(self) -> NDArray[np.uint32] | None:
        if self._state is _State.ON:
            assert self._grammar is not None  # noqa: S101
            fill = getattr(self._grammar, "fill_bitmask", None)
            if callable(fill):
                words = np.asarray(fill(), dtype=np.uint32)
                return words[: self._words]
            words = np.frombuffer(self._grammar.compute_bitmask(), dtype="<u4")
            return words[: self._words].astype(np.uint32)
        if self._state is _State.FORCE:
            return self._only((self._opening[len(self._seen)],))
        if self._state is _State.END and self._eos:
            return self._only(self._eos)
        return None

    def _only(self, tokens: Iterable[int]) -> NDArray[np.uint32]:
        words = np.zeros(self._words, dtype=np.uint32)
        for token in tokens:
            words[token // 32] |= np.uint32(1 << (token % 32))
        return words

    def accept(self, tokens: Iterable[int]) -> None:
        """Advance state through committed ``tokens`` in order."""
        for token in tokens:
            self._accept(token)

    def _accept(self, token: int) -> bool:
        prev_state = self._state
        prev_reasoning = self._reasoning_open
        prev_seen = tuple(self._seen)
        prev_grammar = self._grammar
        has_native_rb = prev_grammar is not None and callable(
            getattr(prev_grammar, "rollback", None)
        )
        fallback_snap = (
            prev_grammar.deep_copy()
            if prev_grammar is not None and prev_state is _State.ON and not has_native_rb
            else None
        )
        grammar_consumed = False
        accepted = True

        if self._reasoning_open is not None:
            opened = token == self._reasoning_open
            self._reasoning_open = None
            if opened:
                self._grammar = None
                self._seen.clear()
                self._state = _State.WAIT_START
                self._history.append(
                    _Step(prev_state, prev_reasoning, prev_seen, prev_grammar, False, fallback_snap)
                )
                return True

        state = self._state
        if state is _State.WAIT_START:
            if token == self._start_after:
                self._begin()
        elif state is _State.FORCE:
            expected = self._opening[len(self._seen)]
            accepted = token == expected
            self._seen.append(token)
            if len(self._seen) == len(self._opening):
                self._start_grammar()
        elif state is _State.WAIT_BRACE:
            self._wait_brace(token)
        elif state is _State.WAIT_OPEN:
            self._seen.append(token)
            del self._seen[: -len(self._opening)]
            if tuple(self._seen) == self._opening:
                self._start_grammar()
        elif state is _State.ON:
            grammar = self._grammar
            assert grammar is not None  # noqa: S101
            if not grammar.consume_token(token) or grammar.is_error():
                _LOG.debug("constraint: the grammar refused a token: %s", grammar.get_error())
                self._state = _State.END
                accepted = False
            else:
                grammar_consumed = True
                if grammar.is_stopped():
                    self._grammar = None
                    if self._opening and self._parallel:
                        self._seen.clear()
                        self._state = _State.WAIT_OPEN
                    else:
                        self._state = _State.END
        elif state is _State.END:
            accepted = token in self._eos

        self._history.append(
            _Step(
                prev_state, prev_reasoning, prev_seen, prev_grammar, grammar_consumed, fallback_snap
            )
        )
        return accepted

    def rollback(self, num_tokens: int) -> None:
        """Undo the last ``num_tokens`` accepted transitions."""
        if num_tokens <= 0:
            return
        steps = min(num_tokens, len(self._history))
        for _ in range(steps):
            step = self._history.pop()
            self._state = step.state
            self._reasoning_open = step.reasoning_open
            self._seen = list(step.seen)
            if step.grammar_snapshot is not None:
                self._grammar = step.grammar_snapshot
            else:
                self._grammar = step.grammar
                if step.grammar_consumed and self._grammar is not None:
                    self._grammar.rollback(1)

    def validate_tokens(self, tokens: Sequence[int]) -> bool:
        """Check whether ``tokens`` form a valid continuation without mutating committed state."""
        if not tokens:
            return True
        if (
            self._state is _State.ON
            and self._reasoning_open is None
            and not (self._opening and self._parallel)
            and self._grammar is not None
            and callable(getattr(self._grammar, "validate_tokens", None))
        ):
            return bool(self._grammar.validate_tokens(tokens))
        taken = 0
        try:
            for tok in tokens:
                taken += 1
                if not self._accept(tok):
                    return False
            return True
        finally:
            self.rollback(taken)

    def window_masks(self, draft_tokens: Sequence[int]) -> list[NDArray[np.uint32] | None]:
        """Compute exact per-row allowed-token masks across a speculative verification window."""
        masks: list[NDArray[np.uint32] | None] = [self.mask()]
        if not draft_tokens:
            return masks
        taken = 0
        try:
            for draft in draft_tokens:
                self._accept(draft)
                taken += 1
                masks.append(self.mask())
        finally:
            self.rollback(taken)
        return masks
