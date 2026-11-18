"""One request's constraint as it runs: which tokens the next draw may pick.

:class:`ConstraintMatcher` is a small state machine around a grammar matcher::

    WAIT_START   free; waits for ``start_after`` (the end of a reasoning block)
    WAIT_OPEN    free; waits for the model to open a tool call (a lazy call)
    FORCE        only the opening marker may follow (a call the request forces)
    ON           the grammar decides (JSON, or the body of a call)
    END          only an end-of-sequence token may follow

A free state needs no mask at all, so a request whose constraint has not started pays nothing and
its decode steps can still replay a CUDA graph. The matcher only learns **committed** tokens, the
ones the scheduler emitted, through :meth:`ConstraintMatcher.accept`; the mask it gives is for
the token after them.

A grammar error (a token the grammar refused, which the mask should have prevented) moves the
matcher to ``END``: the output stays as the model wrote it and is never repaired.
"""

from __future__ import annotations

import logging
from enum import Enum, auto
from typing import TYPE_CHECKING, Protocol

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Collection, Iterable

    from numpy.typing import NDArray

__all__ = ["ConstraintMatcher", "GrammarState"]

_LOG = logging.getLogger(__name__)


class GrammarState(Protocol):
    """The part of a grammar matcher (llguidance's ``LLMatcher``) a constraint uses."""

    def compute_bitmask(self) -> bytes: ...

    def consume_token(self, sampled_token: int) -> bool: ...

    def is_stopped(self) -> bool: ...

    def is_error(self) -> bool: ...

    def get_error(self) -> str: ...

    def deep_copy(self) -> GrammarState: ...


class _State(Enum):
    WAIT_START = auto()
    WAIT_OPEN = auto()
    FORCE = auto()
    ON = auto()
    END = auto()


class ConstraintMatcher:
    """The running constraint of one request.

    Args:
        template: The compiled grammar in its start state; each start of the grammar works on a
            copy, so the template is shared by every request with the same constraint.
        vocab_size: Bits of a mask.
        eos_token_ids: The tokens ``END`` allows.
        start_after: A token that switches the constraint on, or ``None`` to start at once.
        open_token: The opening marker of a tool call, or ``None`` for a JSON constraint.
        forced: Whether the answer must start with a call.
        parallel: Whether the model may write on (more calls) after a call; otherwise ``END``.
    """

    def __init__(
        self,
        template: GrammarState,
        *,
        vocab_size: int,
        eos_token_ids: Collection[int],
        start_after: int | None = None,
        open_token: int | None = None,
        forced: bool = False,
        parallel: bool = True,
    ) -> None:
        self._template = template
        self._words = (vocab_size + 31) // 32
        self._eos = tuple(token for token in eos_token_ids if token < vocab_size)
        self._start_after = start_after
        self._open = open_token
        self._forced = forced
        self._parallel = parallel
        self._grammar: GrammarState | None = None
        self._state = _State.WAIT_START
        if start_after is None:
            self._begin()

    @property
    def active(self) -> bool:
        """Whether the next draw is constrained."""
        if self._state is _State.END:
            return bool(self._eos)
        return self._state in (_State.FORCE, _State.ON)

    @property
    def state(self) -> str:
        """The state's name, for tests and logs."""
        return self._state.name

    def _begin(self) -> None:
        if self._open is None:
            self._start_grammar()
        else:
            self._state = _State.FORCE if self._forced else _State.WAIT_OPEN

    def _start_grammar(self) -> None:
        self._grammar = self._template.deep_copy()
        self._state = _State.ON

    def mask(self) -> NDArray[np.uint32] | None:
        """The allowed tokens of the next draw as ``uint32`` words, or ``None`` if free."""
        if self._state is _State.ON:
            assert self._grammar is not None  # noqa: S101  # ON always has a grammar
            words = np.frombuffer(self._grammar.compute_bitmask(), dtype="<u4")
            return words[: self._words].astype(np.uint32)
        if self._state is _State.FORCE:
            assert self._open is not None  # noqa: S101  # FORCE exists only for tool calls
            return self._only((self._open,))
        if self._state is _State.END and self._eos:
            return self._only(self._eos)
        return None  # free, or an end the model has no token for

    def _only(self, tokens: Iterable[int]) -> NDArray[np.uint32]:
        words = np.zeros(self._words, dtype=np.uint32)
        for token in tokens:
            words[token // 32] |= np.uint32(1 << (token % 32))
        return words

    def accept(self, tokens: Iterable[int]) -> None:
        """Learn committed tokens, in order."""
        for token in tokens:
            self._accept(token)

    def _accept(self, token: int) -> None:
        state = self._state
        if state is _State.WAIT_START:
            if token == self._start_after:
                self._begin()
        elif state in (_State.WAIT_OPEN, _State.FORCE):
            if token == self._open:
                self._start_grammar()
        elif state is _State.ON:
            grammar = self._grammar
            assert grammar is not None  # noqa: S101  # ON always has a grammar
            if not grammar.consume_token(token) or grammar.is_error():
                _LOG.debug("constraint: the grammar refused a token: %s", grammar.get_error())
                self._state = _State.END
            elif grammar.is_stopped():
                self._grammar = None
                if self._open is not None and self._parallel:
                    self._state = _State.WAIT_OPEN
                else:
                    self._state = _State.END
