"""A bounded memory of token-id tuples a tokenizer service has already checked.

The server encodes a prompt, then hands the ids to the engine, which used to check every id
again: tens of milliseconds of Python for a long prompt, on the thread that runs the model. Instead
the service remembers the tuple it returned, by identity, and the engine asks whether it is one of
them::

    ids = service.encode(value, options, vouch=True).token_ids    # remembered here
    ...
    service.take_vouched(ids, options)                            # forgotten here, not re-checked

Identity, not equality: only the very tuple the service built is trusted, and the memory holds
it, so its ``id`` cannot be reused by another object while it is remembered. Each vouching encode
is good for one take (an identical prompt served from the encode cache can be vouched for twice).
The memory is bounded by count and by ids; a tuple it forgot is simply checked again, so the bound
costs speed, never correctness.
"""

from __future__ import annotations

import threading
from collections import OrderedDict

__all__ = ["VouchedIds"]


class VouchedIds:
    """Remembers checked tuples of token ids until they are taken; safe to use from any thread.

    Args:
        max_entries: Most tuples remembered at once; the oldest is forgotten first.
        max_ids: Most ids remembered in all.
    """

    def __init__(self, *, max_entries: int, max_ids: int) -> None:
        if max_entries < 1 or max_ids < 1:
            raise ValueError("max_entries and max_ids must be >= 1")
        self._max_entries = max_entries
        self._max_ids = max_ids
        # id(ids) -> (ids, how many vouches are left for this very tuple)
        self._held: OrderedDict[int, tuple[tuple[int, ...], int]] = OrderedDict()
        self._ids = 0
        self._lock = threading.Lock()

    def __len__(self) -> int:
        return len(self._held)

    def vouch(self, ids: tuple[int, ...]) -> None:
        """Remember ``ids`` as checked, once more; forget the oldest tuples beyond the bounds."""
        with self._lock:
            key = id(ids)
            held = self._held.pop(key, None)
            if held is None:
                self._ids += len(ids)
            self._held[key] = (ids, 1 if held is None else held[1] + 1)
            while self._held and (len(self._held) > self._max_entries or self._ids > self._max_ids):
                _, (dropped, _count) = self._held.popitem(last=False)
                self._ids -= len(dropped)

    def take(self, ids: tuple[int, ...]) -> bool:
        """Use up one vouch for this very tuple; ``False`` if there is none."""
        with self._lock:
            key = id(ids)
            held = self._held.get(key)
            if held is None or held[0] is not ids:
                return False
            if held[1] > 1:
                self._held[key] = (ids, held[1] - 1)
            else:
                del self._held[key]
                self._ids -= len(ids)
            return True

    def clear(self) -> None:
        """Forget everything."""
        with self._lock:
            self._held.clear()
            self._ids = 0
