"""Keeping Python's garbage collector out of latency-sensitive work.

Two uses. While a CUDA graph is captured, a collection that runs a finaliser
which frees a CUDA tensor would put that free into the capture; :func:`paused_gc`
collects once and then keeps the collector off for the block. After start-up,
:func:`freeze_startup_objects` moves every object alive then (the model, the
caches, the compiled graphs) out of the collector's view, so later collections
do not walk them on every decode step.

Nothing here imports torch.
"""

from __future__ import annotations

import gc
from contextlib import contextmanager
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Generator

__all__ = ["freeze_startup_objects", "paused_gc"]


@contextmanager
def paused_gc() -> Generator[None]:
    """Collect, then run the block with automatic collection off; restore it after.

    Nesting is safe: an inner block leaves collection as the outer one set it.
    """
    was_enabled = gc.isenabled()
    gc.collect()
    gc.disable()
    try:
        yield
    finally:
        if was_enabled:
            gc.enable()


def freeze_startup_objects() -> int:
    """Collect once and move every surviving object to the permanent generation.

    Returns:
        How many objects were frozen (``gc.get_freeze_count()``).
    """
    gc.collect()
    gc.freeze()
    return gc.get_freeze_count()
