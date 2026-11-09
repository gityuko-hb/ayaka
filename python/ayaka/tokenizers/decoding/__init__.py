"""Turning generated token ids back into text, one request at a time.

``incremental``  the detokenizer base class and the stream and slow paths.
``byte``         the fast path over a verified token-to-bytes table.
``stop``         stop-string matching shared by every path.

Public names are resolved from their submodule on first access.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ayaka.utils.lazy_exports import build_lazy_exports

if TYPE_CHECKING:
    from ayaka.tokenizers.decoding.byte import ByteIncrementalDetokenizer, verify_byte_table
    from ayaka.tokenizers.decoding.incremental import (
        FastIncrementalDetokenizer,
        IncrementalDetokenizer,
        SlowIncrementalDetokenizer,
    )
    from ayaka.tokenizers.decoding.stop import StopChecker

__all__ = [
    "ByteIncrementalDetokenizer",
    "FastIncrementalDetokenizer",
    "IncrementalDetokenizer",
    "SlowIncrementalDetokenizer",
    "StopChecker",
    "verify_byte_table",
]

_EXPORTS = {
    "byte": ("ByteIncrementalDetokenizer", "verify_byte_table"),
    "incremental": (
        "FastIncrementalDetokenizer",
        "IncrementalDetokenizer",
        "SlowIncrementalDetokenizer",
    ),
    "stop": ("StopChecker",),
}

__getattr__, __dir__ = build_lazy_exports(__name__, _EXPORTS, globals())
