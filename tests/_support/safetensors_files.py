"""Write ``.safetensors`` files for tests, well-formed or deliberately broken.

The writer follows the format and nothing else: header length, JSON header
padded with spaces to a multiple of eight bytes, then the payload. It does not
use the ``safetensors`` package, so the tests that use it need no extra.
"""

from __future__ import annotations

import json
import struct
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

#: A tensor to write: safetensors dtype token, shape, payload bytes.
type TensorData = tuple[str, tuple[int, ...], bytes]

ITEMSIZE = {"F64": 8, "F32": 4, "F16": 2, "BF16": 2, "I64": 8, "I32": 4, "I8": 1, "U8": 1}


def filled(token: str, shape: tuple[int, ...], start: int = 0) -> TensorData:
    """A tensor whose bytes count up from ``start``, so a misplaced read is visible."""
    nbytes = int(np.prod(shape, dtype=np.int64)) * ITEMSIZE[token]
    payload = bytes((start + index) % 251 for index in range(nbytes))
    return token, shape, payload


def write_safetensors(
    path: Path,
    tensors: Mapping[str, TensorData],
    *,
    metadata: Mapping[str, str] | None = None,
) -> Path:
    """Write a valid file with the tensors laid out back to back, in the given order."""
    header: dict[str, object] = {}
    if metadata is not None:
        header["__metadata__"] = dict(metadata)
    payload = bytearray()
    for name, (token, shape, data) in tensors.items():
        header[name] = {
            "dtype": token,
            "shape": list(shape),
            "data_offsets": [len(payload), len(payload) + len(data)],
        }
        payload += data
    return write_raw(path, json.dumps(header).encode(), bytes(payload))


def write_raw(
    path: Path, header: bytes, payload: bytes = b"", *, declared_length: int | None = None
) -> Path:
    """Write ``header`` as given, padded to eight bytes, and ``payload`` after it.

    ``declared_length`` overrides the length written in the first eight bytes,
    to build a header that claims more (or less) than it holds.
    """
    padded = header + b" " * (-len(header) % 8)
    length = len(padded) if declared_length is None else declared_length
    path.write_bytes(struct.pack("<Q", length) + padded + payload)
    return path


def write_entries(path: Path, entries: Mapping[str, object], payload: bytes = b"") -> Path:
    """Write a file whose header holds ``entries`` exactly as given, valid or not."""
    return write_raw(path, json.dumps(dict(entries)).encode(), payload)
