"""Write GGUF files for tests: any metadata, any tensor table, valid or deliberately broken."""

from __future__ import annotations

import struct
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from pathlib import Path

#: Value type ids of the GGUF format.
U8, I8, U16, I16, U32, I32, F32, BOOL, STRING, ARRAY, U64, I64, F64 = range(13)
_FORMATS = {
    U8: "<B",
    I8: "<b",
    U16: "<H",
    I16: "<h",
    U32: "<I",
    I32: "<i",
    F32: "<f",
    BOOL: "<?",
    U64: "<Q",
    I64: "<q",
    F64: "<d",
}

type Value = tuple[int, object]
"""A metadata value and its type id: ``(U32, 7)``, ``(ARRAY, (F32, [...]))``."""


def string(text: str) -> bytes:
    raw = text.encode()
    return struct.pack("<Q", len(raw)) + raw


def value(kind: int, item: object) -> bytes:
    if kind == STRING:
        return string(str(item))
    if kind == ARRAY:
        item_kind, items = item  # type: ignore[misc]  # (type, list)
        body = b"".join(value(item_kind, entry) for entry in items)  # type: ignore[has-type]
        return struct.pack("<IQ", item_kind, len(items)) + body  # type: ignore[has-type]
    return struct.pack(_FORMATS[kind], item)


def tensor_info(name: str, ne: Sequence[int], type_id: int, offset: int) -> bytes:
    """One entry of the table; ``ne`` is in GGUF order (innermost first)."""
    dims = struct.pack("<I", len(ne)) + b"".join(struct.pack("<Q", d) for d in ne)
    return string(name) + dims + struct.pack("<IQ", type_id, offset)


def write_gguf(
    path: Path,
    metadata: Mapping[str, Value],
    tensors: Sequence[tuple[str, Sequence[int], int, int]],
    *,
    data_bytes: int,
    version: int = 3,
    alignment: int = 32,
    magic: bytes = b"GGUF",
    tensor_count: int | None = None,
) -> Path:
    """Write a header, the table of ``(name, ne, type id, offset)`` and ``data_bytes`` of zeros."""
    header = magic + struct.pack(
        "<IQQ", version, len(tensors) if tensor_count is None else tensor_count, len(metadata)
    )
    for key, (kind, item) in metadata.items():
        header += string(key) + struct.pack("<I", kind) + value(kind, item)
    for name, ne, type_id, offset in tensors:
        header += tensor_info(name, ne, type_id, offset)
    padding = -len(header) % alignment
    path.write_bytes(header + b"\0" * padding + b"\0" * data_bytes)
    return path


def llama_metadata(**overrides: Value) -> dict[str, Value]:
    """The metadata of a tiny ``llama`` GGUF: 2 layers, width 32, 4 heads of 8, 2 KV heads."""
    fields: dict[str, Value] = {
        "general.architecture": (STRING, "llama"),
        "llama.context_length": (U32, 128),
        "llama.embedding_length": (U32, 32),
        "llama.block_count": (U32, 2),
        "llama.feed_forward_length": (U32, 64),
        "llama.attention.head_count": (U32, 4),
        "llama.attention.head_count_kv": (U32, 2),
        "llama.rope.freq_base": (F32, 10000.0),
        "llama.attention.layer_norm_rms_epsilon": (F32, 1e-5),
        "tokenizer.ggml.tokens": (ARRAY, (STRING, [f"t{i}" for i in range(256)])),
    }
    fields.update(overrides)
    return fields
