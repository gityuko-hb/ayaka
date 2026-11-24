"""GGUF files: the header, the metadata and the tensor table, read defensively; no tensor data.

A GGUF file (versions 2 and 3, little-endian) is::

    magic "GGUF" | version u32 | tensor count u64 | metadata count u64
    metadata:  key (string) | value type u32 | value          (repeated)
    tensors:   name (string) | dims u32 | ne[dims] u64 | ggml type u32 | offset u64   (repeated)
    padding to ``general.alignment`` (32 by default) | tensor data

A string is a u64 length and that many UTF-8 bytes. ``ne[0]`` is the innermost dimension, so
the PyTorch shape is ``ne`` reversed. A tensor's offset is relative to the start of the data and
a multiple of the alignment; its size follows from its GGML type, whose blocks pack
``block_size`` elements into ``type_size`` bytes.

Everything read is checked before it is used: counts, string and array lengths and dimensions
have limits; every tensor lies inside the file, starts on the alignment and overlaps no other;
an unknown value type or GGML type is refused. :func:`read_gguf` never reads tensor data.

:func:`gguf_hf_config` turns the metadata of a ``llama`` file into the ``config.json`` fields
``ModelConfig.from_hf_config`` reads, so ``ayaka plan`` can plan a GGUF file. Running GGUF
tensors is not implemented.
"""

from __future__ import annotations

import enum
import itertools
import math
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, BinaryIO, Final, cast

from ayaka.core.errors import ModelLoadError

if TYPE_CHECKING:
    from collections.abc import Mapping

__all__ = ["GGMLType", "GGUFFile", "GGUFTensor", "gguf_hf_config", "read_gguf"]

_MAGIC: Final = b"GGUF"
_VERSIONS: Final = (2, 3)
_DEFAULT_ALIGNMENT: Final = 32
#: Limits on what a header may declare, far above any real file, so a corrupt or hostile
#: header fails with a message instead of allocating without bound.
MAX_TENSORS: Final = 1 << 20
MAX_METADATA: Final = 1 << 16
MAX_STRING_BYTES: Final = 1 << 24
MAX_ARRAY_ITEMS: Final = 1 << 24
MAX_DIMS: Final = 8


class GGMLType(enum.Enum):
    """A GGML tensor type: its id in the file, elements per block and bytes per block."""

    F32 = (0, 1, 4)
    F16 = (1, 1, 2)
    Q4_0 = (2, 32, 18)
    Q4_1 = (3, 32, 20)
    Q5_0 = (6, 32, 22)
    Q5_1 = (7, 32, 24)
    Q8_0 = (8, 32, 34)
    Q8_1 = (9, 32, 36)
    Q2_K = (10, 256, 84)
    Q3_K = (11, 256, 110)
    Q4_K = (12, 256, 144)
    Q5_K = (13, 256, 176)
    Q6_K = (14, 256, 210)
    Q8_K = (15, 256, 292)
    IQ2_XXS = (16, 256, 66)
    IQ2_XS = (17, 256, 74)
    IQ3_XXS = (18, 256, 98)
    IQ1_S = (19, 256, 50)
    IQ4_NL = (20, 32, 18)
    IQ3_S = (21, 256, 110)
    IQ2_S = (22, 256, 82)
    IQ4_XS = (23, 256, 136)
    I8 = (24, 1, 1)
    I16 = (25, 1, 2)
    I32 = (26, 1, 4)
    I64 = (27, 1, 8)
    F64 = (28, 1, 8)
    IQ1_M = (29, 256, 56)
    BF16 = (30, 1, 2)
    TQ1_0 = (34, 256, 54)
    TQ2_0 = (35, 256, 66)
    MXFP4 = (39, 32, 17)

    def __init__(self, type_id: int, block_size: int, type_size: int) -> None:
        self.type_id: Final[int] = type_id
        self.block_size: Final[int] = block_size
        self.type_size: Final[int] = type_size

    @classmethod
    def by_id(cls, type_id: int) -> GGMLType:
        """The type stored under ``type_id``.

        Raises:
            ValueError: If no GGML type has that id (removed types included: 4, 5, 31-33, 36-38).
        """
        for member in cls:
            if member.type_id == type_id:
                return member
        raise ValueError(f"unknown GGML type id {type_id}")

    def nbytes(self, shape: tuple[int, ...]) -> int:
        """Bytes of a tensor of ``shape`` (PyTorch order: the last dimension is ``ne[0]``).

        Raises:
            ValueError: If the innermost dimension is not a whole number of blocks.
        """
        inner = shape[-1] if shape else 1
        if inner % self.block_size:
            raise ValueError(
                f"{self.name} packs {self.block_size} elements per block, but the innermost "
                f"dimension is {inner}"
            )
        return math.prod(shape) // self.block_size * self.type_size


@dataclass(frozen=True, slots=True)
class GGUFTensor:
    """One tensor of the table: what it is and where its bytes are.

    Attributes:
        name: The tensor's name in the file (``blk.0.attn_q.weight``).
        shape: PyTorch order, the GGUF dimensions reversed.
        ggml_type: How its elements are stored.
        offset: Absolute byte offset of the first byte in the file.
        nbytes: Bytes it occupies, from its type and shape.
    """

    name: str
    shape: tuple[int, ...]
    ggml_type: GGMLType
    offset: int
    nbytes: int

    @property
    def end(self) -> int:
        """Offset one past the last byte."""
        return self.offset + self.nbytes


@dataclass(frozen=True, slots=True)
class GGUFFile:
    """What a GGUF header says.

    Attributes:
        path: The file.
        version: The GGUF version (2 or 3).
        metadata: Every key and its value: ints, floats, bools, strings and lists of them.
        tensors: The tensor table, in file order.
        alignment: The alignment of the data section and of every tensor in it.
        data_offset: Where the data section starts.
    """

    path: Path
    version: int
    metadata: Mapping[str, object]
    tensors: tuple[GGUFTensor, ...]
    alignment: int
    data_offset: int

    @property
    def weights_bytes(self) -> int:
        """Bytes of every tensor, as stored: the size of the weights the file describes."""
        return sum(tensor.nbytes for tensor in self.tensors)

    def type_counts(self) -> dict[str, int]:
        """How many tensors each GGML type stores, most common first."""
        counts: dict[str, int] = {}
        for tensor in self.tensors:
            counts[tensor.ggml_type.name] = counts.get(tensor.ggml_type.name, 0) + 1
        return dict(sorted(counts.items(), key=lambda item: (-item[1], item[0])))


#: Value types: id -> struct format of a scalar, or the name of a compound type.
_SCALARS: Final = {
    0: "<B",
    1: "<b",
    2: "<H",
    3: "<h",
    4: "<I",
    5: "<i",
    6: "<f",
    7: "<?",
    10: "<Q",
    11: "<q",
    12: "<d",
}
_STRING: Final = 8
_ARRAY: Final = 9


class _Reader:
    def __init__(self, handle: BinaryIO, path: Path, size: int) -> None:
        self._handle = handle
        self._path = path
        self._size = size

    @property
    def position(self) -> int:
        return self._handle.tell()

    def take(self, count: int, what: str) -> bytes:
        if count > self._size - self.position:
            raise self.error(f"{what} runs past the end of the file")
        return self._handle.read(count)

    def unpack(self, fmt: str, what: str) -> object:
        return struct.unpack(fmt, self.take(struct.calcsize(fmt), what))[0]

    def u32(self, what: str) -> int:
        return cast("int", self.unpack("<I", what))

    def u64(self, what: str) -> int:
        return cast("int", self.unpack("<Q", what))

    def string(self, what: str) -> str:
        length = self.u64(f"{what} length")
        if length > MAX_STRING_BYTES:
            raise self.error(f"{what} declares {length} bytes, above {MAX_STRING_BYTES}")
        raw = self.take(length, what)
        try:
            return raw.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise self.error(f"{what} is not UTF-8: {exc}") from exc

    def value(self, value_type: int, what: str, *, depth: int = 0) -> object:
        if value_type in _SCALARS:
            return self.unpack(_SCALARS[value_type], what)
        if value_type == _STRING:
            return self.string(what)
        if value_type == _ARRAY and depth == 0:
            item_type = self.u32(f"{what} item type")
            count = self.u64(f"{what} item count")
            if count > MAX_ARRAY_ITEMS:
                raise self.error(f"{what} declares {count} items, above {MAX_ARRAY_ITEMS}")
            return [self.value(item_type, what, depth=1) for _ in range(count)]
        raise self.error(f"{what} has unknown value type {value_type}")

    def error(self, reason: str) -> ModelLoadError:
        return ModelLoadError(
            f"{self._path}: not a valid GGUF file: {reason} (at byte {self.position})",
            context={"path": str(self._path)},
        )


def read_gguf(path: str | Path) -> GGUFFile:
    """Read the header, metadata and tensor table of a GGUF file; no tensor data is read.

    Raises:
        ModelLoadError: If the file cannot be read, is not GGUF, or its header is inconsistent:
            an unknown version, value type or GGML type, a declared count or length above the
            limits, a tensor outside the file, off the alignment or overlapping another, or a
            repeated metadata key or tensor name.
    """
    file = Path(path)
    try:
        size = file.stat().st_size
        handle = file.open("rb")
    except OSError as exc:
        raise ModelLoadError(f"{file}: cannot read: {exc}", context={"path": str(file)}) from exc
    with handle:
        reader = _Reader(handle, file, size)
        if reader.take(4, "magic") != _MAGIC:
            raise reader.error("the magic is not 'GGUF'")
        version = reader.u32("version")
        if version not in _VERSIONS:
            raise reader.error(f"version {version} is not supported (supported: 2, 3)")
        tensor_count = reader.u64("tensor count")
        metadata_count = reader.u64("metadata count")
        if tensor_count > MAX_TENSORS:
            raise reader.error(f"{tensor_count} tensors declared, above {MAX_TENSORS}")
        if metadata_count > MAX_METADATA:
            raise reader.error(f"{metadata_count} metadata keys declared, above {MAX_METADATA}")
        metadata: dict[str, object] = {}
        for index in range(metadata_count):
            key = reader.string(f"metadata key {index}")
            if key in metadata:
                raise reader.error(f"metadata key {key!r} is repeated")
            metadata[key] = reader.value(reader.u32(f"{key} type"), key)
        raw_tensors = [_tensor_info(reader, index) for index in range(tensor_count)]
        alignment = metadata.get("general.alignment", _DEFAULT_ALIGNMENT)
        if not isinstance(alignment, int) or isinstance(alignment, bool) or alignment < 1:
            raise reader.error(f"general.alignment {alignment!r} is not a positive integer")
        if alignment & (alignment - 1):
            raise reader.error(f"general.alignment {alignment} is not a power of two")
        data_offset = -(-reader.position // alignment) * alignment
        tensors = _place(reader, raw_tensors, data_offset, alignment, size)
    return GGUFFile(
        path=file,
        version=version,
        metadata=metadata,
        tensors=tensors,
        alignment=alignment,
        data_offset=data_offset,
    )


def _tensor_info(reader: _Reader, index: int) -> tuple[str, tuple[int, ...], GGMLType, int]:
    name = reader.string(f"tensor {index} name")
    dims = reader.u32(f"{name} dimension count")
    if not 1 <= dims <= MAX_DIMS:
        raise reader.error(f"{name} has {dims} dimensions (1 to {MAX_DIMS} allowed)")
    ne = [reader.u64(f"{name} dimension") for _ in range(dims)]
    type_id = reader.u32(f"{name} type")
    try:
        ggml_type = GGMLType.by_id(type_id)
    except ValueError as exc:
        raise reader.error(f"{name}: {exc}") from exc
    offset = reader.u64(f"{name} offset")
    return name, tuple(reversed(ne)), ggml_type, offset


def _place(
    reader: _Reader,
    raw: list[tuple[str, tuple[int, ...], GGMLType, int]],
    data_offset: int,
    alignment: int,
    size: int,
) -> tuple[GGUFTensor, ...]:
    tensors: list[GGUFTensor] = []
    seen: set[str] = set()
    for name, shape, ggml_type, relative in raw:
        if name in seen:
            raise reader.error(f"tensor {name!r} is listed twice")
        seen.add(name)
        if relative % alignment:
            raise reader.error(f"{name} starts at {relative}, not a multiple of {alignment}")
        try:
            nbytes = ggml_type.nbytes(shape)
        except ValueError as exc:
            raise reader.error(f"{name}: {exc}") from exc
        tensor = GGUFTensor(name, shape, ggml_type, data_offset + relative, nbytes)
        if tensor.end > size:
            raise reader.error(
                f"{name} ends at byte {tensor.end}, past the end of the {size}-byte file"
            )
        tensors.append(tensor)
    ordered = sorted(tensors, key=lambda tensor: tensor.offset)
    for before, after in itertools.pairwise(ordered):
        if after.offset < before.end:
            raise reader.error(f"{after.name} overlaps {before.name}")
    return tuple(tensors)


def gguf_hf_config(file: GGUFFile) -> dict[str, object]:
    """The ``config.json`` fields of the model a ``llama`` GGUF file describes.

    Read: ``context_length``, ``embedding_length``, ``block_count``, ``feed_forward_length``,
    ``attention.head_count`` and ``head_count_kv``, ``attention.key_length`` (else
    ``rope.dimension_count``, else the width over the heads), ``attention.layer_norm_rms_epsilon``,
    ``rope.freq_base``, a ``linear`` or ``yarn`` ``rope.scaling``, the vocabulary from
    ``tokenizer.ggml.tokens`` and tied embeddings when there is no ``output.weight``.

    Raises:
        ModelLoadError: If the architecture is not ``llama`` or a required key is missing.
    """
    metadata = file.metadata
    architecture = metadata.get("general.architecture")
    if architecture != "llama":
        raise ModelLoadError(
            f"{file.path}: GGUF architecture {architecture!r} is not supported (only 'llama')",
            context={"path": str(file.path), "architecture": str(architecture)},
        )

    def number(key: str, *, default: int | float | None = None) -> int | float:
        value = metadata.get(f"llama.{key}", default)
        if not isinstance(value, int | float) or isinstance(value, bool):
            raise ModelLoadError(
                f"{file.path}: GGUF key llama.{key} is missing or not a number",
                context={"path": str(file.path), "key": f"llama.{key}"},
            )
        return value

    hidden = int(number("embedding_length"))
    heads = int(number("attention.head_count"))
    head_dim = int(
        number(
            "attention.key_length",
            default=int(number("rope.dimension_count", default=hidden // heads)),
        )
    )
    tokens = metadata.get("tokenizer.ggml.tokens")
    counted = len(cast("list[object]", tokens)) if isinstance(tokens, list) else None
    vocab = metadata.get("llama.vocab_size", counted)
    if not isinstance(vocab, int):
        raise ModelLoadError(
            f"{file.path}: GGUF file has neither llama.vocab_size nor tokenizer.ggml.tokens",
            context={"path": str(file.path)},
        )
    raw: dict[str, object] = {
        "architectures": ["LlamaForCausalLM"],
        "model_type": "llama",
        "vocab_size": vocab,
        "hidden_size": hidden,
        "num_hidden_layers": int(number("block_count")),
        "intermediate_size": int(number("feed_forward_length")),
        "num_attention_heads": heads,
        "num_key_value_heads": int(number("attention.head_count_kv", default=heads)),
        "head_dim": head_dim,
        "max_position_embeddings": int(number("context_length")),
        "rms_norm_eps": float(number("attention.layer_norm_rms_epsilon", default=1e-5)),
        "rope_theta": float(number("rope.freq_base", default=10000.0)),
        "hidden_act": "silu",
        "tie_word_embeddings": not any(t.name == "output.weight" for t in file.tensors),
    }
    scaling = metadata.get("llama.rope.scaling.type")
    if scaling in ("linear", "yarn"):
        raw["rope_scaling"] = {
            "rope_type": scaling,
            "factor": float(number("rope.scaling.factor")),
            **(
                {
                    "original_max_position_embeddings": int(
                        number("rope.scaling.original_context_length")
                    )
                }
                if scaling == "yarn"
                else {}
            ),
        }
    return raw
