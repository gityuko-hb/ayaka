"""Read ``.safetensors`` files without trusting them.

File layout, because every check below refers to it::

    [0, 8)         u64 little-endian: header length N
    [8, 8 + N)     UTF-8 JSON: {"name": {"dtype", "shape", "data_offsets": [begin, end]}, ...,
                   "__metadata__": {str: str}}
    [8 + N, EOF)   tensor payload; ``data_offsets`` count from 8 + N

A checkpoint is data from outside, and every header field is chosen by whoever
made the file. Opening a :class:`SafetensorsFile` reads only the header and
refuses a file that is not self-consistent: a header longer than the file or
than :data:`_HEADER_LIMIT`, a repeated key, an unknown dtype, a byte range that
leaves the file or disagrees with ``shape`` and ``dtype``, and two tensors that
claim the same bytes. Tensor bytes are mapped, not read, and only when asked.

Pickle-based checkpoints are never read; that is the point of this format.
"""

from __future__ import annotations

import mmap
import os
import struct
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Final, cast

from ayaka.core.errors import ModelLoadError
from ayaka.utils.dtypes import DType
from ayaka.utils.parsing import parse_json_strict
from ayaka.weights.spec import TensorSpec
from ayaka.weights.tensor import TensorRef

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping
    from os import PathLike
    from types import TracebackType

__all__ = ["SafetensorsFile"]

#: Largest header accepted: the limit of the reference implementation. A real
#: header is a few hundred KiB even for a model with tens of thousands of
#: tensors; the cap turns a forged length into a refusal, not an allocation.
_HEADER_LIMIT: Final = 100 * 2**20
_METADATA_KEY: Final = "__metadata__"


def _unreadable(path: Path, message: str, **context: object) -> ModelLoadError:
    return ModelLoadError(f"{path}: {message}", context={"path": str(path), **context})


class SafetensorsFile:
    """One opened ``.safetensors`` file: its validated header and, on demand, its bytes.

    Construction reads the header only. :meth:`read` maps the file on first
    use. Use it as a context manager, or call :meth:`close`.

    Attributes:
        path: The file.
        size: Size of the file in bytes when the header was read.
        tensors: Tensor name to :class:`~ayaka.weights.tensor.TensorRef`, in
            the order the tensors lie in the file.
        metadata: The free-form ``__metadata__`` strings; empty when absent.

    Raises:
        ayaka.core.errors.ModelLoadError: From the constructor, if the file
            cannot be read or its header is corrupt or names something this
            build cannot read. The message says which of those it is.

    Example:
        >>> import tempfile, pathlib
        >>> header = b'{"w":{"dtype":"U8","shape":[2],"data_offsets":[0,2]}}'
        >>> with tempfile.TemporaryDirectory() as directory:
        ...     path = pathlib.Path(directory, "w.safetensors")
        ...     _ = path.write_bytes(len(header).to_bytes(8, "little") + header + b"\\x01\\x02")
        ...     with SafetensorsFile(path) as file:
        ...         bytes(file.read("w")), file.tensors["w"].shape
        (b'\\x01\\x02', (2,))
    """

    def __init__(self, path: str | PathLike[str]) -> None:
        self.path = Path(path)
        self._mapping: mmap.mmap | None = None
        self._closed = False
        self.size, data_start, raw_header = _read_header_bytes(self.path)
        refs, self.metadata = _parse_header(raw_header, self.path, data_start, self.size)
        self.tensors: Mapping[str, TensorRef] = MappingProxyType({ref.name: ref for ref in refs})

    def read(self, name: str) -> memoryview:
        """Return the bytes of one tensor as a read-only view into the mapped file.

        Nothing is copied. Release every view before :meth:`close`: the file
        cannot be unmapped while one is alive.

        Raises:
            KeyError: If the file has no tensor ``name``.
            ValueError: If the file is closed.
            ayaka.core.errors.ModelLoadError: If the file cannot be mapped, or
                changed size since its header was read.
        """
        ref = self.tensors[name]
        return memoryview(self._map_file())[ref.offset : ref.end]

    def close(self) -> None:
        """Unmap the file. Safe to call twice.

        Raises:
            BufferError: If a view returned by :meth:`read` is still alive; the
                file stays open.
        """
        if self._mapping is not None:
            try:
                self._mapping.close()
            except BufferError:
                raise BufferError(
                    f"{self.path}: release the memoryviews returned by read() before closing"
                ) from None
            self._mapping = None
        self._closed = True

    def __enter__(self) -> SafetensorsFile:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()

    def _map_file(self) -> mmap.mmap:
        if self._closed:
            raise ValueError(f"{self.path}: the file is closed")
        if self._mapping is None:
            try:
                with self.path.open("rb") as handle:
                    mapping = mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ)
            except (OSError, ValueError) as exc:
                raise _unreadable(self.path, f"cannot map the file: {exc}") from exc
            mapped_size = len(mapping)
            if mapped_size != self.size:
                mapping.close()
                raise _unreadable(
                    self.path,
                    f"the file is now {mapped_size} bytes but its header was read at "
                    f"{self.size}; it changed after it was opened",
                )
            self._mapping = mapping
        return self._mapping


def _read_header_bytes(path: Path) -> tuple[int, int, bytes]:
    """Return ``(file size, payload start, raw header)``, reading nothing else."""
    try:
        with path.open("rb") as handle:
            size = os.fstat(handle.fileno()).st_size
            prefix = handle.read(8)
            if len(prefix) < 8:
                raise _unreadable(path, f"{size} bytes is too short to hold a header length")
            (length,) = struct.unpack("<Q", prefix)
            if length > _HEADER_LIMIT:
                raise _unreadable(
                    path,
                    f"the header claims {length} bytes, above the limit of {_HEADER_LIMIT}; "
                    "refusing to read a length that was probably forged",
                )
            if 8 + length > size:
                raise _unreadable(
                    path, f"the header claims {length} bytes but the file has only {size}"
                )
            raw = handle.read(length)
    except OSError as exc:
        raise _unreadable(path, f"cannot read the file: {exc}") from exc
    if len(raw) != length:
        raise _unreadable(path, f"the header is truncated: read {len(raw)} of {length} bytes")
    return size, 8 + length, raw


def _parse_header(
    raw: bytes, path: Path, data_start: int, size: int
) -> tuple[list[TensorRef], Mapping[str, str]]:
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise _unreadable(path, f"the header is not valid UTF-8: {exc}") from exc
    try:
        header = parse_json_strict(text)
    except ValueError as exc:
        raise _unreadable(path, f"the header is not valid JSON: {exc}") from exc
    if not isinstance(header, dict):
        raise _unreadable(path, f"the header is a JSON {type(header).__name__}, not an object")

    entries = dict(cast("dict[str, object]", header))
    metadata = _metadata(entries.pop(_METADATA_KEY, None), path)
    payload = size - data_start
    refs = [
        _tensor_ref(name, entry, path=path, data_start=data_start, payload=payload)
        for name, entry in entries.items()
    ]
    refs.sort(key=lambda ref: (ref.offset, ref.end))
    _check_no_overlap(refs, path)
    return refs, metadata


def _metadata(value: object, path: Path) -> Mapping[str, str]:
    if value is None:
        return MappingProxyType({})
    if isinstance(value, dict):
        items = cast("dict[str, object]", value)
        if all(isinstance(text, str) for text in items.values()):
            return MappingProxyType(cast("dict[str, str]", items))
    raise _unreadable(path, f"{_METADATA_KEY} must be an object whose values are strings")


def _tensor_ref(
    name: str, entry: object, *, path: Path, data_start: int, payload: int
) -> TensorRef:
    def corrupt(message: str) -> ModelLoadError:
        return _unreadable(path, f"tensor {name!r}: {message}", tensor=name)

    if not name.strip():
        raise _unreadable(path, "a tensor has an empty name")
    if not isinstance(entry, dict):
        raise corrupt("the header entry is not an object")
    fields = cast("dict[str, object]", entry)

    dtype = _dtype(fields.get("dtype"), corrupt)
    shape = _int_list(fields.get("shape"))
    if shape is None:
        raise corrupt(f"shape must be a list of non-negative integers, got {fields.get('shape')!r}")
    spec = TensorSpec(name, tuple(shape), dtype)

    bounds = _int_list(fields.get("data_offsets"))
    if bounds is None or len(bounds) != 2:
        raise corrupt(
            f"data_offsets must be [begin, end] of non-negative integers, "
            f"got {fields.get('data_offsets')!r}"
        )
    begin, end = bounds
    if begin > end:
        raise corrupt(f"data_offsets [{begin}, {end}] run backwards")
    if end - begin != spec.nbytes:
        raise corrupt(
            f"data_offsets span {end - begin} bytes but shape {list(shape)} of "
            f"{dtype.safetensors_token} needs {spec.nbytes}"
        )
    if end > payload:
        raise corrupt(
            f"its bytes end at {data_start + end} but the file has only {data_start + payload}; "
            "a truncated download, or a forged offset"
        )
    return TensorRef(spec, path, data_start + begin)


def _int_list(value: object) -> list[int] | None:
    """Return ``value`` if it is a list of non-negative ints (``bool`` is not an int here)."""
    if not isinstance(value, list):
        return None
    items = cast("list[object]", value)
    if all(type(item) is int and item >= 0 for item in items):
        return cast("list[int]", items)
    return None


def _dtype(token: object, corrupt: Callable[[str], ModelLoadError]) -> DType:
    if not isinstance(token, str):
        raise corrupt(f"dtype must be a string, got {token!r}")
    try:
        dtype = DType.by_safetensors_token(token)
    except ValueError:
        # Valid safetensors the format allows but this build has no type for
        # (U16, U32, C64, ...): the engine lacks support, the file is fine.
        raise corrupt(f"dtype {token!r} is not one this build reads") from None
    if dtype.safetensors_token != token:
        raise corrupt(f"dtype {token!r} must be spelled {dtype.safetensors_token!r}")
    if dtype.sub_byte:
        raise corrupt(f"dtype {token!r} is packed below one byte; this build cannot unpack it")
    return dtype


def _check_no_overlap(refs: list[TensorRef], path: Path) -> None:
    previous: TensorRef | None = None
    for ref in refs:
        if ref.nbytes == 0:
            continue
        if previous is not None and ref.offset < previous.end:
            raise _unreadable(
                path,
                f"tensors {previous.name!r} [{previous.offset}, {previous.end}) and "
                f"{ref.name!r} [{ref.offset}, {ref.end}) claim the same bytes",
                tensors=(previous.name, ref.name),
            )
        previous = ref
