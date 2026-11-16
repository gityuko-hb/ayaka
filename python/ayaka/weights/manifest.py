"""The tensors of a whole checkpoint, indexed by name across its files."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Iterator, Mapping
from pathlib import Path
from typing import Final, cast

from ayaka.core.errors import ModelLoadError
from ayaka.utils.parsing import parse_json_strict
from ayaka.weights.safetensors import SafetensorsFile
from ayaka.weights.tensor import TensorRef

__all__ = ["INDEX_FILENAME", "WeightManifest"]

#: The file Hugging Face writes next to the shards of a sharded checkpoint.
INDEX_FILENAME: Final = "model.safetensors.index.json"
_SHARD_SUFFIX: Final = ".safetensors"
_INDEX_LIMIT: Final = 100 * 2**20


class WeightManifest(Mapping[str, TensorRef]):
    """Where every tensor of a checkpoint is, built from headers alone.

    A manifest maps a checkpoint tensor name to its
    :class:`~ayaka.weights.tensor.TensorRef`. Building one reads each shard's
    header and nothing else, so a 400 GB checkpoint costs a few hundred small
    reads, and a checkpoint that is wrong fails in seconds instead of after the
    download is in memory. Iteration follows the physical order: by file, then
    by offset within it, which is the order to read in.

    A manifest holds no open files; open a shard with
    :class:`~ayaka.weights.safetensors.SafetensorsFile` to read its bytes.

    Args:
        root: Directory the checkpoint was read from.
        refs: Every tensor of the checkpoint.

    Raises:
        ayaka.core.errors.ModelLoadError: If two refs have the same name.
    """

    __slots__ = ("_refs", "root")

    def __init__(self, root: str | Path, refs: Iterable[TensorRef]) -> None:
        self.root = Path(root)
        ordered = sorted(refs, key=lambda ref: (str(ref.path), ref.offset))
        by_name: dict[str, TensorRef] = {}
        clashes: dict[str, list[str]] = defaultdict(list)
        for ref in ordered:
            earlier = by_name.setdefault(ref.name, ref)
            if earlier is not ref:
                clashes[ref.name] = [str(earlier.path), str(ref.path)]
        if clashes:
            raise ModelLoadError(
                f"{self.root}: tensors stored in more than one file, with nothing to say "
                f"which is current: {_describe(sorted(clashes))}",
                context={"duplicates": dict(clashes)},
            )
        self._refs = by_name

    @classmethod
    def from_directory(cls, root: str | Path) -> WeightManifest:
        """Index the safetensors checkpoint in a directory.

        With a ``model.safetensors.index.json`` the shards are the ones it
        names, and the index is checked against their headers in both
        directions. Without one, every ``*.safetensors`` file directly in the
        directory is a shard.

        Raises:
            ayaka.core.errors.ModelLoadError: If the directory is missing or
                holds no tensors, a file is corrupt, the index names a shard
                that is absent or lies outside the directory, or the index and
                the shards disagree about which tensors exist. Problems of one
                kind are reported together, not one per attempt.
        """
        directory = Path(root)
        if not directory.is_dir():
            raise ModelLoadError(f"{directory}: not a directory", context={"path": str(directory)})
        index = directory / INDEX_FILENAME
        refs = _from_index(directory, index) if index.is_file() else _from_shards(directory)
        if not refs:
            raise ModelLoadError(
                f"{directory}: no tensors found; expected *.safetensors files or {INDEX_FILENAME}",
                context={"path": str(directory)},
            )
        return cls(directory, refs)

    @property
    def files(self) -> tuple[Path, ...]:
        """The files that hold tensors, in sorted order."""
        return tuple(sorted({ref.path for ref in self._refs.values()}, key=str))

    @property
    def total_bytes(self) -> int:
        """Bytes of tensor data in the whole checkpoint."""
        return sum(ref.nbytes for ref in self._refs.values())

    def __getitem__(self, name: str) -> TensorRef:
        return self._refs[name]

    def __iter__(self) -> Iterator[str]:
        return iter(self._refs)

    def __len__(self) -> int:
        return len(self._refs)

    def __repr__(self) -> str:
        return f"WeightManifest({str(self.root)!r}, {len(self)} tensors, {len(self.files)} files)"


def _from_shards(directory: Path) -> list[TensorRef]:
    refs: list[TensorRef] = []
    for path in sorted(directory.glob(f"*{_SHARD_SUFFIX}")):
        if path.is_file():
            refs.extend(SafetensorsFile(path).tensors.values())
    return refs


def _from_index(directory: Path, index: Path) -> list[TensorRef]:
    weight_map = _read_weight_map(index)
    tensors_of: dict[str, set[str]] = defaultdict(set)
    for tensor, shard in weight_map.items():
        tensors_of[shard].add(tensor)

    absent = [shard for shard in sorted(tensors_of) if not (directory / shard).is_file()]
    if absent:
        raise ModelLoadError(
            f"{index}: names shards that are not in {directory}: {_describe(absent)}; "
            "an interrupted or partial download",
            context={"missing_shards": absent},
        )

    refs: list[TensorRef] = []
    not_in_shard: dict[str, list[str]] = {}
    not_in_index: dict[str, list[str]] = {}
    for shard in sorted(tensors_of):
        stored = SafetensorsFile(directory / shard).tensors
        lacking = sorted(tensors_of[shard].difference(stored))
        stray = sorted(set(stored).difference(tensors_of[shard]))
        if lacking:
            not_in_shard[shard] = lacking
        if stray:
            not_in_index[shard] = stray
        refs.extend(stored[name] for name in sorted(tensors_of[shard].intersection(stored)))
    if not_in_shard or not_in_index:
        raise ModelLoadError(
            f"{index} and the shards disagree"
            + "".join(
                f"; {shard} lacks {_describe(names)} that the index maps to it"
                for shard, names in not_in_shard.items()
            )
            + "".join(
                f"; {shard} holds {_describe(names)} that the index does not map, so they "
                "would never be read"
                for shard, names in not_in_index.items()
            ),
            context={"not_in_shard": not_in_shard, "not_in_index": not_in_index},
        )
    return refs


def _read_weight_map(index: Path) -> dict[str, str]:
    """Read and validate ``weight_map`` of an index file: tensor name to shard path."""

    def bad(message: str) -> ModelLoadError:
        return ModelLoadError(f"{index}: {message}", context={"path": str(index)})

    try:
        with index.open("rb") as handle:
            raw = handle.read(_INDEX_LIMIT + 1)
        if len(raw) > _INDEX_LIMIT:
            raise bad(f"the index is larger than {_INDEX_LIMIT} bytes")
        document = parse_json_strict(raw.decode("utf-8"))
    except OSError as exc:
        raise bad(f"cannot read the index: {exc}") from exc
    except ValueError as exc:  # not UTF-8, not JSON, or a repeated key
        raise bad(f"the index is not valid JSON: {exc}") from exc

    weight_map = (
        cast("dict[str, object]", document).get("weight_map")
        if isinstance(document, dict)
        else None
    )
    if not isinstance(weight_map, dict) or not weight_map:
        raise bad("the index has no non-empty weight_map")
    entries = cast("dict[str, object]", weight_map)
    for tensor, shard in entries.items():
        if not isinstance(shard, str):
            raise bad(f"weight_map maps {tensor!r} to {shard!r}, which is not a file name")
        if (reason := _shard_name_problem(shard)) is not None:
            raise bad(f"weight_map maps {tensor!r} to {shard!r}: {reason}")
    return cast("dict[str, str]", entries)


def _shard_name_problem(name: str) -> str | None:
    """Why ``name`` may not be a shard path inside the checkpoint directory, if it may not.

    The index comes with the checkpoint, so a name in it is untrusted input:
    one that is absolute or climbs with ``..`` would make the loader open a file
    the checkpoint's author chose anywhere on the machine. Symlinks are not
    resolved: a Hugging Face cache snapshot is made of them, on purpose.
    """
    if not name.endswith(_SHARD_SUFFIX):
        return f"a shard must be a {_SHARD_SUFFIX} file"
    if "\\" in name or ":" in name or "\0" in name or name.startswith("/"):
        return "a shard must be a relative path written with '/'"
    if any(part in ("", ".", "..") for part in name.split("/")):
        return "a shard path may not contain empty, '.' or '..' parts"
    return None


def _describe(names: Iterable[str], *, limit: int = 8) -> str:
    """List names for a message: all of them when few, else the first few and a count."""
    names = list(names)
    shown = ", ".join(repr(name) for name in names[:limit])
    return shown if len(names) <= limit else f"{shown} and {len(names) - limit} more"
