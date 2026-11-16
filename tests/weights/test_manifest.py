"""Indexing a checkpoint from headers: shards, index files and the ways they can disagree."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, cast

import pytest

from ayaka.core.errors import ModelLoadError
from ayaka.utils.dtypes import DType
from ayaka.weights import manifest as manifest_module
from ayaka.weights.manifest import INDEX_FILENAME, WeightManifest
from ayaka.weights.spec import TensorSpec
from ayaka.weights.tensor import TensorRef
from safetensors_files import filled, write_raw, write_safetensors

if TYPE_CHECKING:
    from pathlib import Path

pytestmark = pytest.mark.unit


def _write_index(directory: Path, weight_map: Any, **extra: Any) -> Path:
    path = directory / INDEX_FILENAME
    path.write_text(json.dumps({"metadata": {}, "weight_map": weight_map, **extra}))
    return path


def _sharded(directory: Path) -> None:
    """Two shards and a correct index: a and b in the first, c in the second."""
    write_safetensors(
        directory / "model-00001-of-00002.safetensors",
        {"a": filled("F32", (2, 2)), "b": filled("F16", (3,))},
    )
    write_safetensors(directory / "model-00002-of-00002.safetensors", {"c": filled("U8", (5,))})
    _write_index(
        directory,
        {
            "a": "model-00001-of-00002.safetensors",
            "b": "model-00001-of-00002.safetensors",
            "c": "model-00002-of-00002.safetensors",
        },
    )


class TestWithoutAnIndex:
    def test_a_single_file(self, tmp_path: Path) -> None:
        write_safetensors(
            tmp_path / "model.safetensors", {"a": filled("F32", (2,)), "b": filled("F32", (3,))}
        )

        manifest = WeightManifest.from_directory(tmp_path)

        assert set(manifest) == {"a", "b"}
        assert manifest["b"].shape == (3,)
        assert manifest["b"].path == tmp_path / "model.safetensors"
        assert manifest.root == tmp_path

    def test_every_safetensors_file_in_the_directory_is_a_shard(self, tmp_path: Path) -> None:
        write_safetensors(tmp_path / "x.safetensors", {"a": filled("U8", (1,))})
        write_safetensors(tmp_path / "y.safetensors", {"b": filled("U8", (1,))})
        (tmp_path / "notes.txt").write_text("not a shard")

        manifest = WeightManifest.from_directory(tmp_path)

        assert manifest.files == (tmp_path / "x.safetensors", tmp_path / "y.safetensors")

    def test_a_tensor_stored_in_two_files_is_refused(self, tmp_path: Path) -> None:
        write_safetensors(tmp_path / "x.safetensors", {"w": filled("U8", (1,))})
        write_safetensors(tmp_path / "y.safetensors", {"w": filled("U8", (1,))})

        with pytest.raises(ModelLoadError, match=r"more than one file.*'w'") as excinfo:
            WeightManifest.from_directory(tmp_path)

        assert excinfo.value.context["duplicates"] == {
            "w": [str(tmp_path / "x.safetensors"), str(tmp_path / "y.safetensors")]
        }

    def test_a_corrupt_shard_fails_the_whole_manifest(self, tmp_path: Path) -> None:
        write_safetensors(tmp_path / "good.safetensors", {"a": filled("U8", (1,))})
        write_raw(tmp_path / "bad.safetensors", b"not json")

        with pytest.raises(ModelLoadError, match=r"bad\.safetensors.*not valid JSON"):
            WeightManifest.from_directory(tmp_path)

    def test_a_directory_named_like_a_shard_is_not_one(self, tmp_path: Path) -> None:
        (tmp_path / "trap.safetensors").mkdir()
        write_safetensors(tmp_path / "real.safetensors", {"a": filled("U8", (1,))})

        assert set(WeightManifest.from_directory(tmp_path)) == {"a"}

    def test_an_empty_directory(self, tmp_path: Path) -> None:
        with pytest.raises(ModelLoadError, match=r"no tensors found"):
            WeightManifest.from_directory(tmp_path)

    def test_a_directory_of_files_with_no_tensors(self, tmp_path: Path) -> None:
        write_safetensors(tmp_path / "empty.safetensors", {})

        with pytest.raises(ModelLoadError, match=r"no tensors found"):
            WeightManifest.from_directory(tmp_path)

    def test_a_missing_directory(self, tmp_path: Path) -> None:
        with pytest.raises(ModelLoadError, match="not a directory"):
            WeightManifest.from_directory(tmp_path / "absent")

    def test_a_file_instead_of_a_directory(self, tmp_path: Path) -> None:
        file = write_safetensors(tmp_path / "m.safetensors", {"a": filled("U8", (1,))})

        with pytest.raises(ModelLoadError, match="not a directory"):
            WeightManifest.from_directory(file)


class TestWithAnIndex:
    def test_follows_the_index(self, tmp_path: Path) -> None:
        _sharded(tmp_path)

        manifest = WeightManifest.from_directory(tmp_path)

        assert set(manifest) == {"a", "b", "c"}
        assert manifest["c"].path == tmp_path / "model-00002-of-00002.safetensors"
        assert manifest.total_bytes == 16 + 6 + 5
        assert len(manifest.files) == 2

    def test_iterates_in_physical_order(self, tmp_path: Path) -> None:
        write_safetensors(
            tmp_path / "s1.safetensors", {"z": filled("U8", (2,)), "a": filled("U8", (2,))}
        )
        write_safetensors(tmp_path / "s2.safetensors", {"m": filled("U8", (2,))})
        _write_index(
            tmp_path, {"m": "s2.safetensors", "a": "s1.safetensors", "z": "s1.safetensors"}
        )

        manifest = WeightManifest.from_directory(tmp_path)

        assert list(manifest) == ["z", "a", "m"]  # s1 by offset (z first), then s2

    def test_the_index_wins_over_stray_files_it_does_not_name(self, tmp_path: Path) -> None:
        write_safetensors(tmp_path / "s1.safetensors", {"a": filled("U8", (1,))})
        write_safetensors(tmp_path / "unrelated.safetensors", {"a": filled("U8", (1,))})
        _write_index(tmp_path, {"a": "s1.safetensors"})

        assert WeightManifest.from_directory(tmp_path)["a"].path == tmp_path / "s1.safetensors"

    def test_a_shard_in_a_subdirectory(self, tmp_path: Path) -> None:
        (tmp_path / "shards").mkdir()
        write_safetensors(tmp_path / "shards" / "s.safetensors", {"a": filled("U8", (1,))})
        _write_index(tmp_path, {"a": "shards/s.safetensors"})

        assert WeightManifest.from_directory(tmp_path)["a"].path == (
            tmp_path / "shards" / "s.safetensors"
        )

    def test_a_symlinked_shard_is_followed(self, tmp_path: Path) -> None:
        # A Hugging Face cache snapshot is made of symlinks into a blob directory.
        blobs, snapshot = tmp_path / "blobs", tmp_path / "snapshot"
        blobs.mkdir()
        snapshot.mkdir()
        write_safetensors(blobs / "abc123", {"a": filled("U8", (1,))})
        try:
            (snapshot / "s.safetensors").symlink_to(blobs / "abc123")
        except OSError as exc:  # Windows without developer mode cannot create symlinks
            pytest.skip(f"cannot create a symlink on this machine: {exc}")
        _write_index(snapshot, {"a": "s.safetensors"})

        assert "a" in WeightManifest.from_directory(snapshot)

    def test_shards_that_are_absent_are_all_reported_together(self, tmp_path: Path) -> None:
        write_safetensors(tmp_path / "s1.safetensors", {"a": filled("U8", (1,))})
        _write_index(
            tmp_path, {"a": "s1.safetensors", "b": "s2.safetensors", "c": "s3.safetensors"}
        )

        with pytest.raises(
            ModelLoadError, match=r"'s2\.safetensors', 's3\.safetensors'.*partial"
        ) as e:
            WeightManifest.from_directory(tmp_path)

        assert e.value.context["missing_shards"] == ["s2.safetensors", "s3.safetensors"]

    def test_disagreements_in_both_directions_are_reported_together(self, tmp_path: Path) -> None:
        write_safetensors(
            tmp_path / "s1.safetensors", {"a": filled("U8", (1,)), "stray": filled("U8", (1,))}
        )
        write_safetensors(tmp_path / "s2.safetensors", {"c": filled("U8", (1,))})
        _write_index(
            tmp_path,
            {
                "a": "s1.safetensors",
                "ghost": "s1.safetensors",
                "c": "s2.safetensors",
                "d": "s2.safetensors",
            },
        )

        with pytest.raises(ModelLoadError, match="disagree") as excinfo:
            WeightManifest.from_directory(tmp_path)

        message = str(excinfo.value)
        assert "s1.safetensors lacks 'ghost'" in message
        assert "s2.safetensors lacks 'd'" in message
        assert "s1.safetensors holds 'stray'" in message
        assert excinfo.value.context == {
            "not_in_shard": {"s1.safetensors": ["ghost"], "s2.safetensors": ["d"]},
            "not_in_index": {"s1.safetensors": ["stray"]},
        }

    @pytest.mark.parametrize(
        ("shard", "why"),
        [
            ("../outside.safetensors", "'..'"),
            ("a/../../outside.safetensors", "'..'"),
            ("/etc/passwd.safetensors", "relative path"),
            ("C:/x.safetensors", "relative path"),
            ("..\\outside.safetensors", "relative path"),
            ("a//b.safetensors", "empty"),
            ("./s.safetensors", "'.'"),
            ("model.bin", "must be a .safetensors file"),
            ("model.safetensors.pkl", "must be a .safetensors file"),
        ],
    )
    def test_a_shard_name_that_could_leave_the_directory_is_refused(
        self, tmp_path: Path, shard: str, why: str
    ) -> None:
        _write_index(tmp_path, {"a": shard})

        with pytest.raises(ModelLoadError, match=rf"maps 'a' to .*: .*{why}"):
            WeightManifest.from_directory(tmp_path)

    def test_the_file_a_traversal_would_reach_is_never_opened(self, tmp_path: Path) -> None:
        directory = tmp_path / "ckpt"
        directory.mkdir()
        write_safetensors(tmp_path / "outside.safetensors", {"a": filled("U8", (1,))})
        _write_index(directory, {"a": "../outside.safetensors"})

        with pytest.raises(ModelLoadError, match=r"'\.\.'"):
            WeightManifest.from_directory(directory)

    @pytest.mark.parametrize(
        ("weight_map", "match"),
        [
            ({}, "no non-empty weight_map"),
            (None, "no non-empty weight_map"),
            (["a"], "no non-empty weight_map"),
            ({"a": 3}, r"maps 'a' to 3, which is not a file name"),
            ({"a": None}, r"maps 'a' to None"),
        ],
    )
    def test_a_malformed_weight_map(self, tmp_path: Path, weight_map: Any, match: str) -> None:
        _write_index(tmp_path, weight_map)

        with pytest.raises(ModelLoadError, match=match):
            WeightManifest.from_directory(tmp_path)

    def test_an_index_without_weight_map(self, tmp_path: Path) -> None:
        (tmp_path / INDEX_FILENAME).write_text('{"metadata": {}}')

        with pytest.raises(ModelLoadError, match="no non-empty weight_map"):
            WeightManifest.from_directory(tmp_path)

    @pytest.mark.parametrize("text", ["", "{", "[1]", "null"])
    def test_an_index_that_is_not_a_json_object_with_a_map(self, tmp_path: Path, text: str) -> None:
        (tmp_path / INDEX_FILENAME).write_text(text)

        with pytest.raises(ModelLoadError, match=r"index"):
            WeightManifest.from_directory(tmp_path)

    def test_a_repeated_tensor_in_the_index_is_not_silently_dropped(self, tmp_path: Path) -> None:
        (tmp_path / INDEX_FILENAME).write_text(
            '{"weight_map": {"a": "s1.safetensors", "a": "s2.safetensors"}}'
        )

        with pytest.raises(ModelLoadError, match=r"duplicate object keys: \['a'\]"):
            WeightManifest.from_directory(tmp_path)

    def test_an_index_over_the_size_limit_is_not_read_whole(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(manifest_module, "_INDEX_LIMIT", 32)
        _write_index(tmp_path, {f"tensor_{index}": "s.safetensors" for index in range(10)})

        with pytest.raises(ModelLoadError, match="larger than 32 bytes"):
            WeightManifest.from_directory(tmp_path)

    def test_an_index_that_is_not_utf8(self, tmp_path: Path) -> None:
        (tmp_path / INDEX_FILENAME).write_bytes(b'{"weight_map": {"\xff": "s.safetensors"}}')

        with pytest.raises(ModelLoadError, match="not valid JSON"):
            WeightManifest.from_directory(tmp_path)

    def test_a_long_list_is_cut_in_the_message_but_not_in_the_context(self, tmp_path: Path) -> None:
        write_safetensors(tmp_path / "s.safetensors", {"a": filled("U8", (1,))})
        names = {f"t{index:02d}": "s.safetensors" for index in range(20)}
        _write_index(tmp_path, {"a": "s.safetensors", **names})

        with pytest.raises(ModelLoadError, match=r"lacks 't00'.*and 12 more") as excinfo:
            WeightManifest.from_directory(tmp_path)

        not_in_shard = cast("dict[str, list[str]]", excinfo.value.context["not_in_shard"])
        assert len(not_in_shard["s.safetensors"]) == 20


class TestMappingProtocol:
    def _manifest(self, tmp_path: Path) -> WeightManifest:
        _sharded(tmp_path)
        return WeightManifest.from_directory(tmp_path)

    def test_behaves_like_a_read_only_mapping(self, tmp_path: Path) -> None:
        manifest = self._manifest(tmp_path)

        assert len(manifest) == 3
        assert "a" in manifest
        assert "z" not in manifest
        assert manifest.get("z") is None
        assert dict(manifest.items())["b"] is manifest["b"]
        with pytest.raises(KeyError):
            manifest["z"]
        with pytest.raises(TypeError):
            manifest["a"] = manifest["b"]  # type: ignore[index]  # the check under test

    def test_repr_says_what_it_holds(self, tmp_path: Path) -> None:
        assert "3 tensors, 2 files" in repr(self._manifest(tmp_path))

    def test_can_be_built_from_refs(self, tmp_path: Path) -> None:
        refs = [
            TensorRef(TensorSpec("b", (1,), DType.FP32), tmp_path / "x.safetensors", 40),
            TensorRef(TensorSpec("a", (1,), DType.FP32), tmp_path / "x.safetensors", 8),
        ]

        manifest = WeightManifest(tmp_path, refs)

        assert list(manifest) == ["a", "b"]
        assert manifest.total_bytes == 8

    def test_refs_with_the_same_name_are_refused(self, tmp_path: Path) -> None:
        spec = TensorSpec("w", (1,), DType.FP32)
        refs = [
            TensorRef(spec, tmp_path / "x.safetensors", 8),
            TensorRef(spec, tmp_path / "y.safetensors", 8),
        ]

        with pytest.raises(ModelLoadError, match="more than one file"):
            WeightManifest(tmp_path, refs)
