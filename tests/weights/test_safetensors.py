"""The safetensors reader: a valid file reads back exactly, every kind of broken file is refused."""

from __future__ import annotations

import json
import struct
from typing import TYPE_CHECKING, Any

import pytest

from ayaka.core.errors import ModelLoadError
from ayaka.utils.dtypes import DType
from ayaka.weights import safetensors as safetensors_module
from ayaka.weights.safetensors import SafetensorsFile
from safetensors_files import filled, write_entries, write_raw, write_safetensors

if TYPE_CHECKING:
    from pathlib import Path

pytestmark = pytest.mark.unit


def _entry(dtype: Any = "U8", shape: Any = (4,), offsets: Any = (0, 4)) -> dict[str, Any]:
    return {"dtype": dtype, "shape": list(shape), "data_offsets": list(offsets)}


class TestValidFiles:
    def test_reads_back_every_tensor_exactly(self, tmp_path: Path) -> None:
        tensors = {
            "a": filled("F32", (2, 3)),
            "b": filled("I64", (4,), start=7),
            "c": filled("F16", (3, 1, 2), start=19),
        }
        path = write_safetensors(tmp_path / "m.safetensors", tensors)

        with SafetensorsFile(path) as file:
            for name, (_, _, data) in tensors.items():
                assert bytes(file.read(name)) == data

    def test_describes_dtype_shape_and_position(self, tmp_path: Path) -> None:
        path = write_safetensors(
            tmp_path / "m.safetensors",
            {"a": filled("F32", (2, 3)), "b": filled("BF16", (5,))},
        )

        with SafetensorsFile(path) as file:
            a, b = file.tensors["a"], file.tensors["b"]

            assert (a.dtype, a.shape, a.nbytes) == (DType.FP32, (2, 3), 24)
            assert (b.dtype, b.shape, b.nbytes) == (DType.BF16, (5,), 10)
            assert b.offset == a.end
            assert a.path == path
            assert file.size == path.stat().st_size

    def test_lists_tensors_in_file_order_whatever_the_header_order(self, tmp_path: Path) -> None:
        entries = {"second": _entry(offsets=(4, 8)), "first": _entry(offsets=(0, 4))}
        path = write_entries(tmp_path / "m.safetensors", entries, bytes(range(8)))

        with SafetensorsFile(path) as file:
            assert list(file.tensors) == ["first", "second"]
            assert bytes(file.read("second")) == bytes([4, 5, 6, 7])

    def test_reads_a_scalar(self, tmp_path: Path) -> None:
        path = write_safetensors(tmp_path / "m.safetensors", {"s": filled("F32", ())})

        with SafetensorsFile(path) as file:
            assert file.tensors["s"].shape == ()
            assert len(file.read("s")) == 4

    def test_reads_an_empty_tensor_between_two_others(self, tmp_path: Path) -> None:
        path = write_safetensors(
            tmp_path / "m.safetensors",
            {"a": filled("U8", (3,)), "empty": filled("F32", (0, 5)), "b": filled("U8", (2,))},
        )

        with SafetensorsFile(path) as file:
            assert len(file.read("empty")) == 0
            assert bytes(file.read("b")) == bytes([0, 1])

    def test_a_file_with_no_tensors_is_valid(self, tmp_path: Path) -> None:
        with SafetensorsFile(write_raw(tmp_path / "m.safetensors", b"{}")) as file:
            assert dict(file.tensors) == {}
            assert dict(file.metadata) == {}

    def test_keeps_the_metadata(self, tmp_path: Path) -> None:
        path = write_safetensors(
            tmp_path / "m.safetensors", {"a": filled("U8", (1,))}, metadata={"format": "pt"}
        )

        with SafetensorsFile(path) as file:
            assert dict(file.metadata) == {"format": "pt"}
            assert "__metadata__" not in file.tensors

    def test_views_are_read_only(self, tmp_path: Path) -> None:
        path = write_safetensors(tmp_path / "m.safetensors", {"a": filled("U8", (4,))})

        with SafetensorsFile(path) as file:
            view = file.read("a")
            assert view.readonly
            with pytest.raises(TypeError):
                view[0] = 1
            view.release()

    def test_opening_reads_the_header_but_maps_nothing(self, tmp_path: Path) -> None:
        path = write_safetensors(tmp_path / "m.safetensors", {"a": filled("U8", (4,))})

        file = SafetensorsFile(path)

        assert file._mapping is None  # the property under test: the file is not mapped until read
        file.close()


class TestLifecycle:
    def test_unknown_tensor_is_a_key_error(self, tmp_path: Path) -> None:
        path = write_safetensors(tmp_path / "m.safetensors", {"a": filled("U8", (1,))})

        with SafetensorsFile(path) as file, pytest.raises(KeyError, match="nope"):
            file.read("nope")

    def test_cannot_read_after_close(self, tmp_path: Path) -> None:
        path = write_safetensors(tmp_path / "m.safetensors", {"a": filled("U8", (1,))})
        file = SafetensorsFile(path)
        file.close()

        with pytest.raises(ValueError, match="the file is closed"):
            file.read("a")

    def test_close_twice_is_harmless(self, tmp_path: Path) -> None:
        file = SafetensorsFile(write_safetensors(tmp_path / "m.safetensors", {}))
        file.close()
        file.close()

    def test_close_refuses_while_a_view_is_alive_and_works_after_release(
        self, tmp_path: Path
    ) -> None:
        path = write_safetensors(tmp_path / "m.safetensors", {"a": filled("U8", (4,))})
        file = SafetensorsFile(path)
        view = file.read("a")

        with pytest.raises(BufferError, match="release the memoryviews"):
            file.close()
        assert bytes(view) == bytes([0, 1, 2, 3])  # the file stayed open and readable
        view.release()
        file.close()

    def test_a_file_that_changes_size_after_opening_is_not_read(self, tmp_path: Path) -> None:
        path = write_safetensors(tmp_path / "m.safetensors", {"a": filled("U8", (8,))})
        file = SafetensorsFile(path)
        path.write_bytes(path.read_bytes()[:-3])

        with pytest.raises(ModelLoadError, match="changed after it was opened"):
            file.read("a")
        file.close()


class TestUnreadableFiles:
    def test_missing_file(self, tmp_path: Path) -> None:
        with pytest.raises(ModelLoadError, match="cannot read the file") as excinfo:
            SafetensorsFile(tmp_path / "absent.safetensors")

        assert excinfo.value.context["path"] == str(tmp_path / "absent.safetensors")

    def test_directory(self, tmp_path: Path) -> None:
        with pytest.raises(ModelLoadError, match="cannot read the file"):
            SafetensorsFile(tmp_path)

    @pytest.mark.parametrize("size", [0, 5, 7])
    def test_too_short_for_a_header_length(self, tmp_path: Path, size: int) -> None:
        path = tmp_path / "m.safetensors"
        path.write_bytes(b"\0" * size)

        with pytest.raises(ModelLoadError, match=rf"{size} bytes is too short"):
            SafetensorsFile(path)

    def test_header_longer_than_the_file(self, tmp_path: Path) -> None:
        path = write_raw(tmp_path / "m.safetensors", b"{}", declared_length=500)

        with pytest.raises(ModelLoadError, match="claims 500 bytes but the file has only 16"):
            SafetensorsFile(path)

    def test_forged_huge_length_is_refused_without_reading(self, tmp_path: Path) -> None:
        path = tmp_path / "m.safetensors"
        path.write_bytes(struct.pack("<Q", 2**63) + b"{}")

        with pytest.raises(ModelLoadError, match="probably forged"):
            SafetensorsFile(path)

    def test_length_just_over_the_limit(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(safetensors_module, "_HEADER_LIMIT", 16)
        path = write_raw(tmp_path / "m.safetensors", b"{" + b" " * 16 + b"}")

        with pytest.raises(ModelLoadError, match="above the limit of 16"):
            SafetensorsFile(path)


class TestCorruptHeaders:
    def _open(self, tmp_path: Path, header: bytes, payload: bytes = b"") -> SafetensorsFile:
        return SafetensorsFile(write_raw(tmp_path / "m.safetensors", header, payload))

    def test_not_utf8(self, tmp_path: Path) -> None:
        with pytest.raises(ModelLoadError, match="not valid UTF-8"):
            self._open(tmp_path, b'{"a": "\xff"}')

    @pytest.mark.parametrize("header", [b"", b"{", b"{'a': 1}", b"nonsense"])
    def test_not_json(self, tmp_path: Path, header: bytes) -> None:
        with pytest.raises(ModelLoadError, match="not valid JSON"):
            self._open(tmp_path, header)

    @pytest.mark.parametrize(("header", "kind"), [(b"[]", "list"), (b"3", "int"), (b'"x"', "str")])
    def test_not_an_object(self, tmp_path: Path, header: bytes, kind: str) -> None:
        with pytest.raises(ModelLoadError, match=rf"a JSON {kind}, not an object"):
            self._open(tmp_path, header)

    def test_a_repeated_tensor_name_is_not_silently_dropped(self, tmp_path: Path) -> None:
        one, two = json.dumps(_entry()), json.dumps(_entry(offsets=(4, 8)))
        header = f'{{"w": {one}, "w": {two}}}'.encode()

        with pytest.raises(ModelLoadError, match=r"duplicate object keys: \['w'\]"):
            self._open(tmp_path, header, bytes(8))

    def test_a_repeated_field_inside_an_entry_is_refused_too(self, tmp_path: Path) -> None:
        header = b'{"w": {"dtype": "U8", "dtype": "F32", "shape": [4], "data_offsets": [0, 4]}}'

        with pytest.raises(ModelLoadError, match="duplicate object keys"):
            self._open(tmp_path, header, bytes(4))

    def test_absurdly_deep_nesting_is_an_error_not_a_crash(self, tmp_path: Path) -> None:
        header = b'{"w": ' + b"[" * 100_000 + b"]" * 100_000 + b"}"

        with pytest.raises(ModelLoadError, match="nested too deeply"):
            self._open(tmp_path, header)

    def test_an_empty_tensor_name(self, tmp_path: Path) -> None:
        with pytest.raises(ModelLoadError, match="a tensor has an empty name"):
            write_entries_and_open(tmp_path, {"": _entry()}, bytes(4))

    def test_an_entry_that_is_not_an_object(self, tmp_path: Path) -> None:
        with pytest.raises(ModelLoadError, match=r"tensor 'w': the header entry is not an object"):
            write_entries_and_open(tmp_path, {"w": [1, 2]})

    @pytest.mark.parametrize("metadata", [["a"], {"a": 1}, {"a": {"b": "c"}}, "x"])
    def test_metadata_must_be_strings(self, tmp_path: Path, metadata: Any) -> None:
        with pytest.raises(ModelLoadError, match="__metadata__ must be an object"):
            write_entries_and_open(tmp_path, {"__metadata__": metadata})


class TestDtypes:
    def test_a_missing_dtype(self, tmp_path: Path) -> None:
        entry = _entry()
        del entry["dtype"]

        with pytest.raises(ModelLoadError, match=r"dtype must be a string, got None"):
            write_entries_and_open(tmp_path, {"w": entry}, bytes(4))

    def test_a_dtype_that_is_not_text(self, tmp_path: Path) -> None:
        with pytest.raises(ModelLoadError, match=r"dtype must be a string, got 32"):
            write_entries_and_open(tmp_path, {"w": _entry(dtype=32)}, bytes(4))

    @pytest.mark.parametrize("token", ["U16", "U32", "C64", "nonsense"])
    def test_a_dtype_this_build_does_not_read(self, tmp_path: Path, token: str) -> None:
        with pytest.raises(ModelLoadError, match=rf"dtype '{token}' is not one this build reads"):
            write_entries_and_open(tmp_path, {"w": _entry(dtype=token)}, bytes(4))

    @pytest.mark.parametrize("token", ["f32", "bf16", "Bool"])
    def test_a_token_in_the_wrong_case_is_not_forgiven(self, tmp_path: Path, token: str) -> None:
        with pytest.raises(ModelLoadError, match="must be spelled"):
            write_entries_and_open(tmp_path, {"w": _entry(dtype=token)}, bytes(4))

    @pytest.mark.parametrize("token", ["F4", "I4"])
    def test_sub_byte_dtypes_are_named_as_such(self, tmp_path: Path, token: str) -> None:
        with pytest.raises(ModelLoadError, match="packed below one byte"):
            write_entries_and_open(tmp_path, {"w": _entry(dtype=token)}, bytes(4))


class TestShapesAndOffsets:
    @pytest.mark.parametrize("shape", [-1, [-1], [2.0], [True], ["2"], None, "4"])
    def test_a_bad_shape(self, tmp_path: Path, shape: Any) -> None:
        entry = _entry()
        entry["shape"] = shape

        with pytest.raises(ModelLoadError, match="shape must be a list of non-negative integers"):
            write_entries_and_open(tmp_path, {"w": entry}, bytes(4))

    @pytest.mark.parametrize(
        "offsets", [None, [0], [0, 1, 2], [-1, 3], [0, -4], [0.0, 4], [True, 4], ["0", "4"], 4]
    )
    def test_malformed_offsets(self, tmp_path: Path, offsets: Any) -> None:
        entry = _entry()
        entry["data_offsets"] = offsets

        with pytest.raises(ModelLoadError, match=r"data_offsets must be \[begin, end\]"):
            write_entries_and_open(tmp_path, {"w": entry}, bytes(4))

    def test_offsets_that_run_backwards(self, tmp_path: Path) -> None:
        with pytest.raises(ModelLoadError, match=r"\[4, 0\] run backwards"):
            write_entries_and_open(tmp_path, {"w": _entry(offsets=(4, 0))}, bytes(4))

    def test_offsets_that_disagree_with_shape_and_dtype(self, tmp_path: Path) -> None:
        entry = _entry(dtype="F32", shape=(2,), offsets=(0, 4))

        with pytest.raises(ModelLoadError, match=r"span 4 bytes but shape \[2\] of F32 needs 8"):
            write_entries_and_open(tmp_path, {"w": entry}, bytes(8))

    def test_a_shape_whose_size_overflows_is_a_mismatch_not_an_allocation(
        self, tmp_path: Path
    ) -> None:
        entry = _entry(dtype="F32", shape=(2**40, 2**40, 2**40), offsets=(0, 4))

        with pytest.raises(ModelLoadError, match="needs"):
            write_entries_and_open(tmp_path, {"w": entry}, bytes(4))

    def test_bytes_past_the_end_of_the_file(self, tmp_path: Path) -> None:
        path = write_safetensors(tmp_path / "m.safetensors", {"w": filled("U8", (16,))})
        path.write_bytes(path.read_bytes()[:-5])

        with pytest.raises(
            ModelLoadError, match=r"tensor 'w': its bytes end at \d+ but the file"
        ) as e:
            SafetensorsFile(path)
        assert "truncated download" in str(e.value)

    def test_forged_offsets_near_the_u64_limit(self, tmp_path: Path) -> None:
        entry = _entry(shape=(2**64 - 1,), offsets=(0, 2**64 - 1))

        with pytest.raises(ModelLoadError, match="its bytes end at"):
            write_entries_and_open(tmp_path, {"w": entry}, bytes(4))

    def test_two_tensors_claiming_the_same_bytes(self, tmp_path: Path) -> None:
        entries = {"a": _entry(), "b": _entry()}

        with pytest.raises(
            ModelLoadError, match=r"'a' \[\d+, \d+\) and 'b' .* claim the same bytes"
        ) as e:
            write_entries_and_open(tmp_path, entries, bytes(4))
        assert e.value.context["tensors"] == ("a", "b")

    def test_a_tensor_starting_inside_another(self, tmp_path: Path) -> None:
        entries = {"big": _entry(shape=(8,), offsets=(0, 8)), "small": _entry(offsets=(2, 6))}

        with pytest.raises(ModelLoadError, match="claim the same bytes"):
            write_entries_and_open(tmp_path, entries, bytes(8))

    def test_neighbours_that_only_touch_do_not_overlap(self, tmp_path: Path) -> None:
        entries = {"a": _entry(offsets=(0, 4)), "b": _entry(offsets=(4, 8))}

        with write_entries_and_open(tmp_path, entries, bytes(8)) as file:
            assert list(file.tensors) == ["a", "b"]

    def test_an_empty_tensor_inside_another_is_harmless(self, tmp_path: Path) -> None:
        entries = {"big": _entry(offsets=(0, 4)), "empty": _entry(shape=(0,), offsets=(2, 2))}

        with write_entries_and_open(tmp_path, entries, bytes(4)) as file:
            assert len(file.read("empty")) == 0

    def test_errors_carry_the_file_and_the_tensor(self, tmp_path: Path) -> None:
        with pytest.raises(ModelLoadError) as excinfo:
            write_entries_and_open(tmp_path, {"w": _entry(offsets=(4, 0))}, bytes(4))

        assert excinfo.value.context == {"path": str(tmp_path / "m.safetensors"), "tensor": "w"}
        assert str(tmp_path / "m.safetensors") in str(excinfo.value)


def write_entries_and_open(
    tmp_path: Path, entries: dict[str, Any], payload: bytes = b""
) -> SafetensorsFile:
    return SafetensorsFile(write_entries(tmp_path / "m.safetensors", entries, payload))
