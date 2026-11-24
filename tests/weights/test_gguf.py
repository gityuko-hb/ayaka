"""The GGUF reader: header, metadata and tensor table from self-written files, broken ones too."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import pytest

from ayaka.core.errors import ModelLoadError
from ayaka.weights.gguf import GGMLType, gguf_hf_config, read_gguf
from gguf_files import (
    ARRAY,
    BOOL,
    F32,
    F64,
    I8,
    I64,
    STRING,
    U8,
    U32,
    U64,
    llama_metadata,
    write_gguf,
)

if TYPE_CHECKING:
    from pathlib import Path

pytestmark = pytest.mark.unit


def test_metadata_of_every_value_type_round_trips(tmp_path: Path) -> None:
    metadata = {
        "a.u8": (U8, 7),
        "a.i8": (I8, -3),
        "a.u32": (U32, 70000),
        "a.u64": (U64, 2**40),
        "a.i64": (I64, -(2**40)),
        "a.f32": (F32, 0.5),
        "a.f64": (F64, 0.25),
        "a.bool": (BOOL, True),
        "a.text": (STRING, "héllo"),
        "a.list": (ARRAY, (U32, [1, 2, 3])),
        "a.words": (ARRAY, (STRING, ["x", "yz"])),
    }

    found = read_gguf(write_gguf(tmp_path / "m.gguf", metadata, [], data_bytes=0))

    assert found.version == 3
    expected = {
        key: item[1] if kind == ARRAY else item  # type: ignore[index]  # (type, list)
        for key, (kind, item) in metadata.items()
    }
    assert dict(found.metadata) == expected
    assert found.tensors == ()


def test_dimensions_are_reversed_and_sizes_follow_the_types(tmp_path: Path) -> None:
    table = [
        ("token_embd.weight", [256, 10], GGMLType.Q4_K.type_id, 0),  # 10 x 256 elements
        ("blk.0.ffn_down.weight", [32, 4], GGMLType.Q8_0.type_id, 1472),
        ("output_norm.weight", [32], GGMLType.F32.type_id, 1632),
    ]
    found = read_gguf(write_gguf(tmp_path / "m.gguf", {}, table, data_bytes=1760))

    embd, down, norm = found.tensors
    assert (embd.shape, embd.nbytes) == ((10, 256), 10 * 144)
    assert (down.shape, down.nbytes) == ((4, 32), 4 * 34)
    assert (norm.shape, norm.nbytes) == ((32,), 128)
    assert embd.offset == found.data_offset
    assert found.data_offset % 32 == 0
    assert found.weights_bytes == 1440 + 136 + 128
    assert found.type_counts() == {"F32": 1, "Q4_K": 1, "Q8_0": 1}


@pytest.mark.parametrize("ggml", list(GGMLType), ids=lambda t: t.name)
def test_every_type_counts_whole_blocks(ggml: GGMLType) -> None:
    shape = (3, 2 * ggml.block_size)

    assert ggml.nbytes(shape) == 3 * 2 * ggml.type_size
    assert GGMLType.by_id(ggml.type_id) is ggml


def test_the_block_sizes_are_those_of_ggml() -> None:
    # (elements, bytes) per block from ggml's type traits.
    assert {t.name: (t.block_size, t.type_size) for t in GGMLType} == {
        "F32": (1, 4),
        "F16": (1, 2),
        "Q4_0": (32, 18),
        "Q4_1": (32, 20),
        "Q5_0": (32, 22),
        "Q5_1": (32, 24),
        "Q8_0": (32, 34),
        "Q8_1": (32, 36),
        "Q2_K": (256, 84),
        "Q3_K": (256, 110),
        "Q4_K": (256, 144),
        "Q5_K": (256, 176),
        "Q6_K": (256, 210),
        "Q8_K": (256, 292),
        "IQ2_XXS": (256, 66),
        "IQ2_XS": (256, 74),
        "IQ3_XXS": (256, 98),
        "IQ1_S": (256, 50),
        "IQ4_NL": (32, 18),
        "IQ3_S": (256, 110),
        "IQ2_S": (256, 82),
        "IQ4_XS": (256, 136),
        "I8": (1, 1),
        "I16": (1, 2),
        "I32": (1, 4),
        "I64": (1, 8),
        "F64": (1, 8),
        "IQ1_M": (256, 56),
        "BF16": (1, 2),
        "TQ1_0": (256, 54),
        "TQ2_0": (256, 66),
        "MXFP4": (32, 17),
    }


def test_a_custom_alignment_is_honoured(tmp_path: Path) -> None:
    table = [("a", [64], GGMLType.F16.type_id, 0), ("b", [64], GGMLType.F16.type_id, 128)]
    path = write_gguf(
        tmp_path / "m.gguf",
        {"general.alignment": (U32, 64)},
        table,
        data_bytes=256,
        alignment=64,
    )

    found = read_gguf(path)

    assert (found.alignment, found.data_offset % 64) == (64, 0)
    assert found.tensors[1].offset == found.data_offset + 128


def _broken(tmp_path: Path, **kwargs: object) -> Path:
    fields: dict[str, object] = {
        "metadata": {},
        "tensors": [("a", [32], GGMLType.F32.type_id, 0)],
        "data_bytes": 128,
    }
    fields.update(kwargs)
    return write_gguf(tmp_path / "bad.gguf", **fields)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"magic": b"GGML"}, "magic is not 'GGUF'"),
        ({"version": 1}, "version 1 is not supported"),
        ({"tensor_count": 2**21}, "tensors declared, above"),
        ({"tensor_count": 5, "data_bytes": 0}, "runs past the end|has 0 dimensions"),
        ({"data_bytes": 64}, "past the end of the 1"),
        ({"tensors": [("a", [32], 4, 0)]}, "unknown GGML type id 4"),
        ({"tensors": [("a", [32], GGMLType.F32.type_id, 8)]}, "not a multiple of 32"),
        ({"tensors": [("a", [100], GGMLType.Q4_0.type_id, 0)]}, "32 elements per block"),
        ({"tensors": [("a", [], GGMLType.F32.type_id, 0)]}, "has 0 dimensions"),
        (
            {
                "tensors": [
                    ("a", [32], GGMLType.F32.type_id, 0),
                    ("b", [32], GGMLType.F32.type_id, 96),
                ],
                "data_bytes": 256,
            },
            "b overlaps a",
        ),
        (
            {
                "tensors": [
                    ("a", [8], GGMLType.F32.type_id, 0),
                    ("a", [8], GGMLType.F32.type_id, 32),
                ]
            },
            "listed twice",
        ),
        ({"metadata": {"general.alignment": (U32, 48)}}, "not a power of two"),
        ({"metadata": {"general.alignment": (STRING, "32")}}, "not a positive integer"),
    ],
)
def test_a_broken_header_is_refused_with_the_reason(
    tmp_path: Path, kwargs: dict[str, object], message: str
) -> None:
    with pytest.raises(ModelLoadError, match=message):
        read_gguf(_broken(tmp_path, **kwargs))


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ((13).to_bytes(4, "little") + b"\0" * 8, "unknown value type 13"),
        (
            (ARRAY).to_bytes(4, "little")
            + (ARRAY).to_bytes(4, "little")
            + (1).to_bytes(8, "little"),
            "unknown value type 9",
        ),
    ],
    ids=["unknown-type", "nested-array"],
)
def test_an_unknown_or_nested_value_type_is_refused(
    tmp_path: Path, payload: bytes, message: str
) -> None:
    from gguf_files import string

    path = tmp_path / "type.gguf"
    path.write_bytes(
        b"GGUF"
        + (3).to_bytes(4, "little")
        + (0).to_bytes(8, "little")
        + (1).to_bytes(8, "little")
        + string("x")
        + payload
    )

    with pytest.raises(ModelLoadError, match=message):
        read_gguf(path)


def test_a_repeated_metadata_key_is_refused(tmp_path: Path) -> None:
    from gguf_files import string, value

    path = tmp_path / "dup.gguf"
    entry = string("k") + (U32).to_bytes(4, "little") + value(U32, 1)
    path.write_bytes(
        b"GGUF"
        + (3).to_bytes(4, "little")
        + (0).to_bytes(8, "little")
        + (2).to_bytes(8, "little")
        + entry
        + entry
    )

    with pytest.raises(ModelLoadError, match="'k' is repeated"):
        read_gguf(path)


def test_a_huge_string_length_is_refused_before_it_is_read(tmp_path: Path) -> None:
    path = tmp_path / "long.gguf"
    path.write_bytes(
        b"GGUF"
        + (3).to_bytes(4, "little")
        + (0).to_bytes(8, "little")
        + (1).to_bytes(8, "little")
        + (2**40).to_bytes(8, "little")
    )

    with pytest.raises(ModelLoadError, match="above"):
        read_gguf(path)


def test_a_missing_file_is_a_load_error(tmp_path: Path) -> None:
    with pytest.raises(ModelLoadError, match="cannot read"):
        read_gguf(tmp_path / "absent.gguf")


def _llama_table(*, tied: bool = False) -> tuple[list[tuple[str, list[int], int, int]], int]:
    """A mixed-type tensor table of the tiny llama: Q4_K, Q6_K, Q8_0 and F32 tensors."""
    entries: list[tuple[str, list[int], GGMLType]] = [
        ("token_embd.weight", [256, 256], GGMLType.Q6_K),
        ("output_norm.weight", [32], GGMLType.F32),
    ]
    if not tied:
        entries.append(("output.weight", [256, 256], GGMLType.Q6_K))
    for layer in range(2):
        entries += [
            (f"blk.{layer}.attn_norm.weight", [32], GGMLType.F32),
            (f"blk.{layer}.attn_q.weight", [256, 1], GGMLType.Q4_K),
            (f"blk.{layer}.attn_v.weight", [32, 16], GGMLType.Q8_0),
            (f"blk.{layer}.ffn_down.weight", [64, 32], GGMLType.Q8_0),
        ]
    table: list[tuple[str, list[int], int, int]] = []
    offset = 0
    for name, ne, ggml in entries:
        table.append((name, ne, ggml.type_id, offset))
        size = ggml.nbytes(tuple(reversed(ne)))
        offset += -(-size // 32) * 32
    return table, offset


def test_the_sum_of_a_mixed_table_is_the_weights_bytes(tmp_path: Path) -> None:
    table, end = _llama_table()
    found = read_gguf(write_gguf(tmp_path / "m.gguf", llama_metadata(), table, data_bytes=end))

    expected = sum(
        GGMLType.by_id(type_id).nbytes(tuple(reversed(ne))) for _, ne, type_id, _ in table
    )
    assert found.weights_bytes == expected
    assert set(found.type_counts()) == {"Q4_K", "Q6_K", "Q8_0", "F32"}


def test_llama_metadata_becomes_a_config(tmp_path: Path) -> None:
    table, end = _llama_table(tied=True)
    path = write_gguf(tmp_path / "m.gguf", llama_metadata(), table, data_bytes=end)

    raw = gguf_hf_config(read_gguf(path))

    assert raw["architectures"] == ["LlamaForCausalLM"]
    assert (raw["hidden_size"], raw["num_hidden_layers"], raw["intermediate_size"]) == (32, 2, 64)
    assert (raw["num_attention_heads"], raw["num_key_value_heads"], raw["head_dim"]) == (4, 2, 8)
    assert (raw["vocab_size"], raw["max_position_embeddings"]) == (256, 128)
    assert raw["tie_word_embeddings"] is True
    assert math.isclose(raw["rope_theta"], 10000.0)  # type: ignore[arg-type]


def test_rope_scaling_is_read_from_the_metadata(tmp_path: Path) -> None:
    metadata = llama_metadata(
        **{
            "llama.rope.scaling.type": (STRING, "yarn"),
            "llama.rope.scaling.factor": (F32, 4.0),
            "llama.rope.scaling.original_context_length": (U32, 32),
        }
    )
    path = write_gguf(tmp_path / "m.gguf", metadata, [], data_bytes=0)

    raw = gguf_hf_config(read_gguf(path))

    assert raw["rope_scaling"] == {
        "rope_type": "yarn",
        "factor": 4.0,
        "original_max_position_embeddings": 32,
    }


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"general.architecture": (STRING, "qwen2")}, "architecture 'qwen2' is not supported"),
        ({"llama.block_count": (STRING, "2")}, "llama.block_count is missing or not a number"),
        ({"tokenizer.ggml.tokens": (U32, 0)}, "neither llama.vocab_size nor"),
    ],
)
def test_metadata_that_cannot_be_planned_is_refused(
    tmp_path: Path, overrides: dict[str, tuple[int, object]], message: str
) -> None:
    path = write_gguf(tmp_path / "m.gguf", llama_metadata(**overrides), [], data_bytes=0)

    with pytest.raises(ModelLoadError, match=message):
        gguf_hf_config(read_gguf(path))
