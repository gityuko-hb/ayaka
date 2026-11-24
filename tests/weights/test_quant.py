"""The quantization recipe read from ``config.json``: what is accepted, what is refused and why."""

from __future__ import annotations

import copy

import pytest

from ayaka.core.errors import ModelLoadError
from ayaka.utils.dtypes import DType
from ayaka.weights.quant import QuantFormat, read_quant_format

pytestmark = pytest.mark.unit

# Verbatim from the checkpoints' config.json (Hugging Face Hub, read 2026-11-24).
QWEN3_8B_FP8 = {
    "activation_scheme": "dynamic",
    "fmt": "e4m3",
    "quant_method": "fp8",
    "weight_block_size": [128, 128],
}
LLAMA_31_8B_FP8 = {
    "config_groups": {
        "group_0": {
            "input_activations": {
                "block_structure": None,
                "dynamic": False,
                "group_size": None,
                "num_bits": 8,
                "observer": "minmax",
                "observer_kwargs": {},
                "strategy": "tensor",
                "symmetric": True,
                "type": "float",
            },
            "output_activations": None,
            "targets": ["Linear"],
            "weights": {
                "block_structure": None,
                "dynamic": False,
                "group_size": None,
                "num_bits": 8,
                "observer": "minmax",
                "observer_kwargs": {},
                "strategy": "tensor",
                "symmetric": True,
                "type": "float",
            },
        }
    },
    "format": "naive-quantized",
    "global_compression_ratio": 1.4590163747613412,
    "ignore": ["lm_head"],
    "kv_cache_scheme": None,
    "quant_method": "compressed-tensors",
    "quantization_status": "compressed",
}
PHI_4_FP8_DYNAMIC = {
    "config_groups": {
        "group_0": {
            "input_activations": {
                "actorder": None,
                "block_structure": None,
                "dynamic": True,
                "group_size": None,
                "num_bits": 8,
                "observer": None,
                "observer_kwargs": {},
                "strategy": "token",
                "symmetric": True,
                "type": "float",
            },
            "output_activations": None,
            "targets": ["Linear"],
            "weights": {
                "actorder": None,
                "block_structure": None,
                "dynamic": False,
                "group_size": None,
                "num_bits": 8,
                "observer": "mse",
                "observer_kwargs": {},
                "strategy": "channel",
                "symmetric": True,
                "type": "float",
            },
        }
    },
    "format": "float-quantized",
    "global_compression_ratio": 1.2816636668151653,
    "ignore": ["lm_head"],
    "kv_cache_scheme": None,
    "quant_method": "compressed-tensors",
    "quantization_status": "compressed",
}


def test_a_checkpoint_without_a_recipe_is_not_quantized() -> None:
    assert read_quant_format({"architectures": ["LlamaForCausalLM"]}) is None


def test_hugging_face_fp8_is_e4m3_with_128_by_128_blocks() -> None:
    found = read_quant_format({"quantization_config": QWEN3_8B_FP8})

    assert found is not None
    assert found == QuantFormat(
        producer="fp8",
        weight_dtype=DType.FP8_E4M3,
        scale_layout="block",
        block=(128, 128),
        scale_name="weight_scale_inv",
    )
    assert found.name == "fp8_e4m3/block-128x128"
    assert found.bits == 8


def test_compressed_tensors_per_tensor_with_static_inputs_ignores_the_input_scales() -> None:
    found = read_quant_format({"quantization_config": LLAMA_31_8B_FP8})

    assert found is not None
    assert (found.producer, found.scale_layout, found.block) == (
        "compressed-tensors",
        "tensor",
        None,
    )
    assert found.scale_name == "weight_scale"
    assert found.ignored == ("input_scale",)


def test_compressed_tensors_per_channel_with_dynamic_inputs_stores_no_input_scale() -> None:
    found = read_quant_format({"quantization_config": PHI_4_FP8_DYNAMIC})

    assert found is not None
    assert (found.scale_layout, found.ignored, found.name) == ("channel", (), "fp8_e4m3/channel")


def test_compressed_tensors_blocks_take_their_block_structure() -> None:
    recipe = copy.deepcopy(PHI_4_FP8_DYNAMIC)
    weights = recipe["config_groups"]["group_0"]["weights"]  # type: ignore[index]
    weights.update({"strategy": "block", "block_structure": [128, 64]})

    found = read_quant_format({"quantization_config": recipe})

    assert found is not None
    assert (found.scale_layout, found.block) == ("block", (128, 64))


def _with(base: dict[str, object], path: tuple[str, ...], value: object) -> dict[str, object]:
    recipe = copy.deepcopy(base)
    node: dict[str, object] = recipe
    for key in path[:-1]:
        node = node[key]  # type: ignore[assignment]
    node[path[-1]] = value
    return recipe


_WEIGHTS = ("config_groups", "group_0", "weights")


@pytest.mark.parametrize(
    ("recipe", "message"),
    [
        ({"quant_method": "gptq"}, "quant_method 'gptq' is not supported"),
        (_with(QWEN3_8B_FP8, ("fmt",), "e5m2"), "fmt 'e5m2'"),
        (_with(QWEN3_8B_FP8, ("activation_scheme",), "static"), "activation_scheme 'static'"),
        (_with(QWEN3_8B_FP8, ("weight_block_size",), None), "without weight_block_size"),
        (_with(QWEN3_8B_FP8, ("weight_block_size",), [128]), "two positive integers"),
        (
            _with(QWEN3_8B_FP8, ("modules_to_not_convert",), ["lm_head", "model.layers.0.mlp"]),
            r"leaves \['model.layers.0.mlp'\] in 16 bit",
        ),
        (_with(PHI_4_FP8_DYNAMIC, ("format",), "pack-quantized"), "format 'pack-quantized'"),
        (_with(PHI_4_FP8_DYNAMIC, ("kv_cache_scheme",), {"num_bits": 8}), "kv_cache_scheme"),
        (_with(PHI_4_FP8_DYNAMIC, ("ignore",), ["lm_head", "re:.*down_proj"]), "16 bit"),
        (_with(PHI_4_FP8_DYNAMIC, (*_WEIGHTS, "type"), "int"), "type is 'int'"),
        (_with(PHI_4_FP8_DYNAMIC, (*_WEIGHTS, "num_bits"), 4), "num_bits is 4"),
        (_with(PHI_4_FP8_DYNAMIC, (*_WEIGHTS, "symmetric"), False), "not symmetric"),
        (_with(PHI_4_FP8_DYNAMIC, (*_WEIGHTS, "strategy"), "group"), "strategy 'group'"),
        (
            _with(PHI_4_FP8_DYNAMIC, ("config_groups", "group_0", "targets"), ["Attention"]),
            "targets",
        ),
        (
            _with(PHI_4_FP8_DYNAMIC, ("quantization_status",), "frozen"),
            "quantization_status 'frozen'",
        ),
    ],
)
def test_what_is_not_run_is_refused_with_the_reason(recipe: object, message: str) -> None:
    with pytest.raises(ModelLoadError, match=message):
        read_quant_format({"quantization_config": recipe})


def test_two_config_groups_are_refused() -> None:
    recipe = copy.deepcopy(PHI_4_FP8_DYNAMIC)
    groups = recipe["config_groups"]  # type: ignore[index]
    groups["group_1"] = groups["group_0"]

    with pytest.raises(ModelLoadError, match="2 groups"):
        read_quant_format({"quantization_config": recipe})


@pytest.mark.parametrize(
    ("fields", "message"),
    [
        ({"scale_layout": "block", "block": None}, "block is given exactly"),
        ({"scale_layout": "channel", "block": (1, 1)}, "block is given exactly"),
        ({"scale_layout": "block", "block": (0, 128)}, "positive"),
        ({"weight_dtype": DType.INT8}, "8-bit float"),
    ],
)
def test_an_inconsistent_record_is_refused(fields: dict[str, object], message: str) -> None:
    base: dict[str, object] = {
        "producer": "fp8",
        "weight_dtype": DType.FP8_E4M3,
        "scale_layout": "channel",
        "block": None,
        "scale_name": "weight_scale",
    }
    base.update(fields)

    with pytest.raises(ValueError, match=message):
        QuantFormat(**base)  # type: ignore[arg-type]  # the fields, by design
