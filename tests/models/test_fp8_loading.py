"""FP8 checkpoints load as written by their producers and compute what their dequantized weights do.

Each tiny checkpoint is a random Llama-family state whose decoder projections are quantized to
e4m3 in one producer's layout and written under its names (``weight_scale_inv`` blocks,
``weight_scale`` per tensor with ``input_scale``, per channel, Phi-3's fused projections). The
model keeps the weights 8 bit and must equal the formulas run on the dequantized float32 state.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import pytest

torch: Any = pytest.importorskip("torch")

from ayaka.configs.device import DeviceConfig  # noqa: E402
from ayaka.configs.model_source import ModelSourceConfig  # noqa: E402
from ayaka.core.errors import ModelLoadError  # noqa: E402
from ayaka.layers.quantization.fp8 import Fp8Method  # noqa: E402
from ayaka.model_loader.loader import describe_model, load_model  # noqa: E402
from ayaka.utils.dtypes import DType  # noqa: E402
from fp8_support import (  # noqa: E402  # needs the importorskip above
    HF_FP8_BLOCK,
    compressed_tensors,
    fp8_data,
    quantize,
    quantize_state,
)
from gpt2_support import run_paged, tensor_data  # noqa: E402
from llama_support import (  # noqa: E402
    llama_config,
    llama_config_dict,
    llama_reference_hidden,
    llama_state,
)
from safetensors_files import write_safetensors  # noqa: E402

if TYPE_CHECKING:
    from pathlib import Path

    from ayaka.configs.model import ModelConfig

pytestmark = pytest.mark.unit


def _write(
    directory: Path, state: dict[str, Any], config: ModelConfig, recipe: dict[str, object] | None
) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    raw = llama_config_dict(config)
    if recipe is not None:
        raw["quantization_config"] = recipe
    (directory / "config.json").write_text(json.dumps(raw), encoding="utf-8")
    tensors = {
        name: fp8_data(t) if t.dtype == torch.float8_e4m3fn else tensor_data(t, torch.float32)
        for name, t in state.items()
    }
    write_safetensors(directory / "model.safetensors", tensors)
    return directory


def _assert_matches(model: Any, dense: dict[str, Any], config: ModelConfig) -> None:
    generator = torch.Generator().manual_seed(8)
    sequences = [torch.randint(0, 64, (n,), generator=generator) for n in (11, 6)]
    outputs = run_paged(model, sequences, prompt_lengths=[7, 6], chunk=3)
    for sequence, hidden in zip(sequences, outputs, strict=True):
        expected = llama_reference_hidden(dense, config, sequence)
        torch.testing.assert_close(hidden, expected, rtol=1e-4, atol=1e-4)


def _load(directory: Path) -> Any:
    return load_model(ModelSourceConfig(str(directory)), DeviceConfig(dtype=DType.FP32))


@pytest.mark.parametrize(
    ("architecture", "qk_norm"),
    [("LlamaForCausalLM", None), ("Qwen3ForCausalLM", "head")],
    ids=["llama", "qwen3"],
)
def test_hugging_face_fp8_blocks_load_into_the_fused_projections(
    tmp_path: Path, architecture: str, qk_norm: str | None
) -> None:
    config = llama_config(architecture=architecture)
    stored, dense = quantize_state(
        llama_state(config, seed=1, qk_norm=qk_norm), "block", scale_name="weight_scale_inv"
    )
    model = _load(_write(tmp_path / "ckpt", stored, config, HF_FP8_BLOCK))

    qkv = model.layers[0].attn.qkv_proj
    assert isinstance(qkv.method, Fp8Method)
    assert qkv.weight.dtype == torch.float8_e4m3fn
    # q 32, k 16, v 16 rows in 16-row blocks: 2 + 1 + 1 rows of scales, 2 columns of 16.
    assert tuple(qkv.weight_scale.shape) == (4, 2)
    _assert_matches(model, dense, config)


@pytest.mark.parametrize(
    ("strategy", "static_inputs"),
    [("tensor", True), ("channel", False)],
    ids=["tensor-static-inputs", "channel-dynamic-inputs"],
)
def test_compressed_tensors_scales_load_and_input_scales_are_read_past(
    tmp_path: Path, strategy: str, static_inputs: bool
) -> None:
    config = llama_config()
    stored, dense = quantize_state(
        llama_state(config, seed=2),
        strategy,
        scale_name="weight_scale",
        input_scales=static_inputs,
    )
    recipe = compressed_tensors(strategy, static_inputs=static_inputs)
    model = _load(_write(tmp_path / "ckpt", stored, config, recipe))

    # Each per-tensor scale of q, k and v is repeated over its rows of the fused projection.
    assert tuple(model.layers[0].attn.qkv_proj.weight_scale.shape) == (64, 1)
    _assert_matches(model, dense, config)


def test_phi3_fused_fp8_projections_load_whole(tmp_path: Path) -> None:
    config = llama_config(architecture="Phi3ForCausalLM")
    state = llama_state(config, seed=3)
    stored: dict[str, Any] = {}
    dense: dict[str, Any] = dict(state)
    for name, tensor in state.items():
        if ".layers." not in name or not name.endswith("_proj.weight"):
            stored[name] = tensor
    for layer in range(config.num_layers):
        attn, mlp = f"model.layers.{layer}.self_attn.", f"model.layers.{layer}.mlp."
        groups = {
            f"{attn}qkv_proj": [f"{attn}{p}_proj.weight" for p in "qkv"],
            f"{attn}o_proj": [f"{attn}o_proj.weight"],
            f"{mlp}gate_up_proj": [f"{mlp}gate_proj.weight", f"{mlp}up_proj.weight"],
            f"{mlp}down_proj": [f"{mlp}down_proj.weight"],
        }
        for fused, parts in groups.items():
            weight, scale = quantize(torch.cat([state[p] for p in parts]), "channel")
            stored[f"{fused}.weight"], stored[f"{fused}.weight_scale"] = weight, scale
            start = 0
            for part in parts:
                rows = state[part].shape[0]
                dense[part] = weight[start : start + rows].float() * scale[start : start + rows]
                start += rows
    model = _load(_write(tmp_path / "phi3", stored, config, compressed_tensors("channel")))

    _assert_matches(model, dense, config)


def test_the_plan_counts_eight_bit_weights_and_float32_scales(tmp_path: Path) -> None:
    config = llama_config()
    stored, _ = quantize_state(llama_state(config, seed=4), "block", scale_name="weight_scale_inv")
    directory = _write(tmp_path / "ckpt", stored, config, HF_FP8_BLOCK)

    _, built = describe_model(ModelSourceConfig(str(directory)), DeviceConfig(dtype=DType.BF16))
    model: Any = built

    qkv = model.layers[0].attn.qkv_proj
    assert qkv.weight.element_size() == 1
    assert qkv.weight_scale.dtype == torch.float32
    assert model.layers[0].input_layernorm.weight.dtype == torch.bfloat16


def test_parts_that_end_inside_a_block_are_refused(tmp_path: Path) -> None:
    config = llama_config()
    stored, _ = quantize_state(
        llama_state(config, seed=5), "block", scale_name="weight_scale_inv", block=(24, 16)
    )
    recipe = {**HF_FP8_BLOCK, "weight_block_size": [24, 16]}

    with pytest.raises(ModelLoadError, match="a part of 32 rows ends inside a block"):
        _load(_write(tmp_path / "ckpt", stored, config, recipe))


def test_a_recipe_over_16_bit_weights_is_refused(tmp_path: Path) -> None:
    config = llama_config()
    stored, _ = quantize_state(llama_state(config, seed=6), "block", scale_name="weight_scale_inv")
    widened = {n: t.float() if t.dtype == torch.float8_e4m3fn else t for n, t in stored.items()}

    with pytest.raises(ModelLoadError, match="is fp32, expected fp8_e4m3"):
        _load(_write(tmp_path / "ckpt", widened, config, HF_FP8_BLOCK))


def test_eight_bit_weights_without_a_recipe_are_still_refused(tmp_path: Path) -> None:
    config = llama_config()
    stored, _ = quantize_state(llama_state(config, seed=7), "channel", scale_name="weight_scale")

    with pytest.raises(ModelLoadError) as caught:
        _load(_write(tmp_path / "ckpt", stored, config, None))

    assert "weight_scale" in str(caught.value)  # no parameter claims the scales


def test_gpt2_refuses_a_quantized_checkpoint() -> None:
    from ayaka.models.gpt2 import GPT2LMHeadModel
    from ayaka.ops.backend import load_backend
    from ayaka.weights.quant import read_quant_format

    config = llama_config(architecture="GPT2LMHeadModel", hidden_act="gelu_new")
    quant = read_quant_format({"quantization_config": HF_FP8_BLOCK})

    with pytest.raises(ModelLoadError, match="does not run quantized checkpoints"):
        GPT2LMHeadModel(
            config,
            ops=load_backend(["torch"], device="cpu", dtype=DType.FP32),
            dtype=DType.FP32,
            device="meta",
            quant=quant,
        )
