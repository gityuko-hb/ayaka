"""OLMoE and Qwen3-MoE: their configs, their checkpoint layout and what they refuse.

The arithmetic is checked against ``transformers`` in ``test_moe_transformers.py``; this file
needs only torch.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import pytest

from ayaka.configs.model import ModelConfig
from ayaka.core.errors import ConfigError, ModelLoadError

torch: Any = pytest.importorskip("torch")

from ayaka.configs.device import DeviceConfig  # noqa: E402
from ayaka.configs.model_source import ModelSourceConfig  # noqa: E402
from ayaka.layers.ffn.dense import GatedMLP  # noqa: E402
from ayaka.layers.ffn.moe import MoEBlock  # noqa: E402
from ayaka.model_loader.loader import describe_model  # noqa: E402
from ayaka.models.olmoe import OlmoeForCausalLM  # noqa: E402
from ayaka.models.qwen3_moe import Qwen3MoeForCausalLM  # noqa: E402
from ayaka.models.registry import resolve_model  # noqa: E402
from ayaka.ops.backend import load_backend  # noqa: E402
from ayaka.utils.dtypes import DType  # noqa: E402
from ayaka.weights.quant import read_quant_format  # noqa: E402

if TYPE_CHECKING:
    from pathlib import Path

pytestmark = pytest.mark.unit

# The MoE fields of the checkpoints' config.json, verbatim (Hugging Face Hub, read 2026-11-24).
OLMOE_1B_7B_0125_INSTRUCT: dict[str, object] = {
    "architectures": ["OlmoeForCausalLM"],
    "attention_bias": False,
    "clip_qkv": None,
    "hidden_act": "silu",
    "hidden_size": 2048,
    "intermediate_size": 1024,
    "max_position_embeddings": 4096,
    "model_type": "olmoe",
    "norm_topk_prob": False,
    "num_attention_heads": 16,
    "num_experts": 64,
    "num_experts_per_tok": 8,
    "num_hidden_layers": 16,
    "num_key_value_heads": 16,
    "rms_norm_eps": 1e-05,
    "rope_scaling": None,
    "rope_theta": 10000.0,
    "tie_word_embeddings": False,
    "vocab_size": 50304,
}
QWEN3_30B_A3B: dict[str, object] = {
    "architectures": ["Qwen3MoeForCausalLM"],
    "attention_bias": False,
    "decoder_sparse_step": 1,
    "head_dim": 128,
    "hidden_act": "silu",
    "hidden_size": 2048,
    "intermediate_size": 6144,
    "max_position_embeddings": 40960,
    "max_window_layers": 48,
    "mlp_only_layers": [],
    "model_type": "qwen3_moe",
    "moe_intermediate_size": 768,
    "norm_topk_prob": True,
    "num_attention_heads": 32,
    "num_experts": 128,
    "num_experts_per_tok": 8,
    "num_hidden_layers": 48,
    "num_key_value_heads": 4,
    "rms_norm_eps": 1e-06,
    "rope_scaling": None,
    "rope_theta": 1000000.0,
    "sliding_window": None,
    "tie_word_embeddings": False,
    "use_sliding_window": False,
    "vocab_size": 151936,
}


def _tiny(raw: dict[str, object], **overrides: object) -> dict[str, object]:
    return {
        **raw,
        "hidden_size": 32,
        "intermediate_size": 24,
        "moe_intermediate_size": 12,
        "num_hidden_layers": 3,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 8,
        "num_experts": 4,
        "num_experts_per_tok": 2,
        "vocab_size": 64,
        "max_position_embeddings": 128,
        **overrides,
    }


def test_olmoe_reads_as_64_experts_of_width_1024_without_renormalisation() -> None:
    config = ModelConfig.from_hf_config(OLMOE_1B_7B_0125_INSTRUCT)

    assert config.is_moe
    assert (config.num_experts, config.experts_per_token) == (64, 8)
    assert config.moe_intermediate_size == 1024  # OLMoE's experts are intermediate_size wide
    assert config.norm_topk_prob is False
    assert config.dense_layers == ()
    assert config.tie_word_embeddings is False


def test_qwen3_30b_a3b_reads_as_128_experts_of_width_768_renormalised() -> None:
    config = ModelConfig.from_hf_config(QWEN3_30B_A3B)

    assert (config.num_experts, config.experts_per_token, config.moe_intermediate_size) == (
        128,
        8,
        768,
    )
    assert config.norm_topk_prob is True
    assert config.dense_layers == ()


@pytest.mark.parametrize(
    ("overrides", "dense"),
    [({"mlp_only_layers": [0, 2]}, (0, 2)), ({"decoder_sparse_step": 2}, (0, 2)), ({}, ())],
    ids=["mlp-only", "sparse-step", "every-layer"],
)
def test_the_dense_layers_follow_qwen3_moe(
    overrides: dict[str, object], dense: tuple[int, ...]
) -> None:
    config = ModelConfig.from_hf_config(_tiny(QWEN3_30B_A3B, **overrides))

    assert config.dense_layers == dense


def test_a_dense_config_has_no_expert_fields() -> None:
    config = ModelConfig.from_hf_config(_tiny(QWEN3_30B_A3B, num_experts=0))

    assert not config.is_moe
    assert (config.num_experts, config.experts_per_token, config.dense_layers) == (0, 0, ())


def test_clip_qkv_is_refused() -> None:
    with pytest.raises(ConfigError, match=r"clip_qkv 8\.0"):
        ModelConfig.from_hf_config({**OLMOE_1B_7B_0125_INSTRUCT, "clip_qkv": 8.0})


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"num_experts_per_tok": 5}, "between 1 and num_experts"),
        ({"num_experts_per_tok": 0}, "between 1 and num_experts"),
    ],
)
def test_an_inconsistent_moe_config_is_refused(overrides: dict[str, object], message: str) -> None:
    with pytest.raises(ConfigError, match=message):
        ModelConfig.from_hf_config(_tiny(OLMOE_1B_7B_0125_INSTRUCT, **overrides))


def test_the_registry_resolves_both() -> None:
    assert resolve_model("OlmoeForCausalLM") is OlmoeForCausalLM
    assert resolve_model("Qwen3MoeForCausalLM") is Qwen3MoeForCausalLM


def _build(raw: dict[str, object], cls: type[Any]) -> Any:
    ops = load_backend(["torch"], device="cpu", dtype=DType.FP32)
    return cls(ModelConfig.from_hf_config(raw), ops=ops, dtype=DType.FP32, device="meta")


def test_the_checkpoint_lists_each_expert_and_the_router() -> None:
    model = _build(_tiny(OLMOE_1B_7B_0125_INSTRUCT), OlmoeForCausalLM)

    specs = {spec.name: spec.shape for spec in model.checkpoint_specs()}

    assert specs["layers.0.mlp.gate.weight"] == (4, 32)
    for expert in range(4):
        assert specs[f"layers.0.mlp.experts.{expert}.gate_proj.weight"] == (12, 32)
        assert specs[f"layers.0.mlp.experts.{expert}.up_proj.weight"] == (12, 32)
        assert specs[f"layers.0.mlp.experts.{expert}.down_proj.weight"] == (32, 12)
    assert specs["layers.0.attn.q_norm.weight"] == (32,)  # over the whole projection
    assert specs["layers.0.attn.k_norm.weight"] == (16,)
    assert "layers.0.mlp.experts.gate_up_proj" not in specs


def test_qwen3_moe_builds_a_dense_mlp_where_the_config_says() -> None:
    model = _build(_tiny(QWEN3_30B_A3B, mlp_only_layers=[1]), Qwen3MoeForCausalLM)

    kinds = [type(layer.mlp) for layer in model.layers]

    assert kinds == [MoEBlock, GatedMLP, MoEBlock]
    assert model.layers[1].mlp.gate_up_proj.out_features == 2 * 24
    assert model.layers[0].attn.q_norm.weight.shape == (8,)  # Qwen3's per-head norm


def test_a_dense_config_is_refused_by_the_moe_models() -> None:
    with pytest.raises(ModelLoadError, match="needs num_experts"):
        _build(_tiny(QWEN3_30B_A3B, num_experts=0), Qwen3MoeForCausalLM)


def test_a_quantized_moe_checkpoint_is_refused() -> None:
    quant = read_quant_format(
        {
            "quantization_config": {
                "quant_method": "fp8",
                "fmt": "e4m3",
                "activation_scheme": "dynamic",
                "weight_block_size": [128, 128],
            }
        }
    )
    ops = load_backend(["torch"], device="cpu", dtype=DType.FP32)

    with pytest.raises(ModelLoadError, match="does not run quantized checkpoints"):
        OlmoeForCausalLM(
            ModelConfig.from_hf_config(_tiny(OLMOE_1B_7B_0125_INSTRUCT)),
            ops=ops,
            dtype=DType.FP32,
            device="meta",
            quant=quant,
        )


def test_the_planned_weights_of_olmoe_1b_7b_are_its_parameters_in_bf16(tmp_path: Path) -> None:
    (tmp_path / "config.json").write_text(json.dumps(OLMOE_1B_7B_0125_INSTRUCT), encoding="utf-8")

    _, model = describe_model(ModelSourceConfig(str(tmp_path)), DeviceConfig(dtype=DType.BF16))

    parameters = sum(p.numel() for p in model.parameters())
    # 16 layers x (64 experts x 3 x 1024 x 2048 + router 64 x 2048 + attention 4 x 2048^2 + norms)
    # + embedding and head 2 x 50304 x 2048 + final norm.
    per_layer = 64 * 3 * 1024 * 2048 + 64 * 2048 + 4 * 2048 * 2048 + 2 * 2048 + 2 * 2048
    assert parameters == 16 * per_layer + 2 * 50304 * 2048 + 2048
