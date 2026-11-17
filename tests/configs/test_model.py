from __future__ import annotations

from dataclasses import FrozenInstanceError, replace

import pytest

from ayaka.configs.model import ModelConfig, RopeScaling
from ayaka.core.errors import ConfigError, ModelLoadError

GPT2_JSON: dict[str, object] = {
    "architectures": ["GPT2LMHeadModel"],
    "vocab_size": 50257,
    "n_embd": 768,
    "n_layer": 12,
    "n_head": 12,
    "n_inner": None,
    "n_positions": 1024,
    "activation_function": "gelu_new",
    "layer_norm_epsilon": 1e-5,
}

LLAMA_JSON: dict[str, object] = {
    "architectures": ["LlamaForCausalLM"],
    "vocab_size": 128256,
    "hidden_size": 2048,
    "num_hidden_layers": 16,
    "num_attention_heads": 32,
    "num_key_value_heads": 8,
    "intermediate_size": 8192,
    "max_position_embeddings": 131072,
    "hidden_act": "silu",
    "rms_norm_eps": 1e-5,
    "tie_word_embeddings": True,
    "head_dim": 64,
}


def make(**overrides: object) -> ModelConfig:
    values: dict[str, object] = {
        "architecture": "TinyForCausalLM",
        "vocab_size": 64,
        "hidden_size": 32,
        "num_layers": 2,
        "num_attention_heads": 4,
        "num_kv_heads": 2,
        "head_dim": 8,
        "intermediate_size": 64,
        "max_position_embeddings": 128,
        "hidden_act": "silu",
        "norm_eps": 1e-5,
        "tie_word_embeddings": False,
    }
    values.update(overrides)
    return ModelConfig(**values)  # type: ignore[arg-type]  # values are built from known-good types


@pytest.mark.unit
class TestModelConfig:
    def test_is_frozen(self) -> None:
        config = make()

        with pytest.raises(FrozenInstanceError):
            config.num_layers = 3  # type: ignore[misc]  # frozen dataclass

    def test_kv_group_size(self) -> None:
        assert make().kv_group_size == 2
        assert make(num_kv_heads=4).kv_group_size == 1

    def test_integer_eps_is_stored_as_float(self) -> None:
        config = make(norm_eps=1)

        assert config.norm_eps == 1.0
        assert isinstance(config.norm_eps, float)

    def test_serialization_and_fingerprint(self) -> None:
        config = make()

        assert config.to_dict()["architecture"] == "TinyForCausalLM"
        assert config.fingerprint == make().fingerprint
        assert config.fingerprint != replace(config, num_layers=3).fingerprint

    @pytest.mark.parametrize(
        "field",
        [
            "vocab_size",
            "hidden_size",
            "num_layers",
            "num_attention_heads",
            "num_kv_heads",
            "head_dim",
            "intermediate_size",
            "max_position_embeddings",
        ],
    )
    def test_rejects_non_positive_sizes(self, field: str) -> None:
        with pytest.raises(ValueError, match=rf"model\.{field} must be >= 1"):
            make(**{field: 0})

    @pytest.mark.parametrize("value", [True, 1.0, "3"])
    def test_rejects_non_integer_sizes(self, value: object) -> None:
        with pytest.raises(TypeError, match=r"model\.num_layers must be an integer"):
            make(num_layers=value)

    @pytest.mark.parametrize("field", ["architecture", "hidden_act"])
    @pytest.mark.parametrize("value", ["", "  ", None])
    def test_rejects_blank_text(self, field: str, value: object) -> None:
        with pytest.raises(ValueError, match=rf"model\.{field} must be a non-empty string"):
            make(**{field: value})

    @pytest.mark.parametrize("eps", [0.0, -1e-5, float("nan"), float("inf")])
    def test_rejects_bad_eps(self, eps: float) -> None:
        with pytest.raises((ConfigError, ValueError), match="norm_eps"):
            make(norm_eps=eps)

    def test_rejects_non_bool_tie(self) -> None:
        with pytest.raises(TypeError, match="tie_word_embeddings must be bool"):
            make(tie_word_embeddings=1)

    def test_kv_heads_must_not_exceed_heads(self) -> None:
        with pytest.raises(ConfigError) as caught:
            make(num_kv_heads=8)

        assert caught.value.code == "KV_HEADS_EXCEED_HEADS"
        assert caught.value.path == "model.num_kv_heads"

    def test_kv_heads_must_divide_heads(self) -> None:
        with pytest.raises(ConfigError) as caught:
            make(num_attention_heads=6, num_kv_heads=4)

        assert caught.value.code == "KV_HEADS_NOT_DIVISOR"

    def test_a_scaling_needs_a_rotary_embedding(self) -> None:
        with pytest.raises(ConfigError) as caught:
            make(rope_scaling=RopeScaling(kind="linear", factor=2.0))

        assert caught.value.code == "ROPE_SCALING_WITHOUT_ROPE"

    @pytest.mark.parametrize("theta", [0.5, float("inf"), -1.0])
    def test_rope_theta_must_be_above_one(self, theta: float) -> None:
        with pytest.raises(ConfigError, match="rope_theta"):
            make(rope_theta=theta)


@pytest.mark.unit
class TestRopeScaling:
    def test_llama3_needs_the_trained_context(self) -> None:
        with pytest.raises(ConfigError) as caught:
            RopeScaling(kind="llama3", factor=8.0)

        assert caught.value.code == "ROPE_SCALING_INCOMPLETE"

    def test_a_factor_below_one_is_refused(self) -> None:
        with pytest.raises(ConfigError, match=">= 1"):
            RopeScaling(kind="linear", factor=0.5)

    def test_llama3_bands_must_be_ordered(self) -> None:
        with pytest.raises(ConfigError, match="greater than low_freq_factor"):
            RopeScaling(
                kind="llama3",
                factor=8.0,
                original_max_position_embeddings=8192,
                low_freq_factor=4.0,
                high_freq_factor=1.0,
            )

    def test_an_unknown_kind_is_refused(self) -> None:
        with pytest.raises(ConfigError) as caught:
            RopeScaling(kind="dynamic", factor=2.0)  # type: ignore[arg-type]  # deliberately wrong

        assert caught.value.code == "INVALID_CHOICE"


@pytest.mark.unit
class TestFromHfConfig:
    def test_gpt2_spelling(self) -> None:
        config = ModelConfig.from_hf_config(GPT2_JSON)

        assert config == ModelConfig(
            architecture="GPT2LMHeadModel",
            vocab_size=50257,
            hidden_size=768,
            num_layers=12,
            num_attention_heads=12,
            num_kv_heads=12,
            head_dim=64,
            intermediate_size=3072,
            max_position_embeddings=1024,
            hidden_act="gelu_new",
            norm_eps=1e-5,
            tie_word_embeddings=True,
        )

    def test_explicit_gpt2_inner_size_wins(self) -> None:
        config = ModelConfig.from_hf_config({**GPT2_JSON, "n_inner": 1000})

        assert config.intermediate_size == 1000

    def test_llama_spelling(self) -> None:
        config = ModelConfig.from_hf_config(LLAMA_JSON)

        assert (config.architecture, config.num_kv_heads, config.head_dim) == (
            "LlamaForCausalLM",
            8,
            64,
        )
        assert config.intermediate_size == 8192
        assert config.kv_group_size == 4
        assert config.hidden_act == "silu"

    def test_head_dim_is_derived_when_absent(self) -> None:
        raw = {k: v for k, v in LLAMA_JSON.items() if k != "head_dim"}

        assert ModelConfig.from_hf_config(raw).head_dim == 2048 // 32

    def test_untied_embeddings_are_read(self) -> None:
        config = ModelConfig.from_hf_config({**LLAMA_JSON, "tie_word_embeddings": False})

        assert not config.tie_word_embeddings

    def test_does_not_modify_its_input(self) -> None:
        raw = dict(LLAMA_JSON)

        ModelConfig.from_hf_config(raw)

        assert raw == LLAMA_JSON

    def test_lists_every_missing_key(self) -> None:
        with pytest.raises(ModelLoadError) as caught:
            ModelConfig.from_hf_config({"vocab_size": 10})

        assert caught.value.context["missing"] == (
            "architecture",
            "hidden_size",
            "num_layers",
            "num_attention_heads",
            "intermediate_size",
            "max_position_embeddings",
            "hidden_act",
            "norm_eps",
        )
        assert "intermediate_size" in str(caught.value)

    def test_null_values_count_as_missing(self) -> None:
        with pytest.raises(ModelLoadError, match="hidden_act"):
            ModelConfig.from_hf_config({**LLAMA_JSON, "hidden_act": None})

    def test_empty_architectures_list_is_missing(self) -> None:
        with pytest.raises(ModelLoadError, match="architecture"):
            ModelConfig.from_hf_config({**LLAMA_JSON, "architectures": []})

    def test_head_dim_that_cannot_be_derived_is_an_error(self) -> None:
        raw = {k: v for k, v in LLAMA_JSON.items() if k != "head_dim"}
        raw["num_attention_heads"] = 30

        with pytest.raises(ConfigError) as caught:
            ModelConfig.from_hf_config(raw)

        assert caught.value.code == "HEAD_DIM_UNDEFINED"

    def test_gpt2_has_no_rotary_embedding(self) -> None:
        config = ModelConfig.from_hf_config(GPT2_JSON)

        assert (config.rope_theta, config.rope_scaling) == (None, None)

    def test_llama_rope_theta_and_its_default(self) -> None:
        assert ModelConfig.from_hf_config({**LLAMA_JSON, "rope_theta": 500000}).rope_theta == 5e5
        assert ModelConfig.from_hf_config(LLAMA_JSON).rope_theta == 10000.0

    def test_llama_without_the_tie_key_is_untied(self) -> None:
        raw = {k: v for k, v in LLAMA_JSON.items() if k != "tie_word_embeddings"}

        assert not ModelConfig.from_hf_config(raw).tie_word_embeddings

    def test_rope_scaling_in_the_older_spelling(self) -> None:
        raw = {
            **LLAMA_JSON,
            "rope_theta": 500000.0,
            "rope_scaling": {
                "rope_type": "llama3",
                "factor": 32.0,
                "low_freq_factor": 1.0,
                "high_freq_factor": 4.0,
                "original_max_position_embeddings": 8192,
            },
        }

        scaling = ModelConfig.from_hf_config(raw).rope_scaling

        assert scaling == RopeScaling(
            kind="llama3",
            factor=32.0,
            original_max_position_embeddings=8192,
            low_freq_factor=1.0,
            high_freq_factor=4.0,
        )

    def test_rope_parameters_of_transformers_5(self) -> None:
        raw = {
            **LLAMA_JSON,
            "rope_parameters": {"rope_type": "linear", "rope_theta": 1e6, "factor": 2.0},
        }

        config = ModelConfig.from_hf_config(raw)

        assert config.rope_theta == 1e6
        assert config.rope_scaling == RopeScaling(kind="linear", factor=2.0)

    def test_yarn_takes_the_context_when_the_original_is_not_given(self) -> None:
        raw = {**LLAMA_JSON, "rope_scaling": {"type": "yarn", "factor": 4.0}}

        scaling = ModelConfig.from_hf_config(raw).rope_scaling

        assert scaling is not None
        assert scaling.original_max_position_embeddings == 131072

    def test_default_scaling_is_no_scaling(self) -> None:
        raw = {**LLAMA_JSON, "rope_scaling": {"rope_type": "default"}}

        assert ModelConfig.from_hf_config(raw).rope_scaling is None

    @pytest.mark.parametrize(
        "scaling",
        [
            {"rope_type": "dynamic", "factor": 2.0},
            {"rope_type": "longrope", "factor": 2.0},
            {"rope_type": "yarn", "factor": 4.0, "mscale": 0.7},
        ],
    )
    def test_scalings_that_are_not_implemented_are_refused(
        self, scaling: dict[str, object]
    ) -> None:
        with pytest.raises(ConfigError) as caught:
            ModelConfig.from_hf_config({**LLAMA_JSON, "rope_scaling": scaling})

        assert caught.value.code == "ROPE_SCALING_UNSUPPORTED"

    def test_a_partial_rotary_factor_is_refused(self) -> None:
        with pytest.raises(ConfigError, match="part of each head"):
            ModelConfig.from_hf_config({**LLAMA_JSON, "partial_rotary_factor": 0.5})

    def test_projection_biases_are_read(self) -> None:
        config = ModelConfig.from_hf_config({**LLAMA_JSON, "attention_bias": True})

        assert config.attention_bias
        assert not config.mlp_bias

    def test_invalid_values_reach_the_constructor_checks(self) -> None:
        with pytest.raises(ConfigError) as caught:
            ModelConfig.from_hf_config({**LLAMA_JSON, "num_key_value_heads": 5})

        assert caught.value.code == "KV_HEADS_NOT_DIVISOR"
