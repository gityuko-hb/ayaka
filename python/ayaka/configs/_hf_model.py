"""Hugging Face ``config.json`` parsing helpers for :class:`ayaka.configs.model.ModelConfig`."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any, Final, cast

from ayaka.core.errors import ConfigError, ModelLoadError
from ayaka.utils.validation import require_int

if TYPE_CHECKING:
    from ayaka.configs.model import RopeScaling

__all__ = [
    "SLIDING_WINDOW_ARCHITECTURES",
    "UNTIED_BY_DEFAULT",
    "parse_hf_model_fields",
]

#: ``tie_word_embeddings`` when ``config.json`` does not say. Hugging Face's base config
#: ties by default, but some architectures' own config classes do not.
UNTIED_BY_DEFAULT: Final = frozenset(
    {
        "DeepseekV2ForCausalLM",
        "DeepseekV3ForCausalLM",
        "GptOssForCausalLM",
        "LlamaForCausalLM",
        "MiMoForCausalLM",
        "OlmoeForCausalLM",
        "Phi3ForCausalLM",
        "Qwen2ForCausalLM",
        "Qwen3ForCausalLM",
        "Qwen3MoeForCausalLM",
        "Qwen3_5ForConditionalGeneration",
    }
)

#: Architectures that natively execute sliding-window KV attention across the full context.
SLIDING_WINDOW_ARCHITECTURES: Final = frozenset(
    {
        "Gemma4ForConditionalGeneration",
        "GptOssForCausalLM",
    }
)

_HF_KEYS: Final[dict[str, tuple[str, ...]]] = {
    "vocab_size": ("vocab_size",),
    "hidden_size": ("hidden_size", "n_embd"),
    "num_layers": ("num_hidden_layers", "n_layer"),
    "num_attention_heads": ("num_attention_heads", "n_head"),
    "intermediate_size": ("intermediate_size", "n_inner"),
    "max_position_embeddings": ("max_position_embeddings", "n_positions"),
    "hidden_act": ("hidden_act", "hidden_activation", "activation_function"),
    "norm_eps": ("rms_norm_eps", "layer_norm_epsilon", "layer_norm_eps"),
}

#: Architectures with a rotary embedding and the default ``rope_theta`` when omitted.
_ROPE_THETA_DEFAULT: Final = {
    "DeepseekV2ForCausalLM": 10000.0,
    "DeepseekV3ForCausalLM": 10000.0,
    "Gemma4ForConditionalGeneration": 10000.0,
    "LlamaForCausalLM": 10000.0,
    "MiMoForCausalLM": 10000.0,
    "OlmoeForCausalLM": 10000.0,
    "Phi3ForCausalLM": 10000.0,
    "Qwen2ForCausalLM": 10000.0,
    "Qwen3ForCausalLM": 10000.0,
    "Qwen3MoeForCausalLM": 10000.0,
    "Qwen3_5ForConditionalGeneration": 10000000.0,
}


def _unwrap_text_config(config: Mapping[str, object]) -> Mapping[str, object]:
    text = config.get("text_config")
    if isinstance(text, Mapping):
        merged: dict[str, object] = dict(config)
        merged.update(cast("Mapping[str, object]", text))
        return merged
    return config


def _hf_sliding_window(config: Mapping[str, object], num_layers: int) -> int | None:
    window = config.get("sliding_window")
    if window is None:
        return None
    layer_types = config.get("layer_types")
    if isinstance(layer_types, Sequence) and not isinstance(layer_types, str):
        kinds = cast("Sequence[object]", layer_types)
        return cast("int", window) if "sliding_attention" in kinds else None
    if config.get("use_sliding_window") is False:
        return None
    max_window_layers = config.get("max_window_layers")
    if type(max_window_layers) is int and max_window_layers >= num_layers:
        return None
    return cast("int", window)


def _hf_experts(
    config: Mapping[str, object], num_layers: int, intermediate_size: object
) -> dict[str, Any]:
    experts = config.get(
        "num_experts", config.get("num_local_experts", config.get("n_routed_experts"))
    )
    if not experts:
        return {}
    step = config.get("decoder_sparse_step", 1)
    only = config.get("mlp_only_layers") or ()
    listed = (
        {cast("int", layer) for layer in cast("Sequence[object]", only)}
        if isinstance(only, Sequence) and not isinstance(only, str)
        else set[int]()
    )
    first_dense = config.get("first_k_dense_replace", 0)
    if type(first_dense) is int and first_dense > 0:
        listed.update(range(min(first_dense, num_layers)))
    step_value = step if type(step) is int and step >= 1 else 1
    dense = tuple(
        layer for layer in range(num_layers) if layer in listed or (layer + 1) % step_value
    )
    return {
        "num_experts": experts,
        "experts_per_token": config.get("num_experts_per_tok", config.get("experts_per_token", 0)),
        "moe_intermediate_size": config.get("moe_intermediate_size") or intermediate_size,
        "norm_topk_prob": config.get("norm_topk_prob", False),
        "dense_layers": dense,
        "num_shared_experts": config.get("n_shared_experts", 0) or 0,
    }


def _hf_rope_branch(
    raw: Mapping[str, object],
    config: Mapping[str, object],
    architecture: str,
    max_positions: int,
) -> tuple[float | None, RopeScaling | None, float]:
    from ayaka.configs.model import RopeScaling

    theta = raw.get("rope_theta", config.get("rope_theta"))
    if theta is None:
        theta = _ROPE_THETA_DEFAULT.get(architecture)
    if theta is None:
        return None, None, 1.0
    partial = config.get("partial_rotary_factor", raw.get("partial_rotary_factor", 1.0))
    scaling = RopeScaling.from_hf(raw, max_position_embeddings=max_positions)
    partial_val = 1.0 if partial is None else float(cast("float", partial))
    if not (0.0 < partial_val <= 1.0):
        raise ConfigError(
            "model.partial_rotary_factor",
            "INVALID_RANGE",
            f"partial_rotary_factor must be in (0, 1], got {partial_val}",
        )
    return float(cast("float", theta)), scaling, partial_val


def _hf_rope(config: Mapping[str, object], architecture: str, max_positions: int) -> dict[str, Any]:
    parameters = config.get("rope_parameters")
    raw: Mapping[str, object]
    if isinstance(parameters, Mapping):
        params_map = cast("Mapping[str, object]", parameters)
        sliding_cfg = params_map.get("sliding_attention")
        full_cfg = params_map.get("full_attention")
        if isinstance(sliding_cfg, Mapping) or isinstance(full_cfg, Mapping):
            base_map = cast(
                "Mapping[str, object]",
                sliding_cfg if isinstance(sliding_cfg, Mapping) else full_cfg,
            )
            theta, scaling, partial = _hf_rope_branch(base_map, config, architecture, max_positions)
            full_map = cast(
                "Mapping[str, object]",
                full_cfg if isinstance(full_cfg, Mapping) else {},
            )
            full_theta, full_scaling, full_partial = _hf_rope_branch(
                full_map, config, architecture, max_positions
            )
            return {
                "rope_theta": theta,
                "rope_scaling": scaling,
                "partial_rotary_factor": partial,
                "full_rope_theta": full_theta,
                "full_rope_scaling": full_scaling,
                "full_partial_rotary_factor": full_partial,
                "mrope_interleaved": False,
                "mrope_section": (),
            }
        raw = params_map
    else:
        scaling_raw = config.get("rope_scaling")
        raw = cast("Mapping[str, object]", scaling_raw) if isinstance(scaling_raw, Mapping) else {}
    theta, parsed_scaling, partial = _hf_rope_branch(raw, config, architecture, max_positions)
    section_raw = raw.get("mrope_section")
    mrope_section = (
        tuple(int(x) for x in cast("Sequence[int]", section_raw))
        if isinstance(section_raw, Sequence) and not isinstance(section_raw, str)
        else ()
    )
    return {
        "rope_theta": theta,
        "rope_scaling": parsed_scaling,
        "partial_rotary_factor": partial,
        "full_rope_theta": None,
        "full_rope_scaling": None,
        "full_partial_rotary_factor": 1.0,
        "mrope_interleaved": bool(raw.get("mrope_interleaved", False)),
        "mrope_section": mrope_section,
    }


def _layer_types(config: Mapping[str, object]) -> tuple[str, ...]:
    raw = config.get("layer_types")
    if isinstance(raw, Sequence) and not isinstance(raw, str):
        return tuple(str(x) for x in cast("Sequence[object]", raw))
    return ()


def parse_hf_model_fields(raw_config: Mapping[str, object]) -> dict[str, Any]:
    """Parse validated kwargs for :class:`ModelConfig` from a Hugging Face ``config.json``."""
    config = _unwrap_text_config(raw_config)
    found: dict[str, object] = {}
    for name, keys in _HF_KEYS.items():
        for key in keys:
            if config.get(key) is not None:
                found[name] = config[key]
                break
    architectures = config.get("architectures")
    if isinstance(architectures, Sequence) and not isinstance(architectures, str):
        first = next(iter(cast("Sequence[object]", architectures)), None)
        if first is not None:
            found["architecture"] = first
    if "intermediate_size" not in found and "n_embd" in config:
        hidden = found.get("hidden_size")
        if type(hidden) is int:
            found["intermediate_size"] = 4 * hidden
    missing = [name for name in ("architecture", *_HF_KEYS) if name not in found]
    if missing:
        raise ModelLoadError(
            f"config.json lacks required keys for {', '.join(missing)}",
            context={"missing": tuple(missing)},
        )
    heads = require_int(found["num_attention_heads"], "model.num_attention_heads", minimum=1)
    head_dim = config.get("head_dim")
    if head_dim is None:
        hidden_size = require_int(found["hidden_size"], "model.hidden_size", minimum=1)
        if hidden_size % heads:
            raise ConfigError(
                "model.head_dim",
                "HEAD_DIM_UNDEFINED",
                f"config.json has no head_dim and hidden_size {hidden_size} is not a "
                f"multiple of {heads} attention heads",
            )
        head_dim = hidden_size // heads
    if config.get("clip_qkv") is not None:
        raise ConfigError(
            "model.clip_qkv",
            "CLIP_QKV_UNSUPPORTED",
            f"clipping the q, k and v projections (clip_qkv {config['clip_qkv']}) is "
            "not implemented",
        )
    kv_heads = config.get("num_key_value_heads")
    architecture = cast("str", found["architecture"])
    max_positions = cast("int", found["max_position_embeddings"])
    num_layers = cast("int", found["num_layers"])
    rope_fields = _hf_rope(config, architecture, max_positions)
    vision_raw = raw_config.get("vision_config")
    vision_cfg = (
        dict(cast("Mapping[str, Any]", vision_raw)) if isinstance(vision_raw, Mapping) else None
    )
    return {
        "architecture": architecture,
        "vocab_size": cast("int", found["vocab_size"]),
        "hidden_size": cast("int", found["hidden_size"]),
        "num_layers": num_layers,
        "num_attention_heads": heads,
        "num_kv_heads": heads if kv_heads is None else cast("int", kv_heads),
        "head_dim": cast("int", head_dim),
        "intermediate_size": cast("int", found["intermediate_size"]),
        "max_position_embeddings": max_positions,
        "hidden_act": cast("str", found["hidden_act"]),
        "norm_eps": cast("float", found["norm_eps"]),
        "tie_word_embeddings": cast(
            "bool",
            raw_config.get(
                "tie_word_embeddings",
                config.get("tie_word_embeddings", architecture not in UNTIED_BY_DEFAULT),
            ),
        ),
        "attention_bias": cast("bool", config.get("attention_bias", False)),
        "mlp_bias": cast("bool", config.get("mlp_bias", False)),
        "sliding_window": _hf_sliding_window(config, num_layers),
        "layer_types": _layer_types(config),
        "attn_output_gate": bool(config.get("attn_output_gate", False)),
        "linear_num_key_heads": int(cast("int", config.get("linear_num_key_heads") or 0)),
        "linear_num_value_heads": int(cast("int", config.get("linear_num_value_heads") or 0)),
        "linear_key_head_dim": int(cast("int", config.get("linear_key_head_dim") or 0)),
        "linear_value_head_dim": int(cast("int", config.get("linear_value_head_dim") or 0)),
        "linear_conv_kernel_dim": int(cast("int", config.get("linear_conv_kernel_dim") or 0)),
        "global_head_dim": cast("int | None", config.get("global_head_dim")),
        "num_global_kv_heads": cast("int | None", config.get("num_global_key_value_heads")),
        "num_kv_shared_layers": int(cast("int", config.get("num_kv_shared_layers") or 0)),
        "use_double_wide_mlp": bool(config.get("use_double_wide_mlp", False)),
        "hidden_size_per_layer_input": int(
            cast("int", config.get("hidden_size_per_layer_input") or 0)
        ),
        "final_logit_softcapping": cast("float | None", config.get("final_logit_softcapping")),
        "attention_k_eq_v": bool(config.get("attention_k_eq_v", False)),
        "rope_interleave": bool(
            config.get("rope_interleave", architecture == "DeepseekV2ForCausalLM")
        ),
        "swiglu_limit": cast("float | None", config.get("swiglu_limit")),
        "swiglu_alpha": float(cast("float", config.get("swiglu_alpha", 1.702))),
        "kv_lora_rank": int(cast("int", config.get("kv_lora_rank") or 0)),
        "q_lora_rank": cast("int | None", config.get("q_lora_rank")),
        "qk_nope_head_dim": int(cast("int", config.get("qk_nope_head_dim") or 0)),
        "qk_rope_head_dim": int(cast("int", config.get("qk_rope_head_dim") or 0)),
        "v_head_dim": int(cast("int", config.get("v_head_dim") or 0)),
        "vision_config": vision_cfg,
        "image_token_id": cast("int | None", raw_config.get("image_token_id")),
        "boi_token_id": cast("int | None", raw_config.get("boi_token_id")),
        "eoi_token_id": cast("int | None", raw_config.get("eoi_token_id")),
        "vision_start_token_id": cast("int | None", raw_config.get("vision_start_token_id")),
        "vision_end_token_id": cast("int | None", raw_config.get("vision_end_token_id")),
        "vision_soft_tokens_per_image": cast(
            "int | None", raw_config.get("vision_soft_tokens_per_image")
        ),
        "pad_token_id": int(
            cast("int", config.get("pad_token_id", raw_config.get("pad_token_id")))
            if config.get("pad_token_id", raw_config.get("pad_token_id")) is not None
            else 0
        ),
        "use_bidirectional_vision": config.get("use_bidirectional_attention") == "vision",
        **rope_fields,
        **_hf_experts(config, num_layers, found["intermediate_size"]),
    }
