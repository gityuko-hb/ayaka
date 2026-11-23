"""A tiny Llama in the Hugging Face checkpoint layout, and its formulas in plain torch.

Nothing here uses ``transformers`` or Ayaka's layers. ``llama_state`` builds random tensors under
the names and shapes of a Hugging Face ``LlamaForCausalLM`` state dict (separate ``q_proj``,
``k_proj``, ``v_proj``, ``gate_proj``, ``up_proj``), ``llama_reference_hidden`` computes what Llama
does with them from the published formulas (RMSNorm, rotate-half RoPE, grouped-query causal
attention, SiLU-gated MLP) in float32, and ``write_llama`` writes them as a checkpoint directory.

Import this module only from a test that has already done ``pytest.importorskip("torch")``.
"""

from __future__ import annotations

import json
import math
from typing import TYPE_CHECKING

import torch

from ayaka.configs.model import ModelConfig, RopeScaling
from gpt2_support import tensor_data
from safetensors_files import write_safetensors

if TYPE_CHECKING:
    from pathlib import Path

__all__ = [
    "llama_config",
    "llama_config_dict",
    "llama_inverse_frequencies",
    "llama_reference_hidden",
    "llama_state",
    "write_llama",
]


def llama_config(**overrides: object) -> ModelConfig:
    """A 2-layer Llama with 4 query heads sharing 2 key/value heads of width 8."""
    fields: dict[str, object] = {
        "architecture": "LlamaForCausalLM",
        "vocab_size": 64,
        "hidden_size": 32,
        "num_layers": 2,
        "num_attention_heads": 4,
        "num_kv_heads": 2,
        "head_dim": 8,
        "intermediate_size": 48,
        "max_position_embeddings": 128,
        "hidden_act": "silu",
        "norm_eps": 1e-5,
        "tie_word_embeddings": False,
        "rope_theta": 10000.0,
    }
    fields.update(overrides)
    return ModelConfig(**fields)  # type: ignore[arg-type]  # a dict of the fields, by design


def llama_state(
    config: ModelConfig, *, seed: int = 0, qk_norm: str | None = None
) -> dict[str, torch.Tensor]:
    """Random float32 tensors named and shaped like a Hugging Face ``LlamaForCausalLM``.

    ``qk_norm`` adds the query and key norms of Qwen3 (``"head"``: ``[head_dim]``) or OLMoE
    (``"full"``: ``[heads * head_dim]`` and ``[kv_heads * head_dim]``).
    """
    generator = torch.Generator().manual_seed(seed)
    hidden, inner = config.hidden_size, config.intermediate_size
    q_rows = config.num_attention_heads * config.head_dim
    kv_rows = config.num_kv_heads * config.head_dim

    def rand(*shape: int, scale: float = 0.3) -> torch.Tensor:
        return torch.randn(shape, generator=generator) * scale

    state: dict[str, torch.Tensor] = {
        "model.embed_tokens.weight": rand(config.vocab_size, hidden, scale=1.0),
        "model.norm.weight": 1.0 + rand(hidden, scale=0.1),
    }
    if not config.tie_word_embeddings:
        state["lm_head.weight"] = rand(config.vocab_size, hidden)
    for layer in range(config.num_layers):
        block = f"model.layers.{layer}."
        state |= {
            f"{block}input_layernorm.weight": 1.0 + rand(hidden, scale=0.1),
            f"{block}self_attn.q_proj.weight": rand(q_rows, hidden),
            f"{block}self_attn.k_proj.weight": rand(kv_rows, hidden),
            f"{block}self_attn.v_proj.weight": rand(kv_rows, hidden),
            f"{block}self_attn.o_proj.weight": rand(hidden, q_rows),
            f"{block}post_attention_layernorm.weight": 1.0 + rand(hidden, scale=0.1),
            f"{block}mlp.gate_proj.weight": rand(inner, hidden),
            f"{block}mlp.up_proj.weight": rand(inner, hidden),
            f"{block}mlp.down_proj.weight": rand(hidden, inner),
        }
        if qk_norm is not None:
            q_width, k_width = (
                (config.head_dim, config.head_dim) if qk_norm == "head" else (q_rows, kv_rows)
            )
            state |= {
                f"{block}self_attn.q_norm.weight": 1.0 + rand(q_width, scale=0.1),
                f"{block}self_attn.k_norm.weight": 1.0 + rand(k_width, scale=0.1),
            }
        if config.attention_bias:
            state |= {
                f"{block}self_attn.q_proj.bias": rand(q_rows, scale=0.1),
                f"{block}self_attn.k_proj.bias": rand(kv_rows, scale=0.1),
                f"{block}self_attn.v_proj.bias": rand(kv_rows, scale=0.1),
                f"{block}self_attn.o_proj.bias": rand(hidden, scale=0.1),
            }
    return state


def llama_inverse_frequencies(config: ModelConfig) -> torch.Tensor:
    """Llama's inverse frequencies (default, linear or llama3 scaling), written out again."""
    dim = config.head_dim
    theta = config.rope_theta or 10000.0
    inv = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float64) / dim))
    scaling: RopeScaling | None = config.rope_scaling
    if scaling is None:
        return inv
    if scaling.kind == "linear":
        return inv / scaling.factor
    if scaling.kind == "llama3":
        old = scaling.original_max_position_embeddings or 0
        low, high = old / scaling.low_freq_factor, old / scaling.high_freq_factor
        wavelength = 2 * math.pi / inv
        smooth = (old / wavelength - scaling.low_freq_factor) / (
            scaling.high_freq_factor - scaling.low_freq_factor
        )
        blended = (1 - smooth) * inv / scaling.factor + smooth * inv
        out = torch.where(wavelength > low, inv / scaling.factor, inv)
        middle = (wavelength >= high) & (wavelength <= low)
        return torch.where(middle, blended, out)
    raise NotImplementedError(scaling.kind)


def llama_reference_hidden(
    state: dict[str, torch.Tensor], config: ModelConfig, token_ids: torch.Tensor
) -> torch.Tensor:
    """Llama's final hidden states for one whole sequence ``[T]``, in float32, from the formulas."""
    steps = token_ids.shape[0]
    heads, kv_heads, dim = config.num_attention_heads, config.num_kv_heads, config.head_dim

    def rms(x: torch.Tensor, name: str) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + config.norm_eps) * state[name]

    def proj(x: torch.Tensor, name: str) -> torch.Tensor:
        out = x @ state[f"{name}.weight"].T
        bias = state.get(f"{name}.bias")
        return out if bias is None else out + bias

    inv = llama_inverse_frequencies(config).float()
    angles = torch.arange(steps, dtype=torch.float32)[:, None] * inv[None, :]  # [T, dim/2]
    cos = torch.cat([angles.cos(), angles.cos()], dim=-1)[:, None, :]  # [T, 1, dim]
    sin = torch.cat([angles.sin(), angles.sin()], dim=-1)[:, None, :]

    def rotate(x: torch.Tensor) -> torch.Tensor:  # [T, h, dim]
        first, second = x[..., : dim // 2], x[..., dim // 2 :]
        return x * cos + torch.cat([-second, first], dim=-1) * sin

    mask = torch.triu(torch.ones(steps, steps, dtype=torch.bool), diagonal=1)
    x = state["model.embed_tokens.weight"][token_ids]
    for layer in range(config.num_layers):
        block = f"model.layers.{layer}"
        h = rms(x, f"{block}.input_layernorm.weight")
        q, k = proj(h, f"{block}.self_attn.q_proj"), proj(h, f"{block}.self_attn.k_proj")
        q_norm = f"{block}.self_attn.q_norm.weight"
        if q_norm in state and state[q_norm].shape[0] != dim:  # one norm over the projection
            q, k = rms(q, q_norm), rms(k, f"{block}.self_attn.k_norm.weight")
        q, k = q.reshape(steps, heads, dim), k.reshape(steps, kv_heads, dim)
        if q_norm in state and state[q_norm].shape[0] == dim:  # one norm per head
            q, k = rms(q, q_norm), rms(k, f"{block}.self_attn.k_norm.weight")
        q, k = rotate(q), rotate(k)
        v = proj(h, f"{block}.self_attn.v_proj").reshape(steps, kv_heads, dim)
        k = k.repeat_interleave(heads // kv_heads, dim=1)
        v = v.repeat_interleave(heads // kv_heads, dim=1)
        scores = torch.einsum("thd,shd->hts", q, k) / math.sqrt(dim)
        weights = scores.masked_fill(mask, float("-inf")).softmax(dim=-1)
        attended = torch.einsum("hts,shd->thd", weights, v).reshape(steps, heads * dim)
        x = x + proj(attended, f"{block}.self_attn.o_proj")
        h = rms(x, f"{block}.post_attention_layernorm.weight")
        gated = torch.nn.functional.silu(proj(h, f"{block}.mlp.gate_proj")) * proj(
            h, f"{block}.mlp.up_proj"
        )
        x = x + proj(gated, f"{block}.mlp.down_proj")
    return rms(x, "model.norm.weight")


def llama_config_dict(config: ModelConfig) -> dict[str, object]:
    """The ``config.json`` of a Hugging Face Llama with the shape of ``config``."""
    raw: dict[str, object] = {
        "architectures": [config.architecture],
        "model_type": "llama",
        "vocab_size": config.vocab_size,
        "hidden_size": config.hidden_size,
        "num_hidden_layers": config.num_layers,
        "num_attention_heads": config.num_attention_heads,
        "num_key_value_heads": config.num_kv_heads,
        "head_dim": config.head_dim,
        "intermediate_size": config.intermediate_size,
        "max_position_embeddings": config.max_position_embeddings,
        "hidden_act": config.hidden_act,
        "rms_norm_eps": config.norm_eps,
        "tie_word_embeddings": config.tie_word_embeddings,
        "rope_theta": config.rope_theta,
        "attention_bias": config.attention_bias,
        "mlp_bias": config.mlp_bias,
        "bos_token_id": 1,
        "eos_token_id": 2,
    }
    scaling = config.rope_scaling
    if scaling is not None:
        raw["rope_scaling"] = {
            "rope_type": scaling.kind,
            "factor": scaling.factor,
            "original_max_position_embeddings": scaling.original_max_position_embeddings,
            "low_freq_factor": scaling.low_freq_factor,
            "high_freq_factor": scaling.high_freq_factor,
        }
    return raw


def write_llama(
    directory: Path,
    state: dict[str, torch.Tensor],
    config: ModelConfig,
    *,
    dtype: torch.dtype = torch.float32,
) -> Path:
    """Write ``state`` and ``config.json`` as a one-file safetensors checkpoint in ``directory``."""
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "config.json").write_text(json.dumps(llama_config_dict(config)), encoding="utf-8")
    write_safetensors(
        directory / "model.safetensors", {name: tensor_data(t, dtype) for name, t in state.items()}
    )
    return directory
