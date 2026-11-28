"""MiMo-7B: the Qwen2 model, with a multi-token-prediction layer the engine does not use yet.

MiMo-7B checkpoints (``MiMoForCausalLM``, such as ``XiaomiMiMo/MiMo-7B-RL``) are Qwen2 checkpoints
with one addition. The authors' ``modeling_mimo.py`` defines ``MiMoModel(Qwen2Model)`` and
``MiMoForCausalLM(Qwen2ForCausalLM)`` and overrides no forward method: the main path is exactly
:mod:`ayaka.models.qwen2` (biases on the query, key and value projections, none on the output
projection, an untied head, no query/key norm). What they add is ``mtp_layers``, a list of
``num_nextn_predict_layers`` (1) layers that guess the token after the next one from the main
model's final hidden state, for speculative decoding.

Those tensors (``model.mtp_layers.N.*``, 16 for one layer) are **set aside**: the weight mapping
ignores them, so the model is built, sized in ``ayaka plan`` and run as the plain Qwen2 model it is.
They are not read into memory. Using them is the multi-token-prediction drafter of the speculative
decoding work. Every other tensor that no parameter claims is still an error.
"""

from __future__ import annotations

import contextlib
import dataclasses
import warnings
from itertools import groupby
from pathlib import Path
from typing import TYPE_CHECKING, Final, cast

import torch
from torch import nn

from ayaka.core.errors import ModelLoadError
from ayaka.layers.attention.dense import Attention
from ayaka.layers.ffn.dense import GatedMLP
from ayaka.layers.linear import Linear
from ayaka.layers.norm import RMSNorm
from ayaka.models.qwen2 import Qwen2ForCausalLM
from ayaka.weights.manifest import WeightManifest
from ayaka.weights.safetensors import SafetensorsFile

if TYPE_CHECKING:
    from ayaka.configs.model import ModelConfig
    from ayaka.layers.context import ForwardContext
    from ayaka.layers.rope import RotaryEmbedding
    from ayaka.ops.backend import OpsBackend
    from ayaka.utils.dtypes import DType
    from ayaka.weights.mapping import WeightMapping
    from ayaka.weights.tensor import TensorRef

__all__ = [
    "MiMoForCausalLM",
    "MiMoMTPLayer",
    "describe_mimo_mtp_bytes",
    "load_mimo_mtp_layer",
]

#: The multi-token-prediction layers of the checkpoint, which no parameter of this model claims.
_MTP_TENSORS: Final = (r"model\.mtp_layers\..*",)
_UNWRITABLE: Final = "The given buffer is not writable"


class MiMoForCausalLM(Qwen2ForCausalLM):
    """MiMo-7B with a language-model head; the parameters are those of :class:`Qwen2ForCausalLM`."""

    @classmethod
    def weight_mapping(cls, config: ModelConfig) -> WeightMapping:
        """Qwen2's names, with the checkpoint's multi-token-prediction tensors ignored."""
        mapping = super().weight_mapping(config)
        return dataclasses.replace(mapping, ignore=(*mapping.ignore, *_MTP_TENSORS))


class MiMoMTPLayer(nn.Module):
    """One multi-token-prediction decoder block of MiMo-7B (layer 36)."""

    def __init__(
        self,
        config: ModelConfig,
        *,
        rotary: RotaryEmbedding,
        ops: OpsBackend,
        dtype: DType,
        device: torch.device | str | None = None,
        layer_id: int | None = None,
    ) -> None:
        super().__init__()
        hidden = config.hidden_size
        torch_dtype = dtype.torch_dtype
        kv_layer = config.num_layers if layer_id is None else layer_id
        self.token_layernorm = RMSNorm(
            hidden, eps=config.norm_eps, ops=ops, dtype=torch_dtype, device=device
        )
        self.hidden_layernorm = RMSNorm(
            hidden, eps=config.norm_eps, ops=ops, dtype=torch_dtype, device=device
        )
        self.input_proj = Linear(
            2 * hidden, hidden, bias=False, ops=ops, dtype=torch_dtype, device=device
        )
        self.input_layernorm = RMSNorm(
            hidden, eps=config.norm_eps, ops=ops, dtype=torch_dtype, device=device
        )
        self.attn = Attention(
            hidden,
            config.num_attention_heads,
            config.num_kv_heads,
            config.head_dim,
            layer_id=kv_layer,
            bias=True,
            out_bias=False,
            rotary=rotary,
            ops=ops,
            dtype=torch_dtype,
            device=device,
        )
        self.post_attention_layernorm = RMSNorm(
            hidden, eps=config.norm_eps, ops=ops, dtype=torch_dtype, device=device
        )
        self.mlp = GatedMLP(
            hidden,
            config.intermediate_size,
            activation="silu",
            bias=False,
            ops=ops,
            dtype=torch_dtype,
            device=device,
        )
        self.final_layernorm = RMSNorm(
            hidden, eps=config.norm_eps, ops=ops, dtype=torch_dtype, device=device
        )

    def forward(
        self,
        input_embeds: torch.Tensor,
        hidden_states: torch.Tensor,
        context: ForwardContext,
    ) -> torch.Tensor:
        """Predict next-step hidden states from ``(hidden_states[t], embed(token[t+1]))``."""
        tok = self.token_layernorm(input_embeds)
        hid = self.hidden_layernorm(hidden_states)
        x = self.input_proj(torch.cat([hid, tok], dim=-1))
        residual = x
        h = self.input_layernorm(x)
        h = self.attn(h, context)
        self.post_attention_layernorm.fused_add(h, residual)
        h = self.mlp(h) + residual
        return self.final_layernorm(h)


def describe_mimo_mtp_bytes(config: ModelConfig, *, dtype: DType) -> int:
    """Exact device memory in bytes required by one ``MiMoMTPLayer`` weight set."""
    hidden = config.hidden_size
    inner = config.intermediate_size
    q_rows = config.num_attention_heads * config.head_dim
    kv_rows = config.num_kv_heads * config.head_dim
    elements = (
        5 * hidden
        + 2 * hidden * hidden
        + q_rows * hidden
        + q_rows
        + 2 * (kv_rows * hidden + kv_rows)
        + hidden * q_rows
        + 3 * inner * hidden
    )
    return dtype.nbytes(elements)


def _copy_mtp_tensor(file: SafetensorsFile, dst: torch.Tensor, ref: TensorRef) -> None:
    stored = cast("torch.dtype", ref.dtype.torch_dtype)
    view = file.read(ref.name)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=_UNWRITABLE, category=UserWarning)
        tensor = torch.frombuffer(view, dtype=stored).reshape(ref.shape)
    dst.copy_(tensor.to(dtype=dst.dtype))


def load_mimo_mtp_layer(
    directory: str | Path,
    config: ModelConfig,
    *,
    rotary: RotaryEmbedding,
    ops: OpsBackend,
    dtype: DType,
    device: torch.device | str | None = None,
    layer_id: int | None = None,
    mtp_index: int = 0,
) -> MiMoMTPLayer:
    """Build a :class:`MiMoMTPLayer` on ``device`` and stream its 16 checkpoint tensors."""
    if config.num_nextn_predict_layers <= mtp_index:
        raise ModelLoadError(
            f"{config.architecture} checkpoint has {config.num_nextn_predict_layers} MTP "
            f"layer(s); cannot load mtp_index={mtp_index}",
            context={"architecture": config.architecture, "mtp_index": mtp_index},
        )
    layer = MiMoMTPLayer(
        config,
        rotary=rotary,
        ops=ops,
        dtype=dtype,
        device=device,
        layer_id=layer_id,
    )
    prefix = f"model.mtp_layers.{mtp_index}."
    q_rows = config.num_attention_heads * config.head_dim
    kv_rows = config.num_kv_heads * config.head_dim
    inner = config.intermediate_size
    qkv_bias = layer.attn.qkv_proj.bias
    if qkv_bias is None:
        raise ModelLoadError("MiMoMTPLayer attention requires qkv_proj bias")

    targets: dict[str, torch.Tensor] = {
        f"{prefix}token_layernorm.weight": layer.token_layernorm.weight,
        f"{prefix}hidden_layernorm.weight": layer.hidden_layernorm.weight,
        f"{prefix}input_layernorm.weight": layer.input_layernorm.weight,
        f"{prefix}post_attention_layernorm.weight": layer.post_attention_layernorm.weight,
        f"{prefix}final_layernorm.weight": layer.final_layernorm.weight,
        f"{prefix}input_proj.weight": layer.input_proj.weight,
        f"{prefix}self_attn.o_proj.weight": layer.attn.o_proj.weight,
        f"{prefix}mlp.down_proj.weight": layer.mlp.down_proj.weight,
        f"{prefix}self_attn.q_proj.weight": layer.attn.qkv_proj.weight[:q_rows],
        f"{prefix}self_attn.k_proj.weight": layer.attn.qkv_proj.weight[q_rows : q_rows + kv_rows],
        f"{prefix}self_attn.v_proj.weight": layer.attn.qkv_proj.weight[q_rows + kv_rows :],
        f"{prefix}self_attn.q_proj.bias": qkv_bias[:q_rows],
        f"{prefix}self_attn.k_proj.bias": qkv_bias[q_rows : q_rows + kv_rows],
        f"{prefix}self_attn.v_proj.bias": qkv_bias[q_rows + kv_rows :],
        f"{prefix}mlp.gate_proj.weight": layer.mlp.gate_up_proj.weight[:inner],
        f"{prefix}mlp.up_proj.weight": layer.mlp.gate_up_proj.weight[inner:],
    }

    root = Path(directory)
    manifest = WeightManifest.from_directory(root)
    missing = [name for name in targets if name not in manifest]
    if missing:
        raise ModelLoadError(
            f"{root}: missing MTP weight tensor(s): {', '.join(missing[:6])}",
            context={"missing": missing},
        )
    refs = [(name, targets[name], manifest[name]) for name in targets]
    wrong_shape = {
        name: f"{tuple(ref.shape)} != {tuple(dst.shape)}"
        for name, dst, ref in refs
        if tuple(ref.shape) != tuple(dst.shape)
    }
    if wrong_shape:
        raise ModelLoadError(
            f"{root}: unexpected MTP tensor shape(s): {wrong_shape}",
            context={"wrong_shape": wrong_shape},
        )
    refs.sort(key=lambda item: (str(item[2].path), item[2].offset))
    with torch.no_grad():
        for path, group in groupby(refs, key=lambda item: item[2].path):
            file = SafetensorsFile(path)
            try:
                for _name, dst, ref in group:
                    _copy_mtp_tensor(file, dst, ref)
            except BaseException:
                with contextlib.suppress(BufferError):
                    file.close()
                raise
            file.close()
    layer.eval()
    return layer
