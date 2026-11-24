"""DFlash diffusion block-drafting model (e.g. Qwen3-8B & Llama-3.1-8B DFlash checkpoints)."""

from __future__ import annotations

import contextlib
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Final, cast

import torch
from torch import nn
from torch.nn import functional as functional

from ayaka.core.errors import ModelLoadError
from ayaka.layers.ffn.dense import GatedMLP
from ayaka.layers.linear import Linear
from ayaka.layers.norm import RMSNorm
from ayaka.layers.rope import RotaryEmbedding
from ayaka.utils.parsing import parse_json_strict
from ayaka.weights.manifest import WeightManifest
from ayaka.weights.safetensors import SafetensorsFile

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from ayaka.ops.backend import OpsBackend
    from ayaka.utils.dtypes import DType
    from ayaka.weights.tensor import TensorRef

__all__ = [
    "DFlashAttention",
    "DFlashConfig",
    "DFlashDecoderLayer",
    "DFlashDraftModel",
    "describe_dflash_bytes",
    "describe_dflash_kv_bytes_per_token",
    "load_dflash_draft_model",
]

_UNWRITABLE: Final = "The given buffer is not writable"


@dataclass(frozen=True, slots=True)
class DFlashConfig:
    """Configuration parsed from a DFlash draft checkpoint ``config.json``."""

    block_size: int
    mask_token_id: int
    target_layer_ids: tuple[int, ...]
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    rms_norm_eps: float
    rope_theta: float
    max_position_embeddings: int
    vocab_size: int

    @classmethod
    def from_hf_config(cls, raw: Mapping[str, object]) -> DFlashConfig:
        """Build a :class:`DFlashConfig` from a parsed ``config.json`` object."""
        dflash_raw = raw.get("dflash_config")
        if not isinstance(dflash_raw, dict):
            raise ModelLoadError("DFlash config.json missing dflash_config table")
        dflash = cast("Mapping[str, object]", dflash_raw)
        mask_id = dflash.get("mask_token_id", raw.get("mask_token_id"))
        layer_ids_raw = dflash.get("target_layer_ids", raw.get("target_layer_ids"))
        if not isinstance(mask_id, int) or isinstance(mask_id, bool):
            raise ModelLoadError("DFlash config missing integer mask_token_id")
        if not isinstance(layer_ids_raw, list) or not layer_ids_raw:
            raise ModelLoadError("DFlash config missing target_layer_ids")
        layer_ids = tuple(int(cast("int", v)) for v in cast("list[object]", layer_ids_raw))

        def _req_int(key: str, default: int | None = None) -> int:
            val = raw.get(key, default)
            if not isinstance(val, int) or isinstance(val, bool):
                raise ModelLoadError(f"DFlash config missing integer {key}")
            return val

        hidden = _req_int("hidden_size")
        heads = _req_int("num_attention_heads")
        eps_raw = raw.get("rms_norm_eps", 1e-6)
        theta_raw = raw.get("rope_theta", 1000000.0)
        return cls(
            block_size=_req_int("block_size"),
            mask_token_id=mask_id,
            target_layer_ids=layer_ids,
            hidden_size=hidden,
            intermediate_size=_req_int("intermediate_size"),
            num_hidden_layers=_req_int("num_hidden_layers"),
            num_attention_heads=heads,
            num_key_value_heads=_req_int("num_key_value_heads", heads),
            head_dim=_req_int("head_dim", hidden // heads),
            rms_norm_eps=float(cast("float", eps_raw)),
            rope_theta=float(cast("float", theta_raw)),
            max_position_embeddings=_req_int("max_position_embeddings", 32768),
            vocab_size=_req_int("vocab_size"),
        )

    @classmethod
    def from_directory(cls, directory: str | Path) -> DFlashConfig:
        """Read ``config.json`` inside ``directory``."""
        path = Path(directory) / "config.json"
        try:
            text = path.read_text(encoding="utf-8")
        except OSError as exc:
            raise ModelLoadError(f"cannot read DFlash config {path}: {exc}") from exc
        raw = parse_json_strict(text)
        if not isinstance(raw, dict):
            raise ModelLoadError(f"{path}: expected JSON object")
        return cls.from_hf_config(cast("Mapping[str, object]", raw))


def describe_dflash_bytes(config: DFlashConfig, *, dtype: DType) -> int:
    """Exact device parameter bytes required by a :class:`DFlashDraftModel`."""
    hidden = config.hidden_size
    inner = config.intermediate_size
    q_rows = config.num_attention_heads * config.head_dim
    kv_rows = config.num_key_value_heads * config.head_dim
    per_layer = (
        2 * hidden
        + q_rows * hidden
        + kv_rows * hidden
        + kv_rows * hidden
        + hidden * q_rows
        + 2 * config.head_dim
        + 3 * inner * hidden
    )
    elements = (
        hidden
        + len(config.target_layer_ids) * hidden * hidden
        + hidden
        + (config.num_hidden_layers * per_layer)
    )
    return dtype.nbytes(elements)


def describe_dflash_kv_bytes_per_token(config: DFlashConfig, *, dtype: DType) -> int:
    """Device KV cache bytes per committed context token across DFlash layers."""
    return dtype.nbytes(2 * config.num_hidden_layers * config.num_key_value_heads * config.head_dim)


class DFlashAttention(nn.Module):
    """Cross-context + non-causal block self-attention with QK per-head RMSNorm."""

    def __init__(
        self,
        config: DFlashConfig,
        *,
        rotary: RotaryEmbedding,
        ops: OpsBackend,
        dtype: torch.dtype,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__()
        hidden = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        q_rows = self.num_heads * self.head_dim
        kv_rows = self.num_kv_heads * self.head_dim
        self.q_proj = Linear(hidden, q_rows, bias=False, ops=ops, dtype=dtype, device=device)
        self.k_proj = Linear(hidden, kv_rows, bias=False, ops=ops, dtype=dtype, device=device)
        self.v_proj = Linear(hidden, kv_rows, bias=False, ops=ops, dtype=dtype, device=device)
        self.o_proj = Linear(q_rows, hidden, bias=False, ops=ops, dtype=dtype, device=device)
        self.q_norm = RMSNorm(
            self.head_dim, eps=config.rms_norm_eps, ops=ops, dtype=dtype, device=device
        )
        self.k_norm = RMSNorm(
            self.head_dim, eps=config.rms_norm_eps, ops=ops, dtype=dtype, device=device
        )
        self.rotary = rotary

    def project_context_kv(
        self, proj_ctx_hidden: torch.Tensor, positions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Project, normalize, and rotate ``[N_ctx, hidden]`` context K/V tensors."""
        n = int(proj_ctx_hidden.shape[0])
        k = self.k_proj(proj_ctx_hidden).view(n, self.num_kv_heads, self.head_dim)
        k = self.k_norm(k.reshape(-1, self.head_dim)).view(n, self.num_kv_heads, self.head_dim)
        dummy_q = k.new_empty((n, 1, self.head_dim))
        self.rotary(positions, dummy_q, k)
        v = self.v_proj(proj_ctx_hidden).view(n, self.num_kv_heads, self.head_dim)
        return k.transpose(0, 1).contiguous(), v.transpose(0, 1).contiguous()

    def forward(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        ctx_k: torch.Tensor,
        ctx_v: torch.Tensor,
    ) -> torch.Tensor:
        """Attend block queries ``[Q, hidden]`` over cached ``ctx_k, ctx_v`` plus block KV."""
        q_len = int(hidden_states.shape[0])
        q = self.q_proj(hidden_states).view(q_len, self.num_heads, self.head_dim)
        q = self.q_norm(q.reshape(-1, self.head_dim)).view(q_len, self.num_heads, self.head_dim)
        k_blk = self.k_proj(hidden_states).view(q_len, self.num_kv_heads, self.head_dim)
        k_blk = self.k_norm(k_blk.reshape(-1, self.head_dim)).view(
            q_len, self.num_kv_heads, self.head_dim
        )
        self.rotary(positions, q, k_blk)
        v_blk = self.v_proj(hidden_states).view(q_len, self.num_kv_heads, self.head_dim)

        q_b = q.transpose(0, 1).unsqueeze(0)
        k_all = torch.cat([ctx_k, k_blk.transpose(0, 1)], dim=1).unsqueeze(0)
        v_all = torch.cat([ctx_v, v_blk.transpose(0, 1)], dim=1).unsqueeze(0)
        attended = functional.scaled_dot_product_attention(
            q_b, k_all, v_all, is_causal=False, enable_gqa=True
        )
        return self.o_proj(attended.squeeze(0).transpose(0, 1).reshape(q_len, -1))


class DFlashDecoderLayer(nn.Module):
    """One DFlash block decoder layer (norm -> DFlashAttention -> norm -> GatedMLP)."""

    def __init__(
        self,
        config: DFlashConfig,
        *,
        rotary: RotaryEmbedding,
        ops: OpsBackend,
        dtype: torch.dtype,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__()
        hidden = config.hidden_size
        self.input_layernorm = RMSNorm(
            hidden, eps=config.rms_norm_eps, ops=ops, dtype=dtype, device=device
        )
        self.self_attn = DFlashAttention(config, rotary=rotary, ops=ops, dtype=dtype, device=device)
        self.post_attention_layernorm = RMSNorm(
            hidden, eps=config.rms_norm_eps, ops=ops, dtype=dtype, device=device
        )
        self.mlp = GatedMLP(
            hidden,
            config.intermediate_size,
            activation="silu",
            bias=False,
            ops=ops,
            dtype=dtype,
            device=device,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        ctx_k: torch.Tensor,
        ctx_v: torch.Tensor,
    ) -> torch.Tensor:
        h = hidden_states + self.self_attn(
            self.input_layernorm(hidden_states), positions, ctx_k, ctx_v
        )
        return h + self.mlp(self.post_attention_layernorm(h))


class DFlashDraftModel(nn.Module):
    """Five-layer DFlash diffusion drafter sharing target ``embed_tokens`` and ``lm_head``."""

    def __init__(
        self,
        config: DFlashConfig,
        *,
        ops: OpsBackend,
        dtype: DType,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        torch_dtype = dtype.torch_dtype
        hidden = config.hidden_size
        self.rotary = RotaryEmbedding(
            config.head_dim,
            max_positions=config.max_position_embeddings,
            theta=config.rope_theta,
            ops=ops,
            device=device,
        )
        self.fc = Linear(
            len(config.target_layer_ids) * hidden,
            hidden,
            bias=False,
            ops=ops,
            dtype=torch_dtype,
            device=device,
        )
        self.hidden_norm = RMSNorm(
            hidden, eps=config.rms_norm_eps, ops=ops, dtype=torch_dtype, device=device
        )
        self.layers = nn.ModuleList(
            [
                DFlashDecoderLayer(
                    config, rotary=self.rotary, ops=ops, dtype=torch_dtype, device=device
                )
                for _ in range(config.num_hidden_layers)
            ]
        )
        self.norm = RMSNorm(
            hidden, eps=config.rms_norm_eps, ops=ops, dtype=torch_dtype, device=device
        )

    def project_context(self, target_features: torch.Tensor) -> torch.Tensor:
        """Project concatenated target layer features ``[N_ctx, K*H]`` to ``[N_ctx, H]``."""
        return self.hidden_norm(self.fc(target_features))

    def compute_context_kv(
        self, target_features: torch.Tensor, positions: torch.Tensor
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """Project new target context features into per-layer cached ``(K, V)`` tensors."""
        proj = self.project_context(target_features)
        ks: list[torch.Tensor] = []
        vs: list[torch.Tensor] = []
        for layer in self.layers:
            dflash_layer = cast("DFlashDecoderLayer", layer)
            k, v = dflash_layer.self_attn.project_context_kv(proj, positions)
            ks.append(k)
            vs.append(v)
        return ks, vs

    def forward_block(
        self,
        noise_embeds: torch.Tensor,
        positions: torch.Tensor,
        ctx_ks: Sequence[torch.Tensor],
        ctx_vs: Sequence[torch.Tensor],
    ) -> torch.Tensor:
        """Run all DFlash layers on ``[B_blk, H]`` block embeddings over cached context KV."""
        h = noise_embeds
        for idx, layer in enumerate(self.layers):
            dflash_layer = cast("DFlashDecoderLayer", layer)
            h = dflash_layer(h, positions, ctx_ks[idx], ctx_vs[idx])
        return self.norm(h)


def _copy_dflash_tensor(file: SafetensorsFile, dst: torch.Tensor, ref: TensorRef) -> None:
    stored = cast("torch.dtype", ref.dtype.torch_dtype)
    view = file.read(ref.name)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=_UNWRITABLE, category=UserWarning)
        tensor = torch.frombuffer(view, dtype=stored).reshape(ref.shape)
    dst.copy_(tensor.to(dtype=dst.dtype))


def load_dflash_draft_model(
    directory: str | Path,
    *,
    ops: OpsBackend,
    dtype: DType,
    device: torch.device | str | None = None,
) -> DFlashDraftModel:
    """Build and load a :class:`DFlashDraftModel` from a safetensors directory."""
    from itertools import groupby

    root = Path(directory)
    config = DFlashConfig.from_directory(root)
    model = DFlashDraftModel(config, ops=ops, dtype=dtype, device=device)
    inner = config.intermediate_size
    targets: dict[str, torch.Tensor] = {
        "fc.weight": model.fc.weight,
        "hidden_norm.weight": model.hidden_norm.weight,
        "norm.weight": model.norm.weight,
    }
    for idx, layer in enumerate(model.layers):
        dl = cast("DFlashDecoderLayer", layer)
        p = f"layers.{idx}."
        targets[f"{p}input_layernorm.weight"] = dl.input_layernorm.weight
        targets[f"{p}post_attention_layernorm.weight"] = dl.post_attention_layernorm.weight
        targets[f"{p}self_attn.q_proj.weight"] = dl.self_attn.q_proj.weight
        targets[f"{p}self_attn.k_proj.weight"] = dl.self_attn.k_proj.weight
        targets[f"{p}self_attn.v_proj.weight"] = dl.self_attn.v_proj.weight
        targets[f"{p}self_attn.o_proj.weight"] = dl.self_attn.o_proj.weight
        targets[f"{p}self_attn.q_norm.weight"] = dl.self_attn.q_norm.weight
        targets[f"{p}self_attn.k_norm.weight"] = dl.self_attn.k_norm.weight
        targets[f"{p}mlp.gate_proj.weight"] = dl.mlp.gate_up_proj.weight[:inner]
        targets[f"{p}mlp.up_proj.weight"] = dl.mlp.gate_up_proj.weight[inner:]
        targets[f"{p}mlp.down_proj.weight"] = dl.mlp.down_proj.weight

    manifest = WeightManifest.from_directory(root)
    missing = [name for name in targets if name not in manifest]
    if missing:
        raise ModelLoadError(
            f"{root}: missing DFlash tensor(s): {', '.join(missing[:6])}",
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
            f"{root}: unexpected DFlash tensor shape(s): {wrong_shape}",
            context={"wrong_shape": wrong_shape},
        )
    refs.sort(key=lambda item: (str(item[2].path), item[2].offset))
    with torch.no_grad():
        for path, group in groupby(refs, key=lambda item: item[2].path):
            file = SafetensorsFile(path)
            try:
                for _name, dst, ref in group:
                    _copy_dflash_tensor(file, dst, ref)
            except BaseException:
                with contextlib.suppress(BufferError):
                    file.close()
                raise
            file.close()
    model.eval()
    return model
