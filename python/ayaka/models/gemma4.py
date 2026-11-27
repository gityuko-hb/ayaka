"""Gemma 4 sliding+full hybrid attention with shared KV layers and per-layer embeddings."""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar, Final

import torch
from torch import nn
from torch.nn import functional

from ayaka.configs.model import RopeScaling
from ayaka.core.errors import ModelLoadError
from ayaka.core.types import ForwardMode, KVLayoutKind
from ayaka.kvcache.storage.spec import KVGroup, KVSpec
from ayaka.layers.embedding.token import LMHead, TokenEmbedding
from ayaka.layers.ffn.dense import GatedMLP
from ayaka.layers.linear import Linear
from ayaka.layers.norm import RMSNorm, ScalelessRMSNorm
from ayaka.layers.rope import RotaryEmbedding
from ayaka.models._vision_common import VisionLifecycle
from ayaka.models._vision_gemma4 import Gemma4VisionTower
from ayaka.models.base import CausalLM
from ayaka.ops.kvcache import FP8_E4M3_CODEC, RAW_CODEC
from ayaka.utils.math_utils import div_ceil
from ayaka.weights.mapping import WeightMapping

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from ayaka.configs.model import ModelConfig
    from ayaka.core.multimodal import ImageInput
    from ayaka.layers.context import ForwardContext
    from ayaka.ops.attention import AttentionMetadata
    from ayaka.ops.backend import OpsBackend
    from ayaka.ops.kvcache import KVBuffers
    from ayaka.utils.dtypes import DType
    from ayaka.weights.quant import QuantFormat

__all__ = ["Gemma4ForConditionalGeneration"]

_PREFIX: Final = r"model\.(?:language_model\.)?"
_RENAMES: Final = (
    (rf"{_PREFIX}embed_tokens\.weight", r"embed_tokens.weight"),
    (rf"{_PREFIX}embed_tokens_per_layer\.weight", r"embed_tokens_per_layer.weight"),
    (rf"{_PREFIX}per_layer_model_projection\.weight", r"per_layer_model_projection.weight"),
    (rf"{_PREFIX}per_layer_projection_norm\.weight", r"per_layer_projection_norm.weight"),
    (rf"{_PREFIX}norm\.weight", r"norm.weight"),
    (
        rf"{_PREFIX}layers\.(\d+)\.(input_layernorm|post_attention_layernorm|"
        r"pre_feedforward_layernorm|post_feedforward_layernorm|post_per_layer_input_norm|"
        r"per_layer_input_gate|per_layer_projection)\.weight",
        r"layers.\1.\2.weight",
    ),
    (rf"{_PREFIX}layers\.(\d+)\.layer_scalar", r"layers.\1.layer_scalar"),
    (
        rf"{_PREFIX}layers\.(\d+)\.self_attn\.([qkvo]_proj|[qk]_norm)\.weight",
        r"layers.\1.attn.\2.weight",
    ),
    (
        rf"{_PREFIX}layers\.(\d+)\.mlp\.(gate|up|down)_proj\.weight",
        r"layers.\1.mlp.\2_proj.weight",
    ),
)
_IGNORE_TOWERS: Final = (
    r"model\.vision_tower\..*",
    r"model\.embed_vision\..*",
    r"model\.audio_tower\..*",
    r"model\.embed_audio\..*",
    rf"{_PREFIX}layers\.\d+\.self_attn\.rotary_emb\.inv_freq",
)


class _Gemma4Attention(nn.Module):
    """Attention block supporting sliding/full geometry and read-only KV-shared layers."""

    def __init__(
        self,
        config: ModelConfig,
        *,
        layer_id: int,
        is_sliding: bool,
        head_dim: int,
        num_kv_heads: int,
        kv_source_layer: int | None,
        rotary: RotaryEmbedding,
        ops: OpsBackend,
        dtype: torch.dtype,
        device: torch.device | str | None,
    ) -> None:
        super().__init__()
        self.layer_id = layer_id
        self.is_sliding = is_sliding
        self.window = config.sliding_window if is_sliding else None
        self.head_dim = head_dim
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = num_kv_heads
        self.kv_source_layer = kv_source_layer
        self.is_shared = kv_source_layer is not None
        self.use_bidirectional_vision = config.use_bidirectional_vision
        self.rotary = rotary

        self.q_proj = Linear(
            config.hidden_size,
            config.num_attention_heads * head_dim,
            bias=config.attention_bias,
            ops=ops,
            dtype=dtype,
            device=device,
        )
        self.q_norm = RMSNorm(head_dim, eps=config.norm_eps, ops=ops, dtype=dtype, device=device)
        if not self.is_shared:
            self.k_proj: Linear | None = Linear(
                config.hidden_size,
                num_kv_heads * head_dim,
                bias=config.attention_bias,
                ops=ops,
                dtype=dtype,
                device=device,
            )
            self.v_proj: Linear | None = Linear(
                config.hidden_size,
                num_kv_heads * head_dim,
                bias=config.attention_bias,
                ops=ops,
                dtype=dtype,
                device=device,
            )
            self.k_norm: RMSNorm | None = RMSNorm(
                head_dim, eps=config.norm_eps, ops=ops, dtype=dtype, device=device
            )
            self.v_norm: ScalelessRMSNorm | None = ScalelessRMSNorm(head_dim, eps=config.norm_eps)
        else:
            self.k_proj = None
            self.v_proj = None
            self.k_norm = None
            self.v_norm = None
        self.o_proj = Linear(
            config.num_attention_heads * head_dim,
            config.hidden_size,
            bias=config.attention_bias,
            ops=ops,
            dtype=dtype,
            device=device,
        )
        self._kvcache = ops.kvcache
        self._attention = ops.attention

    def forward(self, hidden: torch.Tensor, context: ForwardContext) -> torch.Tensor:
        rows = hidden.shape[0]
        if rows == 0:
            return hidden
        q = self.q_norm(
            self.q_proj(hidden).reshape(rows, self.num_heads, self.head_dim).contiguous()
        )
        cache_layer = self.kv_source_layer if self.kv_source_layer is not None else self.layer_id
        meta = context.attention_for(cache_layer)
        buffers = context.buffers(cache_layer)

        if self.is_shared:
            dummy_k = torch.zeros((rows, 1, self.head_dim), dtype=q.dtype, device=q.device)
            self.rotary(meta.positions, q, dummy_k)
        else:
            if (
                self.k_proj is None
                or self.v_proj is None
                or self.k_norm is None
                or self.v_norm is None
            ):
                raise ModelLoadError(
                    f"Gemma4 non-shared layer {self.layer_id} missing k/v projections"
                )
            k = self.k_norm(
                self.k_proj(hidden).reshape(rows, self.num_kv_heads, self.head_dim).contiguous()
            )
            v = self.v_norm(
                self.v_proj(hidden).reshape(rows, self.num_kv_heads, self.head_dim).contiguous()
            )
            self.rotary(meta.positions, q, k)
            self._kvcache.store(k, v, buffers, meta)

        if (
            self.use_bidirectional_vision
            and self.is_sliding
            and context.mode is ForwardMode.PREFILL
            and context.bidirectional_spans
            and isinstance(meta.block_table, torch.Tensor)
        ):
            attended = self._attend_sliding_bidirectional(
                q, buffers, meta, meta.block_table, context.bidirectional_spans
            )
        else:
            attended = torch.empty_like(q)
            self._attention.attend(q, buffers, meta, attended, scale=1.0, window=self.window)
        return self.o_proj(attended.reshape(rows, self.num_heads * self.head_dim))

    def _attend_sliding_bidirectional(
        self,
        q: torch.Tensor,
        buffers: KVBuffers,
        meta: AttentionMetadata,
        block_table: torch.Tensor,
        spans: tuple[tuple[int, int], ...],
    ) -> torch.Tensor:
        out = torch.empty_like(q)
        rows_list = [
            int(meta.rows_per_request[i].item()) for i in range(meta.rows_per_request.shape[0])
        ]
        lens_list = [int(meta.kv_lengths[i].item()) for i in range(meta.kv_lengths.shape[0])]
        k_pool = buffers.k if buffers.layout is KVLayoutKind.NHD else buffers.k.transpose(1, 2)
        v_pool = buffers.v if buffers.layout is KVLayoutKind.NHD else buffers.v.transpose(1, 2)
        offset = 0
        for req_idx, (n_q, n_k) in enumerate(zip(rows_list, lens_list, strict=True)):
            if n_q <= 0:
                continue
            num_blocks = div_ceil(n_k, buffers.block_size)
            block_ids = block_table[req_idx, :num_blocks].to(torch.long)
            k_seq = k_pool[block_ids].reshape(-1, self.num_kv_heads, self.head_dim)[:n_k]
            v_seq = v_pool[block_ids].reshape(-1, self.num_kv_heads, self.head_dim)[:n_k]
            if buffers.codec == FP8_E4M3_CODEC:
                k_seq = k_seq.to(q.dtype) * buffers.k_scale
                v_seq = v_seq.to(q.dtype) * buffers.v_scale
            q_req = q[offset : offset + n_q].transpose(0, 1).unsqueeze(0)
            k_req = k_seq.transpose(0, 1).unsqueeze(0)
            v_req = v_seq.transpose(0, 1).unsqueeze(0)
            if self.num_kv_heads != self.num_heads:
                rep = self.num_heads // self.num_kv_heads
                k_req = k_req.repeat_interleave(rep, dim=1)
                v_req = v_req.repeat_interleave(rep, dim=1)
            q_pos = torch.arange(n_k - n_q, n_k, device=q.device)[:, None]
            k_pos = torch.arange(n_k, device=q.device)[None, :]
            mask = k_pos <= q_pos
            if self.window is not None:
                mask = mask & ((q_pos - k_pos) < self.window)
            start_pos = n_k - n_q
            for s, e in spans:
                ls, le = max(s - offset, 0), min(e - offset, n_q)
                if le > ls:
                    ks, ke = start_pos + ls, start_pos + le
                    mask[ls:le, ks:ke] = True
            if self.window is not None:
                mask = mask & ((q_pos - k_pos).abs() < self.window)
            att = functional.scaled_dot_product_attention(
                q_req, k_req, v_req, attn_mask=mask.unsqueeze(0).unsqueeze(0), scale=1.0
            )
            out[offset : offset + n_q] = att.squeeze(0).transpose(0, 1)
            offset += n_q
        return out


class _Gemma4Block(nn.Module):
    def __init__(
        self,
        config: ModelConfig,
        layer_id: int,
        *,
        is_sliding: bool,
        head_dim: int,
        num_kv_heads: int,
        kv_source_layer: int | None,
        double_wide_mlp: bool,
        rotary: RotaryEmbedding,
        ops: OpsBackend,
        dtype: torch.dtype,
        device: torch.device | str | None,
    ) -> None:
        super().__init__()
        hidden = config.hidden_size
        self.layer_id = layer_id
        self.ple_dim = config.hidden_size_per_layer_input
        self.input_layernorm = RMSNorm(
            hidden, eps=config.norm_eps, ops=ops, dtype=dtype, device=device
        )
        self.attn = _Gemma4Attention(
            config,
            layer_id=layer_id,
            is_sliding=is_sliding,
            head_dim=head_dim,
            num_kv_heads=num_kv_heads,
            kv_source_layer=kv_source_layer,
            rotary=rotary,
            ops=ops,
            dtype=dtype,
            device=device,
        )
        self.post_attention_layernorm = RMSNorm(
            hidden, eps=config.norm_eps, ops=ops, dtype=dtype, device=device
        )
        self.pre_feedforward_layernorm = RMSNorm(
            hidden, eps=config.norm_eps, ops=ops, dtype=dtype, device=device
        )
        mlp_width = 2 * config.intermediate_size if double_wide_mlp else config.intermediate_size
        self.mlp = GatedMLP(
            hidden,
            mlp_width,
            activation="gelu_tanh",
            bias=config.mlp_bias,
            ops=ops,
            dtype=dtype,
            device=device,
        )
        self.post_feedforward_layernorm = RMSNorm(
            hidden, eps=config.norm_eps, ops=ops, dtype=dtype, device=device
        )
        self.layer_scalar = nn.Parameter(
            torch.empty(1, dtype=dtype, device=device), requires_grad=False
        )
        if self.ple_dim > 0:
            self.per_layer_input_gate: Linear | None = Linear(
                hidden, self.ple_dim, bias=False, ops=ops, dtype=dtype, device=device
            )
            self.per_layer_projection: Linear | None = Linear(
                self.ple_dim, hidden, bias=False, ops=ops, dtype=dtype, device=device
            )
            self.post_per_layer_input_norm: RMSNorm | None = RMSNorm(
                hidden, eps=config.norm_eps, ops=ops, dtype=dtype, device=device
            )
        else:
            self.per_layer_input_gate = None
            self.per_layer_projection = None
            self.post_per_layer_input_norm = None

    def forward(
        self,
        hidden: torch.Tensor,
        per_layer_input: torch.Tensor | None,
        context: ForwardContext,
    ) -> torch.Tensor:
        residual = hidden
        h = self.attn(self.input_layernorm(hidden), context)
        hidden = residual + self.post_attention_layernorm(h)

        residual = hidden
        h = self.mlp(self.pre_feedforward_layernorm(hidden))
        hidden = residual + self.post_feedforward_layernorm(h)

        if (
            self.ple_dim > 0
            and per_layer_input is not None
            and self.per_layer_input_gate is not None
            and self.per_layer_projection is not None
            and self.post_per_layer_input_norm is not None
        ):
            residual = hidden
            gate = functional.gelu(self.per_layer_input_gate(hidden), approximate="tanh")
            h = self.per_layer_projection(gate * per_layer_input)
            hidden = residual + self.post_per_layer_input_norm(h)

        return hidden * self.layer_scalar


def _gemma4_layout(
    config: ModelConfig,
) -> tuple[tuple[str, ...], tuple[int, ...], tuple[int, ...], dict[int, int | None]]:
    types = config.layer_types or tuple("sliding_attention" for _ in range(config.num_layers))
    first_shared = max(0, config.num_layers - config.num_kv_shared_layers)
    last_by_type: dict[str, int] = {}
    for idx in range(first_shared):
        last_by_type[types[idx]] = idx
    sliding_owned: list[int] = []
    full_owned: list[int] = []
    sources: dict[int, int | None] = {}
    for idx, kind in enumerate(types):
        if idx < first_shared:
            sources[idx] = None
            if kind == "sliding_attention":
                sliding_owned.append(idx)
            else:
                full_owned.append(idx)
        else:
            sources[idx] = last_by_type[kind]
    return types, tuple(sliding_owned), tuple(full_owned), sources


class Gemma4ForConditionalGeneration(CausalLM):
    """Gemma 4 language model with heterogeneous sliding/full heads, KV sharing, and PLE."""

    checkpoint_renames: ClassVar[tuple[tuple[str, str], ...]] = _RENAMES
    quantizable: ClassVar[bool] = False
    _embed_scale: torch.Tensor
    _proj_scale: torch.Tensor
    _tok_ple_scale: torch.Tensor
    _inp_ple_scale: torch.Tensor

    def __init__(
        self,
        config: ModelConfig,
        *,
        ops: OpsBackend,
        dtype: DType,
        device: torch.device | str | None = None,
        quant: QuantFormat | None = None,
    ) -> None:
        super().__init__(config, ops=ops, dtype=dtype, device=device, quant=quant)
        sliding_theta = config.rope_theta
        if sliding_theta is None:
            raise ModelLoadError(f"{config.architecture} requires rope_theta")
        types, sliding_owned, full_owned, sources = _gemma4_layout(config)
        self._layer_types = types
        self._sliding_owned = sliding_owned
        self._full_owned = full_owned
        self._first_shared = max(0, config.num_layers - config.num_kv_shared_layers)
        self._full_head_dim = config.global_head_dim or config.head_dim
        self._full_kv_heads = config.num_global_kv_heads or config.num_kv_heads
        self._ple_dim = config.hidden_size_per_layer_input

        self.sliding_rotary = RotaryEmbedding(
            config.head_dim,
            max_positions=config.max_position_embeddings,
            theta=sliding_theta,
            scaling=config.rope_scaling,
            ops=ops,
            device=device,
        )
        full_scaling = config.full_rope_scaling
        if full_scaling is None and config.full_partial_rotary_factor < 1.0:
            full_scaling = RopeScaling(
                kind="proportional",
                factor=1.0,
                partial_rotary_factor=config.full_partial_rotary_factor,
            )
        self.full_rotary = RotaryEmbedding(
            self._full_head_dim,
            max_positions=config.max_position_embeddings,
            theta=config.full_rope_theta or sliding_theta,
            scaling=full_scaling,
            ops=ops,
            device=device,
        )

        torch_dtype = dtype.torch_dtype
        self.embed_tokens = TokenEmbedding(
            config.vocab_size, config.hidden_size, ops=ops, dtype=torch_dtype, device=device
        )
        vision_tower = (
            Gemma4VisionTower(config, dtype=torch_dtype, device="meta")
            if config.has_vision
            else None
        )
        self._vision = VisionLifecycle(
            vision_tower,
            config.vision_spec,
            dtype=torch_dtype,
            device_fn=lambda: self.embed_tokens.weight.device,
        )
        if self._ple_dim > 0:
            self.embed_tokens_per_layer: TokenEmbedding | None = TokenEmbedding(
                config.vocab_size,
                config.num_layers * self._ple_dim,
                ops=ops,
                dtype=torch_dtype,
                device=device,
            )
            self.per_layer_model_projection: Linear | None = Linear(
                config.hidden_size,
                config.num_layers * self._ple_dim,
                bias=False,
                ops=ops,
                dtype=torch_dtype,
                device=device,
            )
            self.per_layer_projection_norm: RMSNorm | None = RMSNorm(
                self._ple_dim, eps=config.norm_eps, ops=ops, dtype=torch_dtype, device=device
            )
        else:
            self.embed_tokens_per_layer = None
            self.per_layer_model_projection = None
            self.per_layer_projection_norm = None

        self.layers = nn.ModuleList(
            _Gemma4Block(
                config,
                idx,
                is_sliding=(types[idx] == "sliding_attention"),
                head_dim=config.head_dim
                if types[idx] == "sliding_attention"
                else self._full_head_dim,
                num_kv_heads=(
                    config.num_kv_heads
                    if types[idx] == "sliding_attention"
                    else self._full_kv_heads
                ),
                kv_source_layer=sources[idx],
                double_wide_mlp=config.use_double_wide_mlp and idx >= self._first_shared,
                rotary=self.sliding_rotary
                if types[idx] == "sliding_attention"
                else self.full_rotary,
                ops=ops,
                dtype=torch_dtype,
                device=device,
            )
            for idx in range(config.num_layers)
        )
        self.norm = RMSNorm(
            config.hidden_size, eps=config.norm_eps, ops=ops, dtype=torch_dtype, device=device
        )
        self.lm_head = LMHead(
            config.hidden_size, config.vocab_size, ops=ops, dtype=torch_dtype, device=device
        )
        for name in ("_embed_scale", "_proj_scale", "_tok_ple_scale", "_inp_ple_scale"):
            self.register_buffer(
                name, torch.empty((), dtype=torch_dtype, device=device), persistent=False
            )
        if self.embed_tokens.weight.device.type != "meta":
            self.initialize_buffers()
        self.tie_weights()

    def initialize_buffers(self) -> None:
        super().initialize_buffers()
        self._embed_scale.fill_(self.config.hidden_size**0.5)
        self._proj_scale.fill_(self.config.hidden_size**-0.5)
        self._tok_ple_scale.fill_(max(self._ple_dim, 1) ** 0.5)
        self._inp_ple_scale.fill_(2.0**-0.5)

    @classmethod
    def weight_mapping(cls, config: ModelConfig) -> WeightMapping:
        first_shared = max(0, config.num_layers - config.num_kv_shared_layers)
        shared_ignores = tuple(
            rf"{_PREFIX}layers\.{idx}\.self_attn\.(k_proj|v_proj|k_norm)\.weight"
            for idx in range(first_shared, config.num_layers)
        )
        extra = (r"lm_head\.weight",) if config.tie_word_embeddings else ()
        return WeightMapping(
            renames=cls.checkpoint_renames,
            ignore=(*_IGNORE_TOWERS, *shared_ignores, *extra),
        )

    def tie_weights(self) -> None:
        if self.config.tie_word_embeddings:
            self.lm_head.tie_weights(self.embed_tokens)

    def stored_parts(self, name: str) -> tuple[tuple[str, int], ...]:
        prefix, dot, suffix = name.rpartition(".")
        if not dot or not prefix.endswith(".mlp.gate_up_proj"):
            return ()
        base = prefix.removesuffix("gate_up_proj")
        layer_str = prefix.split(".")[1]
        layer_id = int(layer_str)
        width = self.config.intermediate_size
        if self.config.use_double_wide_mlp and layer_id >= self._first_shared:
            width *= 2
        return ((f"{base}gate_proj.{suffix}", width), (f"{base}up_proj.{suffix}", width))

    def kv_specs(
        self,
        *,
        block_size: int,
        layout: KVLayoutKind,
        codec: str = RAW_CODEC,
        kv_scale: float = 1.0,
    ) -> tuple[KVGroup, ...]:
        groups: list[KVGroup] = []
        if self._sliding_owned:
            groups.append(
                KVGroup(
                    self._sliding_owned,
                    KVSpec(
                        num_layers=len(self._sliding_owned),
                        num_kv_heads=self.config.num_kv_heads,
                        head_dim=self.config.head_dim,
                        block_size=block_size,
                        dtype=self.compute_dtype,
                        layout=layout,
                        codec=codec,
                        kv_scale=kv_scale,
                        window=self.config.sliding_window,
                    ),
                )
            )
        if self._full_owned:
            groups.append(
                KVGroup(
                    self._full_owned,
                    KVSpec(
                        num_layers=len(self._full_owned),
                        num_kv_heads=self._full_kv_heads,
                        head_dim=self._full_head_dim,
                        block_size=block_size,
                        dtype=self.compute_dtype,
                        layout=layout,
                        codec=codec,
                        kv_scale=kv_scale,
                    ),
                )
            )
        return tuple(groups)

    def _per_layer_inputs(
        self, token_ids: torch.Tensor, inputs_embeds: torch.Tensor
    ) -> torch.Tensor | None:
        if (
            self._ple_dim <= 0
            or self.embed_tokens_per_layer is None
            or self.per_layer_model_projection is None
            or self.per_layer_projection_norm is None
        ):
            return None
        rows = inputs_embeds.shape[0]
        num_layers = self.config.num_layers
        proj = (self.per_layer_model_projection(inputs_embeds) * self._proj_scale).reshape(
            rows, num_layers, self._ple_dim
        )
        proj = self.per_layer_projection_norm(proj)
        safe_ids = self._safe_token_ids(token_ids)
        tok_ple = (self.embed_tokens_per_layer(safe_ids) * self._tok_ple_scale).reshape(
            rows, num_layers, self._ple_dim
        )
        return (proj + tok_ple) * self._inp_ple_scale

    def _safe_token_ids(self, token_ids: torch.Tensor) -> torch.Tensor:
        if self.config.image_token_id is None:
            return token_ids
        return torch.where(
            token_ids == self.config.image_token_id, self.config.pad_token_id, token_ids
        )

    def embed_input_ids(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(self._safe_token_ids(token_ids)) * self._embed_scale

    def has_vision_tower(self) -> bool:
        return self._vision.has_tower

    def vision_loaded(self) -> bool:
        return self._vision.is_loaded

    def vision_weight_bytes(self) -> int:
        return self._vision.weight_bytes()

    def vision_activation_bytes(self) -> int:
        return self._vision.activation_bytes()

    def attach_vision_source(self, directory: Path, *, lazy: bool) -> None:
        self._vision.attach(directory, lazy=lazy)

    def ensure_vision_loaded(self) -> None:
        self._vision.ensure_loaded()

    def release_vision(self) -> None:
        self._vision.release()

    def uses_bidirectional_vision(self) -> bool:
        return self.config.use_bidirectional_vision

    def encode_images(self, images: Sequence[ImageInput]) -> tuple[torch.Tensor, ...]:
        return self._vision.encode(images)

    def forward(self, token_ids: torch.Tensor, context: ForwardContext) -> torch.Tensor:
        if context.input_embeds is not None:
            x = context.input_embeds
        else:
            x = self.embed_tokens(token_ids) * self._embed_scale
        ple = self._per_layer_inputs(token_ids, x)
        for idx, layer in enumerate(self.layers):
            layer_ple = ple[:, idx, :] if ple is not None else None
            x = layer(x, layer_ple, context)
        return self.norm(x)

    def compute_logits(self, hidden_rows: torch.Tensor) -> torch.Tensor:
        logits = self.lm_head(hidden_rows)
        cap = self.config.final_logit_softcapping
        if cap is not None:
            logits = torch.tanh(logits / cap) * cap
        return logits
