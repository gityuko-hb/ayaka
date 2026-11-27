"""Architecture hyperparameters of one model, as read from its ``config.json``.

``ModelConfig`` is a frozen host value: it describes the shape of the network,
not where the weights live (:mod:`ayaka.configs.model_source`) or how they are
run (:mod:`ayaka.configs.device`). Everything the memory plan and the KV cache
geometry need is here, so ``ayaka plan`` can build one from ``config.json``
without loading a tensor.

::

    config.json --from_hf_config()--> ModelConfig --> models (build the network)
                                                  \\-> memory, kvcache (sizes)

Fields that only some architectures have are optional and added with the
milestone that implements them: rotary embeddings (``rope_theta``,
``rope_scaling``) and the projection biases of the Llama family arrived with the Llama models,
the sliding window (``sliding_window``) with Qwen2; mixture-of-experts and vision come later.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Final, Literal, cast

from ayaka.configs._hf_model import (
    SLIDING_WINDOW_ARCHITECTURES,
    parse_hf_model_fields,
)
from ayaka.configs.base import ConfigMixin
from ayaka.core.errors import ConfigError
from ayaka.utils.validation import require_bool, require_int, require_number, require_text

if TYPE_CHECKING:
    from collections.abc import Mapping

__all__ = ["ModelConfig", "RopeScaling", "RopeScalingKind"]

type RopeScalingKind = Literal["linear", "llama3", "yarn", "proportional"]
"""The rotary scalings Ayaka implements; ``config.json`` calls this ``rope_type``."""

_ROPE_KINDS: Final[tuple[RopeScalingKind, ...]] = ("linear", "llama3", "yarn", "proportional")


@dataclass(frozen=True, slots=True)
class RopeScaling(ConfigMixin):
    """How a model stretches its rotary embedding past the context it was trained on.

    Each kind changes the inverse frequencies of the rotation; the formulas are
    in :mod:`ayaka.layers.rope`. Only the fields of the chosen kind are read.

    Attributes:
        kind: ``"linear"`` (positions divided by ``factor``), ``"llama3"``
            (Llama 3.1's per-band interpolation) or ``"yarn"``.
        factor: The context extension factor; at least 1.
        original_max_position_embeddings: Context the model was trained on before
            scaling. Required by ``llama3`` and ``yarn``.
        low_freq_factor: ``llama3`` only.
        high_freq_factor: ``llama3`` only; greater than ``low_freq_factor``.
        beta_fast: ``yarn`` only: the rotation count above which a band is not interpolated.
        beta_slow: ``yarn`` only: the rotation count below which a band is fully interpolated.
        attention_factor: ``yarn`` only: multiplies cos and sin; ``None`` uses
            ``0.1 * ln(factor) + 1``.

    Raises:
        TypeError: If a field has the wrong type.
        ConfigError: ``INVALID_CHOICE`` for an unknown kind, ``INVALID_RANGE`` for
            a number out of range, ``ROPE_SCALING_INCOMPLETE`` when a field the
            kind needs is missing.
    """

    kind: RopeScalingKind
    factor: float
    original_max_position_embeddings: int | None = None
    low_freq_factor: float = 1.0
    high_freq_factor: float = 4.0
    beta_fast: float = 32.0
    beta_slow: float = 1.0
    attention_factor: float | None = None
    truncate: bool = True
    partial_rotary_factor: float = 1.0

    def __post_init__(self) -> None:
        if self.kind not in _ROPE_KINDS:
            raise ConfigError(
                "model.rope_scaling.kind",
                "INVALID_CHOICE",
                f"{self.kind!r} is not one of {list(_ROPE_KINDS)}",
            )
        for name in (
            "factor",
            "low_freq_factor",
            "high_freq_factor",
            "beta_fast",
            "beta_slow",
            "partial_rotary_factor",
        ):
            value = require_number(getattr(self, name), f"model.rope_scaling.{name}")
            if not math.isfinite(value) or value <= 0:
                raise ConfigError(
                    f"model.rope_scaling.{name}", "INVALID_RANGE", "must be finite and positive"
                )
            object.__setattr__(self, name, float(value))
        require_bool(self.truncate, "model.rope_scaling.truncate")
        if self.factor < 1:
            raise ConfigError("model.rope_scaling.factor", "INVALID_RANGE", "must be >= 1")
        if self.partial_rotary_factor > 1:
            raise ConfigError(
                "model.rope_scaling.partial_rotary_factor", "INVALID_RANGE", "must be <= 1"
            )
        if self.attention_factor is not None:
            value = require_number(self.attention_factor, "model.rope_scaling.attention_factor")
            if not math.isfinite(value) or value <= 0:
                raise ConfigError(
                    "model.rope_scaling.attention_factor",
                    "INVALID_RANGE",
                    "must be finite and positive",
                )
            object.__setattr__(self, "attention_factor", float(value))
        if self.original_max_position_embeddings is not None:
            require_int(
                self.original_max_position_embeddings,
                "model.rope_scaling.original_max_position_embeddings",
                minimum=1,
            )
        elif self.kind not in ("linear", "proportional"):
            raise ConfigError(
                "model.rope_scaling.original_max_position_embeddings",
                "ROPE_SCALING_INCOMPLETE",
                f"{self.kind} scaling needs the context the model was trained on",
            )
        if self.kind == "llama3" and self.high_freq_factor <= self.low_freq_factor:
            raise ConfigError(
                "model.rope_scaling.high_freq_factor",
                "INVALID_RANGE",
                "must be greater than low_freq_factor",
            )

    @classmethod
    def from_hf(
        cls, raw: Mapping[str, object], *, max_position_embeddings: int
    ) -> RopeScaling | None:
        """Read the ``rope_scaling`` (or ``rope_parameters``) object of a ``config.json``."""
        kind = raw.get("rope_type", raw.get("type", "default"))
        if kind == "default":
            return None
        if kind not in _ROPE_KINDS:
            raise ConfigError(
                "model.rope_scaling",
                "ROPE_SCALING_UNSUPPORTED",
                f"rope scaling {kind!r} is not implemented; supported: {['default', *_ROPE_KINDS]}",
            )
        unsupported = [
            name for name in ("mscale", "mscale_all_dim") if raw.get(name) not in (None, 0, 0.0)
        ]
        if unsupported:
            raise ConfigError(
                "model.rope_scaling",
                "ROPE_SCALING_UNSUPPORTED",
                f"{kind} scaling with {unsupported} is not implemented",
            )
        original = raw.get("original_max_position_embeddings")
        if original is None and kind == "yarn":
            original = max_position_embeddings

        def number(name: str, default: float) -> float:
            value = raw.get(name)
            return default if value is None else cast("float", value)

        attention_factor = raw.get("attention_factor")
        truncate = raw.get("truncate")
        return cls(
            kind=kind,
            factor=number("factor", 1.0),
            original_max_position_embeddings=cast("int | None", original),
            low_freq_factor=number("low_freq_factor", 1.0),
            high_freq_factor=number("high_freq_factor", 4.0),
            beta_fast=number("beta_fast", 32.0),
            beta_slow=number("beta_slow", 1.0),
            attention_factor=cast("float | None", attention_factor),
            truncate=True if truncate is None else bool(truncate),
            partial_rotary_factor=number("partial_rotary_factor", 1.0),
        )


@dataclass(frozen=True, slots=True)
class ModelConfig(ConfigMixin):
    """Shape of a decoder-only transformer language model.

    Every field is required: a default would be a guess about somebody else's
    checkpoint. :meth:`from_hf_config` fills the fields a ``config.json`` leaves
    implicit (``num_kv_heads``, ``head_dim``) by the rules HuggingFace uses.

    Attributes:
        architecture: Class name from ``config.json`` ``architectures`` (for
            example ``"GPT2LMHeadModel"``). The model registry resolves it.
        vocab_size: Number of rows of the token embedding and the output head.
        hidden_size: Width of the residual stream.
        num_layers: Number of transformer blocks.
        num_attention_heads: Query heads per layer.
        num_kv_heads: Key/value heads per layer. Equal to ``num_attention_heads``
            for multi-head attention, smaller for grouped-query attention; it
            must divide ``num_attention_heads``.
        head_dim: Width of one attention head. Not derived from ``hidden_size``
            here, because some models set it independently.
        intermediate_size: Width of the feed-forward hidden layer.
        max_position_embeddings: Longest sequence the model was trained for.
        hidden_act: Activation name as the checkpoint spells it (``"gelu_new"``,
            ``"silu"``). The ops backend decides which names it implements.
        norm_eps: Epsilon of the normalization layers; finite and positive.
        tie_word_embeddings: Whether the output head reuses the token embedding.
        rope_theta: Base of the rotary embedding's frequencies, or ``None`` for a
            model without one (GPT-2).
        rope_scaling: How the rotary embedding is stretched, or ``None`` for none.
            Only meaningful with ``rope_theta``.
        attention_bias: Whether the query, key, value and output projections have
            a bias. Read for the Llama family (``attention_bias`` in
            ``config.json``); GPT-2 always has biases and ignores it.
        mlp_bias: Whether the feed-forward projections have a bias; Llama family
            only, like ``attention_bias``.
        sliding_window: Tokens a query attends to in the layers that use a sliding
            window, or ``None`` when every layer attends to the whole context. Ayaka
            does not implement windowed attention, so :attr:`context_length` stops at
            the window: up to it, windowed and full causal attention are the same.
        num_experts: Experts in each mixture-of-experts layer; ``0`` for a dense model.
        experts_per_token: Experts each token is routed to (``num_experts_per_tok``); ``0``
            for a dense model.
        moe_intermediate_size: Width of one expert's gate and up projections; ``0`` for a
            dense model.
        norm_topk_prob: Whether the routing weights of a token's chosen experts are divided by
            their sum.
        dense_layers: Layers of an MoE model that have the dense feed-forward block of width
            ``intermediate_size`` instead of experts.

    Raises:
        TypeError: If a field has the wrong type.
        ValueError: If a number is out of range or a text field is blank.
        ConfigError: ``KV_HEADS_EXCEED_HEADS`` or ``KV_HEADS_NOT_DIVISOR`` when
            ``num_kv_heads`` does not fit ``num_attention_heads``; ``INVALID_RANGE``
            when ``norm_eps`` is not positive.
    """

    architecture: str
    vocab_size: int
    hidden_size: int
    num_layers: int
    num_attention_heads: int
    num_kv_heads: int
    head_dim: int
    intermediate_size: int
    max_position_embeddings: int
    hidden_act: str
    norm_eps: float
    tie_word_embeddings: bool
    rope_theta: float | None = None
    rope_scaling: RopeScaling | None = None
    attention_bias: bool = False
    mlp_bias: bool = False
    sliding_window: int | None = None
    num_experts: int = 0
    experts_per_token: int = 0
    moe_intermediate_size: int = 0
    norm_topk_prob: bool = False
    dense_layers: tuple[int, ...] = ()
    num_shared_experts: int = 0
    partial_rotary_factor: float = 1.0
    rope_interleave: bool = False
    layer_types: tuple[str, ...] = ()
    attn_output_gate: bool = False
    mrope_interleaved: bool = False
    mrope_section: tuple[int, ...] = ()
    linear_num_key_heads: int = 0
    linear_num_value_heads: int = 0
    linear_key_head_dim: int = 0
    linear_value_head_dim: int = 0
    linear_conv_kernel_dim: int = 0
    global_head_dim: int | None = None
    num_global_kv_heads: int | None = None
    num_kv_shared_layers: int = 0
    use_double_wide_mlp: bool = False
    hidden_size_per_layer_input: int = 0
    final_logit_softcapping: float | None = None
    full_rope_theta: float | None = None
    full_rope_scaling: RopeScaling | None = None
    full_partial_rotary_factor: float = 1.0
    attention_k_eq_v: bool = False
    swiglu_limit: float | None = None
    swiglu_alpha: float = 1.702
    kv_lora_rank: int = 0
    q_lora_rank: int | None = None
    qk_nope_head_dim: int = 0
    qk_rope_head_dim: int = 0
    v_head_dim: int = 0
    vision_config: Mapping[str, Any] | None = None
    image_token_id: int | None = None
    vision_start_token_id: int | None = None
    vision_end_token_id: int | None = None
    vision_soft_tokens_per_image: int | None = None

    def __post_init__(self) -> None:
        require_text(self.architecture, "model.architecture")
        require_text(self.hidden_act, "model.hidden_act")
        for name in (
            "vocab_size",
            "hidden_size",
            "num_layers",
            "num_attention_heads",
            "num_kv_heads",
            "head_dim",
            "intermediate_size",
            "max_position_embeddings",
        ):
            require_int(getattr(self, name), f"model.{name}", minimum=1)
        require_bool(self.tie_word_embeddings, "model.tie_word_embeddings")
        eps = require_number(self.norm_eps, "model.norm_eps")
        if not math.isfinite(eps) or eps <= 0:
            raise ConfigError("model.norm_eps", "INVALID_RANGE", "must be finite and positive")
        object.__setattr__(self, "norm_eps", eps)
        require_bool(self.attention_bias, "model.attention_bias")
        require_bool(self.mlp_bias, "model.mlp_bias")
        if self.sliding_window is not None:
            require_int(self.sliding_window, "model.sliding_window", minimum=1)
        partial = require_number(self.partial_rotary_factor, "model.partial_rotary_factor")
        if not math.isfinite(partial) or not (0 < partial <= 1):
            raise ConfigError("model.partial_rotary_factor", "INVALID_RANGE", "must be in (0, 1]")
        object.__setattr__(self, "partial_rotary_factor", float(partial))
        self._check_experts()
        if self.rope_theta is not None:
            theta = require_number(self.rope_theta, "model.rope_theta")
            if not math.isfinite(theta) or theta <= 1:
                raise ConfigError("model.rope_theta", "INVALID_RANGE", "must be finite and > 1")
            object.__setattr__(self, "rope_theta", float(theta))
        if self.rope_scaling is not None:
            if not isinstance(cast("object", self.rope_scaling), RopeScaling):
                raise TypeError("model.rope_scaling must be a RopeScaling or None")
            if self.rope_theta is None:
                raise ConfigError(
                    "model.rope_scaling",
                    "ROPE_SCALING_WITHOUT_ROPE",
                    "a rope scaling needs rope_theta",
                )
        if self.num_kv_heads > self.num_attention_heads:
            raise ConfigError(
                "model.num_kv_heads",
                "KV_HEADS_EXCEED_HEADS",
                f"{self.num_kv_heads} key/value heads exceed {self.num_attention_heads} "
                "attention heads",
            )
        if self.num_attention_heads % self.num_kv_heads:
            raise ConfigError(
                "model.num_kv_heads",
                "KV_HEADS_NOT_DIVISOR",
                f"{self.num_kv_heads} key/value heads must divide "
                f"{self.num_attention_heads} attention heads",
            )

    def _check_experts(self) -> None:
        require_int(self.num_experts, "model.num_experts", minimum=0)
        require_int(self.experts_per_token, "model.experts_per_token", minimum=0)
        require_int(self.moe_intermediate_size, "model.moe_intermediate_size", minimum=0)
        require_bool(self.norm_topk_prob, "model.norm_topk_prob")
        if not isinstance(cast("object", self.dense_layers), tuple):
            raise TypeError("model.dense_layers must be a tuple of layer ids")
        for layer in self.dense_layers:
            require_int(layer, "model.dense_layers", minimum=0)
            if layer >= self.num_layers:
                raise ConfigError(
                    "model.dense_layers",
                    "INVALID_RANGE",
                    f"layer {layer} is not one of the {self.num_layers} layers",
                )
        if not self.num_experts:
            if self.experts_per_token or self.moe_intermediate_size or self.dense_layers:
                raise ConfigError(
                    "model.num_experts",
                    "INVALID_RANGE",
                    "a dense model (num_experts 0) has no experts_per_token, "
                    "moe_intermediate_size or dense_layers",
                )
            return
        if not 1 <= self.experts_per_token <= self.num_experts:
            raise ConfigError(
                "model.experts_per_token",
                "INVALID_RANGE",
                f"must be between 1 and num_experts ({self.num_experts}), "
                f"got {self.experts_per_token}",
            )
        if self.moe_intermediate_size < 1:
            raise ConfigError(
                "model.moe_intermediate_size", "INVALID_RANGE", "an MoE model needs a width"
            )

    @property
    def is_moe(self) -> bool:
        """Whether some layer routes tokens to experts."""
        return self.num_experts > 0 and len(self.dense_layers) < self.num_layers

    @property
    def context_length(self) -> int:
        """Longest sequence this engine serves (full for native windowing, capped otherwise)."""
        if self.sliding_window is None or self.architecture in SLIDING_WINDOW_ARCHITECTURES:
            return self.max_position_embeddings
        return min(self.max_position_embeddings, self.sliding_window)

    @property
    def kv_group_size(self) -> int:
        """Query heads that share one key/value head (``1`` for multi-head attention)."""
        return self.num_attention_heads // self.num_kv_heads

    @classmethod
    def from_hf_config(cls, config: Mapping[str, object]) -> ModelConfig:
        """Build a validated config from a parsed Hugging Face ``config.json``."""
        return cls(**parse_hf_model_fields(config))
