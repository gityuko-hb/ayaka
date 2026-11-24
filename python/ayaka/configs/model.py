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
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Final, Literal, cast

from ayaka.configs.base import ConfigMixin
from ayaka.core.errors import ConfigError, ModelLoadError
from ayaka.utils.validation import require_bool, require_int, require_number, require_text

__all__ = ["ModelConfig", "RopeScaling", "RopeScalingKind"]

type RopeScalingKind = Literal["linear", "llama3", "yarn"]
"""The rotary scalings Ayaka implements; ``config.json`` calls this ``rope_type``."""

_ROPE_KINDS: Final[tuple[RopeScalingKind, ...]] = ("linear", "llama3", "yarn")

#: ``tie_word_embeddings`` when ``config.json`` does not say. Hugging Face's base config
#: ties by default, but some architectures' own config classes do not, and a checkpoint
#: written from one of those omits the key. Tying by mistake would silently replace the
#: stored output head by the embedding, so the default follows the architecture.
_UNTIED_BY_DEFAULT: Final = frozenset({"LlamaForCausalLM", "Qwen2ForCausalLM", "Qwen3ForCausalLM"})


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

    def __post_init__(self) -> None:
        if self.kind not in _ROPE_KINDS:
            raise ConfigError(
                "model.rope_scaling.kind",
                "INVALID_CHOICE",
                f"{self.kind!r} is not one of {list(_ROPE_KINDS)}",
            )
        for name in ("factor", "low_freq_factor", "high_freq_factor", "beta_fast", "beta_slow"):
            value = require_number(getattr(self, name), f"model.rope_scaling.{name}")
            if not math.isfinite(value) or value <= 0:
                raise ConfigError(
                    f"model.rope_scaling.{name}", "INVALID_RANGE", "must be finite and positive"
                )
            object.__setattr__(self, name, float(value))
        if self.factor < 1:
            raise ConfigError("model.rope_scaling.factor", "INVALID_RANGE", "must be >= 1")
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
        elif self.kind != "linear":
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
        """Read the ``rope_scaling`` (or ``rope_parameters``) object of a ``config.json``.

        ``rope_type`` (or the older ``type``) ``"default"`` means no scaling and
        returns ``None``. ``original_max_position_embeddings`` falls back to the
        model's ``max_position_embeddings`` for ``yarn``, as Hugging Face does.

        Raises:
            ConfigError: ``ROPE_SCALING_UNSUPPORTED`` for a kind Ayaka does not
                implement (``dynamic``, ``longrope``, ...); any error of the constructor.
        """
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
        if raw.get("truncate") is False:
            unsupported.append("truncate")
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
            return default if value is None else cast("float", value)  # checked by the constructor

        attention_factor = raw.get("attention_factor")
        return cls(
            kind=kind,
            factor=number("factor", 1.0),
            original_max_position_embeddings=cast("int | None", original),
            low_freq_factor=number("low_freq_factor", 1.0),
            high_freq_factor=number("high_freq_factor", 4.0),
            beta_fast=number("beta_fast", 32.0),
            beta_slow=number("beta_slow", 1.0),
            attention_factor=cast("float | None", attention_factor),
        )


# ``config.json`` spells the same hyperparameter differently per family: the first
# present, non-null key wins. GPT-2 style on the right, Llama style on the left.
_HF_KEYS: dict[str, tuple[str, ...]] = {
    "vocab_size": ("vocab_size",),
    "hidden_size": ("hidden_size", "n_embd"),
    "num_layers": ("num_hidden_layers", "n_layer"),
    "num_attention_heads": ("num_attention_heads", "n_head"),
    "intermediate_size": ("intermediate_size", "n_inner"),
    "max_position_embeddings": ("max_position_embeddings", "n_positions"),
    "hidden_act": ("hidden_act", "activation_function"),
    "norm_eps": ("rms_norm_eps", "layer_norm_epsilon", "layer_norm_eps"),
}


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

    @property
    def context_length(self) -> int:
        """The longest sequence this engine serves: the trained context, or the window if shorter.

        Windowed attention is not implemented, so a model whose window is shorter than its
        trained context is served up to the window, where both attentions agree.
        """
        if self.sliding_window is None:
            return self.max_position_embeddings
        return min(self.max_position_embeddings, self.sliding_window)

    @property
    def kv_group_size(self) -> int:
        """Query heads that share one key/value head (``1`` for multi-head attention)."""
        return self.num_attention_heads // self.num_kv_heads

    @classmethod
    def from_hf_config(cls, config: Mapping[str, object]) -> ModelConfig:
        """Build a config from a parsed HuggingFace ``config.json``.

        Reads the Llama spelling of each key and the GPT-2 one (``n_embd``,
        ``n_layer``, ...). Implicit values follow HuggingFace: ``num_kv_heads``
        defaults to ``num_attention_heads``, ``head_dim`` to ``hidden_size //
        num_attention_heads``, ``tie_word_embeddings`` to ``True`` (``False``
        for Llama, whose config class says so), and a GPT-2 style config
        without ``n_inner`` gets ``4 * n_embd``.

        ``sliding_window`` is kept only when some layer uses it: ``layer_types`` lists a
        ``sliding_attention`` layer, or (configs written before ``layer_types``) the window
        is not switched off by ``use_sliding_window: false`` and ``max_window_layers`` leaves
        at least one layer to it, as Qwen2 reads them.

        The rotary embedding is read from ``rope_theta`` and ``rope_scaling``
        (``config.json`` written before transformers 5) or from
        ``rope_parameters`` (written by transformers 5). A Llama config with
        neither gets ``rope_theta`` 10000, the default of its config class.

        Args:
            config: The parsed JSON object. It is not modified.

        Returns:
            The validated config.

        Raises:
            ModelLoadError: ``config`` lacks required keys. The message lists every
                missing one, not just the first.
            ConfigError: ``head_dim`` is not given and ``hidden_size`` is not a
                multiple of ``num_attention_heads``; or any error of the
                constructor.
        """
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
        kv_heads = config.get("num_key_value_heads")
        architecture = cast("str", found["architecture"])
        max_positions = cast("int", found["max_position_embeddings"])
        rope_theta, rope_scaling = _hf_rope(config, architecture, max_positions)
        return cls(
            architecture=architecture,
            vocab_size=cast("int", found["vocab_size"]),
            hidden_size=cast("int", found["hidden_size"]),
            num_layers=cast("int", found["num_layers"]),
            num_attention_heads=heads,
            num_kv_heads=heads if kv_heads is None else cast("int", kv_heads),
            head_dim=cast("int", head_dim),
            intermediate_size=cast("int", found["intermediate_size"]),
            max_position_embeddings=max_positions,
            hidden_act=cast("str", found["hidden_act"]),
            norm_eps=cast("float", found["norm_eps"]),
            tie_word_embeddings=cast(
                "bool",
                config.get("tie_word_embeddings", architecture not in _UNTIED_BY_DEFAULT),
            ),
            rope_theta=rope_theta,
            rope_scaling=rope_scaling,
            attention_bias=cast("bool", config.get("attention_bias", False)),
            mlp_bias=cast("bool", config.get("mlp_bias", False)),
            sliding_window=_hf_sliding_window(config, cast("int", found["num_layers"])),
        )


def _hf_sliding_window(config: Mapping[str, object], num_layers: int) -> int | None:
    """The window of the layers that use one, or ``None`` if no layer does."""
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


#: Architectures with a rotary embedding, and the ``rope_theta`` their config class
#: defaults to when ``config.json`` does not give one.
_ROPE_THETA_DEFAULT: Final = {
    "LlamaForCausalLM": 10000.0,
    "Qwen2ForCausalLM": 10000.0,
    "Qwen3ForCausalLM": 10000.0,
}


def _hf_rope(
    config: Mapping[str, object], architecture: str, max_positions: int
) -> tuple[float | None, RopeScaling | None]:
    """``rope_theta`` and the scaling of a ``config.json``, in either spelling."""
    parameters = config.get("rope_parameters")
    raw: Mapping[str, object]
    if isinstance(parameters, Mapping):
        raw = cast("Mapping[str, object]", parameters)
        theta = raw.get("rope_theta")
    else:
        scaling = config.get("rope_scaling")
        raw = cast("Mapping[str, object]", scaling) if isinstance(scaling, Mapping) else {}
        theta = config.get("rope_theta")
    if theta is None:
        theta = _ROPE_THETA_DEFAULT.get(architecture)
    if theta is None:
        return None, None
    partial = config.get("partial_rotary_factor", raw.get("partial_rotary_factor"))
    if partial not in (None, 1, 1.0):
        raise ConfigError(
            "model.partial_rotary_factor",
            "ROPE_SCALING_UNSUPPORTED",
            f"a rotary embedding over part of each head ({partial}) is not implemented",
        )
    return cast("float", theta), RopeScaling.from_hf(raw, max_position_embeddings=max_positions)
