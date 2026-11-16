"""Architecture hyperparameters of one model, as read from its ``config.json``.

``ModelConfig`` is a frozen host value: it describes the shape of the network,
not where the weights live (:mod:`ayaka.configs.model_source`) or how they are
run (:mod:`ayaka.configs.device`). Everything the memory plan and the KV cache
geometry need is here, so ``ayaka plan`` can build one from ``config.json``
without loading a tensor.

::

    config.json --from_hf_config()--> ModelConfig --> models (build the network)
                                                  \\-> memory, kvcache (sizes)

Fields that only some architectures have (rotary scaling, mixture-of-experts,
sliding windows, vision) are added with the milestone that implements them.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import cast

from ayaka.configs.base import ConfigMixin
from ayaka.core.errors import ConfigError, ModelLoadError
from ayaka.utils.validation import require_bool, require_int, require_number, require_text

__all__ = ["ModelConfig"]

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
    def kv_group_size(self) -> int:
        """Query heads that share one key/value head (``1`` for multi-head attention)."""
        return self.num_attention_heads // self.num_kv_heads

    @classmethod
    def from_hf_config(cls, config: Mapping[str, object]) -> ModelConfig:
        """Build a config from a parsed HuggingFace ``config.json``.

        Reads the Llama spelling of each key and the GPT-2 one (``n_embd``,
        ``n_layer``, ...). Implicit values follow HuggingFace: ``num_kv_heads``
        defaults to ``num_attention_heads``, ``head_dim`` to ``hidden_size //
        num_attention_heads``, ``tie_word_embeddings`` to ``True``, and a GPT-2
        style config without ``n_inner`` gets ``4 * n_embd``.

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
        return cls(
            architecture=cast("str", found["architecture"]),
            vocab_size=cast("int", found["vocab_size"]),
            hidden_size=cast("int", found["hidden_size"]),
            num_layers=cast("int", found["num_layers"]),
            num_attention_heads=heads,
            num_kv_heads=heads if kv_heads is None else cast("int", kv_heads),
            head_dim=cast("int", head_dim),
            intermediate_size=cast("int", found["intermediate_size"]),
            max_position_embeddings=cast("int", found["max_position_embeddings"]),
            hidden_act=cast("str", found["hidden_act"]),
            norm_eps=cast("float", found["norm_eps"]),
            tie_word_embeddings=cast("bool", config.get("tie_word_embeddings", True)),
        )
