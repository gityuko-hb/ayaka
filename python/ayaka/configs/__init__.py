"""Host-side configuration records for model sources and tokenizers.

Every record is frozen and validated at construction. Importing this package is
cheap: each public name is resolved from its submodule on first access.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ayaka.utils.lazy_exports import build_lazy_exports

if TYPE_CHECKING:
    from ayaka.configs.base import (
        ConfigError,
        ConfigMixin,
        FrozenMapping,
        freeze_mapping,
        validate_buckets,
    )
    from ayaka.configs.model_source import ModelSourceConfig, ModelTrustPolicy, RequestFormat
    from ayaka.configs.tokenizer import (
        DetokBackend,
        DetokenizeParams,
        DetokUpdate,
        EncodeResult,
        SpecialTokenPolicyName,
        TokenizerConfig,
        TokenizerMode,
        TruncationSide,
        tokenizer_config_from_source,
    )

__all__ = [
    "ConfigError",
    "ConfigMixin",
    "DetokBackend",
    "DetokUpdate",
    "DetokenizeParams",
    "EncodeResult",
    "FrozenMapping",
    "ModelSourceConfig",
    "ModelTrustPolicy",
    "RequestFormat",
    "SpecialTokenPolicyName",
    "TokenizerConfig",
    "TokenizerMode",
    "TruncationSide",
    "freeze_mapping",
    "tokenizer_config_from_source",
    "validate_buckets",
]

_EXPORTS = {
    "base": (
        "ConfigError",
        "ConfigMixin",
        "FrozenMapping",
        "freeze_mapping",
        "validate_buckets",
    ),
    "model_source": ("ModelSourceConfig", "ModelTrustPolicy", "RequestFormat"),
    "tokenizer": (
        "DetokBackend",
        "DetokUpdate",
        "DetokenizeParams",
        "EncodeResult",
        "SpecialTokenPolicyName",
        "TokenizerConfig",
        "TokenizerMode",
        "TruncationSide",
        "tokenizer_config_from_source",
    ),
}

__getattr__, __dir__ = build_lazy_exports(__name__, _EXPORTS, globals())
