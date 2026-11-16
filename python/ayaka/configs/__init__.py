"""Host-side configuration records for the model, tokenizer, device, cache, scheduler and engine.

Every record is frozen and validated at construction. Importing this package is
cheap: each public name is resolved from its submodule on first access.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ayaka.utils.lazy_exports import build_lazy_exports

if TYPE_CHECKING:
    from ayaka.configs.base import (
        ConfigMixin,
        FrozenMapping,
        freeze_mapping,
        validate_buckets,
    )
    from ayaka.configs.cache import CacheConfig
    from ayaka.configs.device import DeviceConfig
    from ayaka.configs.engine import EngineConfig
    from ayaka.configs.model import ModelConfig
    from ayaka.configs.model_source import ModelSourceConfig, ModelTrustPolicy, RequestFormat
    from ayaka.configs.scheduler import SchedulerConfig, SchedulingPolicyName
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
    "CacheConfig",
    "ConfigMixin",
    "DetokBackend",
    "DetokUpdate",
    "DetokenizeParams",
    "DeviceConfig",
    "EncodeResult",
    "EngineConfig",
    "FrozenMapping",
    "ModelConfig",
    "ModelSourceConfig",
    "ModelTrustPolicy",
    "RequestFormat",
    "SchedulerConfig",
    "SchedulingPolicyName",
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
        "ConfigMixin",
        "FrozenMapping",
        "freeze_mapping",
        "validate_buckets",
    ),
    "cache": ("CacheConfig",),
    "device": ("DeviceConfig",),
    "engine": ("EngineConfig",),
    "model": ("ModelConfig",),
    "model_source": ("ModelSourceConfig", "ModelTrustPolicy", "RequestFormat"),
    "scheduler": ("SchedulerConfig", "SchedulingPolicyName"),
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
