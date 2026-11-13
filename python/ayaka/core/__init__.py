"""Dependency-free public contract for Ayaka.

This package is the lowest contract layer. Apart from the standard library and
other ``ayaka.core`` modules it imports only the standard-library-only
:mod:`ayaka.utils.lazy_exports` and :mod:`ayaka.utils.validation` leaves: no
``torch``, ``triton``, numpy, pydantic, msgspec or higher-level ``ayaka.*``
imports are allowed here.

Importing this package is cheap: each public name is resolved from its submodule
on first access. The package is intentionally small: five modules plus this
facade. Keep it that way. If a new shared type appears, first decide whether it
belongs to a boundary package (cache, engine, execution, tokenizers) instead.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ayaka.utils.lazy_exports import build_lazy_exports

if TYPE_CHECKING:
    from .errors import (
        AyakaError,
        BackendError,
        CancelledError,
        CapabilityError,
        CapacityError,
        ConfigError,
        EngineDeadError,
        EngineError,
        ErrorInfo,
        InvalidRequestError,
        ModelLoadError,
        OverloadedError,
        RequestAborted,
        RequestTimeout,
        UnsupportedFeatureError,
    )
    from .outputs import CompletionOutput, RequestMetrics, RequestOutput, TokenLogprob, Usage
    from .params import (
        LOGIT_BIAS_LIMIT,
        MAX_LOGPROB_TOKEN_IDS,
        TOP_K_DISABLED,
        LogprobMode,
        SamplingParams,
    )
    from .request import MAX_CACHE_SALT_LEN, GenerateRequest, TokenizedRequest
    from .types import (
        FinishReason,
        ForwardMode,
        Iteration,
        KVLayoutKind,
        Layout,
        RequestId,
        SequenceIndex,
        TokenId,
    )

__all__ = [
    "LOGIT_BIAS_LIMIT",
    "MAX_CACHE_SALT_LEN",
    "MAX_LOGPROB_TOKEN_IDS",
    "TOP_K_DISABLED",
    "AyakaError",
    "BackendError",
    "CancelledError",
    "CapabilityError",
    "CapacityError",
    "CompletionOutput",
    "ConfigError",
    "EngineDeadError",
    "EngineError",
    "ErrorInfo",
    "FinishReason",
    "ForwardMode",
    "GenerateRequest",
    "InvalidRequestError",
    "Iteration",
    "KVLayoutKind",
    "Layout",
    "LogprobMode",
    "ModelLoadError",
    "OverloadedError",
    "RequestAborted",
    "RequestId",
    "RequestMetrics",
    "RequestOutput",
    "RequestTimeout",
    "SamplingParams",
    "SequenceIndex",
    "TokenId",
    "TokenLogprob",
    "TokenizedRequest",
    "UnsupportedFeatureError",
    "Usage",
]

_EXPORTS = {
    "errors": (
        "AyakaError",
        "BackendError",
        "CancelledError",
        "CapabilityError",
        "CapacityError",
        "ConfigError",
        "EngineDeadError",
        "EngineError",
        "ErrorInfo",
        "InvalidRequestError",
        "ModelLoadError",
        "OverloadedError",
        "RequestAborted",
        "RequestTimeout",
        "UnsupportedFeatureError",
    ),
    "outputs": ("CompletionOutput", "RequestMetrics", "RequestOutput", "TokenLogprob", "Usage"),
    "params": (
        "LOGIT_BIAS_LIMIT",
        "MAX_LOGPROB_TOKEN_IDS",
        "TOP_K_DISABLED",
        "LogprobMode",
        "SamplingParams",
    ),
    "request": ("MAX_CACHE_SALT_LEN", "GenerateRequest", "TokenizedRequest"),
    "types": (
        "FinishReason",
        "ForwardMode",
        "Iteration",
        "KVLayoutKind",
        "Layout",
        "RequestId",
        "SequenceIndex",
        "TokenId",
    ),
}

__getattr__, __dir__ = build_lazy_exports(__name__, _EXPORTS, globals())
