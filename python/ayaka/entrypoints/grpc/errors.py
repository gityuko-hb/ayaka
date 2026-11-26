"""Bidirectional mapping between :class:`~ayaka.core.errors.AyakaError` and gRPC status codes."""

from __future__ import annotations

import json
from typing import Any, cast

from ayaka.core.errors import (
    AyakaError,
    CancelledError,
    CapabilityError,
    ConfigError,
    ContextLengthError,
    EngineDeadError,
    EngineError,
    ErrorInfo,
    InvalidRequestError,
    OverloadedError,
    UnsupportedFeatureError,
)
from ayaka.utils.import_utils import require_module

__all__ = [
    "METADATA_ERROR_KEY",
    "abort_with_error",
    "error_from_rpc",
    "error_from_wire",
    "error_to_wire",
]

METADATA_ERROR_KEY = "x-ayaka-error-json"


def _grpc() -> Any:
    return require_module(
        "grpc",
        capability="grpc",
        remedy="install the grpc extra: uv sync --extra grpc",
    )


def error_to_wire(exc: BaseException) -> dict[str, object]:
    """Serialize an :class:`AyakaError` (or unexpected error) into structured metadata."""
    if isinstance(exc, ContextLengthError):
        return {
            "kind": "context_length",
            "code": exc.code,
            "message": exc.message,
            "prompt_tokens": exc.prompt_tokens,
            "output_tokens": exc.output_tokens,
            "max_context_tokens": exc.max_context_tokens,
        }
    if isinstance(exc, UnsupportedFeatureError):
        return {
            "kind": "unsupported_feature",
            "code": exc.code,
            "message": exc.message,
            "context": dict(exc.context),
        }
    if isinstance(exc, InvalidRequestError):
        return {
            "kind": "invalid_request",
            "code": exc.code,
            "message": exc.message,
            "context": dict(exc.context),
        }
    if isinstance(exc, OverloadedError):
        return {
            "kind": "overloaded",
            "code": exc.code,
            "message": exc.message,
            "context": dict(exc.context),
        }
    if isinstance(exc, CapabilityError):
        return {
            "kind": "capability",
            "code": exc.code,
            "capability": exc.capability,
            "detail": exc.detail,
            "remedy": exc.remedy,
        }
    if isinstance(exc, CancelledError):
        return {"kind": "cancelled", "code": exc.code, "message": exc.message}
    if isinstance(exc, EngineDeadError):
        return {"kind": "engine_dead", "code": exc.code, "message": exc.message}
    if isinstance(exc, ConfigError):
        return {"kind": "config", "code": exc.code, "path": exc.path, "message": exc.message}
    if isinstance(exc, AyakaError):
        return {
            "kind": "engine",
            "code": exc.code,
            "message": exc.message,
            "context": dict(exc.context),
        }
    return {"kind": "engine", "code": "INTERNAL", "message": f"{type(exc).__name__}: {exc}"}


def error_from_wire(payload: dict[str, object]) -> AyakaError:
    """Reconstruct the matching :class:`AyakaError` subclass from serialized metadata."""
    kind = str(payload.get("kind", "engine"))
    message = str(payload.get("message", "remote error"))
    ctx_raw = payload.get("context")
    ctx = cast("dict[str, Any]", ctx_raw) if isinstance(ctx_raw, dict) else {}
    if kind == "context_length":
        prompt_t = payload.get("prompt_tokens")
        output_t = payload.get("output_tokens")
        max_ctx = payload.get("max_context_tokens")
        return ContextLengthError(
            prompt_tokens=int(prompt_t) if isinstance(prompt_t, int) else 0,
            output_tokens=int(output_t) if isinstance(output_t, int) else 1,
            max_context_tokens=int(max_ctx) if isinstance(max_ctx, int) else 0,
        )
    if kind == "unsupported_feature":
        return UnsupportedFeatureError(message, context=ctx or None)
    if kind == "invalid_request":
        return InvalidRequestError(message, context=ctx or None)
    if kind == "overloaded":
        return OverloadedError(message, context=ctx or None)
    if kind == "capability":
        return CapabilityError(
            str(payload.get("capability", "grpc")),
            detail=str(payload.get("detail", message)),
            remedy=str(payload.get("remedy", "")),
        )
    if kind == "cancelled":
        return CancelledError(message)
    if kind == "engine_dead":
        return EngineDeadError(message)
    if kind == "config":
        return ConfigError(
            str(payload.get("path", "config")),
            str(payload.get("code", "INVALID")),
            message,
        )
    return EngineError(message, context=ctx or None)


def _status_for_error(grpc: Any, exc: BaseException) -> Any:
    if isinstance(exc, (InvalidRequestError, ContextLengthError, ConfigError)):
        return grpc.StatusCode.INVALID_ARGUMENT
    if isinstance(exc, UnsupportedFeatureError):
        return grpc.StatusCode.UNIMPLEMENTED
    if isinstance(exc, CapabilityError):
        return grpc.StatusCode.FAILED_PRECONDITION
    if isinstance(exc, OverloadedError):
        return grpc.StatusCode.RESOURCE_EXHAUSTED
    if isinstance(exc, CancelledError):
        return grpc.StatusCode.CANCELLED
    if isinstance(exc, EngineDeadError):
        return grpc.StatusCode.UNAVAILABLE
    return grpc.StatusCode.INTERNAL


async def abort_with_error(context: Any, exc: BaseException) -> None:
    """Abort an async gRPC RPC context with the mapped status and structured error metadata."""
    grpc = _grpc()
    wire = json.dumps(error_to_wire(exc), separators=(",", ":"), ensure_ascii=False)
    await context.abort(
        _status_for_error(grpc, exc),
        details=str(exc),
        trailing_metadata=((METADATA_ERROR_KEY, wire),),
    )


def error_from_rpc(exc: Any) -> AyakaError:
    """Convert a ``grpc.aio.AioRpcError`` back into the matching :class:`AyakaError`."""
    grpc = _grpc()
    trailing = exc.trailing_metadata() if callable(getattr(exc, "trailing_metadata", None)) else ()
    if trailing:
        for key, value in trailing:
            if key == METADATA_ERROR_KEY:
                raw_text = value.decode("utf-8") if isinstance(value, bytes) else str(value)
                parsed = json.loads(raw_text)
                if isinstance(parsed, dict):
                    return error_from_wire(cast("dict[str, object]", parsed))
    code = exc.code()
    details = exc.details() or "gRPC call failed"
    if code == grpc.StatusCode.INVALID_ARGUMENT:
        return InvalidRequestError(details)
    if code == grpc.StatusCode.UNIMPLEMENTED:
        return UnsupportedFeatureError(details)
    if code == grpc.StatusCode.RESOURCE_EXHAUSTED:
        return OverloadedError(details)
    if code == grpc.StatusCode.CANCELLED:
        return CancelledError(details)
    if code == grpc.StatusCode.UNAVAILABLE:
        return EngineDeadError(details)
    return EngineError(details)


def error_info_to_wire(info: ErrorInfo) -> dict[str, object]:
    """Serialize :class:`~ayaka.core.errors.ErrorInfo`."""
    return {"code": info.code, "message": info.message, "retryable": info.retryable}


def error_info_from_wire(payload: dict[str, object]) -> ErrorInfo:
    """Deserialize :class:`~ayaka.core.errors.ErrorInfo`."""
    return ErrorInfo(
        code=str(payload.get("code", "INTERNAL")),
        message=str(payload.get("message", "")),
        retryable=bool(payload.get("retryable", False)),
    )
