"""Dependency-free error taxonomy for Ayaka.

This module is the root of the package's exception hierarchy. It imports only
the standard library: no ``ayaka.*`` module and no optional dependency may be
imported here, because every layer above may need to raise or catch these
errors.

Transport concerns stay outside. :class:`AyakaError` carries a stable ``code``
and a ``retryable`` flag; the serving layer maps exception types to HTTP status
codes exactly once.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

__all__ = [
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
]


@dataclass(frozen=True, slots=True)
class ErrorInfo:
    """A small, serializable description of one failure.

    Attributes:
        code: Stable machine-readable identifier.
        message: Human-readable message.
        retryable: Whether the same request may succeed when retried later.
    """

    code: str
    message: str
    retryable: bool = False


class AyakaError(Exception):
    """Base class for every error raised by Ayaka.

    Subclasses set :attr:`code` and :attr:`retryable`; instances may attach
    structured context without changing the exception's message.
    """

    code: str = "ayaka_error"
    retryable: bool = False

    def __init__(self, message: str, *, context: Mapping[str, object] | None = None) -> None:
        self.message = message
        self.context: dict[str, object] = dict(context or {})
        super().__init__(message)

    def to_info(self) -> ErrorInfo:
        """Return a serializable view of this failure."""

        return ErrorInfo(code=self.code, message=self.message, retryable=self.retryable)

    def __reduce__(
        self,
    ) -> tuple[Callable[..., object], tuple[object, ...], dict[str, object]]:
        # Exceptions cross worker boundaries. Restore ``context`` as well as
        # ``args`` so an error does not lose its structured fields when pickled.
        return (type(self), (self.message,), {"context": self.context})


class ConfigError(AyakaError, ValueError):
    """A configuration validation failure with a stable path and code."""

    def __init__(
        self,
        path: str,
        code: str,
        message: str,
        *,
        context: Mapping[str, object] | None = None,
    ) -> None:
        self.path = path
        self.code = code
        self.message = message
        self.context: dict[str, object] = dict(context or {})
        Exception.__init__(self, f"{path}: {message} [{code}]")

    def __reduce__(
        self,
    ) -> tuple[Callable[..., object], tuple[object, ...], dict[str, object]]:
        return (type(self), (self.path, self.code, self.message), {"context": self.context})


def _rebuild_capability_error(capability: str, detail: str, remedy: str) -> CapabilityError:
    return CapabilityError(capability, detail=detail, remedy=remedy)


class CapabilityError(AyakaError, ImportError):
    """An optional dependency or backend capability is unavailable."""

    code = "capability_unavailable"

    def __init__(self, capability: str, *, detail: str, remedy: str) -> None:
        self.capability = capability
        self.detail = detail
        self.remedy = remedy
        super().__init__(f"capability {capability!r} unavailable: {detail}. Remedy: {remedy}")

    def __reduce__(
        self,
    ) -> tuple[Callable[..., object], tuple[object, ...], dict[str, object]]:
        return (_rebuild_capability_error, (self.capability, self.detail, self.remedy), {})


class CancelledError(AyakaError, RuntimeError):
    """Work was cancelled through a cooperative cancellation token."""

    code = "cancelled"

    def __init__(self, reason: str = "cancelled") -> None:
        self.reason = reason
        super().__init__(reason)


class InvalidRequestError(AyakaError, ValueError):
    """The caller supplied a request that cannot be admitted."""

    code = "invalid_request"


class UnsupportedFeatureError(InvalidRequestError):
    """The request is valid but asks for a feature this build does not support."""

    code = "unsupported_feature"


class CapacityError(AyakaError, RuntimeError):
    """A bounded resource is exhausted; the same request may succeed later."""

    code = "capacity_exceeded"
    retryable = True


class OverloadedError(CapacityError):
    """The engine or one of its queues is currently overloaded."""

    code = "overloaded"


class EngineError(AyakaError, RuntimeError):
    """An engine invariant or runtime operation failed."""

    code = "engine_error"


class EngineDeadError(EngineError):
    """The engine has failed and will not accept more work."""

    code = "engine_dead"


class RequestAborted(AyakaError, RuntimeError):  # noqa: N818  # public contract name
    """A request was deliberately aborted by its owner or the engine."""

    code = "request_aborted"


class RequestTimeout(AyakaError, RuntimeError):  # noqa: N818  # public contract name
    """A request exceeded its queue, prefill or generation deadline."""

    code = "request_timeout"
    retryable = True


class ModelLoadError(AyakaError, RuntimeError):
    """A checkpoint could not be resolved, validated or materialized."""

    code = "model_load_error"


class BackendError(AyakaError, RuntimeError):
    """An execution, attention or kernel backend failed."""

    code = "backend_error"
