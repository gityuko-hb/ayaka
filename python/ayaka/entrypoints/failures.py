"""Errors as clients see them: one mapping from typed errors to an HTTP status and a kind.

Errors are mapped here, once, from the typed errors of :mod:`ayaka.core.errors`; each protocol
only renders the result in its own shape (``openai_error``, ``anthropic_error``)::

    InvalidRequestError       400   names the field
    UnsupportedFeatureError   400   names the feature
    CapacityError             400   the prompt needs more KV than one request can get here
    OverloadedError           429   with Retry-After
    RequestTimeout            408
    EngineError, not ready    503   with Retry-After; no internal detail, only the request id
    anything else             500   logged with its traceback

A request that was admitted and then ended in error is mapped from its
:class:`~ayaka.entrypoints.conversation.ErrorEvent` by :func:`failure_for_event`.

This module imports aiohttp, from the ``serve`` extra; ``import ayaka`` does not load it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Final, Literal

from aiohttp import web

from ayaka.core.errors import (
    AyakaError,
    CapacityError,
    EngineError,
    InvalidRequestError,
    OverloadedError,
    RequestTimeout,
    UnsupportedFeatureError,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from ayaka.entrypoints.conversation import ErrorEvent


__all__ = [
    "AnswerFailedError",
    "ApiFailure",
    "ErrorRenderer",
    "FailureKind",
    "NotReadyError",
    "failure_for",
    "failure_for_event",
]


#: Seconds a client is told to wait before retrying a 429 or a 503.
_RETRY_AFTER: Final = "1"


class NotReadyError(EngineError):
    """The server is not serving generation requests right now."""

    code = "not_ready"
    retryable = True


class AnswerFailedError(Exception):
    """A request answered in one response ended with an error instead of an answer."""

    def __init__(self, event: ErrorEvent) -> None:
        super().__init__(event.message)
        self.event = event


type FailureKind = Literal[
    "invalid_request",
    "authentication",
    "not_found",
    "too_large",
    "overloaded",
    "timeout",
    "unavailable",
    "internal",
]


@dataclass(frozen=True, slots=True)
class ApiFailure:
    """An error answer before the protocol shapes it.

    Attributes:
        status: The HTTP status.
        kind: What kind of failure it is; each protocol maps it to its own error type.
        code: The stable ``ErrorInfo`` code.
        message: What the client may read.
        param: The request field at fault, when one is.
    """

    status: int
    kind: FailureKind
    code: str
    message: str
    param: str | None = None

    @property
    def headers(self) -> dict[str, str]:
        """Headers the answer carries."""
        return {"Retry-After": _RETRY_AFTER} if self.status in (429, 503) else {}


def failure_for(exc: BaseException, request_id: str) -> ApiFailure:
    """Map an exception to the answer the client gets."""
    if isinstance(exc, AnswerFailedError):
        return failure_for_event(exc.event, request_id)
    if isinstance(exc, web.HTTPException):
        kind: FailureKind = {
            401: "authentication",
            404: "not_found",
            405: "invalid_request",
            413: "too_large",
        }.get(exc.status, "invalid_request")  # type: ignore[assignment]  # literal values
        message = exc.text if exc.text and not exc.text.startswith(str(exc.status)) else exc.reason
        if exc.status == 413:
            message = "the request body is larger than this server accepts"
        return ApiFailure(exc.status, kind, kind, message or exc.reason)
    if isinstance(exc, NotReadyError):
        return ApiFailure(503, "unavailable", exc.code, exc.message)
    if isinstance(exc, UnsupportedFeatureError):
        return ApiFailure(400, "invalid_request", exc.code, exc.message, _param(exc))
    if isinstance(exc, InvalidRequestError):
        return ApiFailure(400, "invalid_request", exc.code, exc.message, _param(exc))
    if isinstance(exc, OverloadedError):
        return ApiFailure(429, "overloaded", exc.code, exc.message)
    if isinstance(exc, CapacityError):
        return ApiFailure(400, "invalid_request", exc.code, exc.message)
    if isinstance(exc, RequestTimeout):
        return ApiFailure(408, "timeout", exc.code, exc.message)
    if isinstance(exc, EngineError):
        return ApiFailure(
            503, "unavailable", exc.code, f"the engine cannot serve requests (request {request_id})"
        )
    code = exc.code if isinstance(exc, AyakaError) else "internal_error"
    return ApiFailure(500, "internal", code, f"internal server error (request {request_id})")


def failure_for_event(event: ErrorEvent, request_id: str) -> ApiFailure:
    """The answer for a request that ended in error before anything was sent."""
    if event.code == "request_timeout":
        return ApiFailure(408, "timeout", event.code, event.message)
    if event.code in ("cancelled", "engine_dead", "not_ready"):
        return ApiFailure(503, "unavailable", event.code, event.message)
    return ApiFailure(500, "internal", event.code, f"{event.message} (request {request_id})")


def _param(exc: AyakaError) -> str | None:
    param = exc.context.get("param")
    return param if isinstance(param, str) else None


type ErrorRenderer = Callable[[ApiFailure], dict[str, object]]
