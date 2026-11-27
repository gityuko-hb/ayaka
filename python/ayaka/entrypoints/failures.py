"""Errors as clients see them: one closed table from error codes to each protocol's shape.

Every code a client can receive has exactly one row in :data:`ERROR_CODES`: its HTTP status, the
OpenAI error ``type``, the Anthropic error ``type`` and the code a Responses ``response.failed``
event carries. :func:`failure_for` maps an exception to a code once; each protocol only renders
the row in its own shape (``openai_error``, ``anthropic_error``, the Responses snapshot). A code
with no row is answered as ``internal_error``, so nothing outside the table reaches a client::

    invalid_request, unsupported_feature    400   names the field or the feature
    context_length_exceeded                 400   prompt tokens, requested output, the limit
    capacity_exceeded                       400   the prompt needs more KV than one request gets
    authentication 401, not_found 404, too_large 413
    overloaded                              429   with Retry-After
    request_timeout                         408
    not_ready, cancelled, engine_error,     503   with Retry-After; no internal detail
    engine_dead, request_aborted
    anything else                           500   ``internal_error``, logged with its traceback

A request that was admitted and then ended in error is mapped from its
:class:`~ayaka.entrypoints.conversation.ErrorEvent` by :func:`failure_for_event`.

This module loads aiohttp only to recognise its HTTP exceptions, inside :func:`failure_for`.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, Final

from ayaka.core.errors import (
    AyakaError,
    CapacityError,
    EngineError,
    InvalidRequestError,
    RequestTimeout,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from ayaka.entrypoints.conversation import ErrorEvent


__all__ = [
    "ERROR_CODES",
    "AnswerFailedError",
    "ApiFailure",
    "ErrorCode",
    "ErrorRenderer",
    "NotReadyError",
    "failure_for",
    "failure_for_event",
    "responses_code",
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


@dataclass(frozen=True, slots=True)
class ErrorCode:
    """How one error code looks in each protocol.

    Attributes:
        status: The HTTP status of an answer that is this error.
        openai_type: The ``error.type`` of an OpenAI error body.
        anthropic_type: The ``error.type`` of an Anthropic error body.
        responses_code: The ``error.code`` of a Responses ``response.failed`` event, from the
            codes OpenAI documents for it.
    """

    status: int
    openai_type: str
    anthropic_type: str
    responses_code: str


def _row(status: int, openai: str, anthropic: str, responses: str) -> ErrorCode:
    return ErrorCode(status, openai, anthropic, responses)


_INVALID: Final = _row(400, "invalid_request_error", "invalid_request_error", "invalid_prompt")
_UNAVAILABLE: Final = _row(503, "service_unavailable_error", "overloaded_error", "server_error")
_INTERNAL: Final = _row(500, "server_error", "api_error", "server_error")

#: Every error code a client can receive, and nothing else.
ERROR_CODES: Final[Mapping[str, ErrorCode]] = MappingProxyType(
    {
        "invalid_request": _INVALID,
        "unsupported_feature": _INVALID,
        "context_length_exceeded": _row(
            400, "invalid_request_error", "invalid_request_error", "context_length_exceeded"
        ),
        "capacity_exceeded": _row(
            400, "invalid_request_error", "invalid_request_error", "server_error"
        ),
        "authentication": _row(401, "authentication_error", "authentication_error", "server_error"),
        "not_found": _row(404, "not_found_error", "not_found_error", "server_error"),
        "too_large": _row(413, "invalid_request_error", "request_too_large", "invalid_prompt"),
        "overloaded": _row(429, "rate_limit_error", "rate_limit_error", "rate_limit_exceeded"),
        "out_of_memory": _UNAVAILABLE,
        "request_timeout": _row(408, "timeout_error", "api_error", "server_error"),
        "preempted_timeout": _row(408, "timeout_error", "api_error", "server_error"),
        "not_ready": _UNAVAILABLE,
        "cancelled": _UNAVAILABLE,
        "request_aborted": _UNAVAILABLE,
        "engine_error": _UNAVAILABLE,
        "engine_dead": _UNAVAILABLE,
        "backend_error": _INTERNAL,
        "capability_unavailable": _INTERNAL,
        "model_load_error": _INTERNAL,
        "internal_error": _INTERNAL,
    }
)


@dataclass(frozen=True, slots=True)
class ApiFailure:
    """An error answer before the protocol shapes it.

    Attributes:
        code: The error code, a key of :data:`ERROR_CODES`.
        message: What the client may read.
        request_id: The id of the request, for the client to quote.
        param: The request field at fault, when one is.

    Raises:
        ValueError: If ``code`` has no row in the table.
    """

    code: str
    message: str
    request_id: str
    param: str | None = None

    def __post_init__(self) -> None:
        if self.code not in ERROR_CODES:
            raise ValueError(f"error code {self.code!r} is not in the error code table")

    @property
    def row(self) -> ErrorCode:
        """How this error looks in each protocol."""
        return ERROR_CODES[self.code]

    @property
    def status(self) -> int:
        """The HTTP status."""
        return self.row.status

    @property
    def headers(self) -> dict[str, str]:
        """Headers the answer carries."""
        return {"Retry-After": _RETRY_AFTER} if self.status in (429, 503) else {}


#: The codes of aiohttp's own HTTP errors, by status.
_HTTP_CODES: Final = {401: "authentication", 404: "not_found", 413: "too_large"}


def failure_for(exc: BaseException, request_id: str) -> ApiFailure:
    """Map an exception to the answer the client gets."""
    from aiohttp import web

    if isinstance(exc, AnswerFailedError):
        return failure_for_event(exc.event, request_id)
    if isinstance(exc, web.HTTPException):
        code = _HTTP_CODES.get(exc.status, "invalid_request")
        message = exc.text if exc.text and not exc.text.startswith(str(exc.status)) else exc.reason
        if exc.status == 413:
            message = "the request body is larger than this server accepts"
        return ApiFailure(code, message or exc.reason, request_id)
    if isinstance(exc, NotReadyError):
        return ApiFailure(exc.code, exc.message, request_id)
    if isinstance(exc, InvalidRequestError):
        return ApiFailure(exc.code, exc.message, request_id, _param(exc))
    if isinstance(exc, (CapacityError, RequestTimeout)):
        return ApiFailure(exc.code, exc.message, request_id)
    if isinstance(exc, EngineError):
        code = exc.code if exc.code in ERROR_CODES else "engine_error"
        return ApiFailure(
            code, f"the engine cannot serve requests (request {request_id})", request_id
        )
    code = exc.code if isinstance(exc, AyakaError) and exc.code in ERROR_CODES else "internal_error"
    return ApiFailure(code, f"internal server error (request {request_id})", request_id)


def failure_for_event(event: ErrorEvent, request_id: str) -> ApiFailure:
    """The answer for a request that ended in error before anything was sent.

    A server error (500) names the request, so the client can quote it; the others say what
    happened as the engine reported it.
    """
    code = event.code if event.code in ERROR_CODES else "internal_error"
    message = event.message
    if ERROR_CODES[code].status == 500:
        message = f"{message} (request {request_id})"
    return ApiFailure(code, message, request_id)


def responses_code(code: str) -> str:
    """The code a Responses ``response.failed`` event carries for an error code."""
    return ERROR_CODES.get(code, _INTERNAL).responses_code


def _param(exc: AyakaError) -> str | None:
    param = exc.context.get("param")
    return param if isinstance(param, str) else None


type ErrorRenderer = Callable[[ApiFailure], dict[str, object]]
