"""The HTTP server: routes, request ids, keys, CORS, readiness and request bodies.

:func:`create_app` builds an aiohttp application around a
:class:`~ayaka.entrypoints.serving.Serving`, the state every handler shares. The server can listen
before the engine is ready: until :meth:`~ayaka.entrypoints.serving.Serving.attach` is called every
generation route answers 503 and ``/health`` says ``starting``.

Routes::

    GET  /health                ready (200), or starting / draining / dead (503); no key needed
    GET  /metrics               Prometheus text; no per-request labels
    GET  /v1/models             the loaded model and its aliases
    POST /v1/chat/completions   OpenAI Chat Completions
    POST /v1/completions        OpenAI Completions
    POST /v1/responses          OpenAI Responses
    POST /v1/messages           Anthropic Messages
    POST /v1/messages/count_tokens   the prompt length of a Messages request

The middleware maps every exception through :func:`~ayaka.entrypoints.failures.failure_for` and
renders it in the shape of the route's protocol.

This module imports aiohttp, from the ``serve`` extra; ``import ayaka`` does not load it.
"""

from __future__ import annotations

import asyncio
import hmac
import logging
import time
import uuid
from typing import TYPE_CHECKING, Final, cast

from aiohttp import web
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from ayaka.core.errors import EngineError, InvalidRequestError
from ayaka.core.params import SamplingParams
from ayaka.entrypoints.failures import AnswerFailedError, ErrorRenderer, failure_for
from ayaka.entrypoints.serving import REQUEST_ID, SERVING, Readiness
from ayaka.utils.parsing import parse_json_strict

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from ayaka.configs.server import ServerConfig
    from ayaka.entrypoints.serving import Serving

__all__ = ["JsonObject", "create_app", "read_body", "sampling_params"]

_LOG = logging.getLogger("ayaka.serve")


# -- reading request bodies -------------------------------------------------------------------


class JsonObject:
    """A JSON object from a request body, read field by field with the field's path in errors.

    Args:
        data: The object.
        path: Where it is in the body, for messages (``messages[2]``); empty at the top.
    """

    def __init__(self, data: dict[str, object], path: str = "") -> None:
        self.data = data
        self.path = path

    @classmethod
    def of(cls, value: object, path: str) -> JsonObject:
        """``value`` as an object, or an error naming ``path``."""
        if not isinstance(value, dict):
            raise _invalid(path, "must be an object")
        return cls(cast("dict[str, object]", value), path)

    def where(self, key: str) -> str:
        """The path of ``key`` in the body."""
        return f"{self.path}.{key}" if self.path else key

    def has(self, key: str) -> bool:
        """Whether the field is present and not null."""
        return self.data.get(key) is not None

    def raw(self, key: str) -> object:
        """The field as it is, ``None`` when absent."""
        return self.data.get(key)

    def string(self, key: str, *, required: bool = False) -> str | None:
        """An optional string field."""
        value = self._get(key, required)
        if value is not None and not isinstance(value, str):
            raise _invalid(self.where(key), "must be a string")
        return value

    def text(self, key: str) -> str:
        """A required string field."""
        value = self.string(key, required=True)
        assert value is not None  # noqa: S101  # required
        return value

    def integer(self, key: str, *, required: bool = False) -> int | None:
        """An optional integer field."""
        value = self._get(key, required)
        if value is not None and (not isinstance(value, int) or isinstance(value, bool)):
            raise _invalid(self.where(key), "must be an integer")
        return value

    def number(self, key: str) -> float | None:
        """An optional number field."""
        value = self.data.get(key)
        if value is None:
            return None
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise _invalid(self.where(key), "must be a number")
        return float(value)

    def flag(self, key: str, default: bool = False) -> bool:
        """An optional boolean field."""
        value = self.data.get(key)
        if value is None:
            return default
        if not isinstance(value, bool):
            raise _invalid(self.where(key), "must be true or false")
        return value

    def child(self, key: str) -> JsonObject | None:
        """An optional object field."""
        value = self.data.get(key)
        return None if value is None else JsonObject.of(value, self.where(key))

    def array(self, key: str, *, required: bool = False) -> list[object] | None:
        """An optional array field."""
        value = self._get(key, required)
        if value is not None and not isinstance(value, list):
            raise _invalid(self.where(key), "must be an array")
        return cast("list[object] | None", value)

    def objects(self, key: str) -> list[JsonObject]:
        """An optional array of objects; empty when absent."""
        items = self.array(key) or []
        return [JsonObject.of(item, f"{self.where(key)}[{i}]") for i, item in enumerate(items)]

    def unknown(self, known: frozenset[str]) -> list[str]:
        """Fields this server does not know, for a debug log line."""
        return sorted(set(self.data) - known)

    def _get(self, key: str, required: bool) -> object:
        value = self.data.get(key)
        if value is None and required:
            raise _invalid(self.where(key), "is required")
        return value


def _invalid(path: str, message: str) -> InvalidRequestError:
    return InvalidRequestError(f"{path}: {message}", context={"param": path})


async def read_body(request: web.Request) -> JsonObject:
    """The request body as a JSON object.

    Raises:
        InvalidRequestError: If it is not one JSON object (repeated keys included).
        HTTPRequestEntityTooLarge: If it is larger than the server accepts.
    """
    raw = await request.read()
    try:
        value = parse_json_strict(raw.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as exc:
        raise InvalidRequestError(f"the body is not valid JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise InvalidRequestError("the body must be a JSON object")
    return JsonObject(cast("dict[str, object]", value))


# -- errors -----------------------------------------------------------------------------------


# -- metrics ----------------------------------------------------------------------------------


# -- the shared state -------------------------------------------------------------------------


def sampling_params(**fields: object) -> SamplingParams:
    """``SamplingParams`` from request fields; a bad value is an invalid request.

    Fields that are ``None`` keep the defaults.
    """
    values = {key: value for key, value in fields.items() if value is not None}
    try:
        return SamplingParams(**values)  # type: ignore[arg-type]  # SamplingParams checks each field
    except (TypeError, ValueError) as exc:
        if isinstance(exc, InvalidRequestError):
            raise
        raise InvalidRequestError(str(exc)) from exc


# -- the application --------------------------------------------------------------------------


_ROUTE_LABELS: Final = (
    "/health",
    "/metrics",
    "/v1/models",
    "/v1/chat/completions",
    "/v1/completions",
    "/v1/messages",
    "/v1/messages/count_tokens",
    "/v1/responses",
)

type Handler = Callable[[web.Request], Awaitable[web.StreamResponse]]


def _renderer(path: str) -> ErrorRenderer:
    """Errors on the Anthropic routes in Anthropic's shape, everywhere else in OpenAI's."""
    if path.startswith("/v1/messages"):
        from ayaka.entrypoints.anthropic.protocol import anthropic_error

        return anthropic_error
    from ayaka.entrypoints.openai.protocol import openai_error

    return openai_error


def _route_label(request: web.Request) -> str:
    return request.path if request.path in _ROUTE_LABELS else "other"


@web.middleware
async def _middleware(request: web.Request, handler: Handler) -> web.StreamResponse:
    serving = request.app[SERVING]
    request_id = uuid.uuid4().hex
    request[REQUEST_ID] = request_id
    started = time.monotonic()
    response: web.StreamResponse
    cors = _cors_origin(serving.config, request)
    try:
        if request.method == "OPTIONS" and cors is not None:
            response = web.Response(status=204, headers=_preflight_headers(request))
        else:
            _authorize(serving.config, request)
            response = await handler(request)
    except asyncio.CancelledError:
        _LOG.info("request %s %s: client went away", request_id, request.path)
        raise
    except Exception as exc:
        failure = failure_for(exc, request_id)
        if failure.status >= 500 and not isinstance(exc, (EngineError, AnswerFailedError)):
            _LOG.exception("request %s %s failed", request_id, request.path)
        else:
            _LOG.info(
                "request %s %s: %d %s", request_id, request.path, failure.status, failure.code
            )
        response = web.json_response(
            _renderer(request.path)(failure), status=failure.status, headers=failure.headers
        )
    if not response.prepared:
        response.headers["x-request-id"] = request_id
        if cors is not None:
            response.headers["Access-Control-Allow-Origin"] = cors
            response.headers["Vary"] = "Origin"
    serving.metrics.http.labels(_route_label(request), str(response.status)).inc()
    _LOG.debug(
        "request %s %s %d in %.3f s",
        request_id,
        request.path,
        response.status,
        time.monotonic() - started,
    )
    return response


def _authorize(config: ServerConfig, request: web.Request) -> None:
    key = config.api_key
    if key is None or request.path == "/health":
        return
    offered = request.headers.get("x-api-key")
    if offered is None:
        scheme, _, value = request.headers.get("Authorization", "").partition(" ")
        offered = value.strip() if scheme.lower() == "bearer" else None
    if offered is None or not hmac.compare_digest(offered.encode(), key.encode()):
        raise web.HTTPUnauthorized(text="a valid API key is required")


def _cors_origin(config: ServerConfig, request: web.Request) -> str | None:
    origin = request.headers.get("Origin")
    allowed = config.cors_origins
    if origin is None or not allowed:
        return None
    if "*" in allowed:
        return "*"
    return origin if origin in allowed else None


def _preflight_headers(request: web.Request) -> dict[str, str]:
    asked = request.headers.get("Access-Control-Request-Headers", "")
    return {
        "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
        "Access-Control-Allow-Headers": asked or "Authorization, Content-Type, x-api-key",
        "Access-Control-Max-Age": "600",
    }


async def _health(request: web.Request) -> web.Response:
    serving = request.app[SERVING]
    if serving.llm is not None and serving.llm.is_dead:
        serving.status = Readiness.DEAD
    status = serving.status
    return web.json_response(
        {"status": status.value}, status=200 if status is Readiness.READY else 503
    )


async def _metrics(request: web.Request) -> web.Response:
    serving = request.app[SERVING]
    body = generate_latest(serving.metrics.registry)
    return web.Response(body=body, headers={"Content-Type": CONTENT_TYPE_LATEST})


async def _models(request: web.Request) -> web.Response:
    serving = request.app[SERVING]
    if "anthropic-version" in request.headers:
        from ayaka.entrypoints.anthropic.messages import models_listing

        return web.json_response(models_listing(serving.model_names, serving.created))
    data = [
        {
            "id": name,
            "object": "model",
            "created": serving.created,
            "owned_by": "ayaka",
            "max_model_len": serving.max_model_len or None,
            # The longest context one request gets on this machine (the KV cache's limit).
            "max_context_tokens": serving.max_context_tokens or None,
            # Whether identical prompt prefixes are read from the cache (``ayaka connect`` step 6).
            "prefix_cache": serving.prefix_cache,
        }
        for name in serving.model_names
    ]
    return web.json_response({"object": "list", "data": data})


def create_app(serving: Serving) -> web.Application:
    """The aiohttp application for ``serving``.

    Args:
        serving: The shared state; attach an engine to it to start answering.
    """
    from ayaka.entrypoints.anthropic import messages as anthropic_messages
    from ayaka.entrypoints.openai import chat as openai_chat
    from ayaka.entrypoints.openai import responses as openai_responses

    app = web.Application(middlewares=[_middleware], client_max_size=serving.config.max_body_bytes)
    app[SERVING] = serving
    app.router.add_get("/health", _health)
    app.router.add_get("/metrics", _metrics)
    app.router.add_get("/v1/models", _models)
    app.router.add_post("/v1/chat/completions", openai_chat.create_chat_completion)
    app.router.add_post("/v1/completions", openai_chat.create_completion)
    app.router.add_post("/v1/responses", openai_responses.create_response)
    app.router.add_post("/v1/messages", anthropic_messages.create_message)
    app.router.add_post("/v1/messages/count_tokens", anthropic_messages.count_tokens)
    return app
