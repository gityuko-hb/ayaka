"""The HTTP server: routes, request ids, keys, errors, readiness and the shared generation path.

:func:`create_app` builds an aiohttp application around a :class:`Serving`, the state every
handler shares: the engine (an :class:`~ayaka.engine.async_llm.AsyncLLM`), the model's output
formats, its names and the metrics. The server can listen before the engine is ready: until
:meth:`Serving.attach` is called every generation route answers 503 and ``/health`` says
``starting``.

Routes::

    GET  /health                ready (200), or starting / draining / dead (503); no key needed
    GET  /metrics               Prometheus text; no per-request labels
    GET  /v1/models             the loaded model and its aliases
    POST /v1/chat/completions   OpenAI Chat Completions
    POST /v1/completions        OpenAI Completions
    POST /v1/responses          OpenAI Responses
    POST /v1/messages           Anthropic Messages
    POST /v1/messages/count_tokens   the prompt length of a Messages request

Errors are mapped here, once, from the typed errors of :mod:`ayaka.core.errors`; each protocol
only renders the result in its own shape::

    InvalidRequestError       400   names the field
    UnsupportedFeatureError   400   names the feature
    CapacityError             400   the prompt needs more KV than one request can get here
    OverloadedError           429   with Retry-After
    RequestTimeout            408
    EngineError, not ready    503   with Retry-After; no internal detail, only the request id
    anything else             500   logged with its traceback

The default log level records request ids, token counts and timings, never prompt or output text.

This module imports aiohttp, from the ``serve`` extra; ``import ayaka`` does not load it.
"""

from __future__ import annotations

import asyncio
import hmac
import logging
import time
import uuid
from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING, Final, Literal, cast

from aiohttp import web
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)

from ayaka.core.constraints import ConstraintKind, StructuredParams, ToolCallSyntax, ToolSchema
from ayaka.core.errors import (
    AyakaError,
    CapacityError,
    EngineError,
    InvalidRequestError,
    OverloadedError,
    RequestTimeout,
    UnsupportedFeatureError,
)
from ayaka.core.params import SamplingParams
from ayaka.core.request import GenerateRequest
from ayaka.core.types import RequestId
from ayaka.entrypoints.conversation import (
    ChoiceEvent,
    ErrorEvent,
    FinishEvent,
    ToolCallEnd,
    UsageEvent,
    collect,
    collect_choices,
)
from ayaka.entrypoints.parsers.output import ChoicesParser, OutputFormat
from ayaka.entrypoints.stream import SlowClientError, collect_events, pump
from ayaka.tokenizers.contracts import EncodeOptions, TextInput, TokenIdsInput
from ayaka.utils.import_utils import has_module
from ayaka.utils.parsing import parse_json_strict

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Sequence

    from ayaka.configs.server import ServerConfig
    from ayaka.engine.async_llm import AsyncLLM, RequestStream
    from ayaka.entrypoints.conversation import (
        Collected,
        Conversation,
        OutputEvent,
        OutputRequest,
    )
    from ayaka.entrypoints.parsers.output import EventParser
    from ayaka.entrypoints.stream import StreamEncoder
    from ayaka.tokenizers.contracts import EncodeInput

__all__ = [
    "REQUEST_ID",
    "SERVING",
    "AnswerFailedError",
    "ApiFailure",
    "Generation",
    "JsonObject",
    "NotReadyError",
    "Readiness",
    "Serving",
    "ServingMetrics",
    "create_app",
    "failure_for",
]

_LOG = logging.getLogger("ayaka.serve")

#: Seconds a client is told to wait before retrying a 429 or a 503.
_RETRY_AFTER: Final = "1"


class Readiness(StrEnum):
    """Where the server is in its life."""

    STARTING = "starting"
    READY = "ready"
    DRAINING = "draining"
    DEAD = "dead"


class NotReadyError(EngineError):
    """The server is not serving generation requests right now."""

    code = "not_ready"
    retryable = True


class AnswerFailedError(Exception):
    """A request answered in one response ended with an error instead of an answer."""

    def __init__(self, event: ErrorEvent) -> None:
        super().__init__(event.message)
        self.event = event


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


# -- metrics ----------------------------------------------------------------------------------


class ServingMetrics:
    """Server metrics on a registry of their own; every label has a small, fixed set of values."""

    def __init__(self) -> None:
        self.registry = CollectorRegistry()
        self.http = Counter(
            "ayaka_http_requests",
            "HTTP requests by route and status",
            ["route", "status"],
            registry=self.registry,
        )
        self.finished = Counter(
            "ayaka_requests_finished",
            "Generation requests by how they ended",
            ["outcome"],
            registry=self.registry,
        )
        self.prompt_tokens = Counter(
            "ayaka_prompt_tokens", "Prompt tokens of finished requests", registry=self.registry
        )
        self.cached_tokens = Counter(
            "ayaka_prompt_cached_tokens",
            "Prompt tokens of finished requests whose KV came from the prefix cache",
            registry=self.registry,
        )
        self.generation_tokens = Counter(
            "ayaka_generation_tokens",
            "Tokens sent to clients by finished requests",
            registry=self.registry,
        )
        self.tool_calls = Counter(
            "ayaka_tool_calls",
            "Function calls the model wrote, by whether their arguments were one JSON object",
            ["valid"],
            registry=self.registry,
        )
        self.running = Gauge(
            "ayaka_requests_running", "Generation requests in flight", registry=self.registry
        )
        self.ready = Gauge(
            "ayaka_server_ready", "1 when the server accepts requests", registry=self.registry
        )
        self.duration = Histogram(
            "ayaka_request_seconds",
            "Time from a generation request's arrival to its last byte",
            buckets=(0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 60, 120, 300),
            registry=self.registry,
        )

    def observe(self, events: Sequence[OutputEvent]) -> None:
        """Count what a batch of events says."""
        for event in events:
            match event:
                case ToolCallEnd(_, valid) | ChoiceEvent(_, ToolCallEnd(_, valid)):
                    self.tool_calls.labels("true" if valid else "false").inc()
                case UsageEvent():
                    self.prompt_tokens.inc(event.prompt_tokens)
                    self.cached_tokens.inc(event.cached_tokens)
                    self.generation_tokens.inc(event.completion_tokens)
                case FinishEvent(reason):
                    self.finished.labels(str(reason)).inc()
                case ErrorEvent(code):
                    self.finished.labels(code if code in _OUTCOMES else "error").inc()
                case _:
                    pass


_OUTCOMES: Final = frozenset({"cancelled", "request_timeout", "engine_dead"})


# -- the shared state -------------------------------------------------------------------------


@dataclass(slots=True)
class Generation:
    """One admitted request: its outputs, how to read them, and what it said so far.

    Attributes:
        request_id: The engine id.
        public_id: The hex part clients see in protocol ids.
        prompt_tokens: Length of the prompt.
        stream: The engine's outputs.
        parser: The request's output parser.
        started: When it was admitted, on the monotonic clock.
        usage: The token counts, once reported.
        outcome: How it ended: a finish reason, an error code, ``slow_client`` or
            ``client_gone``; ``unfinished`` until then.
    """

    request_id: RequestId
    public_id: str
    prompt_tokens: int
    stream: RequestStream
    parser: EventParser
    started: float = field(default_factory=time.monotonic)
    usage: UsageEvent | None = None
    outcome: str = "unfinished"

    def observe(self, events: Sequence[OutputEvent]) -> None:
        """Remember the usage and the end from a batch of events."""
        for event in events:
            match event:
                case UsageEvent():
                    self.usage = event
                case FinishEvent(reason):
                    self.outcome = "tool_calls" if event.tool_calls else str(reason)
                case ErrorEvent(code):
                    self.outcome = code
                case _:
                    pass


class Serving:
    """What every handler shares.

    Args:
        config: The server settings.
        model_name: The name the model was loaded by; reported when ``served_model_names`` is
            empty.
    """

    def __init__(self, config: ServerConfig, *, model_name: str) -> None:
        self.config = config
        self.model_names: tuple[str, ...] = config.served_model_names or (model_name,)
        self.metrics = ServingMetrics()
        self.created = int(time.time())
        self.status = Readiness.STARTING
        self.llm: AsyncLLM | None = None
        self.output_format = OutputFormat()
        self.max_model_len = 0
        self.max_context_tokens = 0
        self.prefix_cache: bool | None = None

    @property
    def model_name(self) -> str:
        """The name responses report."""
        return self.model_names[0]

    # -- life cycle ------------------------------------------------------------------------

    def attach(self, llm: AsyncLLM, *, output_format: OutputFormat | None = None) -> None:
        """Start serving with a started engine."""
        self.llm = llm
        self.max_model_len = llm.engine.tokenizer.max_model_len
        self.max_context_tokens = min(self.max_model_len, llm.engine.kv.max_context_tokens)
        self.prefix_cache = llm.engine.kv.prefix_caching
        if output_format is not None:
            self.output_format = output_format
        self.status = Readiness.READY
        self.metrics.ready.set(1)

    def begin_draining(self) -> None:
        """Refuse new requests; those in flight go on."""
        if self.status is not Readiness.DEAD:
            self.status = Readiness.DRAINING
        self.metrics.ready.set(0)

    def require_ready(self) -> AsyncLLM:
        """The engine, if the server accepts requests now.

        Raises:
            NotReadyError: Otherwise, with the reason.
        """
        llm = self.llm
        if llm is not None and llm.is_dead:
            self.status = Readiness.DEAD
            self.metrics.ready.set(0)
        if self.status is not Readiness.READY or llm is None:
            raise NotReadyError(f"the server is {self.status.value}; retry later")
        return llm

    def check_model(self, name: str | None) -> None:
        """Accept a request for one of the served names (or none at all).

        Raises:
            HTTPNotFound: For any other name.
        """
        if name is None or name in self.model_names:
            return
        raise web.HTTPNotFound(
            text=f"model {name!r} is not served here; available: {', '.join(self.model_names)}"
        )

    # -- the generation path ---------------------------------------------------------------

    async def tokenize_chat(self, conversation: Conversation) -> tuple[int, ...]:
        """Render a conversation with the model's chat template and tokenize it.

        Raises:
            InvalidRequestError: If it cannot be rendered or does not fit the context.
            OverloadedError: If the tokenizer is busy.
        """
        return await self._encode(conversation.template_input())

    async def tokenize_text(self, prompt: str | tuple[int, ...]) -> tuple[int, ...]:
        """Tokenize a raw completion prompt (text, or ids to validate)."""
        value: EncodeInput = (
            TokenIdsInput(prompt) if isinstance(prompt, tuple) else TextInput(prompt)
        )
        return await self._encode(value)

    async def _encode(self, value: EncodeInput) -> tuple[int, ...]:
        llm = self.require_ready()
        service = llm.engine.tokenizer
        try:
            result = await asyncio.to_thread(
                service.encode, value, EncodeOptions(max_output_tokens=1)
            )
        except ValueError as exc:
            if "exceeds context capacity" in str(exc):
                raise InvalidRequestError(
                    f"the prompt does not fit the model context of {self.max_model_len} tokens"
                ) from exc
            raise InvalidRequestError(f"cannot use this prompt: {exc}") from exc
        except RuntimeError as exc:
            raise InvalidRequestError(f"cannot render this prompt: {exc}") from exc
        return result.token_ids

    def output_tokens(self, prompt_tokens: int, requested: int | None) -> int | None:
        """``max_tokens`` for the engine: the request's, capped by the room the context leaves.

        A request may not produce more than the context holds whatever it asks for, so asking for
        more is not an error; asking with no room left is.

        Raises:
            InvalidRequestError: If the prompt leaves no room for one token.
        """
        room = self.max_model_len - prompt_tokens
        if room < 1:
            raise InvalidRequestError(
                f"the prompt has {prompt_tokens} tokens and the model context is "
                f"{self.max_model_len}: no room is left for an answer"
            )
        if requested is None:
            return None
        if requested > room:
            _LOG.debug("max_tokens %d capped to the %d tokens the context leaves", requested, room)
        return min(requested, room)

    def constraint(
        self,
        output: OutputRequest,
        conversation: Conversation | None = None,
        *,
        tools: bool = False,
        parallel_tool_calls: bool = True,
    ) -> StructuredParams | None:
        """The output constraint a request asked for, with this model's syntax; ``None`` if none.

        A JSON answer is constrained from its first token, or from the end of the reasoning
        block when the model reasons. A request with tools gets a lazy tool-call constraint
        when the structured extra is installed and the model's opening marker is one token, so
        a call it writes is a valid call and free text stays free; a request that forces a
        call or asks for strict arguments needs that constraint and is refused without it.

        Raises:
            UnsupportedFeatureError: If the request needs a constraint this server cannot make.
        """
        fmt = self.output_format
        start_after = fmt.reasoning_markers[1].token_id if fmt.reasoning_markers else None
        if output.response is not None:
            schema = output.schema if output.response is ConstraintKind.JSON_SCHEMA else None
            return StructuredParams(kind=output.response, schema=schema, start_after=start_after)
        if not tools or conversation is None or not conversation.tools:
            return None
        markers = fmt.tool_markers
        if fmt.tool_calls is None or markers is None:
            return None  # start() refuses tools without a tool-call parser
        opening, closing = markers
        wanted = output.wants_constraint
        if not wanted and (opening.token_id is None or not has_module("llguidance")):
            # Lazy constraints only for a one-token marker: a spelled one may come in other pieces.
            return None  # free text, as without the structured extra
        spelled: tuple[int, ...] = ()
        tokenizer = self.llm.engine.tokenizer.tokenizer if self.llm is not None else None
        if opening.token_id is None and tokenizer is not None:
            spelled = tuple(tokenizer.encode(opening.text, add_special_tokens=False))
        return StructuredParams(
            kind=ConstraintKind.TOOL_CALL,
            tools=tuple(ToolSchema(tool.name, tool.parameters) for tool in conversation.tools),
            syntax=ToolCallSyntax(
                opening.text, closing.text, opening.token_id, closing.token_id, spelled
            ),
            tool_choice=output.tool_choice,
            strict=output.strict,
            parallel=parallel_tool_calls,
            start_after=start_after,
        )

    async def start(
        self,
        prompt: tuple[int, ...],
        sampling: SamplingParams,
        *,
        public_id: str,
        tools: bool,
        parallel_tool_calls: bool = True,
        structured: StructuredParams | None = None,
    ) -> Generation:
        """Admit a tokenized request; ``sampling.n`` choices get one output parser each.

        Raises:
            UnsupportedFeatureError, InvalidRequestError, OverloadedError: As the engine's
                admission.
            NotReadyError: If the server stopped accepting requests.
        """
        llm = self.require_ready()
        if tools and self.output_format.tool_calls is None:
            raise UnsupportedFeatureError(
                "tools: this server has no tool-call parser for the model; "
                "start it with --tool-call-parser",
                context={"features": ("tools",), "param": "tools"},
            )
        request_id = RequestId(f"ayaka-{public_id}")
        request = GenerateRequest(
            request_id=request_id, prompt=prompt, sampling=sampling, structured=structured
        )
        try:
            stream = await llm.add_request(request)
        except EngineError as exc:
            if self.status is not Readiness.READY or not llm.is_running:
                raise NotReadyError(f"the server is {self.status.value}; retry later") from exc
            raise
        parsers = [
            self.output_format.parser(
                call_id_prefix=f"call_{public_id[:24]}" + (f"_{index}" if sampling.n > 1 else ""),
                tools=tools,
                parallel_tool_calls=parallel_tool_calls,
            )
            for index in range(sampling.n)
        ]
        parser: EventParser = parsers[0] if sampling.n == 1 else ChoicesParser(parsers)
        self.metrics.running.inc()
        return Generation(request_id, public_id, len(prompt), stream, parser)

    async def stream(
        self, request: web.Request, generation: Generation, encoder: StreamEncoder
    ) -> web.StreamResponse:
        """Answer with server-sent events until the terminal event.

        A client that does not read within the deadline, or that went away, has its connection
        closed; :meth:`finish` then cancels the request in the engine.
        """
        config = self.config
        response = web.StreamResponse(
            headers={
                "Content-Type": "text/event-stream; charset=utf-8",
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
                "x-request-id": request[REQUEST_ID],
            }
        )
        await response.prepare(request)
        try:
            await pump(
                generation.stream,
                generation.parser,
                encoder,
                response,
                keepalive=config.keepalive_seconds,
                coalesce=config.sse_coalesce_ms / 1000,
                write_deadline=config.slow_client_seconds,
                observe=self._observer(generation),
            )
        except SlowClientError:
            generation.outcome = "slow_client"
            self.metrics.finished.labels("slow_client").inc()
            _LOG.info("request %s: the client read too slowly; cancelled", request[REQUEST_ID])
            if request.transport is not None:
                request.transport.close()
            return response
        except ConnectionError:
            generation.outcome = "client_gone"
            self.metrics.finished.labels("client_gone").inc()
            return response
        await response.write_eof()
        return response

    async def collect(self, generation: Generation) -> Collected:
        """The whole answer of a request answered in one response.

        Raises:
            AnswerFailedError: If the request ended with an error.
        """
        events = await collect_events(
            generation.stream, generation.parser, observe=self._observer(generation)
        )
        answer = collect(events)
        if answer.error is not None:
            raise AnswerFailedError(answer.error)
        return answer

    async def collect_choices(self, generation: Generation, count: int) -> list[Collected]:
        """Every choice of a request with ``n > 1``, answered in one response.

        Raises:
            AnswerFailedError: If the request ended with an error.
        """
        events = await collect_events(
            generation.stream, generation.parser, observe=self._observer(generation)
        )
        answers = collect_choices(events, count)
        for answer in answers:
            if answer.error is not None:
                raise AnswerFailedError(answer.error)
        return answers

    def _observer(self, generation: Generation) -> Callable[[Sequence[OutputEvent]], None]:
        def observe(events: Sequence[OutputEvent]) -> None:
            generation.observe(events)
            self.metrics.observe(events)

        return observe

    def finish(self, generation: Generation) -> None:
        """Release a generation: cancel it in the engine if it has not ended, and log it.

        The log line carries the request id, how it ended and token counts, never text.
        """
        if not generation.stream.finished and self.llm is not None:
            self.llm.abort(generation.request_id)
            if generation.outcome == "unfinished":
                generation.outcome = "client_gone"
        self.metrics.running.dec()
        seconds = time.monotonic() - generation.started
        self.metrics.duration.observe(seconds)
        usage = generation.usage
        _LOG.info(
            "request %s: %s, %d prompt and %d generated tokens in %.2f s",
            generation.public_id,
            generation.outcome,
            usage.prompt_tokens if usage is not None else generation.prompt_tokens,
            usage.completion_tokens if usage is not None else 0,
            seconds,
        )


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

SERVING: Final = web.AppKey("ayaka.serving", Serving)
"""The application key holding the :class:`Serving` state."""

REQUEST_ID: Final = web.RequestKey("ayaka.request_id", str)
"""The key of the request id in an aiohttp request."""

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
