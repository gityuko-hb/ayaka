"""The state every HTTP handler shares and the generation path they all take.

:class:`Serving` holds the engine (an :class:`~ayaka.engine.async_llm.AsyncLLM`), the model's
output formats, its names and the metrics. A handler reads its protocol's request into a
:class:`~ayaka.entrypoints.conversation.Conversation`, and then goes through the same steps::

    normalized -> tokenize_chat -> output_tokens -> constraint -> start
        -> stream | collect -> finish

Until :meth:`Serving.attach` is called every generation route answers 503 and ``/health`` says
``starting``.

The default log level records request ids, token counts and timings, never prompt or output text.

This module imports aiohttp, from the ``serve`` extra; ``import ayaka`` does not load it.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING, Final

from aiohttp import web

from ayaka.core.constraints import ConstraintKind, StructuredParams, ToolSchema
from ayaka.core.errors import EngineError, InvalidRequestError, UnsupportedFeatureError
from ayaka.core.request import GenerateRequest
from ayaka.core.types import RequestId
from ayaka.entrypoints.conversation import (
    Conversation,
    ErrorEvent,
    FinishEvent,
    UsageEvent,
    collect,
    collect_choices,
)
from ayaka.entrypoints.failures import AnswerFailedError, NotReadyError
from ayaka.entrypoints.metrics import ServingMetrics
from ayaka.entrypoints.normalize import Normalizer, identify_client
from ayaka.entrypoints.parsers.formats import suggest_tool_call_parser
from ayaka.entrypoints.parsers.output import ChoicesParser, OutputFormat
from ayaka.entrypoints.stream import SlowClientError, collect_events, pump
from ayaka.tokenizers.contracts import EncodeOptions, TextInput, TokenIdsInput
from ayaka.utils.import_utils import has_module

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

    from ayaka.configs.server import ServerConfig
    from ayaka.core.params import SamplingParams
    from ayaka.engine.async_llm import AsyncLLM, RequestStream
    from ayaka.entrypoints.conversation import Collected, OutputEvent, OutputRequest
    from ayaka.entrypoints.parsers.output import EventParser
    from ayaka.entrypoints.stream import StreamEncoder
    from ayaka.tokenizers.contracts import EncodeInput


__all__ = ["REQUEST_ID", "SERVING", "Generation", "Readiness", "Serving"]

_LOG = logging.getLogger("ayaka.serve")


class Readiness(StrEnum):
    """Where the server is in its life."""

    STARTING = "starting"
    READY = "ready"
    DRAINING = "draining"
    DEAD = "dead"


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
        self.normalizer = Normalizer(
            enabled=config.normalize_prompts, metrics=self.metrics.normalized
        )
        self.created = int(time.time())
        self.status = Readiness.STARTING
        self.llm: AsyncLLM | None = None
        self.output_format = OutputFormat()
        self.max_model_len = 0
        self.max_context_tokens = 0
        self.prefix_cache: bool | None = None
        self.suggested_tool_call_parser: str | None = None

    def normalized(
        self, conversation: Conversation, headers: Mapping[str, str], request_id: str = ""
    ) -> Conversation:
        """The conversation with the recognised client's churn rules applied.

        The rules run before tokenization, so the prefix cache sees the client's fixed prompt
        without the values that change per session or per version. A prompt from an unknown
        client, or with normalization off, comes back unchanged.
        """
        conversation, _ = self.normalizer.normalize(
            conversation, identify_client(headers), request_id=request_id
        )
        return conversation

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
        self.suggested_tool_call_parser = suggest_tool_call_parser(
            llm.engine.tokenizer.chat_template
        )
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
        block when the model reasons (:meth:`OutputFormat.constraint_start`: the parser and the
        grammar follow the same rule). A request with tools gets a lazy tool-call constraint
        when the structured extra is installed and the model's opening marker is one token, so
        a call it writes is a valid call and free text stays free; a request that forces a
        call or asks for strict arguments needs that constraint and is refused without it. A
        format without markers (``llama3_json``) has no marker a lazy constraint could start on:
        a forced call is constrained from the first token, strict arguments of an optional call
        are refused.

        Raises:
            UnsupportedFeatureError: If the request needs a constraint this server cannot make.
        """
        fmt = self.output_format
        start_after, reasoning_open = fmt.constraint_start()
        if output.response is not None:
            schema = output.schema if output.response is ConstraintKind.JSON_SCHEMA else None
            return StructuredParams(
                kind=output.response,
                schema=schema,
                start_after=start_after,
                reasoning_open=reasoning_open,
            )
        if not tools or conversation is None or not conversation.tools:
            return None
        if fmt.tool_calls is None:
            return None  # start() refuses tools without a tool-call parser
        markers = fmt.tool_markers
        wanted = output.wants_constraint
        spelled: tuple[int, ...] = ()
        if markers is None:
            # A format without markers (llama3_json) has no token a lazy grammar could start on:
            # only a forced call can be constrained, from the first token of the answer.
            if not wanted:
                return None  # free text; the parser finds the call
            if output.tool_choice == "auto":
                raise UnsupportedFeatureError(
                    f"strict: the tool-call parser {fmt.tool_calls.name!r} writes calls without "
                    "markers, so the arguments of an optional call cannot be constrained; force "
                    "the call with tool_choice",
                    context={"features": ("structured tool calls",), "param": "tools"},
                )
        else:
            opening = markers[0]
            if not wanted and (opening.token_id is None or not has_module("llguidance")):
                # Lazy constraints only for a one-token marker: a spelled one may come in other
                # pieces.
                return None  # free text, as without the structured extra
            tokenizer = self.llm.engine.tokenizer.tokenizer if self.llm is not None else None
            if opening.token_id is None and tokenizer is not None:
                spelled = tuple(tokenizer.encode(opening.text, add_special_tokens=False))
        return StructuredParams(
            kind=ConstraintKind.TOOL_CALL,
            tools=tuple(ToolSchema(tool.name, tool.parameters) for tool in conversation.tools),
            syntax=fmt.call_syntax(spelled),
            tool_choice=output.tool_choice,
            strict=output.strict,
            parallel=parallel_tool_calls,
            start_after=start_after,
            reasoning_open=reasoning_open,
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
            # A request that ended early (a client gone or too slow) has no usage event: count
            # what reached the parser. Tokens the engine made after that are not in it.
            usage.completion_tokens if usage is not None else generation.parser.generated_tokens,
            seconds,
        )


SERVING: Final = web.AppKey("ayaka.serving", Serving)
"""The application key holding the :class:`Serving` state."""


REQUEST_ID: Final = web.RequestKey("ayaka.request_id", str)
"""The key of the request id in an aiohttp request."""
