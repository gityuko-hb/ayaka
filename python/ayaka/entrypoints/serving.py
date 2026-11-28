"""The state every HTTP handler shares and the generation path they all take.

:class:`Serving` holds the engine (an :class:`~ayaka.engine.seam.EngineClient`: the server reads
the engine through that contract and nothing else), the tokenizer, the model's output formats, its
names and the metrics. A handler reads its protocol's request into a
:class:`~ayaka.entrypoints.conversation.Conversation`, and then goes through the same steps::

    normalized -> tokenize_chat -> output_tokens -> constraint -> start
        -> stream | collect -> finish

Until :meth:`Serving.attach` is called every generation route answers 503 and ``/health`` says
``starting``. Once ready, ``/health`` says ``stalled`` while the engine has work and has made no
progress for ``stall_seconds`` (:meth:`Serving.stalled`); the server keeps serving, it only
reports. :meth:`Serving.shutdown` drains the engine and then cuts the streams whose client
stopped reading, so they cannot hold the process.

The default log level records request ids, token counts and timings, never prompt or output text.

This module imports aiohttp, from the ``serve`` extra; ``import ayaka`` does not load it.
"""

from __future__ import annotations

import asyncio
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, replace
from enum import StrEnum
from typing import TYPE_CHECKING, Final

from aiohttp import web

from ayaka.core.constraints import ConstraintKind, StructuredParams, ToolSchema
from ayaka.core.errors import (
    ContextLengthError,
    EngineError,
    InvalidRequestError,
    UnsupportedFeatureError,
)
from ayaka.core.request import GenerateRequest
from ayaka.core.types import RequestId
from ayaka.engine.seam import EngineState
from ayaka.entrypoints.conversation import (
    Conversation,
    ErrorEvent,
    FinishEvent,
    UsageEvent,
)
from ayaka.entrypoints.delivery import ResponseDelivery
from ayaka.entrypoints.failures import NotReadyError
from ayaka.entrypoints.metrics import ServingMetrics
from ayaka.entrypoints.normalize import Normalizer, identify_client
from ayaka.entrypoints.parsers.formats import suggest_tool_call_parser
from ayaka.entrypoints.parsers.output import ChoicesParser, OutputFormat
from ayaka.entrypoints.parsers.output_grammar import (
    OutputGrammarContext,
    ToolStrictLevel,
    build_tool_constraint,
)
from ayaka.entrypoints.parsers.parameters import ToolSchemas
from ayaka.entrypoints.parsers.unified import OutputInterpretation
from ayaka.entrypoints.render.selection import PromptRenderer
from ayaka.multimodal.tokens import expand_image_placeholders
from ayaka.tokenizers.contracts import EncodeOptions, TextInput, TokenIdsInput
from ayaka.utils.import_utils import has_module

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

    from ayaka.configs.server import ServerConfig
    from ayaka.configs.tokenizer import EncodeResult
    from ayaka.core.multimodal import ImageInput, ImageSpan, VisionSpec
    from ayaka.core.params import SamplingParams
    from ayaka.engine.async_llm import RequestStream
    from ayaka.engine.seam import EngineClient
    from ayaka.entrypoints.conversation import Collected, OutputEvent, OutputRequest
    from ayaka.entrypoints.parsers.output import EventParser
    from ayaka.entrypoints.render.selection import RenderedChat, RenderRequest
    from ayaka.entrypoints.stream import StreamEncoder
    from ayaka.tokenizers.contracts import EncodeInput
    from ayaka.tokenizers.service import TokenizerService


__all__ = ["REQUEST_ID", "SERVING", "Generation", "Readiness", "Serving"]

_LOG = logging.getLogger("ayaka.serve")

#: Seconds the event streams get at shutdown to write their last frames, once the engine has
#: ended every request; a stream still writing after that has a client that stopped reading.
_FLUSH_SECONDS: Final = 1.0


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
        self.llm: EngineClient | None = None
        self.tokenizer: TokenizerService | None = None
        self.renderer: PromptRenderer | None = None
        self.output_format = OutputFormat()
        self.max_model_len = 0
        self.max_context_tokens = 0
        self.prefix_cache: bool | None = None
        self.vision_spec: VisionSpec | None = None
        self.suggested_tool_call_parser: str | None = None
        # Rendering and encoding run on this one thread, in arrival order: threads of their own
        # would only take turns at the GIL (and the tokenizer's lock) with each other and with
        # the event loop.
        self._preprocess = ThreadPoolExecutor(max_workers=1, thread_name_prefix="ayaka-preprocess")
        self._delivery = ResponseDelivery(config, self.metrics)
        self.watchdog = self._delivery.watchdog
        self.metrics.progress_age.set_function(self.progress_age)

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

    def attach(
        self,
        llm: EngineClient,
        tokenizer: TokenizerService,
        *,
        output_format: OutputFormat | None = None,
    ) -> None:
        """Start serving with a started engine and the tokenizer that makes its prompts."""
        self.llm = llm
        self.tokenizer = tokenizer
        info = llm.info
        self.max_model_len = info.max_model_len
        self.max_context_tokens = info.max_context_tokens
        self.prefix_cache = info.prefix_caching
        self.vision_spec: VisionSpec | None = info.vision
        self.renderer = PromptRenderer(
            tokenizer, model_name=self.model_name, vision_spec=self.vision_spec
        )
        self.suggested_tool_call_parser = suggest_tool_call_parser(tokenizer.chat_template)
        if output_format is not None:
            self.output_format = output_format
        self.status = Readiness.READY
        self.metrics.ready.set(1)

    def begin_draining(self) -> None:
        """Refuse new requests; those in flight go on."""
        if self.status is not Readiness.DEAD:
            self.status = Readiness.DRAINING
        self.metrics.ready.set(0)

    def progress_age(self) -> float:
        """Seconds since the engine last made progress while it had work; 0 when it has none."""
        llm = self.llm
        if llm is None:
            return 0.0
        health = llm.health()
        if not health.busy or health.state is not EngineState.RUNNING:
            return 0.0
        return max(0.0, time.monotonic() - health.progress_at)

    def stalled(self) -> bool:
        """Whether the engine has had work without progress for longer than ``stall_seconds``."""
        return self.progress_age() > self.config.stall_seconds

    async def shutdown(self, grace: float, *, flush: float = _FLUSH_SECONDS) -> int:
        """Stop serving: refuse new requests, drain the engine, then cut the stuck streams.

        The requests in flight get ``grace`` seconds to finish and are cancelled after that. The
        event streams then get ``flush`` seconds to write their last frames; a stream still open
        after that is blocked on a client that stopped reading, and its connection is cut.

        Returns:
            How many connections were cut.
        """
        self.begin_draining()
        if self.llm is not None:
            await self.llm.shutdown(grace)
        self._preprocess.shutdown(wait=False, cancel_futures=True)
        return await self._delivery.cut_stuck_streams(flush)

    def require_ready(self) -> EngineClient:
        """The engine, if the server accepts requests now.

        Raises:
            NotReadyError: Otherwise, with the reason.
        """
        llm = self.llm
        if llm is not None and llm.health().state is EngineState.DEAD:
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

    async def prepare_chat(
        self, conversation: Conversation, *, max_tokens: int | None = None, vouch: bool = True
    ) -> tuple[tuple[int, ...], tuple[ImageSpan, ...]]:
        """Render and tokenize ``conversation``, expanding any attached image placeholders."""
        if conversation.images and self.vision_spec is None:
            raise UnsupportedFeatureError(
                "image input requires a vision-capable model",
                context={"features": ("image_url",)},
            )
        token_ids = await self._encode(conversation.template_data, max_tokens, vouch=vouch)
        if not conversation.images:
            return token_ids, ()
        assert self.vision_spec is not None  # noqa: S101  # checked above
        expanded, spans = expand_image_placeholders(
            token_ids, conversation.images, self.vision_spec
        )
        wanted = max(max_tokens or 1, 1)
        if len(expanded) + wanted > self.max_model_len:
            raise ContextLengthError(
                prompt_tokens=len(expanded),
                output_tokens=wanted,
                max_context_tokens=self.max_model_len,
            )
        return expanded, spans

    async def tokenize_chat(
        self, conversation: Conversation, *, max_tokens: int | None = None, vouch: bool = True
    ) -> tuple[int, ...]:
        """Render a conversation with the model's chat template and tokenize it.

        Args:
            conversation: What to render.
            max_tokens: The output limit the client asked for; it is only reported in the
                error of a prompt that does not fit, since :meth:`output_tokens` caps it.
            vouch: The ids go to the engine next (``False`` when only their count is wanted):
                the tokenizer remembers it checked them, and the engine does not check every id
                again.

        Raises:
            ContextLengthError: If the prompt leaves no room for one output token.
            InvalidRequestError: If it cannot be rendered.
            OverloadedError: If the tokenizer is busy.
        """
        prompt, _ = await self.prepare_chat(conversation, max_tokens=max_tokens, vouch=vouch)
        return prompt

    async def render_chat(
        self,
        request: RenderRequest,
        *,
        max_tokens: int | None = None,
        vouch: bool = True,
    ) -> RenderedChat:
        """Render and tokenize a chat request with request-scoped thinking metadata."""
        self.require_ready()
        renderer = self.renderer
        assert renderer is not None  # noqa: S101  # attach() sets it with the engine
        try:
            return await asyncio.get_running_loop().run_in_executor(
                self._preprocess,
                lambda: renderer.render(request, max_tokens=max_tokens, vouch=vouch),
            )
        except ContextLengthError as exc:
            raise ContextLengthError(
                prompt_tokens=exc.prompt_tokens,
                output_tokens=max(max_tokens or 1, 1),
                max_context_tokens=exc.max_context_tokens,
            ) from None

    async def tokenize_text(
        self, prompt: str | tuple[int, ...], *, max_tokens: int | None = None
    ) -> tuple[int, ...]:
        """Tokenize a raw completion prompt (text, or ids to validate); as :meth:`tokenize_chat`."""

        def value() -> EncodeInput:
            return TokenIdsInput(prompt) if isinstance(prompt, tuple) else TextInput(prompt)

        return await self._encode(value, max_tokens, vouch=True)

    async def _encode(
        self, make: Callable[[], EncodeInput], max_tokens: int | None, *, vouch: bool
    ) -> tuple[int, ...]:
        """Build the input with ``make`` and encode it, both on the preprocess thread."""
        self.require_ready()
        service = self.tokenizer
        assert service is not None  # noqa: S101  # attach() sets it with the engine

        def run() -> EncodeResult:
            return service.encode(make(), EncodeOptions(max_output_tokens=1), vouch=vouch)

        try:
            result = await asyncio.get_running_loop().run_in_executor(self._preprocess, run)
        except ContextLengthError as exc:
            raise ContextLengthError(
                prompt_tokens=exc.prompt_tokens,
                output_tokens=max(max_tokens or 1, 1),
                max_context_tokens=exc.max_context_tokens,
            ) from None
        except ValueError as exc:
            raise InvalidRequestError(f"cannot use this prompt: {exc}") from exc
        except RuntimeError as exc:
            raise InvalidRequestError(f"cannot render this prompt: {exc}") from exc
        return result.token_ids

    def output_tokens(self, prompt_tokens: int, requested: int | None) -> int | None:
        """``max_tokens`` for the engine: the request's, capped by the room the context leaves.

        A request may not produce more than the context holds whatever it asks for, so asking for
        more is not an error; asking with no room left is.

        Raises:
            ContextLengthError: If the prompt leaves no room for one token.
        """
        room = self.max_model_len - prompt_tokens
        if room < 1:
            raise ContextLengthError(
                prompt_tokens=prompt_tokens,
                output_tokens=max(requested or 1, 1),
                max_context_tokens=self.max_model_len,
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
        rendered: RenderedChat | None = None,
    ) -> StructuredParams | None:
        """Build a request-synchronized :class:`StructuredParams`, or ``None`` if unconstrained."""
        fmt = self.output_format
        start_after, reasoning_open = fmt.constraint_start(
            thinking_enabled=rendered.thinking_enabled if rendered is not None else None,
            starts_inside_reasoning=(
                rendered.starts_inside_reasoning if rendered is not None else None
            ),
        )
        if output.response is not None:
            schema = output.schema if output.response is ConstraintKind.JSON_SCHEMA else None
            return StructuredParams(
                kind=output.response,
                schema=schema,
                pattern=output.pattern,
                choices=output.choices,
                grammar=output.grammar,
                structural_tag=output.structural_tag,
                start_after=start_after,
                reasoning_open=reasoning_open,
            )
        if not tools or conversation is None or not conversation.tools or fmt.tool_calls is None:
            return None
        markers, wanted = fmt.tool_markers, output.wants_constraint
        has_structured = has_module("llguidance") or has_module("xgrammar")
        spelled: tuple[int, ...] = ()
        if markers is None:
            if not wanted and not has_structured:
                return None
        else:
            opening = markers[0]
            if not wanted and (
                (fmt.tool_calls.style == "json" and opening.token_id is None) or not has_structured
            ):
                return None
            tokenizer = self.tokenizer.tokenizer if self.tokenizer is not None else None
            if opening.token_id is None and tokenizer is not None:
                spelled = tuple(tokenizer.encode(opening.text, add_special_tokens=False))
        ctx = OutputGrammarContext(
            output_format=fmt,
            tools=tuple(ToolSchema(tool.name, tool.parameters) for tool in conversation.tools),
            tool_choice=output.tool_choice,
            strict_request=output.strict,
            tool_strict_level=ToolStrictLevel(self.config.tool_strict_level),
            parallel_tool_calls=parallel_tool_calls,
            start_after=start_after,
            reasoning_open=reasoning_open,
        )
        try:
            return build_tool_constraint(ctx, spelled_open=spelled)
        except InvalidRequestError:
            if markers is None and not wanted:
                return None
            raise

    async def start(
        self,
        prompt: tuple[int, ...],
        sampling: SamplingParams,
        *,
        public_id: str,
        tools: bool,
        tools_in_prompt: bool = False,
        parallel_tool_calls: bool = True,
        structured: StructuredParams | None = None,
        images: tuple[ImageInput, ...] = (),
        image_spans: tuple[ImageSpan, ...] = (),
        rendered: RenderedChat | None = None,
        tool_schemas: ToolSchemas | None = None,
    ) -> Generation:
        """Admit a tokenized request; ``sampling.n`` choices get one output parser each.

        ``tools_in_prompt`` without ``tools`` is ``tool_choice: none``: the prompt shows the tools
        but the model may not call them, so the answer ends where it opens a call (see
        :meth:`_stop_at_tool_call`).

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
        if tools_in_prompt and not tools:
            sampling = self._stop_at_tool_call(sampling)
        request_id = RequestId(f"ayaka-{public_id}")
        request = GenerateRequest(
            request_id=request_id,
            prompt=prompt,
            sampling=sampling,
            structured=structured,
            images=images,
            image_spans=image_spans,
        )
        try:
            try:
                stream = await llm.add(request)
            except InvalidRequestError as exc:
                if structured is None or not _best_effort(structured):
                    raise
                # The request did not ask for this constraint: a tool schema the grammar cannot
                # compile leaves the answer free, as it was before the constraint existed.
                _LOG.info("request %s: no output constraint: %s", public_id, exc)
                request = replace(request, structured=None)
                stream = await llm.add(request)
        except EngineError as exc:
            if self.status is not Readiness.READY or llm.health().state is not EngineState.RUNNING:
                raise NotReadyError(f"the server is {self.status.value}; retry later") from exc
            raise
        if rendered is not None:
            resolved_schemas = (
                tool_schemas
                if tool_schemas is not None
                else (
                    ToolSchemas.from_tools(structured.tools)
                    if structured is not None and structured.tools
                    else None
                )
            )
            interpretation = OutputInterpretation(
                force_reasoning=rendered.force_reasoning,
                previous_content=rendered.previous_content,
                stream_reasoning=rendered.stream_reasoning,
                tools=tools,
                parallel_tool_calls=parallel_tool_calls,
            )
            parsers: list[EventParser] = [
                self.output_format.processor(
                    interpretation,
                    call_id_prefix=f"call_{public_id[:24]}"
                    + (f"_{index}" if sampling.n > 1 else ""),
                    tool_schemas=resolved_schemas,
                    withhold_call=tools_in_prompt and not tools,
                )
                for index in range(sampling.n)
            ]
        else:
            parsers = [
                self.output_format.parser(
                    call_id_prefix=f"call_{public_id[:24]}"
                    + (f"_{index}" if sampling.n > 1 else ""),
                    tools=tools,
                    parallel_tool_calls=parallel_tool_calls,
                )
                for index in range(sampling.n)
            ]
        parser: EventParser = parsers[0] if sampling.n == 1 else ChoicesParser(parsers)
        self.metrics.running.inc()
        return Generation(request_id, public_id, len(prompt), stream, parser)

    def _stop_at_tool_call(self, sampling: SamplingParams) -> SamplingParams:
        """Stop at the single-token opening call marker, keeping decode inside CUDA graphs."""
        markers = self.output_format.tool_markers
        opening = markers[0].token_id if markers is not None else None
        if opening is None:
            return sampling
        return replace(sampling, stop_token_ids=sampling.stop_token_ids | {opening})

    async def stream(
        self, request: web.Request, generation: Generation, encoder: StreamEncoder
    ) -> web.StreamResponse:
        """Answer with server-sent events until the terminal event.

        A client that stops taking what is written to it for ``slow_client_seconds``, or that went
        away, has its connection cut; :meth:`finish` then cancels the request in the engine.
        """
        return await self._delivery.stream(
            request, generation, encoder, request_id=request[REQUEST_ID]
        )

    async def collect(self, generation: Generation) -> Collected:
        """The whole answer of a request answered in one response.

        Raises:
            AnswerFailedError: If the request ended with an error.
        """
        return await self._delivery.collect(generation)

    async def collect_choices(self, generation: Generation, count: int) -> list[Collected]:
        """Every choice of a request with ``n > 1``, answered in one response.

        Raises:
            AnswerFailedError: If the request ended with an error.
        """
        return await self._delivery.collect_choices(generation, count)

    def finish(self, generation: Generation) -> None:
        """Release a generation: cancel it in the engine if it has not ended, and log it.

        The log line carries the request id, how it ended and token counts, never text.
        """
        self._delivery.finish(generation, self.llm)


SERVING: Final = web.AppKey("ayaka.serving", Serving)
"""The application key holding the :class:`Serving` state."""


REQUEST_ID: Final = web.RequestKey("ayaka.request_id", str)
"""The key of the request id in an aiohttp request."""


def _best_effort(structured: StructuredParams) -> bool:
    """Whether the server added this constraint on its own: an optional call without markers.

    The request asked for nothing a grammar must guarantee, so a grammar that cannot be built
    leaves the answer free instead of failing it.
    """
    syntax = structured.syntax
    return (
        structured.kind is ConstraintKind.TOOL_CALL
        and syntax is not None
        and syntax.bare
        and structured.tool_choice == "auto"
        and not structured.strict
    )
