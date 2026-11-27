"""Engine-free gRPC prompt rendering and incremental output parsing service.

Runs without ``torch``: loads only :class:`~ayaka.tokenizers.service.TokenizerService`,
:class:`~ayaka.entrypoints.render.selection.PromptRenderer`, and parser/grammar builders.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Final, cast

from ayaka.core.constraints import StructuredParams, ToolSchema
from ayaka.core.errors import (
    CapabilityError,
    InvalidRequestError,
    UnsupportedFeatureError,
)
from ayaka.core.outputs import CompletionOutput, RequestOutput, Usage
from ayaka.core.types import FinishReason, RequestId
from ayaka.entrypoints.conversation import ConversationBuilder, OutputRequest
from ayaka.entrypoints.grpc.errors import abort_with_error, error_from_rpc, error_info_from_wire
from ayaka.entrypoints.grpc.messages import (
    MAX_GRPC_MESSAGE_BYTES,
    decode_chat_event,
    decode_structured_params,
    deserialize_proto,
    encode_chat_event,
    encode_structured_params,
    grpc_options,
    serialize_proto,
)
from ayaka.entrypoints.parsers.output import OutputFormat
from ayaka.entrypoints.parsers.output_grammar import (
    OutputGrammarContext,
    ToolStrictLevel,
    build_tool_constraint,
)
from ayaka.entrypoints.parsers.parameters import ToolSchemas
from ayaka.entrypoints.parsers.unified import OutputInterpretation
from ayaka.entrypoints.render.reasoning import ThinkingRequest, parse_reasoning_effort
from ayaka.entrypoints.render.selection import PromptRenderer, RenderRequest
from ayaka.utils.import_utils import require_module

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Iterable

    from ayaka.entrypoints.conversation import Conversation, OutputEvent
    from ayaka.tokenizers.service import TokenizerService

__all__ = [
    "FINGERPRINT_METADATA_KEY",
    "RENDER_SERVICE_NAME",
    "RemoteRenderResult",
    "RenderGrpcClient",
    "RenderGrpcServer",
    "compute_render_fingerprint",
]

RENDER_SERVICE_NAME: Final = "ayaka.v1.RenderService"
FINGERPRINT_METADATA_KEY: Final = "x-render-fingerprint"
_LOG = logging.getLogger("ayaka.grpc.render")


def _grpc() -> Any:
    try:
        return require_module(
            "grpc",
            capability="grpc",
            remedy="install the grpc extra: uv sync --extra grpc",
        )
    except CapabilityError as exc:
        raise UnsupportedFeatureError(
            "gRPC render service needs the grpc extra (grpcio), which is not installed",
            context={"features": ("grpc",)},
        ) from exc


def compute_render_fingerprint(
    *,
    model_name: str,
    tokenizer: TokenizerService,
    reasoning_parser: str | None,
    tool_call_parser: str | None,
    tool_strict_level: str,
) -> str:
    """Compute a deterministic fingerprint of rendering template and parser settings."""
    payload = json.dumps(
        {
            "model": model_name,
            "chat_template": tokenizer.chat_template or "",
            "reasoning_parser": reasoning_parser or "",
            "tool_call_parser": tool_call_parser or "",
            "tool_strict_level": tool_strict_level,
        },
        separators=(",", ":"),
        ensure_ascii=False,
    )
    return hashlib.blake2b(payload.encode("utf-8"), digest_size=16).hexdigest()


@dataclass(frozen=True, slots=True)
class RemoteRenderResult:
    """Prompt token ids, synchronized constraint, and gate returned by :class:`RenderGrpcClient`."""

    input_ids: tuple[int, ...]
    structured: StructuredParams | None
    start_after: int | None
    reasoning_open: int | None
    thinking_enabled: bool
    force_reasoning: bool | None
    starts_inside_reasoning: bool
    previous_content: str
    stream_reasoning: bool
    render_fingerprint: str
    prefill_text: str


class RenderGrpcServer:
    """Async gRPC server exposing engine-free :data:`RENDER_SERVICE_NAME`."""

    def __init__(
        self,
        tokenizer: TokenizerService,
        *,
        model_name: str,
        reasoning_parser: str | None = None,
        tool_call_parser: str | None = None,
        tool_strict_level: str = "auto",
        target: str = "127.0.0.1:0",
        max_message_bytes: int = MAX_GRPC_MESSAGE_BYTES,
    ) -> None:
        self._tokenizer = tokenizer
        self._model_name = model_name
        self._renderer = PromptRenderer(tokenizer, model_name=model_name)
        self._output_format = OutputFormat.resolve(
            reasoning=reasoning_parser,
            tool_calls=tool_call_parser,
            tokenizer=tokenizer.tokenizer,
        )
        self._strict_level = ToolStrictLevel(tool_strict_level)
        self._fingerprint = compute_render_fingerprint(
            model_name=model_name,
            tokenizer=tokenizer,
            reasoning_parser=reasoning_parser,
            tool_call_parser=tool_call_parser,
            tool_strict_level=tool_strict_level,
        )
        self._target = target
        self._max_bytes = max_message_bytes
        self._server: Any = None
        self._bound_target = target
        self._port = 0

    @property
    def fingerprint(self) -> str:
        """Render configuration fingerprint expected in ``x-render-fingerprint`` metadata."""
        return self._fingerprint

    @property
    def target(self) -> str:
        """Bound gRPC address."""
        return self._bound_target

    @property
    def port(self) -> int:
        """Bound TCP port, or ``0`` before :meth:`start` / on UDS."""
        return self._port

    async def start(self) -> str:
        """Start the engine-free render gRPC server and return the reachable address."""
        grpc = _grpc()
        _ = require_module("grpc.aio", capability="grpc", remedy="uv sync --extra grpc")
        server = grpc.aio.server(options=grpc_options(self._max_bytes))
        handlers = {
            "RenderChat": grpc.unary_unary_rpc_method_handler(
                self._render_chat,
                request_deserializer=deserialize_proto,
                response_serializer=serialize_proto,
            ),
            "ParseOutput": grpc.stream_stream_rpc_method_handler(
                self._parse_output,
                request_deserializer=deserialize_proto,
                response_serializer=serialize_proto,
            ),
        }
        server.add_generic_rpc_handlers(
            (grpc.method_handlers_generic_handler(RENDER_SERVICE_NAME, handlers),)
        )
        bound = int(server.add_insecure_port(self._target))
        if bound == 0 and not self._target.startswith("unix:"):
            raise RuntimeError(f"failed to bind gRPC render server to {self._target!r}")
        await server.start()
        self._server = server
        if self._target.startswith("unix:"):
            self._port = 0
            self._bound_target = self._target
        else:
            host = self._target.rsplit(":", 1)[0]
            self._port = bound
            self._bound_target = f"{host}:{bound}"
        _LOG.info("gRPC render server listening on %s", self._bound_target)
        return self._bound_target

    async def shutdown(self, grace: float = 0.0) -> None:
        """Stop the render server cleanly."""
        if self._server is not None:
            await self._server.stop(grace=grace)
            self._server = None

    async def _render_chat(self, payload: dict[str, object], context: Any) -> dict[str, object]:
        meta = cast("dict[str, Any]", dict(context.invocation_metadata() or ()))
        client_fp = meta.get(FINGERPRINT_METADATA_KEY)
        if isinstance(client_fp, bytes):
            client_fp = client_fp.decode("utf-8", errors="replace")
        if client_fp is not None and client_fp != self._fingerprint:
            await abort_with_error(
                context,
                InvalidRequestError(
                    f"x-render-fingerprint mismatch: expected {self._fingerprint!r}, "
                    f"got {client_fp!r}"
                ),
            )
            return {}
        try:
            conversation = _conversation_from_wire(payload)
            effort_raw = payload.get("reasoning_effort")
            effort = (
                parse_reasoning_effort(str(effort_raw), where="reasoning_effort")
                if isinstance(effort_raw, str) and effort_raw
                else None
            )
            thinking_enabled = payload.get("thinking_enabled")
            thinking = ThinkingRequest(
                effort=effort,
                enabled=(thinking_enabled if isinstance(thinking_enabled, bool) else None),
            )
            kwargs_raw = payload.get("chat_template_kwargs")
            template_kwargs = (
                cast("dict[str, object]", kwargs_raw) if isinstance(kwargs_raw, dict) else {}
            )
            continue_final = bool(payload.get("continue_final_message", False))
            tool_choice = str(payload.get("tool_choice", "auto"))
            tools = tuple(conversation.tools)
            render_req = RenderRequest(
                conversation=conversation,
                thinking=thinking,
                chat_template_kwargs=template_kwargs,
                continue_final_message=continue_final,
                named_tool_choice=tool_choice not in ("auto", "required"),
            )
            rendered = self._renderer.render(render_req)
            start_after, reasoning_open = self._output_format.constraint_start(
                thinking_enabled=rendered.thinking_enabled,
                starts_inside_reasoning=rendered.starts_inside_reasoning,
            )
            structured = self._build_structured(
                payload,
                tools,
                tool_choice=tool_choice,
                start_after=start_after,
                reasoning_open=reasoning_open,
            )
        except Exception as exc:
            await abort_with_error(context, exc)
            return {}

        await context.send_initial_metadata(((FINGERPRINT_METADATA_KEY, self._fingerprint),))
        return {
            "input_ids": list(rendered.token_ids),
            "structured_output": (
                encode_structured_params(structured) if structured is not None else None
            ),
            "engine_reasoning_gate": {
                "start_after": start_after,
                "reasoning_open": reasoning_open,
                "thinking_enabled": rendered.thinking_enabled,
                "force_reasoning": rendered.force_reasoning,
                "starts_inside_reasoning": rendered.starts_inside_reasoning,
                "previous_content": rendered.previous_content,
                "stream_reasoning": rendered.stream_reasoning,
            },
            "render_fingerprint": self._fingerprint,
            "prefill_text": rendered.previous_content,
        }

    def _build_structured(
        self,
        payload: dict[str, object],
        tools: tuple[Any, ...],
        *,
        tool_choice: str,
        start_after: int | None,
        reasoning_open: int | None,
    ) -> StructuredParams | None:
        req_out = payload.get("requested_output")
        if isinstance(req_out, dict):
            parsed = decode_structured_params(cast("dict[str, object]", req_out))
            return StructuredParams(
                kind=parsed.kind,
                schema=parsed.schema,
                tools=parsed.tools,
                syntax=parsed.syntax,
                tool_choice=parsed.tool_choice,
                strict=parsed.strict,
                parallel=parsed.parallel,
                start_after=start_after,
                reasoning_open=reasoning_open,
                pattern=parsed.pattern,
                choices=parsed.choices,
                grammar=parsed.grammar,
                structural_tag=parsed.structural_tag,
            )
        if not tools or self._output_format.tool_calls is None:
            return None
        strict_tools = bool(payload.get("strict_tools", False))
        parallel = bool(payload.get("parallel_tool_calls", True))
        output_req = OutputRequest(tool_choice=tool_choice, strict=strict_tools)
        if not output_req.wants_constraint and self._output_format.tool_markers is not None:
            opening = self._output_format.tool_markers[0]
            if self._output_format.tool_calls.style == "json" and opening.token_id is None:
                return None
        spelled: tuple[int, ...] = ()
        if self._output_format.tool_markers is not None:
            opening = self._output_format.tool_markers[0]
            if opening.token_id is None and self._tokenizer.tokenizer is not None:
                spelled = tuple(
                    self._tokenizer.tokenizer.encode(opening.text, add_special_tokens=False)
                )
        ctx = OutputGrammarContext(
            output_format=self._output_format,
            tools=tuple(ToolSchema(t.name, t.parameters) for t in tools),
            tool_choice=tool_choice,
            strict_request=strict_tools,
            tool_strict_level=self._strict_level,
            parallel_tool_calls=parallel,
            start_after=start_after,
            reasoning_open=reasoning_open,
        )
        return build_tool_constraint(ctx, spelled_open=spelled)

    async def _parse_output(
        self, request_iterator: AsyncIterator[dict[str, object]], context: Any
    ) -> AsyncIterator[dict[str, object]]:
        _ = context
        processor = None
        async for chunk in request_iterator:
            if processor is None:
                gate_raw = cast("dict[str, object]", chunk.get("gate") or {})
                force_raw = gate_raw.get("force_reasoning")
                tools_raw = cast("list[dict[str, object]]", chunk.get("tools") or [])
                schemas = ToolSchemas.from_tools(tools_raw) if tools_raw else None
                interp = OutputInterpretation(
                    force_reasoning=force_raw if isinstance(force_raw, bool) else None,
                    previous_content=str(gate_raw.get("previous_content", "")),
                    stream_reasoning=bool(gate_raw.get("stream_reasoning", True)),
                    tools=bool(tools_raw),
                    parallel_tool_calls=bool(chunk.get("parallel_tool_calls", True)),
                )
                processor = self._output_format.processor(
                    interp,
                    call_id_prefix=str(chunk.get("call_id_prefix", "call_0")),
                    tool_schemas=schemas,
                )
            finished = bool(chunk.get("finished", False))
            reason_raw = chunk.get("finish_reason")
            reason = FinishReason(str(reason_raw)) if isinstance(reason_raw, str) else None
            usage_raw = chunk.get("usage")
            usage = (
                Usage(
                    prompt_tokens=int(cast("dict[str, Any]", usage_raw)["prompt_tokens"]),
                    completion_tokens=int(cast("dict[str, Any]", usage_raw)["completion_tokens"]),
                    cached_tokens=int(cast("dict[str, Any]", usage_raw).get("cached_tokens", 0)),
                    cache_written_tokens=int(
                        cast("dict[str, Any]", usage_raw).get("cache_written_tokens", 0)
                    ),
                )
                if isinstance(usage_raw, dict)
                else Usage()
            )
            err_raw = chunk.get("error")
            error = (
                error_info_from_wire(cast("dict[str, object]", err_raw))
                if isinstance(err_raw, dict)
                else None
            )
            ro = RequestOutput(
                request_id=RequestId("grpc-parse"),
                outputs=(
                    CompletionOutput(
                        index=0,
                        text=str(chunk.get("text", "")),
                        token_ids=tuple(
                            int(t) for t in cast("list[int]", chunk.get("token_ids") or [])
                        ),
                        finish_reason=reason,
                    ),
                ),
                finished=finished,
                usage=usage,
                error=error,
            )
            for ev in processor.feed(ro):
                yield encode_chat_event(ev)


class RenderGrpcClient:
    """Async gRPC client calling :class:`RenderGrpcServer`."""

    def __init__(self, target: str, *, max_message_bytes: int = MAX_GRPC_MESSAGE_BYTES) -> None:
        self._target = target
        self._max_bytes = max_message_bytes
        self._channel: Any = None
        self._render_rpc: Any = None
        self._parse_rpc: Any = None
        self._fingerprint: str | None = None

    async def connect(self) -> RenderGrpcClient:
        grpc = _grpc()
        _ = require_module("grpc.aio", capability="grpc", remedy="uv sync --extra grpc")
        channel = grpc.aio.insecure_channel(self._target, options=grpc_options(self._max_bytes))
        svc = RENDER_SERVICE_NAME
        self._channel = channel
        self._render_rpc = channel.unary_unary(
            f"/{svc}/RenderChat",
            request_serializer=serialize_proto,
            response_deserializer=deserialize_proto,
        )
        self._parse_rpc = channel.stream_stream(
            f"/{svc}/ParseOutput",
            request_serializer=serialize_proto,
            response_deserializer=deserialize_proto,
        )
        return self

    async def render_chat(
        self,
        conversation: Conversation,
        *,
        reasoning_effort: str | None = None,
        thinking_enabled: bool | None = None,
        chat_template_kwargs: dict[str, object] | None = None,
        continue_final_message: bool = False,
        tool_choice: str = "auto",
        strict_tools: bool = False,
        parallel_tool_calls: bool = True,
        expected_fingerprint: str | None = None,
    ) -> RemoteRenderResult:
        """Render and tokenize ``conversation`` on the remote render server."""
        if self._render_rpc is None:
            raise RuntimeError("RenderGrpcClient is not connected")
        grpc = _grpc()
        payload = _conversation_to_wire(conversation)
        payload["reasoning_effort"] = reasoning_effort or ""
        payload["thinking_enabled"] = thinking_enabled
        payload["chat_template_kwargs"] = chat_template_kwargs or {}
        payload["continue_final_message"] = continue_final_message
        payload["tool_choice"] = tool_choice
        payload["strict_tools"] = strict_tools
        payload["parallel_tool_calls"] = parallel_tool_calls
        fp = expected_fingerprint or self._fingerprint
        metadata = ((FINGERPRINT_METADATA_KEY, fp),) if fp is not None else ()
        try:
            resp = await self._render_rpc(payload, metadata=metadata)
        except grpc.aio.AioRpcError as exc:
            raise error_from_rpc(exc) from None
        gate = cast("dict[str, object]", resp["engine_reasoning_gate"])
        struct_raw = resp.get("structured_output")
        self._fingerprint = str(resp["render_fingerprint"])
        force_raw = gate.get("force_reasoning")
        start_after_raw = gate.get("start_after")
        open_raw = gate.get("reasoning_open")
        return RemoteRenderResult(
            input_ids=tuple(int(t) for t in cast("list[int]", resp["input_ids"])),
            structured=(
                decode_structured_params(cast("dict[str, object]", struct_raw))
                if isinstance(struct_raw, dict)
                else None
            ),
            start_after=int(start_after_raw) if isinstance(start_after_raw, int) else None,
            reasoning_open=int(open_raw) if isinstance(open_raw, int) else None,
            thinking_enabled=bool(gate["thinking_enabled"]),
            force_reasoning=force_raw if isinstance(force_raw, bool) else None,
            starts_inside_reasoning=bool(gate["starts_inside_reasoning"]),
            previous_content=str(gate.get("previous_content", "")),
            stream_reasoning=bool(gate.get("stream_reasoning", True)),
            render_fingerprint=self._fingerprint,
            prefill_text=str(resp.get("prefill_text", "")),
        )

    async def parse_output(self, chunks: Iterable[dict[str, object]]) -> AsyncIterator[OutputEvent]:
        """Stream token chunks to ``ParseOutput`` and yield decoded semantic events."""
        if self._parse_rpc is None:
            raise RuntimeError("RenderGrpcClient is not connected")

        async def _req_iter() -> AsyncIterator[dict[str, object]]:
            for chunk in chunks:
                yield chunk

        call = self._parse_rpc(_req_iter())
        async for raw_event in call:
            decoded = decode_chat_event(raw_event)
            if decoded is not None:
                yield decoded

    async def close(self) -> None:
        if self._channel is not None:
            await self._channel.close()
            self._channel = None


def _conversation_to_wire(conversation: Conversation) -> dict[str, object]:
    return {
        "system": conversation.system,
        "messages": [
            {
                "role": turn.role,
                "content": turn.text,
                "tool_calls": [
                    {"id": tc.id, "name": tc.name, "arguments": dict(tc.arguments)}
                    for tc in turn.tool_calls
                ],
                "tool_call_id": turn.tool_call_id,
                "name": turn.name,
                "reasoning": turn.reasoning,
            }
            for turn in conversation.turns
        ],
        "tools": [
            {"name": tool.name, "parameters": dict(tool.parameters)} for tool in conversation.tools
        ],
    }


def _conversation_from_wire(payload: dict[str, object]) -> Conversation:
    builder = ConversationBuilder()
    system = payload.get("system")
    if isinstance(system, str) and system:
        builder.system(system)
    for msg in cast("list[dict[str, object]]", payload.get("messages") or []):
        role = str(msg.get("role", "user"))
        content = str(msg.get("content", ""))
        if role in ("system", "developer"):
            builder.system(content)
        elif role == "user":
            builder.user(content)
        elif role == "assistant":
            from ayaka.entrypoints.conversation import ToolCall

            calls = [
                ToolCall(
                    id=str(c["id"]),
                    name=str(c["name"]),
                    arguments=cast("dict[str, object]", c.get("arguments") or {}),
                )
                for c in cast("list[dict[str, object]]", msg.get("tool_calls") or [])
            ]
            reason = msg.get("reasoning")
            builder.assistant(
                content,
                tool_calls=calls,
                reasoning=reason if isinstance(reason, str) else None,
            )
        elif role == "tool":
            builder.tool_result(
                str(msg.get("tool_call_id", "")),
                content,
            )
    for tool in cast("list[dict[str, object]]", payload.get("tools") or []):
        from ayaka.entrypoints.conversation import ToolSpec

        builder.tool(
            ToolSpec(
                name=str(tool["name"]),
                description=None,
                parameters=cast("dict[str, object]", tool.get("parameters") or {}),
            )
        )
    return builder.build()
