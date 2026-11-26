"""Protobuf-framed wire codecs matching ``proto/ayaka/v1/{inference,render}.proto``."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, Final, cast

from ayaka.core.constraints import (
    ConstraintKind,
    StructuralTagItem,
    StructuralTagSpec,
    StructuredParams,
    ToolCallSyntax,
    ToolSchema,
)
from ayaka.core.errors import CapabilityError, UnsupportedFeatureError
from ayaka.core.outputs import CompletionOutput, RequestMetrics, RequestOutput, TokenLogprob, Usage
from ayaka.core.params import LogprobMode, SamplingParams
from ayaka.core.request import GenerateRequest
from ayaka.core.types import FinishReason, RequestId
from ayaka.engine.seam import EngineHealth, EngineInfo, EngineState
from ayaka.entrypoints.conversation import (
    ErrorEvent,
    FinishEvent,
    ReasoningDelta,
    TextDelta,
    ToolCallDelta,
    ToolCallEnd,
    ToolCallStart,
    UsageEvent,
)
from ayaka.entrypoints.grpc.errors import error_info_from_wire, error_info_to_wire
from ayaka.utils.import_utils import require_module

if TYPE_CHECKING:
    from ayaka.entrypoints.conversation import OutputEvent

__all__ = [
    "MAX_GRPC_MESSAGE_BYTES",
    "decode_chat_event",
    "decode_engine_health",
    "decode_engine_info",
    "decode_generate_request",
    "decode_request_output",
    "decode_structured_params",
    "deserialize_proto",
    "encode_chat_event",
    "encode_engine_health",
    "encode_engine_info",
    "encode_generate_request",
    "encode_request_output",
    "encode_structured_params",
    "grpc_options",
    "serialize_proto",
]

#: Default 64 MiB limit on send and receive gRPC messages.
MAX_GRPC_MESSAGE_BYTES: Final[int] = 64 * 1024 * 1024


def _wrappers_pb2() -> Any:
    try:
        return require_module(
            "google.protobuf.wrappers_pb2",
            capability="grpc",
            remedy="install the grpc extra: uv sync --extra grpc",
        )
    except CapabilityError as exc:
        raise UnsupportedFeatureError(
            "gRPC transport needs the grpc extra (grpcio and protobuf), which is not installed",
            context={"features": ("grpc",)},
        ) from exc


def grpc_options(max_bytes: int = MAX_GRPC_MESSAGE_BYTES) -> tuple[tuple[str, int], ...]:
    """Channel and server gRPC options configuring the 64 MiB message ceiling."""
    return (
        ("grpc.max_send_message_length", max_bytes),
        ("grpc.max_receive_message_length", max_bytes),
    )


def serialize_proto(payload: dict[str, object]) -> bytes:
    """Encode ``payload`` into a protobuf ``google.protobuf.BytesValue`` frame."""
    wrappers = _wrappers_pb2()
    raw = json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    msg = wrappers.BytesValue(value=raw)
    return cast("bytes", msg.SerializeToString())


def deserialize_proto(raw: bytes) -> dict[str, object]:
    """Decode a protobuf ``google.protobuf.BytesValue`` frame into a mapping."""
    wrappers = _wrappers_pb2()
    msg = wrappers.BytesValue()
    msg.ParseFromString(raw)
    decoded = json.loads(bytes(msg.value).decode("utf-8"))
    if not isinstance(decoded, dict):
        raise ValueError("expected a JSON object inside protobuf frame")
    return cast("dict[str, object]", decoded)


def encode_sampling_params(params: SamplingParams) -> dict[str, object]:
    """Serialize :class:`~ayaka.core.params.SamplingParams`."""
    bias_pairs: list[tuple[int, float]] = (
        list(params.logit_bias.items())
        if isinstance(params.logit_bias, dict)
        else list(cast("tuple[tuple[int, float], ...]", params.logit_bias or ()))
    )
    return {
        "n": params.n,
        "temperature": params.temperature,
        "top_p": params.top_p,
        "top_k": params.top_k,
        "min_p": params.min_p,
        "repetition_penalty": params.repetition_penalty,
        "frequency_penalty": params.frequency_penalty,
        "presence_penalty": params.presence_penalty,
        "max_tokens": params.max_tokens,
        "min_tokens": params.min_tokens,
        "ignore_eos": params.ignore_eos,
        "stop": list(params.stop),
        "stop_token_ids": sorted(params.stop_token_ids),
        "include_stop_str_in_output": params.include_stop_str_in_output,
        "seed": params.seed,
        "allowed_token_ids": (
            sorted(params.allowed_token_ids) if params.allowed_token_ids is not None else None
        ),
        "logit_bias": [[int(t), float(b)] for t, b in bias_pairs],
        "logprobs": params.logprobs,
        "prompt_logprobs": params.prompt_logprobs,
        "logprob_token_ids": (
            list(params.logprob_token_ids) if params.logprob_token_ids is not None else None
        ),
        "logprob_mode": str(params.logprob_mode),
        "return_sampling_support": params.return_sampling_support,
    }


def decode_sampling_params(data: dict[str, object]) -> SamplingParams:
    """Deserialize :class:`~ayaka.core.params.SamplingParams`."""
    allowed_raw = data.get("allowed_token_ids")
    logprob_raw = data.get("logprob_token_ids")
    bias_raw = data.get("logit_bias")
    logit_bias: tuple[tuple[int, float], ...] | None = None
    if isinstance(bias_raw, list) and bias_raw:
        logit_bias = tuple(
            (int(pair[0]), float(pair[1])) for pair in cast("list[list[float]]", bias_raw)
        )
    return SamplingParams(
        n=_as_int(data.get("n"), 1),
        temperature=_as_float(data.get("temperature"), 1.0),
        top_p=_as_float(data.get("top_p"), 1.0),
        top_k=_as_int(data.get("top_k"), -1),
        min_p=_as_float(data.get("min_p"), 0.0),
        repetition_penalty=_as_float(data.get("repetition_penalty"), 1.0),
        frequency_penalty=_as_float(data.get("frequency_penalty"), 0.0),
        presence_penalty=_as_float(data.get("presence_penalty"), 0.0),
        max_tokens=_as_opt_int(data.get("max_tokens")),
        min_tokens=_as_int(data.get("min_tokens"), 0),
        ignore_eos=bool(data.get("ignore_eos", False)),
        stop=tuple(cast("list[str]", data.get("stop") or [])),
        stop_token_ids=frozenset(
            int(t) for t in cast("list[int]", data.get("stop_token_ids") or [])
        ),
        include_stop_str_in_output=bool(data.get("include_stop_str_in_output", False)),
        seed=_as_opt_int(data.get("seed")),
        allowed_token_ids=(
            frozenset(int(t) for t in cast("list[int]", allowed_raw))
            if isinstance(allowed_raw, list)
            else None
        ),
        logit_bias=logit_bias,
        logprobs=_as_opt_int(data.get("logprobs")),
        prompt_logprobs=_as_opt_int(data.get("prompt_logprobs")),
        logprob_token_ids=(
            tuple(int(t) for t in cast("list[int]", logprob_raw))
            if isinstance(logprob_raw, list)
            else None
        ),
        logprob_mode=LogprobMode(str(data.get("logprob_mode", LogprobMode.RAW))),
        return_sampling_support=bool(data.get("return_sampling_support", False)),
    )


def encode_structured_params(params: StructuredParams) -> dict[str, object]:
    """Serialize all seven :class:`~ayaka.core.constraints.ConstraintKind` specs."""
    syntax = (
        {
            "open": params.syntax.open,
            "close": params.syntax.close,
            "open_token": params.syntax.open_token,
            "close_token": params.syntax.close_token,
            "open_ids": list(params.syntax.open_ids),
            "arguments_key": params.syntax.arguments_key,
            "separator": params.syntax.separator,
        }
        if params.syntax is not None
        else None
    )
    st = (
        {
            "triggers": list(params.structural_tag.triggers),
            "at_least_one": params.structural_tag.at_least_one,
            "stop_after_first": params.structural_tag.stop_after_first,
            "tags": [
                {
                    "begin": item.begin,
                    "end": item.end,
                    "schema_json": (
                        json.dumps(item.schema, separators=(",", ":"), ensure_ascii=False)
                        if item.schema is not None
                        else None
                    ),
                    "grammar": item.grammar,
                }
                for item in params.structural_tag.tags
            ],
        }
        if params.structural_tag is not None
        else None
    )
    return {
        "kind": str(params.kind),
        "schema_json": (
            json.dumps(params.schema, separators=(",", ":"), ensure_ascii=False)
            if params.schema is not None
            else None
        ),
        "tools": [
            {
                "name": t.name,
                "parameters_json": json.dumps(
                    t.parameters, separators=(",", ":"), ensure_ascii=False
                ),
            }
            for t in params.tools
        ],
        "syntax": syntax,
        "tool_choice": params.tool_choice,
        "strict": params.strict,
        "parallel": params.parallel,
        "start_after": params.start_after,
        "reasoning_open": params.reasoning_open,
        "pattern": params.pattern,
        "choices": list(params.choices),
        "grammar": params.grammar,
        "structural_tag": st,
    }


def decode_structured_params(data: dict[str, object]) -> StructuredParams:
    """Deserialize :class:`~ayaka.core.constraints.StructuredParams` preserving JSON key order."""
    kind = ConstraintKind(str(data["kind"]))
    schema_raw = data.get("schema_json")
    schema = (
        cast("dict[str, object]", json.loads(schema_raw)) if isinstance(schema_raw, str) else None
    )
    tools = tuple(
        ToolSchema(
            name=str(item["name"]),
            parameters=cast("dict[str, object]", json.loads(str(item["parameters_json"]))),
        )
        for item in cast("list[dict[str, object]]", data.get("tools") or [])
    )
    syn_raw = data.get("syntax")
    syntax: ToolCallSyntax | None = None
    if isinstance(syn_raw, dict):
        smap = cast("dict[str, object]", syn_raw)
        syntax = ToolCallSyntax(
            open=str(smap.get("open", "")),
            close=str(smap.get("close", "")),
            open_token=_as_opt_int(smap.get("open_token")),
            close_token=_as_opt_int(smap.get("close_token")),
            open_ids=tuple(int(t) for t in cast("list[int]", smap.get("open_ids") or [])),
            arguments_key=str(smap.get("arguments_key", "arguments")),
            separator=str(smap.get("separator", "")),
        )
    st_raw = data.get("structural_tag")
    structural_tag: StructuralTagSpec | None = None
    if isinstance(st_raw, dict):
        st_map = cast("dict[str, object]", st_raw)
        items: list[StructuralTagItem] = []
        for entry in cast("list[dict[str, object]]", st_map.get("tags") or []):
            entry_schema = entry.get("schema_json")
            items.append(
                StructuralTagItem(
                    begin=str(entry["begin"]),
                    end=str(entry.get("end", "")),
                    schema=(
                        cast("dict[str, object]", json.loads(entry_schema))
                        if isinstance(entry_schema, str)
                        else None
                    ),
                    grammar=(
                        str(entry["grammar"]) if isinstance(entry.get("grammar"), str) else None
                    ),
                )
            )
        structural_tag = StructuralTagSpec(
            tags=tuple(items),
            triggers=tuple(cast("list[str]", st_map.get("triggers") or [])),
            at_least_one=bool(st_map.get("at_least_one", False)),
            stop_after_first=bool(st_map.get("stop_after_first", False)),
        )
    return StructuredParams(
        kind=kind,
        schema=schema,
        tools=tools,
        syntax=syntax,
        tool_choice=str(data.get("tool_choice", "auto")),
        strict=bool(data.get("strict", False)),
        parallel=bool(data.get("parallel", True)),
        start_after=_as_opt_int(data.get("start_after")),
        reasoning_open=_as_opt_int(data.get("reasoning_open")),
        pattern=str(data["pattern"]) if isinstance(data.get("pattern"), str) else None,
        choices=tuple(cast("list[str]", data.get("choices") or [])),
        grammar=str(data["grammar"]) if isinstance(data.get("grammar"), str) else None,
        structural_tag=structural_tag,
    )


def encode_generate_request(request: GenerateRequest) -> dict[str, object]:
    """Serialize :class:`~ayaka.core.request.GenerateRequest`."""
    prompt = request.prompt
    return {
        "request_id": str(request.request_id),
        "prompt_ids": list(prompt) if isinstance(prompt, tuple) else None,
        "prompt_text": prompt if isinstance(prompt, str) else None,
        "sampling": encode_sampling_params(request.sampling),
        "structured": (
            encode_structured_params(request.structured) if request.structured is not None else None
        ),
        "client_request_id": request.client_request_id,
        "priority": request.priority,
        "cache_salt": request.cache_salt,
        "prefix_cache": request.prefix_cache,
    }


def decode_generate_request(data: dict[str, object]) -> GenerateRequest:
    """Deserialize :class:`~ayaka.core.request.GenerateRequest`."""
    prompt_ids = data.get("prompt_ids")
    prompt: tuple[int, ...] | str = (
        tuple(int(t) for t in cast("list[int]", prompt_ids))
        if isinstance(prompt_ids, list)
        else str(data.get("prompt_text") or "")
    )
    struct_raw = data.get("structured")
    structured = (
        decode_structured_params(cast("dict[str, object]", struct_raw))
        if isinstance(struct_raw, dict)
        else None
    )
    return GenerateRequest(
        request_id=RequestId(str(data["request_id"])),
        prompt=prompt,
        sampling=decode_sampling_params(cast("dict[str, object]", data["sampling"])),
        structured=structured,
        client_request_id=(
            str(data["client_request_id"])
            if isinstance(data.get("client_request_id"), str)
            else None
        ),
        priority=_as_int(data.get("priority"), 0),
        cache_salt=str(data["cache_salt"]) if isinstance(data.get("cache_salt"), str) else None,
        prefix_cache=bool(data.get("prefix_cache", True)),
    )


def _encode_logprob(lp: TokenLogprob) -> dict[str, object]:
    return {
        "token_id": lp.token_id,
        "logprob": lp.logprob,
        "rank": lp.rank,
        "decoded": lp.decoded,
        "top": [[tid, val] for tid, val in lp.top],
        "top_decoded": list(lp.top_decoded),
    }


def _decode_logprob(data: dict[str, object]) -> TokenLogprob:
    decoded_raw = data.get("decoded")
    top_raw = cast("list[list[float]]", data.get("top") or [])
    return TokenLogprob(
        token_id=_as_int(data["token_id"], 0),
        logprob=_as_float(data["logprob"], 0.0),
        rank=_as_opt_int(data.get("rank")),
        decoded=str(decoded_raw) if isinstance(decoded_raw, str) else None,
        top=tuple((int(pair[0]), float(pair[1])) for pair in top_raw),
        top_decoded=tuple(cast("list[str]", data.get("top_decoded") or [])),
    )


def encode_request_output(output: RequestOutput) -> dict[str, object]:
    """Serialize :class:`~ayaka.core.outputs.RequestOutput`."""
    return {
        "request_id": str(output.request_id),
        "prompt_token_ids": (
            list(output.prompt_token_ids) if output.prompt_token_ids is not None else None
        ),
        "outputs": [
            {
                "index": c.index,
                "text": c.text,
                "token_ids": list(c.token_ids),
                "finish_reason": str(c.finish_reason) if c.finish_reason is not None else None,
                "stop_reason": c.stop_reason,
                "cumulative_logprob": c.cumulative_logprob,
                "logprobs": [_encode_logprob(lp) for lp in c.logprobs] if c.logprobs else None,
            }
            for c in output.outputs
        ],
        "finished": output.finished,
        "usage": {
            "prompt_tokens": output.usage.prompt_tokens,
            "completion_tokens": output.usage.completion_tokens,
            "cached_tokens": output.usage.cached_tokens,
            "cache_written_tokens": output.usage.cache_written_tokens,
        },
        "metrics": (
            {
                "arrival_time": output.metrics.arrival_time,
                "queued_time": output.metrics.queued_time,
                "prefill_time": output.metrics.prefill_time,
                "decode_time": output.metrics.decode_time,
                "ttft": output.metrics.ttft,
                "tpot": output.metrics.tpot,
            }
            if output.metrics is not None
            else None
        ),
        "error": error_info_to_wire(output.error) if output.error is not None else None,
    }


def decode_request_output(data: dict[str, object]) -> RequestOutput:
    """Deserialize :class:`~ayaka.core.outputs.RequestOutput`."""
    prompt_raw = data.get("prompt_token_ids")
    outputs = tuple(
        CompletionOutput(
            index=_as_int(item["index"], 0),
            text=str(item.get("text", "")),
            token_ids=tuple(int(t) for t in cast("list[int]", item.get("token_ids") or [])),
            finish_reason=(
                FinishReason(str(item["finish_reason"]))
                if isinstance(item.get("finish_reason"), str)
                else None
            ),
            stop_reason=cast("int | str | None", item.get("stop_reason")),
            cumulative_logprob=_as_opt_float(item.get("cumulative_logprob")),
            logprobs=(
                tuple(
                    _decode_logprob(lp) for lp in cast("list[dict[str, object]]", item["logprobs"])
                )
                if isinstance(item.get("logprobs"), list)
                else None
            ),
        )
        for item in cast("list[dict[str, object]]", data.get("outputs") or [])
    )
    usage_raw = data.get("usage")
    usage = (
        Usage(
            prompt_tokens=_as_int(cast("dict[str, object]", usage_raw).get("prompt_tokens"), 0),
            completion_tokens=_as_int(
                cast("dict[str, object]", usage_raw).get("completion_tokens"), 0
            ),
            cached_tokens=_as_int(cast("dict[str, object]", usage_raw).get("cached_tokens"), 0),
            cache_written_tokens=_as_int(
                cast("dict[str, object]", usage_raw).get("cache_written_tokens"), 0
            ),
        )
        if isinstance(usage_raw, dict)
        else Usage()
    )
    metrics_raw = data.get("metrics")
    metrics = (
        RequestMetrics(
            arrival_time=_as_float(cast("dict[str, object]", metrics_raw).get("arrival_time"), 0.0),
            queued_time=_as_float(cast("dict[str, object]", metrics_raw).get("queued_time"), 0.0),
            prefill_time=_as_float(cast("dict[str, object]", metrics_raw).get("prefill_time"), 0.0),
            decode_time=_as_float(cast("dict[str, object]", metrics_raw).get("decode_time"), 0.0),
            ttft=_as_float(cast("dict[str, object]", metrics_raw).get("ttft"), 0.0),
            tpot=_as_float(cast("dict[str, object]", metrics_raw).get("tpot"), 0.0),
        )
        if isinstance(metrics_raw, dict)
        else None
    )
    err_raw = data.get("error")
    error = (
        error_info_from_wire(cast("dict[str, object]", err_raw))
        if isinstance(err_raw, dict)
        else None
    )
    return RequestOutput(
        request_id=RequestId(str(data["request_id"])),
        prompt_token_ids=(
            tuple(int(t) for t in cast("list[int]", prompt_raw))
            if isinstance(prompt_raw, list)
            else None
        ),
        outputs=outputs,
        finished=bool(data.get("finished", False)),
        usage=usage,
        metrics=metrics,
        error=error,
    )


def encode_engine_health(health: EngineHealth) -> dict[str, object]:
    return {"state": str(health.state), "progress_at": health.progress_at, "busy": health.busy}


def decode_engine_health(data: dict[str, object]) -> EngineHealth:
    return EngineHealth(
        state=EngineState(str(data["state"])),
        progress_at=_as_float(data.get("progress_at"), 0.0),
        busy=bool(data.get("busy", False)),
    )


def encode_engine_info(info: EngineInfo) -> dict[str, object]:
    return {
        "max_model_len": info.max_model_len,
        "max_context_tokens": info.max_context_tokens,
        "prefix_caching": info.prefix_caching,
    }


def decode_engine_info(data: dict[str, object]) -> EngineInfo:
    return EngineInfo(
        max_model_len=_as_int(data.get("max_model_len"), 0),
        max_context_tokens=_as_int(data.get("max_context_tokens"), 0),
        prefix_caching=bool(data.get("prefix_caching", False)),
    )


def encode_chat_event(event: OutputEvent) -> dict[str, object]:
    """Serialize a semantic parser :data:`~ayaka.entrypoints.conversation.OutputEvent`."""
    if isinstance(event, TextDelta):
        return {"kind": "text", "text": event.text}
    if isinstance(event, ReasoningDelta):
        return {"kind": "reasoning", "text": event.text}
    if isinstance(event, ToolCallStart):
        return {"kind": "tool_start", "index": event.index, "id": event.id, "name": event.name}
    if isinstance(event, ToolCallDelta):
        return {"kind": "tool_delta", "index": event.index, "arguments": event.arguments}
    if isinstance(event, ToolCallEnd):
        return {"kind": "tool_end", "index": event.index, "valid": event.valid}
    if isinstance(event, UsageEvent):
        return {
            "kind": "usage",
            "prompt_tokens": event.prompt_tokens,
            "completion_tokens": event.completion_tokens,
            "reasoning_tokens": event.reasoning_tokens,
            "cached_tokens": event.cached_tokens,
        }
    if isinstance(event, FinishEvent):
        return {
            "kind": "finish",
            "reason": str(event.reason),
            "tool_calls": event.tool_calls,
        }
    if isinstance(event, ErrorEvent):
        return {"kind": "error", "code": event.code, "message": event.message}
    return {"kind": "other"}


def decode_chat_event(data: dict[str, object]) -> OutputEvent | None:
    """Deserialize a semantic :data:`~ayaka.entrypoints.conversation.OutputEvent`."""
    kind = str(data.get("kind", ""))
    if kind == "text":
        return TextDelta(str(data["text"]))
    if kind == "reasoning":
        return ReasoningDelta(str(data["text"]))
    if kind == "tool_start":
        return ToolCallStart(
            index=_as_int(data["index"], 0),
            id=str(data["id"]),
            name=str(data["name"]),
        )
    if kind == "tool_delta":
        return ToolCallDelta(index=_as_int(data["index"], 0), arguments=str(data["arguments"]))
    if kind == "tool_end":
        return ToolCallEnd(index=_as_int(data["index"], 0), valid=bool(data["valid"]))
    if kind == "usage":
        return UsageEvent(
            prompt_tokens=_as_int(data["prompt_tokens"], 0),
            completion_tokens=_as_int(data["completion_tokens"], 0),
            reasoning_tokens=_as_int(data.get("reasoning_tokens"), 0),
            cached_tokens=_as_int(data.get("cached_tokens"), 0),
        )
    if kind == "finish":
        return FinishEvent(
            reason=FinishReason(str(data["reason"])),
            tool_calls=bool(data.get("tool_calls", False)),
        )
    if kind == "error":
        return ErrorEvent(code=str(data["code"]), message=str(data["message"]))
    return None


def _as_int(value: object, default: int) -> int:
    return (
        int(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else default
    )


def _as_opt_int(value: object) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    return int(value) if isinstance(value, (int, float)) else None


def _as_float(value: object, default: float) -> float:
    return (
        float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else default
    )


def _as_opt_float(value: object) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    return float(value) if isinstance(value, (int, float)) else None
