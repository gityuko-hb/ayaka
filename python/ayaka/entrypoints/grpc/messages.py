"""Protobuf-framed wire codecs matching ``proto/ayaka/v1/{inference,render}.proto``."""

from __future__ import annotations

import base64
import json
from typing import TYPE_CHECKING, Any, Final, cast

from ayaka.core.errors import CapabilityError, UnsupportedFeatureError
from ayaka.core.multimodal import ImageInput, ImageSpan, VisionSpec
from ayaka.core.outputs import CompletionOutput, RequestMetrics, RequestOutput, Usage
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
from ayaka.entrypoints.grpc._codecs import (
    as_float,
    as_int,
    as_opt_float,
    as_opt_int,
    decode_logprob,
    decode_sampling,
    decode_structured,
    encode_logprob,
    encode_sampling,
    encode_structured,
)
from ayaka.entrypoints.grpc.errors import error_info_from_wire, error_info_to_wire
from ayaka.utils.import_utils import require_module

if TYPE_CHECKING:
    from ayaka.core.constraints import StructuredParams
    from ayaka.core.params import SamplingParams
    from ayaka.entrypoints.conversation import OutputEvent

__all__ = [
    "MAX_GRPC_MESSAGE_BYTES",
    "decode_chat_event",
    "decode_engine_health",
    "decode_engine_info",
    "decode_generate_request",
    "decode_image_input",
    "decode_image_span",
    "decode_request_output",
    "decode_structured_params",
    "decode_vision_spec",
    "deserialize_proto",
    "encode_chat_event",
    "encode_engine_health",
    "encode_engine_info",
    "encode_generate_request",
    "encode_image_input",
    "encode_image_span",
    "encode_request_output",
    "encode_structured_params",
    "encode_vision_spec",
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
    return encode_sampling(params)


def decode_sampling_params(data: dict[str, object]) -> SamplingParams:
    """Deserialize :class:`~ayaka.core.params.SamplingParams`."""
    return decode_sampling(data)


def encode_structured_params(params: StructuredParams) -> dict[str, object]:
    """Serialize all seven :class:`~ayaka.core.constraints.ConstraintKind` specs."""
    return encode_structured(params)


def decode_structured_params(data: dict[str, object]) -> StructuredParams:
    """Deserialize :class:`~ayaka.core.constraints.StructuredParams` preserving JSON key order."""
    return decode_structured(data)


def encode_image_input(image: ImageInput) -> dict[str, object]:
    """Serialize :class:`~ayaka.core.multimodal.ImageInput`."""
    return {
        "data": base64.b64encode(image.data).decode("ascii"),
        "media_type": image.media_type,
    }


def decode_image_input(data: dict[str, object]) -> ImageInput:
    """Deserialize :class:`~ayaka.core.multimodal.ImageInput`."""
    raw_bytes = base64.b64decode(str(data["data"]).encode("ascii"))
    return ImageInput(
        data=raw_bytes,
        media_type=str(data.get("media_type", "image/png")),
    )


def encode_image_span(span: ImageSpan) -> dict[str, object]:
    """Serialize :class:`~ayaka.core.multimodal.ImageSpan`."""
    return {
        "image_index": span.image_index,
        "token_offset": span.token_offset,
        "token_count": span.token_count,
        "grid_thw": list(span.grid_thw),
    }


def decode_image_span(data: dict[str, object]) -> ImageSpan:
    """Deserialize :class:`~ayaka.core.multimodal.ImageSpan`."""
    grid_raw = cast("list[int]", data.get("grid_thw") or [1, 1, 1])
    if len(grid_raw) != 3:
        raise ValueError(f"image span grid_thw must have 3 elements, got {len(grid_raw)}")
    return ImageSpan(
        image_index=as_int(data.get("image_index"), 0),
        token_offset=as_int(data.get("token_offset"), 0),
        token_count=as_int(data.get("token_count"), 1),
        grid_thw=(int(grid_raw[0]), int(grid_raw[1]), int(grid_raw[2])),
    )


def encode_vision_spec(spec: VisionSpec) -> dict[str, object]:
    """Serialize :class:`~ayaka.core.multimodal.VisionSpec`."""
    return {
        "family": spec.family,
        "image_token_id": spec.image_token_id,
        "boi_token_id": spec.boi_token_id,
        "eoi_token_id": spec.eoi_token_id,
        "patch_size": spec.patch_size,
        "temporal_patch_size": spec.temporal_patch_size,
        "spatial_merge_size": spec.spatial_merge_size,
        "min_pixels": spec.min_pixels,
        "max_pixels": spec.max_pixels,
        "max_soft_tokens": spec.max_soft_tokens,
        "mrope_section": list(spec.mrope_section),
    }


def decode_vision_spec(data: dict[str, object]) -> VisionSpec:
    """Deserialize :class:`~ayaka.core.multimodal.VisionSpec`."""
    return VisionSpec(
        family=str(data["family"]),
        image_token_id=as_int(data.get("image_token_id"), 0),
        boi_token_id=as_opt_int(data.get("boi_token_id")),
        eoi_token_id=as_opt_int(data.get("eoi_token_id")),
        patch_size=as_int(data.get("patch_size"), 16),
        temporal_patch_size=as_int(data.get("temporal_patch_size"), 2),
        spatial_merge_size=as_int(data.get("spatial_merge_size"), 2),
        min_pixels=as_int(data.get("min_pixels"), 65536),
        max_pixels=as_int(data.get("max_pixels"), 16777216),
        max_soft_tokens=as_int(data.get("max_soft_tokens"), 280),
        mrope_section=tuple(int(v) for v in cast("list[int]", data.get("mrope_section") or [])),
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
        "images": [encode_image_input(img) for img in request.images],
        "image_spans": [encode_image_span(span) for span in request.image_spans],
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
    images = tuple(
        decode_image_input(item)
        for item in cast("list[dict[str, object]]", data.get("images") or [])
    )
    image_spans = tuple(
        decode_image_span(item)
        for item in cast("list[dict[str, object]]", data.get("image_spans") or [])
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
        priority=as_int(data.get("priority"), 0),
        cache_salt=str(data["cache_salt"]) if isinstance(data.get("cache_salt"), str) else None,
        prefix_cache=bool(data.get("prefix_cache", True)),
        images=images,
        image_spans=image_spans,
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
                "logprobs": [encode_logprob(lp) for lp in c.logprobs] if c.logprobs else None,
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
            index=as_int(item["index"], 0),
            text=str(item.get("text", "")),
            token_ids=tuple(int(t) for t in cast("list[int]", item.get("token_ids") or [])),
            finish_reason=(
                FinishReason(str(item["finish_reason"]))
                if isinstance(item.get("finish_reason"), str)
                else None
            ),
            stop_reason=cast("int | str | None", item.get("stop_reason")),
            cumulative_logprob=as_opt_float(item.get("cumulative_logprob")),
            logprobs=(
                tuple(
                    decode_logprob(lp) for lp in cast("list[dict[str, object]]", item["logprobs"])
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
            prompt_tokens=as_int(cast("dict[str, object]", usage_raw).get("prompt_tokens"), 0),
            completion_tokens=as_int(
                cast("dict[str, object]", usage_raw).get("completion_tokens"), 0
            ),
            cached_tokens=as_int(cast("dict[str, object]", usage_raw).get("cached_tokens"), 0),
            cache_written_tokens=as_int(
                cast("dict[str, object]", usage_raw).get("cache_written_tokens"), 0
            ),
        )
        if isinstance(usage_raw, dict)
        else Usage()
    )
    metrics_raw = data.get("metrics")
    metrics = (
        RequestMetrics(
            arrival_time=as_float(cast("dict[str, object]", metrics_raw).get("arrival_time"), 0.0),
            queued_time=as_float(cast("dict[str, object]", metrics_raw).get("queued_time"), 0.0),
            prefill_time=as_float(cast("dict[str, object]", metrics_raw).get("prefill_time"), 0.0),
            decode_time=as_float(cast("dict[str, object]", metrics_raw).get("decode_time"), 0.0),
            ttft=as_float(cast("dict[str, object]", metrics_raw).get("ttft"), 0.0),
            tpot=as_float(cast("dict[str, object]", metrics_raw).get("tpot"), 0.0),
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
        progress_at=as_float(data.get("progress_at"), 0.0),
        busy=bool(data.get("busy", False)),
    )


def encode_engine_info(info: EngineInfo) -> dict[str, object]:
    return {
        "max_model_len": info.max_model_len,
        "max_context_tokens": info.max_context_tokens,
        "prefix_caching": info.prefix_caching,
        "device": info.device,
        "dtype": info.dtype,
        "backends": list(info.backends),
        "cuda_graph": info.cuda_graph,
        "vision": encode_vision_spec(info.vision) if info.vision is not None else None,
    }


def decode_engine_info(data: dict[str, object]) -> EngineInfo:
    backends_raw = data.get("backends")
    backends = (
        tuple(str(b) for b in cast("list[object]", backends_raw))
        if isinstance(backends_raw, list)
        else ()
    )
    vision_raw = data.get("vision")
    vision = (
        decode_vision_spec(cast("dict[str, object]", vision_raw))
        if isinstance(vision_raw, dict)
        else None
    )
    return EngineInfo(
        max_model_len=as_int(data.get("max_model_len"), 0),
        max_context_tokens=as_int(data.get("max_context_tokens"), 0),
        prefix_caching=bool(data.get("prefix_caching", False)),
        device=str(data.get("device") or ""),
        dtype=str(data.get("dtype") or ""),
        backends=backends,
        cuda_graph=bool(data.get("cuda_graph", False)),
        vision=vision,
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
            index=as_int(data["index"], 0),
            id=str(data["id"]),
            name=str(data["name"]),
        )
    if kind == "tool_delta":
        return ToolCallDelta(index=as_int(data["index"], 0), arguments=str(data["arguments"]))
    if kind == "tool_end":
        return ToolCallEnd(index=as_int(data["index"], 0), valid=bool(data["valid"]))
    if kind == "usage":
        return UsageEvent(
            prompt_tokens=as_int(data["prompt_tokens"], 0),
            completion_tokens=as_int(data["completion_tokens"], 0),
            reasoning_tokens=as_int(data.get("reasoning_tokens"), 0),
            cached_tokens=as_int(data.get("cached_tokens"), 0),
        )
    if kind == "finish":
        return FinishEvent(
            reason=FinishReason(str(data["reason"])),
            tool_calls=bool(data.get("tool_calls", False)),
        )
    if kind == "error":
        return ErrorEvent(code=str(data["code"]), message=str(data["message"]))
    return None
