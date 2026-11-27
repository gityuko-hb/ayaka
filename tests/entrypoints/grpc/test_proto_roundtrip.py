"""Tests for protobuf frame round-trips, error mapping, 64 MiB limits, and lazy imports."""

from __future__ import annotations

import subprocess
import sys

import pytest

from ayaka.core.constraints import (
    ConstraintKind,
    StructuralTagItem,
    StructuralTagSpec,
    StructuredParams,
    ToolCallSyntax,
    ToolSchema,
)
from ayaka.core.errors import (
    ContextLengthError,
    EngineDeadError,
    ErrorInfo,
    InvalidRequestError,
    OverloadedError,
    UnsupportedFeatureError,
)
from ayaka.core.outputs import CompletionOutput, RequestMetrics, RequestOutput, TokenLogprob, Usage
from ayaka.core.params import LogprobMode, SamplingParams
from ayaka.core.request import GenerateRequest
from ayaka.core.types import FinishReason, RequestId
from ayaka.engine.seam import EngineHealth, EngineInfo, EngineState
from ayaka.entrypoints.conversation import (
    FinishEvent,
    ReasoningDelta,
    TextDelta,
    ToolCallDelta,
    ToolCallEnd,
    ToolCallStart,
    UsageEvent,
)
from ayaka.entrypoints.grpc.errors import error_from_wire, error_to_wire
from ayaka.entrypoints.grpc.messages import (
    MAX_GRPC_MESSAGE_BYTES,
    decode_chat_event,
    decode_engine_health,
    decode_engine_info,
    decode_generate_request,
    decode_request_output,
    deserialize_proto,
    encode_chat_event,
    encode_engine_health,
    encode_engine_info,
    encode_generate_request,
    encode_request_output,
    grpc_options,
    serialize_proto,
)

pytestmark = pytest.mark.unit


def test_grpc_options_default_to_64_mib() -> None:
    assert MAX_GRPC_MESSAGE_BYTES == 64 * 1024 * 1024
    opts = dict(grpc_options())
    assert opts["grpc.max_send_message_length"] == 64 * 1024 * 1024
    assert opts["grpc.max_receive_message_length"] == 64 * 1024 * 1024


def test_protobuf_roundtrip_for_requests_outputs_and_all_seven_constraints() -> None:
    pytest.importorskip("google.protobuf")

    schema = {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}
    constraints = [
        StructuredParams(kind=ConstraintKind.JSON_SCHEMA, schema=schema),
        StructuredParams(kind=ConstraintKind.JSON_OBJECT),
        StructuredParams(
            kind=ConstraintKind.TOOL_CALL,
            tools=(ToolSchema("get_weather", schema),),
            syntax=ToolCallSyntax("<tool_call>", "</tool_call>", open_token=260, close_token=261),
            tool_choice="required",
            strict=True,
        ),
        StructuredParams(kind=ConstraintKind.REGEX, pattern="[a-z]+"),
        StructuredParams(kind=ConstraintKind.CHOICE, choices=("yes", "no")),
        StructuredParams(kind=ConstraintKind.GRAMMAR, grammar='start: "ok"\n'),
        StructuredParams(
            kind=ConstraintKind.STRUCTURAL_TAG,
            structural_tag=StructuralTagSpec(
                tags=(
                    StructuralTagItem(
                        begin="<tool_call><function=get_weather>",
                        schema=schema,
                        end="</function></tool_call>",
                    ),
                ),
                triggers=("<tool_call>",),
                at_least_one=True,
                stop_after_first=True,
            ),
        ),
    ]

    for idx, structured in enumerate(constraints):
        req = GenerateRequest(
            request_id=RequestId(f"req-{idx}"),
            prompt=(10, 20, 30),
            sampling=SamplingParams(
                temperature=0.6,
                max_tokens=64,
                seed=2**45 + idx,
                allowed_token_ids=frozenset({1, 2, 3}),
                logit_bias={5: -1.5},
                logprobs=2,
                logprob_mode=LogprobMode.SAMPLING,
            ),
            structured=structured,
            client_request_id=f"client-{idx}",
            priority=idx,
            cache_salt="salt-a",
            prefix_cache=True,
        )
        wire = serialize_proto(encode_generate_request(req))
        decoded = decode_generate_request(deserialize_proto(wire))
        assert decoded == req

    out = RequestOutput(
        request_id=RequestId("req-0"),
        outputs=(
            CompletionOutput(
                index=0,
                text="ok",
                token_ids=(12, 13),
                finish_reason=FinishReason.STOP,
                logprobs=(
                    TokenLogprob(
                        token_id=12,
                        logprob=-0.1,
                        rank=1,
                        top=((12, -0.1),),
                    ),
                ),
            ),
        ),
        finished=True,
        usage=Usage(prompt_tokens=3, completion_tokens=2, cached_tokens=2),
        metrics=RequestMetrics(arrival_time=1.0, prefill_time=0.2, decode_time=0.3),
        error=ErrorInfo(code="INTERNAL", message="ok", retryable=False),
    )
    out_dec = decode_request_output(deserialize_proto(serialize_proto(encode_request_output(out))))
    assert out_dec == out

    health = EngineHealth(state=EngineState.RUNNING, progress_at=12.5, busy=True)
    assert (
        decode_engine_health(deserialize_proto(serialize_proto(encode_engine_health(health))))
        == health
    )
    info = EngineInfo(max_model_len=8192, max_context_tokens=4096, prefix_caching=True)
    assert decode_engine_info(deserialize_proto(serialize_proto(encode_engine_info(info)))) == info


def test_error_wire_mapping_and_chat_events_roundtrip() -> None:
    errors = [
        InvalidRequestError("bad field", context={"param": "temperature"}),
        ContextLengthError(prompt_tokens=100, output_tokens=20, max_context_tokens=64),
        UnsupportedFeatureError("missing", context={"features": ["grpc"]}),
        OverloadedError("busy"),
        EngineDeadError("dead"),
    ]
    for err in errors:
        restored = error_from_wire(error_to_wire(err))
        assert type(restored) is type(err)
        assert restored.message == err.message

    events = [
        ReasoningDelta("step"),
        TextDelta("hi"),
        ToolCallStart(0, "c0", "fn"),
        ToolCallDelta(0, '{"a": 1}'),
        ToolCallEnd(0, valid=True),
        UsageEvent(prompt_tokens=4, completion_tokens=3, reasoning_tokens=1, cached_tokens=2),
        FinishEvent(FinishReason.STOP, tool_calls=True),
    ]
    for ev in events:
        assert decode_chat_event(encode_chat_event(ev)) == ev


def test_import_ayaka_loads_no_heavy_or_grpc_modules() -> None:
    code = (
        "import sys, ayaka\n"
        "heavy = ('torch', 'aiohttp', 'llguidance', 'xgrammar', 'grpc', 'google.protobuf')\n"
        "bad = [m for m in heavy if m in sys.modules]\n"
        "assert not bad, f'unexpected eager imports: {bad}'\n"
    )
    subprocess.check_call([sys.executable, "-c", code])  # noqa: S603  # fixed python probe
