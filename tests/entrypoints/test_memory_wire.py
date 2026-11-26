"""Unit tests for memory wire error codes, finish limits, and over-long truncation."""

from __future__ import annotations

import asyncio
import json
from typing import TYPE_CHECKING, Any

import pytest

from ayaka.configs.server import ServerConfig
from ayaka.core.errors import ConfigError, OutOfMemoryError, PreemptedTimeoutError
from ayaka.core.outputs import CompletionOutput, RequestOutput
from ayaka.core.types import FinishReason, RequestId
from ayaka.entrypoints.anthropic.messages import MessagesStreamEncoder
from ayaka.entrypoints.anthropic.protocol import anthropic_error
from ayaka.entrypoints.conversation import (
    Conversation,
    ErrorEvent,
    FinishEvent,
    ToolCall,
    Turn,
)
from ayaka.entrypoints.failures import ERROR_CODES, failure_for, failure_for_event, responses_code
from ayaka.entrypoints.metrics import ServingMetrics
from ayaka.entrypoints.openai.chat import ChatStreamEncoder, _Meta  # pyright: ignore[reportPrivateUsage]
from ayaka.entrypoints.openai.protocol import openai_error
from ayaka.entrypoints.openai.responses import (
    ResponsesStreamEncoder,
    _Snapshot,  # pyright: ignore[reportPrivateUsage]
    parse_responses_request,
)
from ayaka.entrypoints.parsers.output import ChoicesParser, OutputFormat
from ayaka.entrypoints.truncation import group_turns, truncate_conversation
from ayaka.entrypoints.wire.reader import JsonObject

torch: Any = pytest.importorskip("torch")
pytest.importorskip("aiohttp")

from serving_support import serve  # noqa: E402
from tiny_model import TinyCheckpoint, write_tiny_gpt2  # noqa: E402

if TYPE_CHECKING:
    from aiohttp.test_utils import TestClient

pytestmark = pytest.mark.unit


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory: pytest.TempPathFactory) -> TinyCheckpoint:
    return write_tiny_gpt2(tmp_path_factory.mktemp("wire-mem-gpt2"))


def test_memory_error_codes_in_closed_table_and_protocols() -> None:
    oom = OutOfMemoryError("cuda out of memory")
    pt = PreemptedTimeoutError("preempted request waited too long")

    assert oom.code == "out_of_memory"
    assert pt.code == "preempted_timeout"
    assert {"out_of_memory", "preempted_timeout"} <= set(ERROR_CODES)

    f_oom = failure_for(oom, "req1")
    assert f_oom.status == 503
    assert f_oom.headers == {"Retry-After": "1"}
    assert openai_error(f_oom)["error"] == {
        "message": "cuda out of memory",
        "type": "service_unavailable_error",
        "param": None,
        "code": "out_of_memory",
    }
    assert anthropic_error(f_oom)["error"] == {
        "type": "overloaded_error",
        "message": "cuda out of memory",
    }
    assert responses_code("out_of_memory") == "server_error"

    f_pt = failure_for(pt, "req2")
    assert f_pt.status == 408
    assert f_pt.headers == {}
    assert openai_error(f_pt)["error"] == {
        "message": "preempted request waited too long",
        "type": "timeout_error",
        "param": None,
        "code": "preempted_timeout",
    }
    assert anthropic_error(f_pt)["error"] == {
        "type": "api_error",
        "message": "preempted request waited too long",
    }


def test_memory_error_events_render_in_all_stream_encoders() -> None:
    for code, expected_http_type in (
        ("out_of_memory", "service_unavailable_error"),
        ("preempted_timeout", "timeout_error"),
    ):
        event = ErrorEvent(code, f"detail-{code}", retryable=True)
        chat_enc = ChatStreamEncoder(_Meta("chatcmpl-1", 1, "m"), include_usage=False)
        chat_payload = json.loads(chat_enc.encode(event).decode().removeprefix("data: ").strip())
        assert chat_payload["error"]["code"] == code
        assert chat_payload["error"]["type"] == expected_http_type

        resp_enc = ResponsesStreamEncoder(_Snapshot("resp_1", 1, "m", {}), "1")
        resp_text = resp_enc.encode(event).decode()
        assert "event: response.failed" in resp_text
        assert f"detail-{code}" in resp_text

        msg_enc = MessagesStreamEncoder("msg_1", "m", prompt_tokens=4)
        msg_text = msg_enc.encode(event).decode()
        assert "event: error" in msg_text
        assert f"detail-{code}" in msg_text

        assert failure_for_event(event, "1").code == code


def test_finish_event_limit_memory_and_metrics() -> None:
    parser = OutputFormat().parser(call_id_prefix="c", tools=False)
    events = parser.feed(
        RequestOutput(
            request_id=RequestId("r1"),
            outputs=(CompletionOutput(index=0, token_ids=(1,), finish_reason=FinishReason.LENGTH),),
            finished=True,
            error=OutOfMemoryError("kv pool full").to_info(),
        )
    )
    finish = next(e for e in events if isinstance(e, FinishEvent))
    assert finish.reason is FinishReason.LENGTH
    assert finish.limit == "memory"

    choices = ChoicesParser(
        [
            OutputFormat().parser(call_id_prefix="c0", tools=False),
            OutputFormat().parser(call_id_prefix="c1", tools=False),
        ]
    )
    c_events = choices.feed(
        RequestOutput(
            request_id=RequestId("r2"),
            outputs=(
                CompletionOutput(index=0, token_ids=(1,), finish_reason=FinishReason.STOP),
                CompletionOutput(index=1, token_ids=(2,), finish_reason=FinishReason.LENGTH),
            ),
            finished=True,
            error=OutOfMemoryError("kv pool full").to_info(),
        )
    )
    overall = c_events[-1]
    assert isinstance(overall, FinishEvent)
    assert overall.limit == "memory"

    metrics = ServingMetrics()
    metrics.observe(events)
    assert metrics.memory_limit_finishes._value.get() == 1.0  # pyright: ignore[reportPrivateUsage]


def test_group_turns_keeps_tool_calls_with_their_results() -> None:
    turns = (
        Turn(role="user", text="first"),
        Turn(role="assistant", text="", tool_calls=(ToolCall("c1", "read", {}),)),
        Turn(role="tool", text="file", tool_call_id="c1", name="read"),
        Turn(role="user", text="latest"),
    )
    groups = group_turns(turns)
    assert len(groups) == 3
    assert [t.role for t in groups[1]] == ["assistant", "tool"]

    conv = Conversation(system="sys", turns=turns)
    trimmed, dropped = truncate_conversation(
        conv,
        lambda c: (len(c.turns) <= 1, len(c.turns) * 10),
        max_prompt_tokens=15,
    )
    assert dropped == 3
    assert trimmed.system == "sys"
    assert trimmed.turns == (Turn(role="user", text="latest"),)


def test_overlong_policy_config_validation() -> None:
    assert ServerConfig().overlong_policy == "reject"
    assert ServerConfig(overlong_policy="truncate").overlong_policy == "truncate"
    with pytest.raises(ConfigError, match="overlong_policy"):
        ServerConfig(overlong_policy="drop")  # type: ignore[arg-type]


def test_responses_truncation_auto_accepted_and_rejects_unknown() -> None:
    req = parse_responses_request(
        JsonObject({"model": "tiny", "input": "hi", "truncation": "auto"})
    )
    assert req.truncation == "auto"

    with pytest.raises(Exception, match="truncation"):
        parse_responses_request(JsonObject({"model": "tiny", "input": "hi", "truncation": "head"}))


def test_http_rejects_overlong_by_default_and_truncates_when_opted_in(
    checkpoint: TinyCheckpoint,
) -> None:
    long_turns = [
        {"role": "system", "content": "keep-sys"},
        {"role": "user", "content": "old-turn-" + "a" * 80},
        {"role": "assistant", "content": "old-reply-" + "b" * 80},
        {"role": "user", "content": "hi"},
    ]

    async def check(reject: bool) -> int:
        cfg = ServerConfig(overlong_policy="reject" if reject else "truncate")
        async with serve(checkpoint.directory, config=cfg) as (client, _, _):
            cli: TestClient[Any, Any] = client
            resp = await cli.post(
                "/v1/chat/completions",
                json={"model": "tiny", "messages": long_turns, "max_tokens": 4, "temperature": 0},
            )
            return resp.status

    assert asyncio.run(check(reject=True)) == 400
    assert asyncio.run(check(reject=False)) == 200
