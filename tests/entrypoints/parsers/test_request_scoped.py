"""Tests for request-scoped output interpretation (none vs thinking) and grammar sync."""

from __future__ import annotations

import asyncio
import json
from typing import TYPE_CHECKING, Any

import pytest

from ayaka.core.outputs import CompletionOutput, RequestOutput, Usage
from ayaka.core.types import FinishReason, RequestId
from ayaka.entrypoints.conversation import (
    FinishEvent,
    ReasoningDelta,
    TextDelta,
    UsageEvent,
)
from ayaka.entrypoints.parsers.formats import REASONING_FORMATS, TOOL_CALL_FORMATS
from ayaka.entrypoints.parsers.markers import Marker
from ayaka.entrypoints.parsers.output import OutputFormat
from ayaka.entrypoints.parsers.unified import OutputInterpretation
from tokenizer_fakes import FakeTokenizer

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from pathlib import Path

pytestmark = pytest.mark.unit


class _ToggleTokenizer(FakeTokenizer):
    """Fake tokenizer with an ``enable_thinking`` template and single-token markers."""

    chat_template = "{% if enable_thinking is defined and enable_thinking is false %}off{% endif %}"

    def __init__(self) -> None:
        super().__init__(stream=False)

    def encode(self, text: str, *, add_special_tokens: bool = True) -> list[int]:
        markers = {"<think>": 260, "</think>": 261, "<tool_call>": 262, "</tool_call>": 263}
        if text in markers:
            return [markers[text]]
        return super().encode(text, add_special_tokens=add_special_tokens)

    def apply_chat_template(
        self,
        messages: Sequence[Mapping[str, Any]],
        *,
        tools: Sequence[Mapping[str, Any]] | None = None,
        add_generation_prompt: bool = True,
        chat_template: str | None = None,
        **kwargs: Any,
    ) -> str:
        prompt = super().apply_chat_template(
            messages,
            tools=tools,
            add_generation_prompt=add_generation_prompt,
            chat_template=chat_template,
            **kwargs,
        )
        if kwargs.get("enable_thinking") is False:
            return prompt + "<think>\n\n</think>\n\n"
        return prompt


def test_processor_none_mode_passes_think_tags_as_plain_text() -> None:
    fmt = OutputFormat(
        reasoning=REASONING_FORMATS["think"],
        reasoning_markers=(Marker("<think>", 1001), Marker("</think>", 1002)),
        tool_calls=TOOL_CALL_FORMATS["hermes"],
        tool_markers=(Marker("<tool_call>", 1003), Marker("</tool_call>", 1004)),
    )
    proc = fmt.processor(
        OutputInterpretation(force_reasoning=False, tools=True),
        call_id_prefix="call_1",
    )
    events = proc.feed(
        RequestOutput(
            request_id=RequestId("r1"),
            outputs=(
                CompletionOutput(
                    index=0,
                    token_ids=(1001, ord("a"), 1002, ord("b")),
                    text="<think>a</think>b",
                    finish_reason=FinishReason.STOP,
                ),
            ),
            finished=True,
            usage=Usage(prompt_tokens=3, completion_tokens=4),
        )
    )
    assert not any(isinstance(e, ReasoningDelta) for e in events)
    assert TextDelta("<think>a</think>b") in events
    usage = next(e for e in events if isinstance(e, UsageEvent))
    assert usage.reasoning_tokens == 0


def test_processor_attributes_reasoning_tokens_excluding_markers() -> None:
    fmt = OutputFormat(
        reasoning=REASONING_FORMATS["think"],
        reasoning_markers=(Marker("<think>", 1001), Marker("</think>", 1002)),
    )
    proc = fmt.processor(OutputInterpretation(force_reasoning=None), call_id_prefix="call_1")
    events = proc.feed(
        RequestOutput(
            request_id=RequestId("r1"),
            outputs=(
                CompletionOutput(
                    index=0,
                    token_ids=(1001, ord("x"), ord("y"), 1002, ord("z")),
                    text="<think>xy</think>z",
                    finish_reason=FinishReason.STOP,
                ),
            ),
            finished=True,
            usage=Usage(prompt_tokens=2, completion_tokens=5),
        )
    )
    usage = next(e for e in events if isinstance(e, UsageEvent))
    assert usage.reasoning_tokens == 2  # excludes <think> and </think>
    assert events[0] == ReasoningDelta("xy")
    assert events[1] == TextDelta("z")
    assert events[-1] == FinishEvent(FinishReason.STOP)


def test_constraint_start_syncs_with_request_scoped_thinking() -> None:
    fmt = OutputFormat(
        reasoning=REASONING_FORMATS["think"],
        reasoning_markers=(Marker("<think>", 1001), Marker("</think>", 1002)),
    )
    assert fmt.constraint_start(thinking_enabled=False) == (None, None)
    assert fmt.constraint_start(thinking_enabled=True, starts_inside_reasoning=True) == (1002, None)
    assert fmt.constraint_start(thinking_enabled=True, starts_inside_reasoning=False) == (
        1002,
        1001,
    )


def test_concurrent_none_and_thinking_requests_on_one_server(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    pytest.importorskip("aiohttp")
    pytest.importorskip("llguidance")

    from serving_support import script, serve, write_long_gpt2
    from tokenizer_fakes import loaded_for

    checkpoint = write_long_gpt2(tmp_path)
    fmt = OutputFormat.resolve(reasoning="think", tool_calls="hermes")
    shared_script = script("<think>", "planning", "</think>", "ok")
    tool_script = script(
        "<think>",
        "planning",
        "</think>",
        "<tool_call>",
        '{"name":"get_weather","arguments":{"city":"Hue"}}',
        "</tool_call>",
    )
    json_script = script('{"ok": true}')

    async def main() -> None:
        tok = _ToggleTokenizer()
        async with serve(
            checkpoint,
            tokenizer=loaded_for(tok, stream=False),
            output_format=fmt,
            scripts=[shared_script, shared_script, tool_script, json_script],
            max_tokens_per_step=64,
            num_blocks=600,
        ) as (client, _, llm):
            req_off = {
                "messages": [{"role": "user", "content": "hi"}],
                "reasoning_effort": "none",
            }
            req_on = {
                "messages": [{"role": "user", "content": "hi"}],
                "reasoning_effort": "high",
            }
            res_off, res_on = await asyncio.gather(
                client.post("/v1/chat/completions", json=req_off),
                client.post("/v1/chat/completions", json=req_on),
            )
            body_off = await res_off.json()
            body_on = await res_on.json()

            msg_off = body_off["choices"][0]["message"]
            assert "reasoning_content" not in msg_off
            assert msg_off["content"] == "<think>planning</think>ok"

            msg_on = body_on["choices"][0]["message"]
            assert msg_on["reasoning_content"] == "planning"
            assert msg_on["content"] == "ok"

            res_tool = await client.post(
                "/v1/chat/completions",
                json={
                    "messages": [{"role": "user", "content": "weather?"}],
                    "reasoning_effort": "high",
                    "tools": [
                        {
                            "type": "function",
                            "function": {
                                "name": "get_weather",
                                "parameters": {
                                    "type": "object",
                                    "properties": {"city": {"type": "string"}},
                                },
                            },
                        }
                    ],
                },
            )
            body_tool = await res_tool.json()
            msg_tool = body_tool["choices"][0]["message"]
            assert msg_tool["reasoning_content"] == "planning"
            assert msg_tool["tool_calls"][0]["function"]["name"] == "get_weather"
            assert json.loads(msg_tool["tool_calls"][0]["function"]["arguments"]) == {"city": "Hue"}

            # Verify structured constraint sync when thinking is disabled on Anthropic Messages.
            res_anthropic = await client.post(
                "/v1/messages",
                json={
                    "model": "tiny",
                    "max_tokens": 32,
                    "messages": [{"role": "user", "content": "hi"}],
                    "thinking": {"type": "disabled"},
                    "output_format": {
                        "type": "json_schema",
                        "schema": {
                            "type": "object",
                            "properties": {"ok": {"type": "boolean"}},
                            "required": ["ok"],
                        },
                    },
                },
            )
            assert res_anthropic.status == 200
            assert llm.requests[-1].structured is not None
            assert llm.requests[-1].structured.start_after is None
            assert llm.requests[-1].structured.reasoning_open is None

    asyncio.run(main())
