"""Tests for engine-free RenderGrpcServer + RenderGrpcClient (no torch imported)."""

from __future__ import annotations

import asyncio
import json
import subprocess
import sys

import pytest

from ayaka.configs.tokenizer import TokenizerConfig
from ayaka.core.errors import InvalidRequestError
from ayaka.core.types import FinishReason
from ayaka.entrypoints.conversation import (
    ConversationBuilder,
    FinishEvent,
    ReasoningDelta,
    ToolCallDelta,
    ToolCallEnd,
    ToolCallStart,
    ToolSpec,
    UsageEvent,
)
from ayaka.entrypoints.grpc.render import RenderGrpcClient, RenderGrpcServer
from ayaka.entrypoints.render.reasoning import ReasoningEffort, ThinkingRequest
from ayaka.entrypoints.render.selection import PromptRenderer, RenderRequest
from ayaka.tokenizers.service import TokenizerService
from tokenizer_fakes import ToolTokenizer, loaded_for

pytestmark = pytest.mark.unit


def test_render_server_parity_fingerprint_and_parse_output() -> None:
    pytest.importorskip("grpc")
    pytest.importorskip("google.protobuf")

    async def main() -> None:
        tok = ToolTokenizer(stream=False)
        service = TokenizerService(
            TokenizerConfig(tokenizer="tiny-tool", skip_tokenizer_init=False),
            max_model_len=512,
            model_vocab_size=tok.vocab_size,
            loaded=loaded_for(tok, stream=False),
        )
        server = RenderGrpcServer(
            service,
            model_name="tiny-tool",
            reasoning_parser="think",
            tool_call_parser="qwen3_coder",
            tool_strict_level="parameter",
            target="127.0.0.1:0",
        )
        bound = await server.start()
        client = await RenderGrpcClient(bound).connect()
        try:
            builder = ConversationBuilder()
            builder.user("weather in Hue for 3 days")
            builder.tool(
                ToolSpec(
                    name="get_weather",
                    description="Weather lookup",
                    parameters={
                        "type": "object",
                        "properties": {"city": {"type": "string"}, "days": {"type": "integer"}},
                        "required": ["city", "days"],
                    },
                )
            )
            conv = builder.build()

            remote = await client.render_chat(conv, reasoning_effort="high", tool_choice="required")
            # In-process bit-for-bit token parity with PromptRenderer + TokenizerService
            renderer = PromptRenderer(service, model_name="tiny-tool")
            local_rendered = renderer.render(
                RenderRequest(
                    conversation=conv, thinking=ThinkingRequest(effort=ReasoningEffort.HIGH)
                )
            )

            assert remote.input_ids == local_rendered.token_ids
            assert remote.render_fingerprint == server.fingerprint
            assert remote.thinking_enabled is True
            assert remote.structured is not None
            assert remote.structured.strict is True

            # Mismatched fingerprint is rejected cleanly.
            with pytest.raises(InvalidRequestError, match="x-render-fingerprint"):
                await client.render_chat(conv, expected_fingerprint="wrong-fingerprint")

            # ParseOutput streams semantic events matching local XML/reasoning parsers.
            gate = {
                "force_reasoning": remote.force_reasoning,
                "previous_content": remote.previous_content,
                "stream_reasoning": remote.stream_reasoning,
            }
            tools_wire = [{"name": t.name, "parameters": dict(t.parameters)} for t in conv.tools]
            chunks = [
                {
                    "call_id_prefix": "call_1",
                    "gate": gate,
                    "tools": tools_wire,
                    "parallel_tool_calls": True,
                    "token_ids": [1, 260, 2],
                    "text": "<think>plan</think><tool_call><function=get_weather>",
                    "finished": False,
                },
                {
                    "token_ids": [3, 261],
                    "text": (
                        "<parameter=city>Hue</parameter>"
                        "<parameter=days>3</parameter></function></tool_call>"
                    ),
                    "finished": True,
                    "finish_reason": str(FinishReason.STOP),
                    "usage": {
                        "prompt_tokens": len(local_rendered.token_ids),
                        "completion_tokens": 4,
                    },
                },
            ]
            events = [ev async for ev in client.parse_output(chunks)]
            assert ReasoningDelta("plan") in events
            starts = [ev for ev in events if isinstance(ev, ToolCallStart)]
            deltas = [ev for ev in events if isinstance(ev, ToolCallDelta)]
            ends = [ev for ev in events if isinstance(ev, ToolCallEnd)]
            assert starts == [ToolCallStart(0, "call_1_0", "get_weather")]
            assert json.loads("".join(d.arguments for d in deltas)) == {"city": "Hue", "days": 3}
            assert ends == [ToolCallEnd(0, valid=True)]
            assert any(isinstance(ev, UsageEvent) for ev in events)
            assert any(isinstance(ev, FinishEvent) and ev.tool_calls for ev in events)
        finally:
            await client.close()
            await server.shutdown()
            service.close()

    asyncio.run(main())


def test_render_server_imports_zero_torch() -> None:
    pytest.importorskip("grpc")
    code = (
        "import sys\n"
        "from ayaka.entrypoints.grpc.render import RenderGrpcServer, RenderGrpcClient\n"
        "from ayaka.entrypoints.cli import main\n"
        "assert 'torch' not in sys.modules, 'torch must not load in render-server modules'\n"
    )
    subprocess.check_call([sys.executable, "-c", code])  # noqa: S603  # fixed python probe
