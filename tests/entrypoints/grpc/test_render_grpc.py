"""Tests for engine-free RenderGrpcServer + RenderGrpcClient (no torch imported)."""

from __future__ import annotations

import asyncio
import json
import subprocess
import sys
from typing import TYPE_CHECKING, Any

import pytest

from ayaka.configs.tokenizer import TokenizerConfig
from ayaka.core.errors import InvalidRequestError, UnsupportedFeatureError
from ayaka.core.multimodal import ImageInput, VisionSpec
from ayaka.core.types import FinishReason
from ayaka.entrypoints.conversation import (
    ConversationBuilder,
    FinishEvent,
    ReasoningDelta,
    TextDelta,
    ToolCallDelta,
    ToolCallEnd,
    ToolCallStart,
    ToolSpec,
    UsageEvent,
)
from ayaka.entrypoints.grpc.render import RenderGrpcClient, RenderGrpcServer
from ayaka.entrypoints.render.reasoning import ReasoningEffort, ThinkingRequest
from ayaka.entrypoints.render.selection import PromptRenderer, RenderRequest
from ayaka.multimodal.png import make_ide_screenshot_png
from ayaka.tokenizers.service import TokenizerService
from tokenizer_fakes import BOS, ToolTokenizer, loaded_for

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

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


def test_render_server_multimodal_images_and_tool_choice_none() -> None:
    pytest.importorskip("grpc")
    pytest.importorskip("google.protobuf")

    class _VisionToolTokenizer(ToolTokenizer):
        _IMAGE_TOKEN = 262
        _IMAGE_MARKER = "<image>"

        @property
        def vocab_size(self) -> int:
            return 263

        @property
        def max_token_id(self) -> int:
            return 262

        @property
        def valid_token_ids(self) -> frozenset[int]:
            return frozenset(range(263))

        def apply_chat_template(
            self,
            messages: Sequence[Mapping[str, Any]],
            *,
            tools: Sequence[Mapping[str, Any]] | None = None,
            add_generation_prompt: bool = True,
            chat_template: str | None = None,
            **kwargs: Any,
        ) -> str:
            normalized = []
            for m in messages:
                content = m["content"]
                if isinstance(content, list):
                    text = "".join(
                        self._IMAGE_MARKER
                        if part.get("type") == "image"
                        else str(part.get("text", ""))
                        for part in content
                    )
                    normalized.append({**m, "content": text})
                else:
                    normalized.append(m)
            return super().apply_chat_template(
                normalized,
                tools=tools,
                add_generation_prompt=add_generation_prompt,
                chat_template=chat_template,
                **kwargs,
            )

        def encode(self, text: str, *, add_special_tokens: bool = True) -> list[int]:
            self.encode_calls += 1
            ids: list[int] = []
            parts = text.split(self._IMAGE_MARKER)
            for idx, part in enumerate(parts):
                if part:
                    ids.extend(super().encode(part, add_special_tokens=False))
                if idx < len(parts) - 1:
                    ids.append(self._IMAGE_TOKEN)
            return [BOS, *ids] if add_special_tokens else ids

    spec = VisionSpec(
        family="qwen3_5",
        image_token_id=262,
        patch_size=16,
        spatial_merge_size=2,
        min_pixels=4096,
        max_pixels=65536,
    )

    async def main() -> None:
        tok = _VisionToolTokenizer(stream=False)
        service = TokenizerService(
            TokenizerConfig(tokenizer="tiny-vision-tool", skip_tokenizer_init=False),
            max_model_len=1024,
            model_vocab_size=tok.vocab_size,
            loaded=loaded_for(tok, stream=False),
        )
        img = ImageInput(make_ide_screenshot_png(64, 64))
        img_builder = ConversationBuilder()
        img_builder.user("describe", images=(img,))
        img_conv = img_builder.build()

        non_vision = RenderGrpcServer(
            service,
            model_name="tiny-tool",
            tool_call_parser="hermes",
            target="127.0.0.1:0",
        )
        nv_addr = await non_vision.start()
        nv_client = await RenderGrpcClient(nv_addr).connect()
        try:
            with pytest.raises(UnsupportedFeatureError, match="vision-capable"):
                await nv_client.render_chat(img_conv)
        finally:
            await nv_client.close()
            await non_vision.shutdown()

        vision_srv = RenderGrpcServer(
            service,
            model_name="tiny-vision-tool",
            tool_call_parser="hermes",
            vision_spec=spec,
            target="127.0.0.1:0",
        )
        assert vision_srv.fingerprint != non_vision.fingerprint
        v_addr = await vision_srv.start()
        v_client = await RenderGrpcClient(v_addr).connect()
        try:
            remote_img = await v_client.render_chat(img_conv)
            local_img = PromptRenderer(
                service, model_name="tiny-vision-tool", vision_spec=spec
            ).render(RenderRequest(conversation=img_conv))
            assert remote_img.input_ids == local_img.token_ids
            assert remote_img.image_spans == local_img.image_spans
            assert len(remote_img.image_spans) == 1

            tool_builder = ConversationBuilder()
            tool_builder.user("just text")
            tool_builder.tool(
                ToolSpec(
                    name="get_weather",
                    description="Weather",
                    parameters={"type": "object", "properties": {}},
                )
            )
            tool_conv = tool_builder.build()
            remote_none = await v_client.render_chat(tool_conv, tool_choice="none")
            assert remote_none.structured is None

            chunks = [
                {
                    "call_id_prefix": "call_none",
                    "gate": {
                        "force_reasoning": remote_none.force_reasoning,
                        "previous_content": remote_none.previous_content,
                        "stream_reasoning": remote_none.stream_reasoning,
                    },
                    "tools": [],
                    "withhold_call": True,
                    "tools_allowed": False,
                    "token_ids": [65, 260],
                    "text": "A<tool_call>",
                    "finished": True,
                    "finish_reason": str(FinishReason.STOP),
                    "stop_reason": 260,
                    "usage": {"prompt_tokens": len(remote_none.input_ids), "completion_tokens": 2},
                }
            ]
            events = [ev async for ev in v_client.parse_output(chunks)]
            texts = "".join(ev.text for ev in events if isinstance(ev, TextDelta))
            assert texts == "A"
            assert not any(isinstance(ev, ToolCallStart) for ev in events)
            assert any(isinstance(ev, FinishEvent) and not ev.tool_calls for ev in events)
        finally:
            await v_client.close()
            await vision_srv.shutdown()
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
