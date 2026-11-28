"""CPU HTTP tests verifying multimodal image delivery across all three chat routes."""

from __future__ import annotations

import asyncio
import base64
import contextlib
from dataclasses import replace
from typing import TYPE_CHECKING, Any

import pytest

pytest.importorskip("torch")
pytest.importorskip("aiohttp")

from aiohttp.test_utils import TestClient, TestServer

from ayaka.configs.server import ServerConfig
from ayaka.core.multimodal import VisionSpec
from ayaka.engine.build import build_engine
from ayaka.entrypoints.http import create_app
from ayaka.entrypoints.serving import Serving
from ayaka.multimodal.png import make_ide_screenshot_png
from ayaka.multimodal.preprocess import image_token_count
from fake_engine import FakeEngine, script
from serving_support import write_long_gpt2
from tiny_model import tiny_engine_config
from tokenizer_fakes import BOS, SPECIALS, FakeTokenizer, loaded_for

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Awaitable, Callable, Mapping, Sequence
    from pathlib import Path

    from ayaka.tokenizers.ports import DecodeStreamLike

pytestmark = pytest.mark.unit

_IMAGE_TOKEN = 260
_IMAGE_MARKER = "<image>"
_VISION_TABLE: tuple[bytes, ...] = (
    *(bytes([i]) for i in range(256)),
    *(s.encode() for s in SPECIALS),
    _IMAGE_MARKER.encode(),
)

_VISION_SPEC = VisionSpec(
    family="qwen3_5",
    image_token_id=_IMAGE_TOKEN,
    patch_size=16,
    spatial_merge_size=2,
    min_pixels=4096,
    max_pixels=65536,
)
_OVERSIZED_VISION_SPEC = VisionSpec(
    family="qwen3_5",
    image_token_id=_IMAGE_TOKEN,
    patch_size=16,
    spatial_merge_size=2,
    min_pixels=262144,
    max_pixels=16777216,
)


class _VisionTokenizer(FakeTokenizer):
    """Byte fake tokenizer extended with a single ``<image>`` placeholder token."""

    @property
    def vocab_size(self) -> int:
        return 261

    @property
    def max_token_id(self) -> int:
        return 260

    @property
    def valid_token_ids(self) -> frozenset[int]:
        return frozenset(range(261))

    def apply_chat_template(
        self,
        messages: Sequence[Mapping[str, Any]],
        *,
        tools: Sequence[Mapping[str, Any]] | None = None,
        add_generation_prompt: bool = True,
        chat_template: str | None = None,
        **kwargs: Any,
    ) -> str:
        del tools, chat_template, kwargs
        parts: list[str] = []
        for message in messages:
            content = message["content"]
            if isinstance(content, list):
                rendered_parts = [
                    _IMAGE_MARKER if part.get("type") == "image" else str(part.get("text", ""))
                    for part in content
                ]
                rendered_content = "".join(rendered_parts)
            else:
                rendered_content = str(content)
            parts.append(f"<|im_start|>{message['role']}\n{rendered_content}<|im_end|>\n")
        if add_generation_prompt:
            parts.append("<|im_start|>assistant\n")
        return "".join(parts)

    def encode(self, text: str, *, add_special_tokens: bool = True) -> list[int]:
        self.encode_calls += 1
        ids: list[int] = []
        i = 0
        while i < len(text):
            if text.startswith(_IMAGE_MARKER, i):
                ids.append(_IMAGE_TOKEN)
                i += len(_IMAGE_MARKER)
            else:
                ids.extend(text[i].encode())
                i += 1
        return [BOS, *ids] if add_special_tokens else ids

    def decode(
        self,
        ids: Sequence[int] | int,
        *,
        skip_special_tokens: bool = False,
        spaces_between_special_tokens: bool = True,
    ) -> str:
        del spaces_between_special_tokens
        sequence = [ids] if isinstance(ids, int) else list(ids)
        special = set(SPECIALS.values())
        return b"".join(
            _VISION_TABLE[i] for i in sequence if not (skip_special_tokens and i in special)
        ).decode("utf-8", errors="replace")

    def new_decode_stream(self, *, skip_special_tokens: bool = False) -> DecodeStreamLike | None:
        del skip_special_tokens
        return None

    def get_vocab(self) -> dict[str, int]:
        return {**{str(i): i for i in range(260)}, _IMAGE_MARKER: _IMAGE_TOKEN}

    def get_decoded_vocab(self) -> tuple[bytes, ...]:
        return _VISION_TABLE

    def token_bytes_table(self) -> tuple[bytes, ...]:
        return _VISION_TABLE


@contextlib.asynccontextmanager
async def _serve_vision(
    directory: Path, *, vision: VisionSpec | None
) -> AsyncGenerator[tuple[TestClient[Any, Any], Serving, FakeEngine]]:
    core = build_engine(
        tiny_engine_config(directory, num_blocks=600),
        tokenizer=loaded_for(_VisionTokenizer(), stream=False),
    )
    info = replace(core.info, vision=vision)
    llm = FakeEngine(info, [script("ok")])
    serving = Serving(ServerConfig(), model_name="tiny")
    serving.attach(llm, core.tokenizer)
    server = TestServer(create_app(serving), handler_cancellation=True)
    try:
        async with TestClient(server) as client:
            yield client, serving, llm
    finally:
        await llm.shutdown()
        core.shutdown()


def _run_with_server(
    directory: Path,
    vision: VisionSpec | None,
    body: Callable[[TestClient[Any, Any], FakeEngine], Awaitable[None]],
) -> None:
    async def main() -> None:
        async with _serve_vision(directory, vision=vision) as (client, _, engine):
            await body(client, engine)

    asyncio.run(main())


def _chat_body(b64: str) -> dict[str, Any]:
    return {
        "model": "tiny",
        "max_tokens": 4,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
                    {"type": "text", "text": "Describe"},
                ],
            }
        ],
    }


def _responses_body(b64: str) -> dict[str, Any]:
    return {
        "model": "tiny",
        "max_output_tokens": 4,
        "input": [
            {
                "role": "user",
                "content": [
                    {"type": "input_image", "image_url": f"data:image/png;base64,{b64}"},
                    {"type": "input_text", "text": "Describe"},
                ],
            }
        ],
    }


def _messages_body(b64: str) -> dict[str, Any]:
    return {
        "model": "tiny",
        "max_tokens": 4,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {"type": "base64", "media_type": "image/png", "data": b64},
                    },
                    {"type": "text", "text": "Describe"},
                ],
            }
        ],
    }


_ROUTES = [
    pytest.param("/v1/chat/completions", _chat_body, id="chat"),
    pytest.param("/v1/responses", _responses_body, id="responses"),
    pytest.param("/v1/messages", _messages_body, id="messages"),
]


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return write_long_gpt2(tmp_path_factory.mktemp("vision-gpt2"), positions=1024)


@pytest.fixture(scope="module")
def short_checkpoint(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return write_long_gpt2(tmp_path_factory.mktemp("vision-short-gpt2"), positions=128)


@pytest.mark.parametrize(("route", "make_body"), _ROUTES)
def test_image_request_delivers_expanded_tokens_images_and_spans(
    checkpoint: Path, route: str, make_body: Callable[[str], dict[str, Any]]
) -> None:
    png_bytes = make_ide_screenshot_png(64, 64)
    b64 = base64.b64encode(png_bytes).decode("ascii")
    expected_tokens, expected_grid = image_token_count(64, 64, _VISION_SPEC)

    async def body(client: TestClient[Any, Any], engine: FakeEngine) -> None:
        response = await client.post(route, json=make_body(b64))
        assert response.status == 200, await response.text()
        assert len(engine.requests) == 1

        req = engine.requests[0]
        assert len(req.images) == 1
        assert req.images[0].data == png_bytes
        assert req.images[0].media_type == "image/png"

        assert len(req.image_spans) == 1
        span = req.image_spans[0]
        assert span.image_index == 0
        assert span.token_count == expected_tokens
        assert span.grid_thw == expected_grid

        assert isinstance(req.prompt, tuple)
        assert req.prompt[span.token_offset : span.token_offset + span.token_count] == (
            (_IMAGE_TOKEN,) * expected_tokens
        )
        assert req.prompt.count(_IMAGE_TOKEN) == expected_tokens

    _run_with_server(checkpoint, _VISION_SPEC, body)


@pytest.mark.parametrize(("route", "make_body"), _ROUTES)
def test_non_vision_model_refuses_image_request(
    checkpoint: Path, route: str, make_body: Callable[[str], dict[str, Any]]
) -> None:
    b64 = base64.b64encode(make_ide_screenshot_png(64, 64)).decode("ascii")

    async def body(client: TestClient[Any, Any], engine: FakeEngine) -> None:
        response = await client.post(route, json=make_body(b64))
        assert response.status == 400
        assert "vision" in (await response.text()).lower()
        assert engine.requests == []

    _run_with_server(checkpoint, None, body)


@pytest.mark.parametrize(("route", "make_body"), _ROUTES)
def test_expanded_image_exceeding_context_is_refused(
    short_checkpoint: Path, route: str, make_body: Callable[[str], dict[str, Any]]
) -> None:
    b64 = base64.b64encode(make_ide_screenshot_png(64, 64)).decode("ascii")

    async def body(client: TestClient[Any, Any], engine: FakeEngine) -> None:
        response = await client.post(route, json=make_body(b64))
        assert response.status == 400
        text = (await response.text()).lower()
        assert "context" in text or "token" in text
        assert engine.requests == []

    _run_with_server(short_checkpoint, _OVERSIZED_VISION_SPEC, body)
