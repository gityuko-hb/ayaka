"""CPU wire protocol tests for multimodal OpenAI and Anthropic image payloads."""

from __future__ import annotations

import base64

import pytest

from ayaka.core.errors import InvalidRequestError
from ayaka.entrypoints.anthropic.protocol import parse_messages_request
from ayaka.entrypoints.openai.protocol import parse_chat_request
from ayaka.entrypoints.openai.responses import parse_responses_request
from ayaka.entrypoints.wire.reader import JsonObject
from ayaka.multimodal.png import make_ide_screenshot_png

pytestmark = pytest.mark.unit


def _png_b64() -> str:
    return base64.b64encode(make_ide_screenshot_png(64, 64)).decode("ascii")


def test_openai_chat_accepts_base64_data_url_and_refuses_remote_url() -> None:
    b64 = _png_b64()
    body = JsonObject.of(
        {
            "model": "qwen3.5",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
                        {"type": "text", "text": "Read the terminal status."},
                    ],
                }
            ],
        },
        "body",
    )
    req = parse_chat_request(body)
    assert len(req.conversation.images) == 1
    assert req.conversation.messages()[0]["content"] == [
        {"type": "image"},
        {"type": "text", "text": "Read the terminal status."},
    ]

    remote = JsonObject.of(
        {
            "model": "qwen3.5",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": "https://evil.invalid/x.png"}}
                    ],
                }
            ],
        },
        "body",
    )
    with pytest.raises(InvalidRequestError, match="refused"):
        parse_chat_request(remote)


def test_openai_responses_accepts_base64_and_refuses_remote_url() -> None:
    b64 = _png_b64()
    body = JsonObject.of(
        {
            "model": "qwen3.5",
            "input": [
                {
                    "role": "user",
                    "content": [
                        {"type": "input_image", "image_url": f"data:image/png;base64,{b64}"},
                        {"type": "input_text", "text": "Describe"},
                    ],
                }
            ],
        },
        "body",
    )
    req = parse_responses_request(body)
    assert len(req.conversation.images) == 1

    remote = JsonObject.of(
        {
            "model": "qwen3.5",
            "input": [
                {
                    "role": "user",
                    "content": [{"type": "input_image", "image_url": "http://10.0.0.1/img.png"}],
                }
            ],
        },
        "body",
    )
    with pytest.raises(InvalidRequestError, match="refused"):
        parse_responses_request(remote)


def test_anthropic_messages_accepts_base64_and_refuses_remote_url() -> None:
    b64 = _png_b64()
    body = JsonObject.of(
        {
            "model": "gemma4",
            "max_tokens": 32,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {"type": "base64", "media_type": "image/png", "data": b64},
                        },
                        {"type": "text", "text": "What text is shown?"},
                    ],
                }
            ],
        },
        "body",
    )
    req = parse_messages_request(body)
    assert len(req.conversation.images) == 1

    remote = JsonObject.of(
        {
            "model": "gemma4",
            "max_tokens": 32,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {"type": "url", "url": "https://example.com/a.png"},
                        }
                    ],
                }
            ],
        },
        "body",
    )
    with pytest.raises(InvalidRequestError, match="refused"):
        parse_messages_request(remote)
