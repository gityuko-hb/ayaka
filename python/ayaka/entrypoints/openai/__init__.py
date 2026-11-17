"""The OpenAI protocols: Chat Completions, Completions and Responses.

Public names are resolved from the module that defines them on first access; importing this
package loads neither aiohttp nor torch.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ayaka.utils.lazy_exports import build_lazy_exports

if TYPE_CHECKING:
    from ayaka.entrypoints.openai.chat import (
        ChatStreamEncoder,
        CompletionStreamEncoder,
        create_chat_completion,
        create_completion,
    )
    from ayaka.entrypoints.openai.protocol import (
        ChatRequest,
        CompletionRequest,
        parse_chat_request,
        parse_completion_request,
    )
    from ayaka.entrypoints.openai.responses import (
        ResponsesRequest,
        ResponsesStreamEncoder,
        create_response,
        parse_responses_request,
    )

__all__ = [
    "ChatRequest",
    "ChatStreamEncoder",
    "CompletionRequest",
    "CompletionStreamEncoder",
    "ResponsesRequest",
    "ResponsesStreamEncoder",
    "create_chat_completion",
    "create_completion",
    "create_response",
    "parse_chat_request",
    "parse_completion_request",
    "parse_responses_request",
]

_EXPORTS = {
    "chat": (
        "ChatStreamEncoder",
        "CompletionStreamEncoder",
        "create_chat_completion",
        "create_completion",
    ),
    "protocol": (
        "ChatRequest",
        "CompletionRequest",
        "parse_chat_request",
        "parse_completion_request",
    ),
    "responses": (
        "ResponsesRequest",
        "ResponsesStreamEncoder",
        "create_response",
        "parse_responses_request",
    ),
}

__getattr__, __dir__ = build_lazy_exports(__name__, _EXPORTS, globals())
