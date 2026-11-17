"""The Anthropic Messages protocol, with ``count_tokens``.

Public names are resolved from the module that defines them on first access; importing this
package loads neither aiohttp nor torch.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ayaka.utils.lazy_exports import build_lazy_exports

if TYPE_CHECKING:
    from ayaka.entrypoints.anthropic.messages import (
        MessagesStreamEncoder,
        count_tokens,
        create_message,
    )
    from ayaka.entrypoints.anthropic.protocol import MessagesRequest, parse_messages_request

__all__ = [
    "MessagesRequest",
    "MessagesStreamEncoder",
    "count_tokens",
    "create_message",
    "parse_messages_request",
]

_EXPORTS = {
    "messages": ("MessagesStreamEncoder", "count_tokens", "create_message"),
    "protocol": ("MessagesRequest", "parse_messages_request"),
}

__getattr__, __dir__ = build_lazy_exports(__name__, _EXPORTS, globals())
