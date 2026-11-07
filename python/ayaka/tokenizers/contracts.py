"""Immutable data crossing the tokenizer boundary; no I/O, no tokenizer code.

Everything here is a frozen value that is validated on construction, so a bad
request fails at the door and never reaches a worker::

    caller builds                  service consumes            service returns
    -----------------              ----------------            ----------------
    TextInput      \\
    RenderedInput   \\                                         EncodeResult
    ChatInput        >-- EncodeInput + EncodeOptions --> ...     (ayaka.configs.tokenizer)
    TemplateInput   /
    TokenIdsInput  /

    factory builds
    --------------
    TokenizerCapabilities  -- what the *loaded* adapter can really do
    LoadedTokenizer        -- adapter + capabilities, kept consistent

Types that only describe configuration or results live in
:mod:`ayaka.configs.tokenizer`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, cast

from ayaka.utils.validation import require_bool, require_frozen, require_int

if TYPE_CHECKING:
    from ayaka.tokenizers.ports import TokenizerLike

__all__ = [
    "Backend",
    "ChatInput",
    "ChatMessage",
    "EncodeInput",
    "EncodeOptions",
    "LoadedTokenizer",
    "RenderedInput",
    "TemplateInput",
    "TextInput",
    "TokenIdsInput",
    "TokenizerCapabilities",
    "encode_input_weight",
]

Backend = Literal["hf", "slow", "tiktoken", "disabled"]
"""The adapter that was actually loaded. ``disabled`` means token-ID-only mode."""

_ROLES = ("system", "user", "assistant")
_MAX_NAMESPACE_CHARS = 256
_ID_BYTES = 8  # budget charged per token id of a TokenIdsInput


@dataclass(frozen=True, slots=True)
class TextInput:
    """Plain text to tokenize, such as a completion prompt.

    Attributes:
        text: The prompt. Literal special tokens in it are *not* guarded: callers
            of the completion API may legitimately send pre-templated text.
        add_special_tokens: Add the model's BOS/EOS tokens around the text.
    """

    text: str
    add_special_tokens: bool = True

    def __post_init__(self) -> None:
        if type(self.text) is not str:
            raise TypeError("text must be str")
        require_bool(self.add_special_tokens, "add_special_tokens")


@dataclass(frozen=True, slots=True)
class RenderedInput:
    """Text already produced by a chat template.

    Control tokens in it (``<|im_start|>`` ...) are meant to become real special
    tokens, and no BOS/EOS is added on top: the template already placed them.
    Use this, not :class:`TextInput`, for text that a template rendered.

    Attributes:
        text: The rendered prompt.
    """

    text: str

    def __post_init__(self) -> None:
        if type(self.text) is not str:
            raise TypeError("text must be str")


@dataclass(frozen=True, slots=True)
class ChatMessage:
    """One plain-text chat turn.

    Attributes:
        role: ``system``, ``user`` or ``assistant``. Tool turns go through
            :class:`TemplateInput`.
        content: The message text.
    """

    role: Literal["system", "user", "assistant"]
    content: str

    def __post_init__(self) -> None:
        if self.role not in _ROLES:
            raise ValueError(f"role must be one of {_ROLES}; use TemplateInput for tool turns")
        if type(self.content) is not str:
            raise TypeError("chat content must be text")


@dataclass(frozen=True, slots=True)
class ChatInput:
    """A conversation to render with the model's chat template, then tokenize.

    Attributes:
        messages: Non-empty tuple of :class:`ChatMessage`.
        add_generation_prompt: Append the assistant-turn prefix so the model
            starts answering. Mutually exclusive with ``continue_final_message``.
        continue_final_message: Continue the last assistant message instead of
            starting a new turn. Requires that message to be from the assistant.
        chat_template: Jinja template overriding the tokenizer's own, or ``None``.
    """

    messages: tuple[ChatMessage, ...]
    add_generation_prompt: bool = True
    continue_final_message: bool = False
    chat_template: str | None = None

    def __post_init__(self) -> None:
        require_frozen(self, "chat")
        if type(self.messages) is not tuple or not self.messages:
            raise ValueError("messages must be a non-empty tuple")
        if any(not isinstance(m, ChatMessage) for m in self.messages):  # pyright: ignore[reportUnnecessaryIsInstance]
            raise TypeError("messages must contain ChatMessage values")
        require_bool(self.add_generation_prompt, "add_generation_prompt")
        require_bool(self.continue_final_message, "continue_final_message")
        if self.add_generation_prompt and self.continue_final_message:
            raise ValueError("generation prompt and final-message continuation conflict")
        if self.continue_final_message and self.messages[-1].role != "assistant":
            raise ValueError("continuation requires a final assistant message")
        if self.chat_template is not None and (
            type(self.chat_template) is not str or not self.chat_template
        ):
            raise ValueError("chat_template must be non-empty text")


@dataclass(frozen=True, slots=True)
class TokenIdsInput:
    """Pre-tokenized prompt; skips tokenization and goes straight to validation.

    Attributes:
        token_ids: Prompt token ids as an immutable tuple.
    """

    token_ids: tuple[int, ...]

    def __post_init__(self) -> None:
        if type(self.token_ids) is not tuple:
            raise TypeError("token_ids must be an immutable tuple")
        for token in self.token_ids:
            require_int(token, "token id")


@dataclass(frozen=True, slots=True)
class TemplateInput:
    """Chat history as JSON, for tool calls and model-specific template fields.

    Serving validates roles and content before building this. The JSON stays a
    string so the value is immutable and cheap to send to a worker process.

    Attributes:
        messages_json: JSON array of message objects (non-empty).
        tools_json: JSON array of tool definitions; ``"[]"`` for none.
    """

    messages_json: str
    tools_json: str = "[]"

    def __post_init__(self) -> None:
        if type(self.messages_json) is not str or type(self.tools_json) is not str:
            raise TypeError("template inputs must be JSON strings")
        messages = cast("object", json.loads(self.messages_json))
        tools = cast("object", json.loads(self.tools_json))
        if not isinstance(messages, list) or not messages:
            raise ValueError("messages must be a non-empty JSON array")
        if any(not isinstance(m, dict) for m in cast("list[object]", messages)):
            raise ValueError("messages must contain objects")
        if not isinstance(tools, list) or any(
            not isinstance(t, dict) for t in cast("list[object]", tools)
        ):
            raise ValueError("tools must be a JSON array of objects")


type EncodeInput = TextInput | RenderedInput | ChatInput | TemplateInput | TokenIdsInput
"""Anything :class:`~ayaka.tokenizers.service.TokenizerService` can encode."""


@dataclass(frozen=True, slots=True)
class EncodeOptions:
    """How much room the prompt may take, decided before a request is created.

    Attributes:
        max_output_tokens: Tokens reserved for generation (``>= 1``). The prompt
            may use at most ``max_model_len - max_output_tokens``.
        max_input_tokens: Optional extra cap on the prompt length.
        truncate: Cut an over-long prompt (side set by ``truncation_side``)
            instead of rejecting it.
        cache_namespace: Isolates encode-cache entries, at most 256 characters
            (for example a tenant id). Entries never match across namespaces.
    """

    max_output_tokens: int = 128
    max_input_tokens: int | None = None
    truncate: bool = False
    cache_namespace: str = ""

    def __post_init__(self) -> None:
        require_int(self.max_output_tokens, "max_output_tokens", minimum=1)
        if self.max_input_tokens is not None:
            require_int(self.max_input_tokens, "max_input_tokens", minimum=1)
        require_bool(self.truncate, "truncate")
        if (
            type(self.cache_namespace) is not str
            or len(self.cache_namespace) > _MAX_NAMESPACE_CHARS
        ):
            raise ValueError(
                f"cache_namespace must be a string of at most {_MAX_NAMESPACE_CHARS} characters"
            )


@dataclass(frozen=True, slots=True)
class TokenizerCapabilities:
    """What the loaded adapter can really do; never inferred from a requested mode.

    Attributes:
        backend: Which adapter was loaded (see :data:`Backend`).
        text_encode: Can turn text into token ids.
        text_decode: Can turn token ids into text.
        chat: Can render a chat template.
        decode_stream: Offers a native incremental decode stream.
        byte_decode: Has a byte table that passed verification.
        chunked_encode: Adapter guarantees identical ids when a long prompt is
            split at newline candidates. Advertised only; the service does not
            use it yet.
    """

    backend: Backend
    text_encode: bool
    text_decode: bool
    chat: bool
    decode_stream: bool
    byte_decode: bool
    chunked_encode: bool = False

    def __post_init__(self) -> None:
        if self.backend not in ("hf", "slow", "tiktoken", "disabled"):
            raise ValueError("unknown loaded backend")
        flags = (
            self.text_encode,
            self.text_decode,
            self.chat,
            self.decode_stream,
            self.byte_decode,
            self.chunked_encode,
        )
        for flag in flags:
            require_bool(flag, "capability")
        if self.backend == "disabled" and any(flags):
            raise ValueError("disabled backend cannot advertise text capabilities")
        if (self.decode_stream or self.byte_decode) and not self.text_decode:
            raise ValueError("decoder capability requires text_decode")


@dataclass(frozen=True, slots=True)
class LoadedTokenizer:
    """A loaded adapter together with its capabilities.

    A resource bundle, not a serializable config and not a KV-cache owner.

    Attributes:
        tokenizer: The adapter, or ``None`` in token-ID-only mode.
        capabilities: What that adapter can do. ``backend == "disabled"`` exactly
            when ``tokenizer`` is ``None``.
    """

    tokenizer: TokenizerLike | None
    capabilities: TokenizerCapabilities

    def __post_init__(self) -> None:
        if not isinstance(self.capabilities, TokenizerCapabilities):  # pyright: ignore[reportUnnecessaryIsInstance]
            raise TypeError("capabilities must be TokenizerCapabilities")
        if (self.tokenizer is None) != (self.capabilities.backend == "disabled"):
            raise ValueError("adapter and capability state disagree")


def encode_input_weight(value: EncodeInput) -> int:
    """Logical size of an input in bytes, as charged against admission budgets.

    Text counts as UTF-8 bytes, chat as content plus role plus template override,
    and token ids as 8 bytes each. It is a stable estimate used to bound queued
    work, not the exact memory used.

    Raises:
        TypeError: If ``value`` is not an :data:`EncodeInput`.
    """
    if isinstance(value, TextInput | RenderedInput):
        return len(value.text.encode())
    if isinstance(value, ChatInput):
        body = sum(len(m.content.encode()) + len(m.role) for m in value.messages)
        return body + len((value.chat_template or "").encode())
    if isinstance(value, TemplateInput):
        return len(value.messages_json.encode()) + len(value.tools_json.encode())
    if isinstance(value, TokenIdsInput):  # pyright: ignore[reportUnnecessaryIsInstance]
        return _ID_BYTES * len(value.token_ids)
    raise TypeError("unsupported encode input")
