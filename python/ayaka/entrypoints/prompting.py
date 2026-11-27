"""Turning a request's conversation into prompt token ids, within the server's limits.

:class:`PromptPreparation` is the part of :class:`~ayaka.entrypoints.serving.Serving` that renders
and tokenizes: every chat endpoint goes through :meth:`~PromptPreparation.render_chat` (with the
request's own thinking) or :meth:`~PromptPreparation.tokenize_chat`, and both hold the over-long
policy. A prompt beyond one request's KV capacity is refused; with ``truncate`` (or the server's
``overlong_policy="truncate"``) the oldest non-system turns are dropped until it fits. Rendering
and encoding run on the server's one preprocess thread, in arrival order.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from ayaka.core.errors import (
    CapacityError,
    ContextLengthError,
    InvalidRequestError,
)
from ayaka.entrypoints.truncation import encode_truncated_chat, render_truncated_chat
from ayaka.tokenizers.contracts import EncodeOptions, TextInput, TokenIdsInput

if TYPE_CHECKING:
    from collections.abc import Callable
    from concurrent.futures import ThreadPoolExecutor

    from ayaka.configs.server import ServerConfig
    from ayaka.configs.tokenizer import EncodeResult
    from ayaka.engine.seam import EngineClient
    from ayaka.entrypoints.conversation import Conversation
    from ayaka.entrypoints.render.selection import PromptRenderer, RenderedChat, RenderRequest
    from ayaka.tokenizers.contracts import EncodeInput
    from ayaka.tokenizers.service import TokenizerService

__all__ = ["PromptPreparation"]


class PromptPreparation:
    """Rendering and tokenizing requests; ``Serving`` sets the state they read.

    Attributes:
        config: The server settings (``overlong_policy``).
        tokenizer: The tokenizer service, once an engine is attached.
        renderer: The request-scoped prompt renderer, once an engine is attached.
        max_model_len: The model's context, from the engine.
        max_context_tokens: The KV one request can hold, from the engine (0: no limit).
    """

    config: ServerConfig
    tokenizer: TokenizerService | None
    renderer: PromptRenderer | None
    max_model_len: int
    max_context_tokens: int
    _preprocess: ThreadPoolExecutor

    def require_ready(self) -> EngineClient:
        """The engine, or an error a client can read; :class:`Serving` defines it."""
        raise NotImplementedError

    async def tokenize_chat(
        self,
        conversation: Conversation,
        *,
        max_tokens: int | None = None,
        vouch: bool = True,
        truncate: bool = False,
    ) -> tuple[int, ...]:
        """Render a conversation with the model's chat template and tokenize it.

        Args:
            conversation: What to render.
            max_tokens: The output limit the client asked for; it is only reported in the
                error of a prompt that does not fit, since :meth:`output_tokens` caps it.
            vouch: The ids go to the engine next (``False`` when only their count is wanted):
                the tokenizer remembers it checked them, and the engine does not check every id
                again.
            truncate: Drop oldest non-system turns when the conversation exceeds the effective
                context or KV capacity. Also enabled by ``ServerConfig.overlong_policy="truncate"``.

        Raises:
            ContextLengthError: If the prompt leaves no room for one output token.
            CapacityError: If the prompt exceeds the KV capacity of one request on this machine.
            InvalidRequestError: If it cannot be rendered.
            OverloadedError: If the tokenizer is busy.
        """
        should_truncate = vouch and (truncate or self.config.overlong_policy == "truncate")
        if should_truncate:
            return await self._encode_truncated(conversation, max_tokens, vouch=vouch)
        token_ids = await self._encode(conversation.template_data, max_tokens, vouch=vouch)
        if vouch and self.max_context_tokens > 0 and len(token_ids) > self.max_context_tokens:
            raise CapacityError(
                f"prompt ({len(token_ids)} tokens) exceeds the {self.max_context_tokens} tokens "
                "of KV cache this engine can give one request"
            )
        return token_ids

    async def _encode_truncated(
        self, conversation: Conversation, max_tokens: int | None, *, vouch: bool
    ) -> tuple[int, ...]:
        self.require_ready()
        service = self.tokenizer
        assert service is not None  # noqa: S101  # attach() sets it with the engine
        max_prompt_tokens = max(1, self._prompt_limit() - 1)
        try:
            result = await asyncio.get_running_loop().run_in_executor(
                self._preprocess,
                lambda: encode_truncated_chat(
                    service, conversation, max_prompt_tokens=max_prompt_tokens, vouch=vouch
                ),
            )
        except ContextLengthError as exc:
            raise ContextLengthError(
                prompt_tokens=exc.prompt_tokens,
                output_tokens=max(max_tokens or 1, 1),
                max_context_tokens=exc.max_context_tokens,
            ) from None
        except ValueError as exc:
            raise InvalidRequestError(f"cannot use this prompt: {exc}") from exc
        except RuntimeError as exc:
            raise InvalidRequestError(f"cannot render this prompt: {exc}") from exc
        return result.token_ids

    async def render_chat(
        self,
        request: RenderRequest,
        *,
        max_tokens: int | None = None,
        vouch: bool = True,
        truncate: bool = False,
    ) -> RenderedChat:
        """Render and tokenize a chat request with request-scoped thinking metadata.

        The over-long policy of :meth:`tokenize_chat` holds here too: with ``truncate`` (or
        ``ServerConfig.overlong_policy="truncate"``) the oldest non-system turns are dropped until
        the prompt fits, and a prompt beyond one request's KV capacity is refused.

        Raises:
            ContextLengthError: If the prompt leaves no room for one output token.
            CapacityError: If the prompt exceeds the KV capacity of one request on this machine.
            InvalidRequestError: If it cannot be rendered.
        """
        self.require_ready()
        renderer = self.renderer
        assert renderer is not None  # noqa: S101  # attach() sets it with the engine
        should_truncate = vouch and (truncate or self.config.overlong_policy == "truncate")
        limit = max(1, self._prompt_limit() - 1)

        def run() -> RenderedChat:
            if should_truncate:
                return render_truncated_chat(
                    renderer, request, max_prompt_tokens=limit, vouch=vouch
                )
            return renderer.render(request, max_tokens=max_tokens, vouch=vouch)

        try:
            rendered = await asyncio.get_running_loop().run_in_executor(self._preprocess, run)
        except ContextLengthError as exc:
            raise ContextLengthError(
                prompt_tokens=exc.prompt_tokens,
                output_tokens=max(max_tokens or 1, 1),
                max_context_tokens=exc.max_context_tokens,
            ) from None
        count = len(rendered.token_ids)
        if vouch and self.max_context_tokens > 0 and count > self.max_context_tokens:
            raise CapacityError(
                f"prompt ({count} tokens) exceeds the {self.max_context_tokens} tokens "
                "of KV cache this engine can give one request"
            )
        return rendered

    def _prompt_limit(self) -> int:
        """Tokens a prompt may take: the model's context, or less if one request's KV is less."""
        if self.max_context_tokens > 0:
            return min(self.max_model_len, self.max_context_tokens)
        return self.max_model_len

    async def tokenize_text(
        self, prompt: str | tuple[int, ...], *, max_tokens: int | None = None
    ) -> tuple[int, ...]:
        """Tokenize a raw completion prompt (text, or ids to validate); as :meth:`tokenize_chat`."""

        def value() -> EncodeInput:
            return TokenIdsInput(prompt) if isinstance(prompt, tuple) else TextInput(prompt)

        return await self._encode(value, max_tokens, vouch=True)

    async def _encode(
        self, make: Callable[[], EncodeInput], max_tokens: int | None, *, vouch: bool
    ) -> tuple[int, ...]:
        """Build the input with ``make`` and encode it, both on the preprocess thread."""
        self.require_ready()
        service = self.tokenizer
        assert service is not None  # noqa: S101  # attach() sets it with the engine

        def run() -> EncodeResult:
            return service.encode(make(), EncodeOptions(max_output_tokens=1), vouch=vouch)

        try:
            result = await asyncio.get_running_loop().run_in_executor(self._preprocess, run)
        except ContextLengthError as exc:
            raise ContextLengthError(
                prompt_tokens=exc.prompt_tokens,
                output_tokens=max(max_tokens or 1, 1),
                max_context_tokens=exc.max_context_tokens,
            ) from None
        except ValueError as exc:
            raise InvalidRequestError(f"cannot use this prompt: {exc}") from exc
        except RuntimeError as exc:
            raise InvalidRequestError(f"cannot render this prompt: {exc}") from exc
        return result.token_ids
