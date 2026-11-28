"""From a caller's request to a request the scheduler can admit.

:class:`InputProcessor` is the door of the engine. A request that gets past it has a validated token
prompt and sampling parameters the engine can honour, and every way it can fail is a typed error of
:mod:`ayaka.core.errors`, so a caller never has to parse a message or catch a bare ``ValueError``:

* ``UnsupportedFeatureError``: the request asks for something this engine does not do yet, checked
  first and before any tokenizing work.
* ``InvalidRequestError``: the prompt or the parameters are wrong for this model: text that cannot
  be tokenized, ids the model does not have, a prompt and ``max_tokens`` that exceed the context.
* ``OverloadedError``: the tokenizer queue is full; try again later.

Tokenizing runs in the :class:`~ayaka.tokenizers.service.TokenizerService`; this class only decides
what to ask it and translates what comes back. A token prompt the service itself encoded with
``vouch=True`` (the server's path) is taken as checked: only its length is looked at again.

Nothing here touches engine state, so :class:`InputProcessor` may run on any thread; the engine
prepares requests away from the thread that runs its steps.
"""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING, Final

from ayaka.core.errors import (
    ContextLengthError,
    InvalidRequestError,
    OverloadedError,
    UnsupportedFeatureError,
)
from ayaka.core.params import SamplingParams
from ayaka.core.request import TokenizedRequest
from ayaka.multimodal.tokens import expand_image_placeholders
from ayaka.sampling.structured.compiler import GrammarCompiler
from ayaka.tokenizers.contracts import (
    ChatInput,
    ChatMessage,
    EncodeOptions,
    TextInput,
    TokenIdsInput,
)
from ayaka.tokenizers.errors import TokenizerOverloaded

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from ayaka.core.multimodal import ImageInput, ImageSpan, VisionSpec
    from ayaka.core.request import GenerateRequest
    from ayaka.core.types import RequestId
    from ayaka.tokenizers.contracts import EncodeInput
    from ayaka.tokenizers.service import TokenizerService

__all__ = ["MAX_LOGPROBS", "InputProcessor", "require_supported"]

#: What the sampler and the output path cannot do yet: a name for the message and a test that says
#: whether a request asks for it. A feature leaves this table when its implementation lands.
_UNSUPPORTED: Final[tuple[tuple[str, Callable[[SamplingParams], bool]], ...]] = (
    ("prompt_logprobs", lambda p: p.prompt_logprobs is not None),
    ("return_sampling_support", lambda p: p.return_sampling_support),
)

#: Most alternatives a request may ask for per token (OpenAI's ``top_logprobs`` limit).
MAX_LOGPROBS: Final = 20


def require_supported(sampling: SamplingParams, *, can_decode_text: bool = True) -> None:
    """Refuse a request that asks for a feature this engine does not provide.

    Args:
        sampling: The request's parameters.
        can_decode_text: Whether the engine has a tokenizer that can turn tokens into text;
            stop strings need it.

    Raises:
        UnsupportedFeatureError: Naming every unsupported feature the request uses.
    """
    missing = [name for name, used in _UNSUPPORTED if used(sampling)]
    if sampling.stop and not can_decode_text:
        missing.append("stop strings without a tokenizer")
    if missing:
        raise UnsupportedFeatureError(
            f"this engine does not support {', '.join(missing)} yet",
            context={"features": tuple(missing)},
        )


class InputProcessor:
    """Validates a request and turns its prompt into token ids.

    Args:
        service: The tokenizer of the engine.
        max_model_len: The model's context length in tokens.
        vocab_size: The number of ids the model can embed and emit.
        compiler: Compiles output constraints; built over the service's tokenizer when absent.
    """

    def __init__(
        self,
        service: TokenizerService,
        *,
        max_model_len: int,
        vocab_size: int,
        compiler: GrammarCompiler | None = None,
        structured_backend: str = "llguidance",
        vision: VisionSpec | None = None,
    ) -> None:
        self._service = service
        self._max_model_len = max_model_len
        self._vocab_size = vocab_size
        self._vision = vision
        self._compiler = compiler or GrammarCompiler(
            service.tokenizer,
            vocab_size=vocab_size,
            eos_token_ids=service.eos_token_ids,
            backend="xgrammar" if structured_backend == "xgrammar" else "llguidance",
        )

    @property
    def compiler(self) -> GrammarCompiler:
        """The constraint compiler; the engine asks it for each request's matcher."""
        return self._compiler

    def process(self, request: GenerateRequest) -> TokenizedRequest:
        """Tokenize and validate one request.

        Raises:
            UnsupportedFeatureError: If the request needs a feature the engine lacks.
            ContextLengthError: If the prompt and ``max_tokens`` do not fit the model.
            InvalidRequestError: If the prompt cannot be tokenized.
            OverloadedError: If the tokenizer queue is full.
        """
        require_supported(request.sampling, can_decode_text=self._service.tokenizer is not None)
        if request.images and self._vision is None:
            raise UnsupportedFeatureError(
                "image input requires a vision-capable model",
                context={"features": ("image_url",)},
            )
        prompt = request.prompt if isinstance(request.prompt, tuple) else TextInput(request.prompt)
        tokenized = self._tokenize(
            prompt,
            request_id=request.request_id,
            sampling=request.sampling,
            client_request_id=request.client_request_id,
            priority=request.priority,
            cache_salt=request.cache_salt,
            prefix_cache=request.prefix_cache,
            images=request.images,
            image_spans=request.image_spans,
        )
        if request.structured is None:
            return tokenized
        # Compiled after tokenizing, so a prompt that does not fit is refused without compiling.
        constraint_id = self._compiler.compile(request.structured)
        return replace(tokenized, structured=request.structured, constraint_id=constraint_id)

    def process_chat(
        self,
        request_id: RequestId,
        messages: Sequence[ChatMessage],
        sampling: SamplingParams | None = None,
        *,
        priority: int = 0,
    ) -> TokenizedRequest:
        """Render a conversation with the model's chat template, then tokenize it.

        Raises:
            UnsupportedFeatureError: As :meth:`process`.
            InvalidRequestError: As :meth:`process`, and if the conversation is empty or its text
                holds a literal special token (a control token cannot be sent as content).
            OverloadedError: As :meth:`process`.
        """
        params = sampling if sampling is not None else SamplingParams()
        require_supported(params, can_decode_text=self._service.tokenizer is not None)
        images = tuple(img for msg in messages for img in msg.images)
        if images and self._vision is None:
            raise UnsupportedFeatureError(
                "image input requires a vision-capable model",
                context={"features": ("image_url",)},
            )
        try:
            conversation = ChatInput(messages=tuple(messages))
        except (TypeError, ValueError) as exc:
            raise InvalidRequestError(f"invalid conversation: {exc}") from exc
        return self._tokenize(
            conversation,
            request_id=request_id,
            sampling=params,
            client_request_id=None,
            priority=priority,
            cache_salt=None,
            prefix_cache=True,
            images=images,
            image_spans=(),
        )

    def _tokenize(
        self,
        prompt: EncodeInput | tuple[int, ...],
        *,
        request_id: RequestId,
        sampling: SamplingParams,
        client_request_id: str | None,
        priority: int,
        cache_salt: str | None,
        prefix_cache: bool,
        images: tuple[ImageInput, ...] = (),
        image_spans: tuple[ImageSpan, ...] = (),
    ) -> TokenizedRequest:
        limit = self._max_model_len
        wanted = 1 if sampling.max_tokens is None else sampling.max_tokens
        if wanted >= limit:
            raise InvalidRequestError(
                f"max_tokens ({wanted}) must be below the model context of {limit} tokens"
            )
        options = EncodeOptions(max_output_tokens=wanted)
        try:
            token_ids = (
                self._service.take_vouched(prompt, options) if isinstance(prompt, tuple) else None
            )
            if token_ids is None:
                value = TokenIdsInput(prompt) if isinstance(prompt, tuple) else prompt
                token_ids = self._service.encode(value, options).token_ids
        except TokenizerOverloaded as exc:
            raise OverloadedError(f"the tokenizer is busy: {exc}") from exc
        except ContextLengthError:
            raise
        except ValueError as exc:
            raise InvalidRequestError(f"cannot use this prompt: {exc}") from exc
        if images and not image_spans:
            assert self._vision is not None  # noqa: S101  # checked by caller
            token_ids, image_spans = expand_image_placeholders(token_ids, images, self._vision)
            if len(token_ids) + wanted > limit:
                raise ContextLengthError(
                    prompt_tokens=len(token_ids),
                    output_tokens=wanted,
                    max_context_tokens=limit,
                )
        sampling.validate_for_model(
            vocab_size=self._vocab_size,
            prompt_tokens=len(token_ids),
            max_model_len=limit,
            max_logprobs=MAX_LOGPROBS,
        )
        return TokenizedRequest(
            request_id=request_id,
            prompt_token_ids=token_ids,
            sampling=sampling,
            client_request_id=client_request_id,
            priority=priority,
            cache_salt=cache_salt,
            prefix_cache=prefix_cache,
            images=images,
            image_spans=image_spans,
        )
