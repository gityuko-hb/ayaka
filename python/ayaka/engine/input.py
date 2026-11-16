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
what to ask it and translates what comes back.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

from ayaka.core.errors import InvalidRequestError, OverloadedError, UnsupportedFeatureError
from ayaka.core.params import SamplingParams
from ayaka.core.request import TokenizedRequest
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

    from ayaka.core.request import GenerateRequest
    from ayaka.core.types import RequestId
    from ayaka.tokenizers.contracts import EncodeInput
    from ayaka.tokenizers.service import TokenizerService

__all__ = ["InputProcessor", "require_supported"]

#: What the sampler and the output path cannot do yet: a name for the message and a test that says
#: whether a request asks for it. A feature leaves this table when its implementation lands.
_UNSUPPORTED: Final[tuple[tuple[str, Callable[[SamplingParams], bool]], ...]] = (
    ("n > 1 (several sequences per request)", lambda p: p.n > 1),
    ("repetition_penalty", lambda p: p.repetition_penalty != 1.0),
    ("frequency_penalty", lambda p: p.frequency_penalty != 0.0),
    ("presence_penalty", lambda p: p.presence_penalty != 0.0),
    ("logit_bias", lambda p: bool(p.logit_bias)),
    ("allowed_token_ids", lambda p: p.allowed_token_ids is not None),
    ("logprobs", lambda p: p.logprobs is not None),
    ("prompt_logprobs", lambda p: p.prompt_logprobs is not None),
    ("logprob_token_ids", lambda p: p.logprob_token_ids is not None),
    ("return_sampling_support", lambda p: p.return_sampling_support),
)


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
    """

    def __init__(self, service: TokenizerService, *, max_model_len: int, vocab_size: int) -> None:
        self._service = service
        self._max_model_len = max_model_len
        self._vocab_size = vocab_size

    def process(self, request: GenerateRequest) -> TokenizedRequest:
        """Tokenize and validate one request.

        Raises:
            UnsupportedFeatureError: If the request needs a feature the engine lacks.
            InvalidRequestError: If the prompt cannot be tokenized or does not fit the model.
            OverloadedError: If the tokenizer queue is full.
        """
        require_supported(request.sampling, can_decode_text=self._service.tokenizer is not None)
        prompt: EncodeInput = (
            TokenIdsInput(request.prompt)
            if isinstance(request.prompt, tuple)
            else TextInput(request.prompt)
        )
        return self._tokenize(
            prompt,
            request_id=request.request_id,
            sampling=request.sampling,
            client_request_id=request.client_request_id,
            priority=request.priority,
            cache_salt=request.cache_salt,
            prefix_cache=request.prefix_cache,
        )

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
        )

    def _tokenize(
        self,
        prompt: EncodeInput,
        *,
        request_id: RequestId,
        sampling: SamplingParams,
        client_request_id: str | None,
        priority: int,
        cache_salt: str | None,
        prefix_cache: bool,
    ) -> TokenizedRequest:
        limit = self._max_model_len
        wanted = 1 if sampling.max_tokens is None else sampling.max_tokens
        if wanted >= limit:
            raise InvalidRequestError(
                f"max_tokens ({wanted}) must be below the model context of {limit} tokens"
            )
        try:
            encoded = self._service.encode(prompt, EncodeOptions(max_output_tokens=wanted))
        except TokenizerOverloaded as exc:
            raise OverloadedError(f"the tokenizer is busy: {exc}") from exc
        except ValueError as exc:
            raise InvalidRequestError(f"cannot use this prompt: {exc}") from exc
        sampling.validate_for_model(
            vocab_size=self._vocab_size,
            prompt_tokens=len(encoded.token_ids),
            max_model_len=limit,
            max_logprobs=0,
        )
        return TokenizedRequest(
            request_id=request_id,
            prompt_token_ids=encoded.token_ids,
            sampling=sampling,
            client_request_id=client_request_id,
            priority=priority,
            cache_salt=cache_salt,
            prefix_cache=prefix_cache,
        )
