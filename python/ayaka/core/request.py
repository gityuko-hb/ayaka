"""Request envelopes: untrusted API input and the tokenized engine input.

``GenerateRequest`` is what a caller hands to the API or serving layer.
``TokenizedRequest`` is what the engine consumes after the tokenizer has
resolved the prompt. They are separate types because their invariants differ:
one may still contain text, the other may only contain validated token ids.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Final

from ayaka.utils.validation import require_bool, require_int, require_text, require_token_tuple

from .errors import InvalidRequestError
from .params import SamplingParams

if TYPE_CHECKING:
    from .types import RequestId, TokenId

__all__ = ["MAX_CACHE_SALT_LEN", "GenerateRequest", "TokenizedRequest"]

#: Maximum length of a prefix-cache salt. Mirrors the cache layer's limit.
MAX_CACHE_SALT_LEN: Final[int] = 256


def _validate_envelope(
    request_id: RequestId,
    client_request_id: str | None,
    priority: int,
    cache_salt: str | None,
    prefix_cache: bool,
) -> None:
    require_text(request_id, "request_id")
    if client_request_id is not None:
        require_text(client_request_id, "client_request_id")
    require_int(priority, "priority", minimum=None)
    require_bool(prefix_cache, "prefix_cache")
    if cache_salt is not None:
        require_text(cache_salt, "cache_salt")
        if len(cache_salt) > MAX_CACHE_SALT_LEN:
            raise InvalidRequestError(f"cache_salt must be at most {MAX_CACHE_SALT_LEN} characters")


def _validate_token_ids(value: object, field_name: str) -> None:
    token_ids = require_token_tuple(value, field_name)
    if not token_ids:
        raise InvalidRequestError(f"{field_name} must not be empty")


@dataclass(frozen=True, slots=True, kw_only=True)
class GenerateRequest:
    """A caller-facing generation request.

    Attributes:
        request_id: Server-assigned identity; never mutated after construction.
        prompt: Raw text or already-tokenized prompt ids.
        sampling: Validated generation parameters.
        client_request_id: Optional caller-provided id, echoed only.
        priority: Scheduler priority; higher values are served first.
        cache_salt: Isolation salt for the prefix-cache hash chain.
        prefix_cache: Whether this request may reuse cached prompt blocks.
    """

    request_id: RequestId
    prompt: str | tuple[int, ...]
    sampling: SamplingParams = field(default_factory=SamplingParams)
    client_request_id: str | None = None
    priority: int = 0
    cache_salt: str | None = None
    prefix_cache: bool = True

    def __post_init__(self) -> None:
        _validate_envelope(
            self.request_id,
            self.client_request_id,
            self.priority,
            self.cache_salt,
            self.prefix_cache,
        )
        if isinstance(self.prompt, str):
            require_text(self.prompt, "prompt")
        else:
            _validate_token_ids(self.prompt, "prompt token id")

    @property
    def is_tokenized(self) -> bool:
        """Whether the prompt is already a tuple of token ids."""

        return isinstance(self.prompt, tuple)

    @property
    def prompt_length(self) -> int:
        """Number of prompt tokens when the prompt is already tokenized."""

        if isinstance(self.prompt, tuple):
            return len(self.prompt)
        raise ValueError("prompt_length is unavailable until the prompt is tokenized")


@dataclass(frozen=True, slots=True, kw_only=True)
class TokenizedRequest:
    """A request after tokenization, ready for admission.

    Attributes:
        request_id: Server-assigned identity.
        prompt_token_ids: Validated prompt token ids.
        sampling: Validated generation parameters.
        client_request_id: Optional caller-provided id, echoed only.
        priority: Scheduler priority; higher values are served first.
        cache_salt: Isolation salt for the prefix-cache hash chain.
        prefix_cache: Whether this request may reuse cached prompt blocks.
    """

    request_id: RequestId
    prompt_token_ids: tuple[TokenId, ...]
    sampling: SamplingParams = field(default_factory=SamplingParams)
    client_request_id: str | None = None
    priority: int = 0
    cache_salt: str | None = None
    prefix_cache: bool = True

    def __post_init__(self) -> None:
        _validate_envelope(
            self.request_id,
            self.client_request_id,
            self.priority,
            self.cache_salt,
            self.prefix_cache,
        )
        _validate_token_ids(self.prompt_token_ids, "prompt_token_id")

    @property
    def prompt_length(self) -> int:
        """Number of prompt tokens."""

        return len(self.prompt_token_ids)

    @property
    def max_total_length(self) -> int | None:
        """Upper bound on prompt plus generated tokens, or ``None`` when unbounded."""

        max_tokens = self.sampling.max_tokens
        if max_tokens is None:
            return None
        return self.prompt_length + max_tokens
