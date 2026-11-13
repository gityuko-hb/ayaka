"""Public result values: usage, metrics, logprobs and request outputs.

The same :class:`RequestOutput` shape is used for non-streaming results and for
streaming chunks. A streaming chunk has ``finished=False`` and carries only the
new text/token ids; the terminal chunk has ``finished=True`` and a finish
reason on at least one of its completion outputs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, cast

from ayaka.utils.validation import (
    require_bool,
    require_int,
    require_number,
    require_text,
    require_token_tuple,
)

from .errors import ErrorInfo, InvalidRequestError

if TYPE_CHECKING:
    from .types import FinishReason, RequestId, SequenceIndex, TokenId

__all__ = [
    "CompletionOutput",
    "RequestMetrics",
    "RequestOutput",
    "TokenLogprob",
    "Usage",
]


def _require_top_logprobs(value: object) -> tuple[tuple[TokenId, float], ...]:
    if not isinstance(value, tuple):
        raise InvalidRequestError("top must be a tuple of (token_id, logprob) pairs")
    pairs: list[tuple[TokenId, float]] = []
    for item in cast("tuple[object, ...]", value):
        if not isinstance(item, tuple):
            raise InvalidRequestError("top entries must be (token_id, logprob) pairs")
        pair = cast("tuple[object, ...]", item)
        if len(pair) != 2:
            raise InvalidRequestError("top entries must be (token_id, logprob) pairs")
        token, logprob = pair
        pairs.append(
            (
                require_int(token, "top token id", minimum=0),
                require_number(logprob, "top logprob"),
            )
        )
    return tuple(pairs)


@dataclass(frozen=True, slots=True)
class Usage:
    """Token accounting for one finished request."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    cached_tokens: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "prompt_tokens",
            require_int(self.prompt_tokens, "prompt_tokens", minimum=0),
        )
        object.__setattr__(
            self,
            "completion_tokens",
            require_int(self.completion_tokens, "completion_tokens", minimum=0),
        )
        object.__setattr__(
            self,
            "cached_tokens",
            require_int(self.cached_tokens, "cached_tokens", minimum=0),
        )
        if self.cached_tokens > self.prompt_tokens:
            raise InvalidRequestError("cached_tokens must be <= prompt_tokens")

    @property
    def total_tokens(self) -> int:
        """Prompt plus completion tokens."""

        return self.prompt_tokens + self.completion_tokens


@dataclass(frozen=True, slots=True)
class RequestMetrics:
    """Engine-side timing for one request, in seconds or monotonic timestamps.

    The engine stamps these values; core never reads a clock. All durations are
    non-negative, and ``ttft``/``tpot`` are supplied by the engine only when
    they are meaningful.
    """

    arrival_time: float = 0.0
    queued_time: float = 0.0
    prefill_time: float = 0.0
    decode_time: float = 0.0
    ttft: float = 0.0
    tpot: float = 0.0

    def __post_init__(self) -> None:
        for name in (
            "arrival_time",
            "queued_time",
            "prefill_time",
            "decode_time",
            "ttft",
            "tpot",
        ):
            object.__setattr__(
                self,
                name,
                require_number(getattr(self, name), name, minimum=0.0),
            )


@dataclass(frozen=True, slots=True)
class TokenLogprob:
    """The logprob of one selected token and, optionally, its top alternatives."""

    token_id: TokenId
    logprob: float
    rank: int | None = None
    decoded: str | None = None
    top: tuple[tuple[TokenId, float], ...] = ()

    def __post_init__(self) -> None:
        require_int(self.token_id, "token_id", minimum=0)
        require_number(self.logprob, "logprob")
        if self.rank is not None:
            require_int(self.rank, "rank", minimum=0)
        object.__setattr__(self, "top", _require_top_logprobs(self.top))


@dataclass(frozen=True, slots=True)
class CompletionOutput:
    """One generated sequence.

    For streaming, ``text`` and ``token_ids`` contain only the new fragment.
    For a non-streaming result, they contain the complete sequence.
    ``token_ids_logprobs`` carries logprobs for explicitly requested token ids
    when ``SamplingParams.logprob_token_ids`` is set.
    """

    index: SequenceIndex
    token_ids: tuple[TokenId, ...] = ()
    text: str = ""
    finish_reason: FinishReason | None = None
    stop_reason: TokenId | str | None = None
    cumulative_logprob: float | None = None
    logprobs: tuple[TokenLogprob, ...] | None = None
    token_ids_logprobs: tuple[TokenLogprob, ...] | None = None

    def __post_init__(self) -> None:
        require_int(self.index, "index", minimum=0)
        object.__setattr__(self, "token_ids", require_token_tuple(self.token_ids, "token_ids"))
        if self.stop_reason is not None:
            if isinstance(self.stop_reason, str):
                require_text(self.stop_reason, "stop_reason")
            else:
                require_int(self.stop_reason, "stop_reason", minimum=0)
        if self.cumulative_logprob is not None:
            require_number(self.cumulative_logprob, "cumulative_logprob")


@dataclass(frozen=True, slots=True)
class RequestOutput:
    """One request's public output.

    Attributes:
        request_id: Identity assigned at the API/serving door.
        prompt_token_ids: Prompt ids when known, otherwise ``None``.
        outputs: One entry per generated sequence.
        finished: ``False`` for a streaming chunk, ``True`` for the terminal chunk.
        usage: Token accounting; meaningful at the terminal chunk.
        metrics: Engine timing, when available.
        error: Structured failure, when the request ended with an error.
    """

    request_id: RequestId
    prompt_token_ids: tuple[TokenId, ...] | None = None
    outputs: tuple[CompletionOutput, ...] = ()
    finished: bool = False
    usage: Usage = field(default_factory=Usage)
    metrics: RequestMetrics | None = None
    error: ErrorInfo | None = None

    def __post_init__(self) -> None:
        require_text(self.request_id, "request_id")
        if self.prompt_token_ids is not None:
            require_token_tuple(self.prompt_token_ids, "prompt_token_ids")
        object.__setattr__(self, "finished", require_bool(self.finished, "finished"))
