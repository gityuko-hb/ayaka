"""Sampling parameters: user intent, validated without a tokenizer or torch.

``SamplingParams`` is a frozen value object. It carries decoding hyperparameters,
generation limits and logprob-reporting options, but no tokenizer state, model
facts, device tensors or derived caches. The engine resolves stop strings and
EOS ids against the tokenizer at the boundary.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Final, cast

from ayaka.utils.validation import (
    require_bool,
    require_float,
    require_int,
    require_nonempty_text,
    require_token_set,
)

from .errors import InvalidRequestError

__all__ = [
    "LOGIT_BIAS_LIMIT",
    "MAX_LOGPROB_TOKEN_IDS",
    "TOP_K_DISABLED",
    "LogprobMode",
    "SamplingParams",
]

#: Sentinel for a disabled top-k filter. ``0`` is accepted as an alias.
TOP_K_DISABLED: Final[int] = -1

#: OpenAI-compatible magnitude limit for a single ``logit_bias`` entry.
LOGIT_BIAS_LIMIT: Final[float] = 100.0

#: Maximum number of explicit token ids whose logprobs may be requested.
MAX_LOGPROB_TOKEN_IDS: Final[int] = 128

_MAX_SEED: Final[int] = (1 << 63) - 1
_MAX_TEMPERATURE: Final[float] = 2.0
_MAX_PENALTY: Final[float] = 2.0
_GREEDY_TEMPERATURE_EPS: Final[float] = 1e-5


class LogprobMode(StrEnum):
    """Which distribution logprobs describe."""

    RAW = "raw"
    SAMPLING = "sampling"


def _require_logit_bias(
    value: object,
) -> tuple[tuple[int, float], ...] | None:
    if value is None:
        return None
    if isinstance(value, Mapping):
        items = tuple(cast("Mapping[object, object]", value).items())
    elif isinstance(value, (tuple, list)):
        items = tuple(cast("tuple[object, ...] | list[object]", value))
    else:
        raise InvalidRequestError(
            "logit_bias must be a mapping of token ids to biases or a sequence of pairs"
        )

    pairs: list[tuple[int, float]] = []
    for item in items:
        if not isinstance(item, tuple):
            raise InvalidRequestError("logit_bias entries must be (token_id, bias) pairs")
        pair = cast("tuple[object, ...]", item)
        if len(pair) != 2:
            raise InvalidRequestError("logit_bias entries must be (token_id, bias) pairs")
        token, bias = pair
        token_id = require_int(token, "logit_bias token", minimum=0)
        bias_value = require_float(bias, "logit_bias value")
        if abs(bias_value) > LOGIT_BIAS_LIMIT:
            raise InvalidRequestError(
                f"logit_bias value must be in [-{LOGIT_BIAS_LIMIT:g}, {LOGIT_BIAS_LIMIT:g}]"
            )
        pairs.append((token_id, bias_value))

    if len({token for token, _ in pairs}) != len(pairs):
        raise InvalidRequestError("logit_bias contains duplicate token ids")
    return tuple(sorted(pairs))


def _require_optional_token_ids(value: object, field: str) -> tuple[int, ...] | None:
    if value is None:
        return None
    if not isinstance(value, (tuple, list)):
        raise InvalidRequestError(f"{field} must be a sequence of token ids")
    items = tuple(cast("tuple[object, ...] | list[object]", value))
    return tuple(require_int(token, field, minimum=0) for token in items)


@dataclass(frozen=True, slots=True, kw_only=True)
class SamplingParams:
    """Validated generation intent for one request.

    Attributes:
        n: Number of independent output sequences; must be 1 for greedy decoding.
        temperature: Softmax temperature; values below ``1e-5`` select greedy decoding.
        top_p: Nucleus-sampling cumulative probability in ``(0, 1]``.
        top_k: Top-k cutoff, :data:`TOP_K_DISABLED`, or ``0`` (disabled alias).
        min_p: Minimum probability relative to the argmax, in ``[0, 1]``.
        repetition_penalty: Multiplicative penalty for repeated tokens.
        frequency_penalty: Linear penalty proportional to token frequency.
        presence_penalty: Flat penalty for tokens already present.
        seed: Optional deterministic seed; ``-1`` and ``None`` use engine entropy.
        max_tokens: Maximum generated tokens, or ``None`` for the engine maximum.
        min_tokens: Number of tokens generated before stop/EOS is applied.
        stop: Stop strings matched against decoded text.
        stop_token_ids: Additional token ids that terminate generation.
        ignore_eos: Generate past EOS even when the model emits it.
        include_stop_str_in_output: Keep a matched stop string in the text.
        logprobs: Top logprobs per output token; ``-1`` means the full vocabulary.
        prompt_logprobs: Top logprobs per prompt token; ``-1`` means the full vocabulary.
        logprob_token_ids: Explicit token ids whose logprobs are always reported.
        logprob_mode: Whether logprobs describe the raw or filtered distribution.
        return_sampling_support: Capture the filtered positive support distribution.
        logit_bias: Additive biases applied to raw logits.
        allowed_token_ids: When set, only these token ids may be sampled.
    """

    n: int = 1
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = TOP_K_DISABLED
    min_p: float = 0.0
    repetition_penalty: float = 1.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    seed: int | None = None
    max_tokens: int | None = 128
    min_tokens: int = 0
    stop: tuple[str, ...] = ()
    stop_token_ids: frozenset[int] = frozenset()
    ignore_eos: bool = False
    include_stop_str_in_output: bool = False
    logprobs: int | None = None
    prompt_logprobs: int | None = None
    logprob_token_ids: tuple[int, ...] | None = None
    logprob_mode: LogprobMode = LogprobMode.RAW
    return_sampling_support: bool = False
    logit_bias: Mapping[int, float] | tuple[tuple[int, float], ...] | None = None
    allowed_token_ids: frozenset[int] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "n", require_int(self.n, "n", minimum=1))
        object.__setattr__(self, "temperature", require_float(self.temperature, "temperature"))
        object.__setattr__(self, "top_p", require_float(self.top_p, "top_p"))
        top_k = require_int(self.top_k, "top_k", minimum=None)
        if top_k == 0:
            top_k = TOP_K_DISABLED
        object.__setattr__(self, "top_k", top_k)
        object.__setattr__(self, "min_p", require_float(self.min_p, "min_p"))
        object.__setattr__(
            self,
            "repetition_penalty",
            require_float(self.repetition_penalty, "repetition_penalty"),
        )
        object.__setattr__(
            self,
            "frequency_penalty",
            require_float(self.frequency_penalty, "frequency_penalty"),
        )
        object.__setattr__(
            self,
            "presence_penalty",
            require_float(self.presence_penalty, "presence_penalty"),
        )
        if self.max_tokens is not None:
            object.__setattr__(
                self,
                "max_tokens",
                require_int(self.max_tokens, "max_tokens", minimum=1),
            )
        object.__setattr__(
            self,
            "min_tokens",
            require_int(self.min_tokens, "min_tokens", minimum=0),
        )

        if self.seed is not None:
            seed = require_int(self.seed, "seed", minimum=None)
            if seed == -1:
                object.__setattr__(self, "seed", None)
            else:
                if seed < 0:
                    raise InvalidRequestError("seed must be -1 or >= 0")
                if seed > _MAX_SEED:
                    raise InvalidRequestError(f"seed must be <= {_MAX_SEED}")
                object.__setattr__(self, "seed", seed)

        if not 0.0 <= self.temperature <= _MAX_TEMPERATURE:
            raise InvalidRequestError(f"temperature must be in [0, {_MAX_TEMPERATURE}]")
        if not 0.0 < self.top_p <= 1.0:
            raise InvalidRequestError("top_p must be in (0, 1]")
        if self.top_k != TOP_K_DISABLED and self.top_k < 1:
            raise InvalidRequestError("top_k must be -1/0 (disabled) or >= 1")
        if not 0.0 <= self.min_p <= 1.0:
            raise InvalidRequestError("min_p must be in [0, 1]")
        if not -_MAX_PENALTY <= self.frequency_penalty <= _MAX_PENALTY:
            raise InvalidRequestError(
                f"frequency_penalty must be in [-{_MAX_PENALTY}, {_MAX_PENALTY}]"
            )
        if not -_MAX_PENALTY <= self.presence_penalty <= _MAX_PENALTY:
            raise InvalidRequestError(
                f"presence_penalty must be in [-{_MAX_PENALTY}, {_MAX_PENALTY}]"
            )
        if self.repetition_penalty <= 0.0:
            raise InvalidRequestError("repetition_penalty must be > 0")
        if self.max_tokens is not None and self.min_tokens > self.max_tokens:
            raise InvalidRequestError("min_tokens must be <= max_tokens")
        if self.is_greedy and self.n != 1:
            raise InvalidRequestError("n must be 1 when sampling is greedy")

        raw_stop = cast("object", self.stop)
        if isinstance(raw_stop, str):
            stop_values = (require_nonempty_text(raw_stop, "stop"),)
        elif isinstance(raw_stop, (tuple, list)):
            items = cast("tuple[object, ...] | list[object]", raw_stop)
            stop_values = tuple(require_nonempty_text(item, "stop") for item in items)
        else:
            raise InvalidRequestError("stop must be a string or a sequence of strings")
        object.__setattr__(self, "stop", stop_values)
        object.__setattr__(
            self,
            "stop_token_ids",
            require_token_set(self.stop_token_ids, "stop_token_ids", allow_empty=True),
        )

        object.__setattr__(
            self,
            "ignore_eos",
            require_bool(self.ignore_eos, "ignore_eos"),
        )
        object.__setattr__(
            self,
            "include_stop_str_in_output",
            require_bool(
                self.include_stop_str_in_output,
                "include_stop_str_in_output",
            ),
        )

        if self.logprobs is not None:
            object.__setattr__(
                self,
                "logprobs",
                require_int(self.logprobs, "logprobs", minimum=-1),
            )
        if self.prompt_logprobs is not None:
            object.__setattr__(
                self,
                "prompt_logprobs",
                require_int(self.prompt_logprobs, "prompt_logprobs", minimum=-1),
            )
        logprob_ids = _require_optional_token_ids(self.logprob_token_ids, "logprob_token_ids")
        if logprob_ids is not None and len(logprob_ids) > MAX_LOGPROB_TOKEN_IDS:
            raise InvalidRequestError(
                f"logprob_token_ids must contain at most {MAX_LOGPROB_TOKEN_IDS} ids"
            )
        object.__setattr__(self, "logprob_token_ids", logprob_ids)
        if (
            self.logprobs is not None
            and self.logprobs >= 0
            and logprob_ids is not None
            and len(logprob_ids) != self.logprobs
        ):
            raise InvalidRequestError(
                "logprobs must equal len(logprob_token_ids) when both are set"
            )
        raw_mode = cast("object", self.logprob_mode)
        if isinstance(raw_mode, LogprobMode):
            logprob_mode = raw_mode
        else:
            candidate: object = (
                raw_mode if isinstance(raw_mode, str) else getattr(raw_mode, "value", None)
            )
            if not isinstance(candidate, str):
                raise InvalidRequestError("logprob_mode must be a LogprobMode or its string value")
            try:
                logprob_mode = LogprobMode(candidate)
            except ValueError as exc:
                raise InvalidRequestError("logprob_mode must be 'raw' or 'sampling'") from exc
        object.__setattr__(self, "logprob_mode", logprob_mode)
        object.__setattr__(
            self,
            "return_sampling_support",
            require_bool(self.return_sampling_support, "return_sampling_support"),
        )
        object.__setattr__(self, "logit_bias", _require_logit_bias(self.logit_bias))
        allowed = self.allowed_token_ids
        object.__setattr__(
            self,
            "allowed_token_ids",
            None
            if allowed is None
            else require_token_set(allowed, "allowed_token_ids", allow_empty=False),
        )

    @property
    def is_greedy(self) -> bool:
        """Whether sampling is deterministic argmax."""

        return self.temperature < _GREEDY_TEMPERATURE_EPS or self.top_k == 1

    def validate_for_model(
        self,
        *,
        vocab_size: int,
        prompt_tokens: int = 0,
        max_model_len: int | None = None,
        max_logprobs: int | None = None,
    ) -> None:
        """Validate the parts of this request that need model facts.

        This is the pure second validation phase: it takes primitive model
        facts, never imports a model or tokenizer, and never mutates the
        parameters. The engine calls it after tokenization and before
        admission.

        Args:
            vocab_size: Number of logits produced by the model.
            prompt_tokens: Number of prompt tokens after tokenization.
            max_model_len: Optional total context limit.
            max_logprobs: Optional cap on ``logprobs``/``prompt_logprobs``.

        Raises:
            InvalidRequestError: If a token id or requested width is out of range.
        """

        vocab = require_int(vocab_size, "vocab_size", minimum=1)
        prompt = require_int(prompt_tokens, "prompt_tokens", minimum=0)

        if max_model_len is not None and self.max_tokens is not None:
            context_limit = require_int(max_model_len, "max_model_len", minimum=1)
            if prompt + self.max_tokens > context_limit:
                raise InvalidRequestError(
                    f"prompt_tokens + max_tokens must be <= max_model_len ({context_limit})"
                )

        if max_logprobs is not None:
            logprob_limit = require_int(max_logprobs, "max_logprobs", minimum=-1)
            if logprob_limit == -1:
                logprob_limit = vocab
            if self.logprobs is not None:
                width = vocab if self.logprobs == -1 else self.logprobs
                if width > logprob_limit:
                    raise InvalidRequestError(f"logprobs must be <= max_logprobs ({logprob_limit})")
            if self.prompt_logprobs is not None:
                width = vocab if self.prompt_logprobs == -1 else self.prompt_logprobs
                if width > logprob_limit:
                    raise InvalidRequestError(
                        f"prompt_logprobs must be <= max_logprobs ({logprob_limit})"
                    )

        for field_name, tokens in (
            ("stop_token_ids", self.stop_token_ids),
            ("logprob_token_ids", self.logprob_token_ids or ()),
        ):
            for token in tokens:
                if token >= vocab:
                    raise InvalidRequestError(
                        f"{field_name} token id {token} must be < vocab_size {vocab}"
                    )

        if self.logit_bias is not None:
            for token, _ in cast("tuple[tuple[int, float], ...]", self.logit_bias):
                if token >= vocab:
                    raise InvalidRequestError(
                        f"logit_bias token id {token} must be < vocab_size {vocab}"
                    )

        if self.allowed_token_ids is not None:
            for token in self.allowed_token_ids:
                if token >= vocab:
                    raise InvalidRequestError(
                        f"allowed_token_ids token id {token} must be < vocab_size {vocab}"
                    )

    @property
    def token_ids_logprobs(self) -> tuple[int, ...] | None:
        """Legacy alias for :attr:`logprob_token_ids`."""

        return self.logprob_token_ids
