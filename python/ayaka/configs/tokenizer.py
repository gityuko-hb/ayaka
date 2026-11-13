"""Tokenizer settings and the small value objects exchanged with a tokenizer service.

Only host-side configuration and result records live here: no tokenizer is loaded
and no package such as ``transformers`` is imported. Every class is frozen and
validated at construction.

::

    ModelSourceConfig                      (ayaka.configs.model_source)
          |
          |  tokenizer_config_from_source()
          v
    TokenizerConfig  -- how to load and run the tokenizer
          |
          +--> text in:   EncodeResult      token ids + cache/truncation info
          |
          +--> ids in:    DetokenizeParams  how to decode and where to stop
                                |
                                v
                          DetokUpdate       text appended by one decode step

Request-time records (``EncodeResult``, ``DetokenizeParams``, ``DetokUpdate``) are
independent of ``TokenizerConfig``; the service combines them.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal, cast

from ayaka.configs.base import ConfigMixin
from ayaka.core.errors import ConfigError
from ayaka.utils.validation import require_bool, require_frozen, require_int, require_text

if TYPE_CHECKING:
    from ayaka.configs.model_source import ModelSourceConfig

TokenizerMode = Literal[
    "auto",  # pick the implementation that fits the model's tokenizer files
    "hf",  # HuggingFace fast tokenizer; needs `transformers`
    "slow",  # HuggingFace pure-Python tokenizer; needs `transformers`
    "tiktoken",  # OpenAI tiktoken BPE; needs `tiktoken`
    "mistral",  # Mistral tokenizer; needs `mistral-common`
]
"""Tokenizer implementation requested by the caller.

Only the name is validated here. Whether the matching optional package is installed
is checked by whatever loads the tokenizer, not by this config.
"""

DetokBackend = Literal[
    "auto",  # pick the detokenizer that fits the selected tokenizer
    "bytes",  # decode whole token-id sequences to bytes, then to text
    "stream",  # incremental decoding, one token at a time
    "window",  # incremental decoding over a sliding window of recent tokens
]
"""Detokenization strategy used when turning generated token ids back into text.

These names are placeholders for the intended backends; no implementation exists yet.
"""

TruncationSide = Literal["left", "right"]
"""Which end of an over-long prompt is dropped: ``left`` keeps the end, ``right`` the start."""

SpecialTokenPolicyName = Literal[
    "reject",  # raise SpecialTokenInContent (default): a literal control token is an injection
    "escape",  # insert a zero-width space so the text can no longer become a control token
    "allow",  # leave chat content untouched; templates are then open to token hijacking
]
"""What to do when chat content contains a literal special token such as ``<|im_end|>``.

Applies to chat input only (``ChatInput`` and ``TemplateInput``). Plain completion text
and already-rendered prompts are never guarded. See :mod:`ayaka.tokenizers.security`.
"""

_MODES: tuple[str, ...] = ("auto", "hf", "slow", "tiktoken", "mistral")
_BACKENDS: tuple[str, ...] = ("auto", "bytes", "stream", "window")
_SIDES: tuple[str, ...] = ("left", "right")
_POLICIES: tuple[str, ...] = ("reject", "escape", "allow")


@dataclass(frozen=True, slots=True)
class TokenizerConfig(ConfigMixin):
    """How to load a tokenizer and how its encode path is batched and cached.

    Attributes:
        tokenizer: Hub repo id or local directory holding the tokenizer files.
        mode: Tokenizer implementation. See :data:`TokenizerMode`.
        revision: Hub branch, tag or commit to load, or ``None`` for the default.
        trust_remote_code: Allow Python code shipped with the tokenizer to run.
        download_dir: Hub download directory, or ``None`` for the default cache.
        offline: Never touch the network; resolve from the local cache only. The HF adapter
            enforces this with :func:`ayaka.utils.hf_utils.hub_offline`, because passing
            ``local_files_only`` alone still let ``huggingface_hub`` request an agent
            registry from the Hub on a machine with a cold cache.
        truncation_side: Which end of an over-long prompt is dropped.
        special_token_policy: How chat content that contains a literal special token
            is handled, for every adapter. Defaults to ``"reject"``. See
            :data:`SpecialTokenPolicyName`.
        skip_tokenizer_init: Token-ID-only mode: no tokenizer is created, so text
            cannot be cached or decoded. Incompatible with the session cache and
            with any ``detokenize_backend`` other than ``"auto"``.
        detokenize_backend: Decoding strategy. See :data:`DetokBackend`.
        encode_pool_workers: Worker count for the encode pool; ``0`` means none.
        encode_batch_window_ms: How long (milliseconds, finite, ``>= 0``) the
            encoder waits to gather requests into one batch.
        encode_max_batch: Maximum requests per encode batch.
        long_prompt_chars: Prompt length in characters above which a prompt would be
            treated as long. Not read by the service yet.
        chunk_chars: Chunk size in characters for encoding long prompts
            (:mod:`ayaka.tokenizers.encoding.chunked`). Not read by the service yet.
        enable_session_cache: Reuse earlier encoding work across requests of a session.
        session_cache_capacity: Maximum number of entries in the session cache.
        verify_session_cache: Cross-check cached results against a fresh encode.
            Requires ``enable_session_cache``.
        encode_max_pending: Maximum number of queued encode requests.
        encode_max_pending_bytes: Maximum total size, in bytes, of queued encode requests.
        session_cache_max_bytes: Maximum total size, in bytes, of the session cache.

    Raises:
        TypeError: If a field has the wrong type (for example a non-``bool`` flag).
        ValueError: If a text or numeric field is empty or out of range.
        ConfigError: If a choice is invalid (``INVALID_CHOICE``) or fields conflict
            (``CACHE_DISABLED``, ``TOKENIZER_DISABLED``).
    """

    tokenizer: str | Path
    mode: TokenizerMode = "auto"
    revision: str | None = None
    trust_remote_code: bool = False
    download_dir: str | None = None
    offline: bool = False
    truncation_side: TruncationSide = "left"
    special_token_policy: SpecialTokenPolicyName = "reject"  # noqa: S105  # a policy name
    skip_tokenizer_init: bool = False
    detokenize_backend: DetokBackend = "auto"
    encode_pool_workers: int = 0
    encode_batch_window_ms: float = 1.0
    encode_max_batch: int = 32
    long_prompt_chars: int = 256 * 1024
    chunk_chars: int = 64 * 1024
    enable_session_cache: bool = False
    session_cache_capacity: int = 4096
    verify_session_cache: bool = False
    encode_max_pending: int = 256
    encode_max_pending_bytes: int = 16 << 20
    session_cache_max_bytes: int = 16 << 20

    def __post_init__(self) -> None:
        if not isinstance(cast("object", self.tokenizer), (str, Path)):
            raise TypeError("tokenizer must be a string or Path")
        require_text(str(self.tokenizer), "tokenizer")
        for name, allowed in (
            ("mode", _MODES),
            ("truncation_side", _SIDES),
            ("special_token_policy", _POLICIES),
            ("detokenize_backend", _BACKENDS),
        ):
            if getattr(self, name) not in allowed:
                raise ConfigError(
                    f"tokenizer.{name}", "INVALID_CHOICE", f"must be one of {allowed}"
                )
        for name in ("revision", "download_dir"):
            value = getattr(self, name)
            if value is not None:
                require_text(value, name)
        for name in (
            "trust_remote_code",
            "offline",
            "skip_tokenizer_init",
            "enable_session_cache",
            "verify_session_cache",
        ):
            require_bool(getattr(self, name), name)
        require_int(self.encode_pool_workers, "encode_pool_workers")
        for name in (
            "encode_max_batch",
            "long_prompt_chars",
            "chunk_chars",
            "session_cache_capacity",
            "encode_max_pending",
            "encode_max_pending_bytes",
            "session_cache_max_bytes",
        ):
            require_int(getattr(self, name), name, minimum=1)
        window = self.encode_batch_window_ms
        if type(window) not in (float, int) or not math.isfinite(window) or window < 0:
            raise ValueError("encode_batch_window_ms must be finite and non-negative")
        if self.verify_session_cache and not self.enable_session_cache:
            raise ConfigError(
                "tokenizer.verify_session_cache",
                "CACHE_DISABLED",
                "verification requires the session cache",
            )
        if self.skip_tokenizer_init and (
            self.enable_session_cache or self.detokenize_backend != "auto"
        ):
            raise ConfigError(
                "tokenizer.skip_tokenizer_init",
                "TOKENIZER_DISABLED",
                "token-ID-only mode cannot cache text or select a decoder",
            )


@dataclass(frozen=True, slots=True)
class DetokenizeParams:
    """Per-request options for turning token ids back into text.

    Attributes:
        skip_special_tokens: Drop special tokens (BOS, EOS, ...) from the output.
        spaces_between_special_tokens: Put a space between adjacent special tokens
            that are kept in the output.
        stop: Stop strings; decoding ends at the first one found in the text.
        include_stop_str_in_output: Keep the matched stop string in the output.
        accumulate_text: Keep the full decoded text, not only the new delta.
    """

    skip_special_tokens: bool = True
    spaces_between_special_tokens: bool = True
    stop: tuple[str, ...] = ()
    include_stop_str_in_output: bool = False
    accumulate_text: bool = False

    def __post_init__(self) -> None:
        require_frozen(self, "detokenize params")
        for name in (
            "skip_special_tokens",
            "spaces_between_special_tokens",
            "include_stop_str_in_output",
            "accumulate_text",
        ):
            require_bool(getattr(self, name), name)
        if type(self.stop) is not tuple:
            raise TypeError("stop must be a tuple")
        for value in self.stop:
            if type(value) is not str or not value:
                raise ValueError("stop strings must be non-empty strings")

    def max_stop_len(self) -> int:
        """Length of the longest stop string, or ``0`` when there are none."""
        return max(map(len, self.stop), default=0)


@dataclass(frozen=True, slots=True)
class DetokUpdate:
    """Append-only text produced by one decode step.

    Attributes:
        delta: New text to append to the output so far.
        stop_matched: The stop string that ended decoding, or ``None``.
        stalled: The step produced no text yet (for example an incomplete
            multi-byte character) and needs more tokens.
        consumed_tokens: Tokens used by this step. It stops at the first text
            match, so it can be fewer than the tokens supplied.
    """

    delta: str = ""
    stop_matched: str | None = None
    stalled: bool = False
    consumed_tokens: int = 0

    def __post_init__(self) -> None:
        require_frozen(self, "detokenizer update")
        if type(self.delta) is not str:
            raise TypeError("delta must be text")
        if self.stop_matched is not None and (
            type(self.stop_matched) is not str or not self.stop_matched
        ):
            raise ValueError("stop_matched must be non-empty text")
        require_int(self.consumed_tokens, "consumed_tokens")
        require_bool(self.stalled, "stalled")


@dataclass(frozen=True, slots=True)
class EncodeResult:
    """Token ids for one prompt, with immutable bookkeeping.

    Ids are a tuple, so service callers never share a mutable list.

    Attributes:
        token_ids: The encoded ids.
        reused_prefix_len: Leading ids reused from the encode cache. This is
            encode-cache reuse only, never KV-cache blocks acquired. Cannot
            exceed ``len(token_ids)``.
        truncated: Whether the prompt was cut to fit a length limit.
        chunks: Number of chunks the prompt was encoded in (``>= 1``).
        extra: Extra metadata as unique ``(key, value)`` pairs; values are
            ``str``, ``int`` or ``bool``. Replaces a mutable metadata dict.
    """

    token_ids: tuple[int, ...]
    reused_prefix_len: int = 0
    truncated: bool = False
    chunks: int = 1
    extra: tuple[tuple[str, str | int | bool], ...] = ()

    def __post_init__(self) -> None:
        require_frozen(self, "encode result")
        if type(self.token_ids) is not tuple or type(self.extra) is not tuple:
            raise TypeError("token_ids and extra must be tuples")
        for token in self.token_ids:
            require_int(token, "token id")
        require_int(self.reused_prefix_len, "reused_prefix_len")
        if self.reused_prefix_len > len(self.token_ids):
            raise ValueError("reused prefix exceeds encoded length")
        require_int(self.chunks, "chunks", minimum=1)
        require_bool(self.truncated, "truncated")
        keys: list[str] = []
        for pair in self.extra:
            if type(pair) is not tuple or len(pair) != 2:
                raise TypeError("extra entries must be pairs")
            key, value = pair
            require_text(key, "extra key")
            if type(value) not in (str, int, bool):
                raise TypeError("extra values must be string, int or bool")
            keys.append(key)
        if len(set(keys)) != len(keys):
            raise ValueError("duplicate extra key")


def tokenizer_config_from_source(
    source: ModelSourceConfig,
    *,
    mode: TokenizerMode = "auto",
    truncation_side: TruncationSide = "left",
    special_token_policy: SpecialTokenPolicyName = "reject",  # noqa: S107  # a policy name
    skip_tokenizer_init: bool = False,
    detokenize_backend: DetokBackend = "auto",
    encode_pool_workers: int = 0,
    encode_batch_window_ms: float = 1.0,
    encode_max_batch: int = 32,
    long_prompt_chars: int = 256 * 1024,
    chunk_chars: int = 64 * 1024,
    enable_session_cache: bool = False,
    session_cache_capacity: int = 4096,
    verify_session_cache: bool = False,
    encode_max_pending: int = 256,
    encode_max_pending_bytes: int = 16 << 20,
    session_cache_max_bytes: int = 16 << 20,
) -> TokenizerConfig:
    """Build a :class:`TokenizerConfig` from a model source plus tokenizer settings.

    Copied from ``source``: the tokenizer path (``tokenizer_path``), the tokenizer
    revision (``effective_tokenizer_revision``, empty becomes ``None``),
    ``allows_remote_code``, ``cache_dir`` (empty becomes ``None``) and ``offline``.
    Everything else comes from the keyword arguments, which have the same meaning
    and defaults as the :class:`TokenizerConfig` fields.
    """
    return TokenizerConfig(
        tokenizer=source.tokenizer_path,
        mode=mode,
        revision=source.effective_tokenizer_revision or None,
        trust_remote_code=source.allows_remote_code,
        download_dir=source.cache_dir or None,
        offline=source.offline,
        truncation_side=truncation_side,
        special_token_policy=special_token_policy,
        skip_tokenizer_init=skip_tokenizer_init,
        detokenize_backend=detokenize_backend,
        encode_pool_workers=encode_pool_workers,
        encode_batch_window_ms=encode_batch_window_ms,
        encode_max_batch=encode_max_batch,
        long_prompt_chars=long_prompt_chars,
        chunk_chars=chunk_chars,
        enable_session_cache=enable_session_cache,
        session_cache_capacity=session_cache_capacity,
        verify_session_cache=verify_session_cache,
        encode_max_pending=encode_max_pending,
        encode_max_pending_bytes=encode_max_pending_bytes,
        session_cache_max_bytes=session_cache_max_bytes,
    )
