"""Turning a :class:`~ayaka.configs.tokenizer.TokenizerConfig` into a loaded tokenizer.

The factory is the one place that knows which adapter serves which mode, and it
reports what the *loaded* adapter can really do instead of echoing the request::

    TokenizerConfig
         |  validate_loading_config   static checks, no files, no network
         v
    skip_tokenizer_init? --yes--> LoadedTokenizer(None, backend="disabled")
         | no
         |-- mode "tiktoken" --> TiktokenTokenizer --+
         |-- mode "mistral"  --> rejected (no adapter) |
         '-- auto / hf / slow --> HfTokenizer ---------+
                                                       v
                                  verify_byte_path(), new_decode_stream()
                                                       v
                                  LoadedTokenizer(adapter, TokenizerCapabilities)

An explicit ``detokenize_backend`` that the loaded adapter cannot provide fails here,
at start-up, not on the first request.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ayaka.core.errors import ConfigError
from ayaka.tokenizers.contracts import LoadedTokenizer, TokenizerCapabilities

if TYPE_CHECKING:
    from ayaka.configs.tokenizer import TokenizerConfig

__all__ = ["DefaultTokenizerFactory", "validate_loading_config"]


def validate_loading_config(config: TokenizerConfig) -> None:
    """Reject combinations that cannot work, before any file or network access.

    Only explicit choices are checked; ``auto`` is resolved later against the
    loaded adapter.

    Raises:
        ConfigError: ``UNSUPPORTED_BACKEND`` when the mode has no adapter, or the
            requested ``detokenize_backend`` is impossible for the mode.
    """
    if config.skip_tokenizer_init:
        return
    mode, backend = config.mode, config.detokenize_backend
    if mode == "mistral":
        raise ConfigError(
            "tokenizer.mode", "UNSUPPORTED_BACKEND", "no adapter exists for mode 'mistral' yet"
        )
    if backend == "stream" and mode in ("slow", "tiktoken"):
        raise ConfigError(
            "tokenizer.detokenize_backend",
            "UNSUPPORTED_BACKEND",
            f"mode '{mode}' has no native decode stream",
        )
    if backend == "bytes" and mode == "slow":
        raise ConfigError(
            "tokenizer.detokenize_backend",
            "UNSUPPORTED_BACKEND",
            "a slow tokenizer has no byte table",
        )


class DefaultTokenizerFactory:
    """Loads HuggingFace and tiktoken tokenizers and resolves their capabilities."""

    @classmethod
    def from_config(cls, config: TokenizerConfig) -> LoadedTokenizer:
        """Load the tokenizer ``config`` describes.

        Raises:
            ConfigError: If the configuration cannot work (see
                :func:`validate_loading_config`) or an explicit ``bytes`` or
                ``stream`` backend is unavailable on the loaded adapter.
        """
        validate_loading_config(config)
        if config.skip_tokenizer_init:
            return LoadedTokenizer(
                None, TokenizerCapabilities("disabled", False, False, False, False, False)
            )
        if config.mode == "tiktoken":
            return cls._load_tiktoken(config)
        return cls._load_hf(config)

    @staticmethod
    def _check_backend(config: TokenizerConfig, *, byte_decode: bool, stream: bool) -> None:
        backend = config.detokenize_backend
        if backend == "bytes" and not byte_decode:
            raise ConfigError(
                "tokenizer.detokenize_backend",
                "UNSUPPORTED_BACKEND",
                "byte-table verification failed",
            )
        if backend == "stream" and not stream:
            raise ConfigError(
                "tokenizer.detokenize_backend",
                "UNSUPPORTED_BACKEND",
                "DecodeStream is unavailable",
            )

    @classmethod
    def _load_hf(cls, config: TokenizerConfig) -> LoadedTokenizer:
        from ayaka.tokenizers.adapters.hf import HfTokenizer

        tokenizer = HfTokenizer.from_config(config)
        byte_decode = tokenizer.verify_byte_path()
        stream = tokenizer.new_decode_stream() is not None
        cls._check_backend(config, byte_decode=byte_decode, stream=stream)
        capabilities = TokenizerCapabilities(
            "hf" if tokenizer.is_fast else "slow",
            text_encode=True,
            text_decode=True,
            chat=True,
            decode_stream=stream,
            byte_decode=byte_decode,
            chunked_encode=tokenizer.supports_chunked_encode,
        )
        return LoadedTokenizer(tokenizer, capabilities)

    @classmethod
    def _load_tiktoken(cls, config: TokenizerConfig) -> LoadedTokenizer:
        from ayaka.tokenizers.adapters.tiktoken import TiktokenTokenizer

        tokenizer = TiktokenTokenizer.from_config(config)
        byte_decode = tokenizer.verify_byte_path()
        cls._check_backend(config, byte_decode=byte_decode, stream=False)
        capabilities = TokenizerCapabilities(
            "tiktoken",
            text_encode=True,
            text_decode=True,
            chat=True,
            decode_stream=False,
            byte_decode=byte_decode,
            chunked_encode=tokenizer.supports_chunked_encode,
        )
        return LoadedTokenizer(tokenizer, capabilities)
