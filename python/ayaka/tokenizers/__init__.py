"""Text in, validated token ids out; token ids in, append-only text out.

The tokenizer layer sits *before* a request exists. It admits work within a budget,
checks every id against the model, and never touches the scheduler or the GPU::

    configs.tokenizer ---------------------------------------------.
    (TokenizerConfig, EncodeResult, DetokenizeParams, DetokUpdate)  |
                                                                    v
    contracts   inputs, options, capabilities       service     admission, batching,
    ports       Protocols the engine depends on     |           cache, validation
    errors      TokenizerOverloaded, ...            |              |
    security    special-token guard                 |              '--> decoding.incremental
                                                    v                    byte | stream | slow
    factory --loads--> adapters.hf | adapters.tiktoken          \\__ stop strings (decoding.stop)

    process_pool   the same service and decoders, in separate processes
    encoding       long-prompt chunking (not wired in yet) and media chat

Importing this package is cheap: each public name is resolved from its
submodule on first access. ``transformers``, ``tokenizers``, ``tiktoken`` and
``jinja2`` are loaded lazily by the adapters that use them.

Typical start-up::

    loaded = DefaultTokenizerFactory.from_config(config)
    with TokenizerService(config, max_model_len=..., model_vocab_size=..., loaded=loaded) as svc:
        result = svc.encode(TextInput("Hello"), EncodeOptions(max_output_tokens=64))
        detok = svc.new_detokenizer(DetokenizeParams(stop=("</s>",)))
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ayaka.utils.lazy_exports import build_lazy_exports

if TYPE_CHECKING:
    from ayaka.tokenizers.contracts import (
        Backend,
        ChatInput,
        ChatMessage,
        EncodeInput,
        EncodeOptions,
        LoadedTokenizer,
        RenderedInput,
        TemplateInput,
        TextInput,
        TokenIdsInput,
        TokenizerCapabilities,
        encode_input_weight,
    )
    from ayaka.tokenizers.decoding.byte import ByteIncrementalDetokenizer, verify_byte_table
    from ayaka.tokenizers.decoding.incremental import (
        FastIncrementalDetokenizer,
        IncrementalDetokenizer,
        SlowIncrementalDetokenizer,
    )
    from ayaka.tokenizers.decoding.stop import StopChecker
    from ayaka.tokenizers.encoding.chunked import (
        encode_chunked,
        safe_split_points,
        split_into_chunks,
    )
    from ayaka.tokenizers.encoding.media import encode_media_chat
    from ayaka.tokenizers.errors import SpecialTokenInContent, TokenizerOverloaded
    from ayaka.tokenizers.factory import DefaultTokenizerFactory, validate_loading_config
    from ayaka.tokenizers.ports import (
        BytesTokenizerLike,
        ByteTokenizerLike,
        DecodeStreamLike,
        DetokenizerLike,
        RenderedEncoderLike,
        TokenizerFactory,
        TokenizerLike,
    )
    from ayaka.tokenizers.process_pool import ProcessDetokenizer, ProcessTokenizerPool
    from ayaka.tokenizers.security import SpecialTokenGuard, SpecialTokenPolicy
    from ayaka.tokenizers.service import TokenizerService

__all__ = [
    "Backend",
    "ByteIncrementalDetokenizer",
    "ByteTokenizerLike",
    "BytesTokenizerLike",
    "ChatInput",
    "ChatMessage",
    "DecodeStreamLike",
    "DefaultTokenizerFactory",
    "DetokenizerLike",
    "EncodeInput",
    "EncodeOptions",
    "FastIncrementalDetokenizer",
    "IncrementalDetokenizer",
    "LoadedTokenizer",
    "ProcessDetokenizer",
    "ProcessTokenizerPool",
    "RenderedEncoderLike",
    "RenderedInput",
    "SlowIncrementalDetokenizer",
    "SpecialTokenGuard",
    "SpecialTokenInContent",
    "SpecialTokenPolicy",
    "StopChecker",
    "TemplateInput",
    "TextInput",
    "TokenIdsInput",
    "TokenizerCapabilities",
    "TokenizerFactory",
    "TokenizerLike",
    "TokenizerOverloaded",
    "TokenizerService",
    "encode_chunked",
    "encode_input_weight",
    "encode_media_chat",
    "safe_split_points",
    "split_into_chunks",
    "validate_loading_config",
    "verify_byte_table",
]

_EXPORTS = {
    "contracts": (
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
    ),
    "decoding.byte": ("ByteIncrementalDetokenizer", "verify_byte_table"),
    "decoding.incremental": (
        "FastIncrementalDetokenizer",
        "IncrementalDetokenizer",
        "SlowIncrementalDetokenizer",
    ),
    "decoding.stop": ("StopChecker",),
    "encoding.chunked": ("encode_chunked", "safe_split_points", "split_into_chunks"),
    "encoding.media": ("encode_media_chat",),
    "errors": ("SpecialTokenInContent", "TokenizerOverloaded"),
    "factory": ("DefaultTokenizerFactory", "validate_loading_config"),
    "ports": (
        "ByteTokenizerLike",
        "BytesTokenizerLike",
        "DecodeStreamLike",
        "DetokenizerLike",
        "RenderedEncoderLike",
        "TokenizerFactory",
        "TokenizerLike",
    ),
    "process_pool": ("ProcessDetokenizer", "ProcessTokenizerPool"),
    "security": ("SpecialTokenGuard", "SpecialTokenPolicy"),
    "service": ("TokenizerService",),
}

__getattr__, __dir__ = build_lazy_exports(__name__, _EXPORTS, globals())
