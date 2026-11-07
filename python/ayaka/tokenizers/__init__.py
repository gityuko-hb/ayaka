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

Import from the submodules you need; this package imports nothing itself, and
``transformers``, ``tokenizers``, ``tiktoken`` and ``jinja2`` are loaded lazily by
the adapters that use them.

Typical start-up::

    loaded = DefaultTokenizerFactory.from_config(config)
    with TokenizerService(config, max_model_len=..., model_vocab_size=..., loaded=loaded) as svc:
        result = svc.encode(TextInput("Hello"), EncodeOptions(max_output_tokens=64))
        detok = svc.new_detokenizer(DetokenizeParams(stop=("</s>",)))
"""
