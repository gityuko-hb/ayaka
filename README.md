# Ayaka

High-performance LLM inference engine with CUDA/C++ native extensions.

> **Status:** early development. The API will change.

## Serving

`ayaka serve` answers OpenAI (Chat Completions, Completions, Responses) and Anthropic (Messages) clients over HTTP.
It needs the `serve` extra:

```sh
uv sync --extra cpu --extra serve
uv run ayaka serve path/to/model --tool-call-parser hermes --reasoning-parser think
```

It listens on `127.0.0.1:8000` unless told otherwise; `ayaka serve --help` lists the settings.

JSON answers (`response_format`, `text.format`, `output_format`), forced tool calls (`tool_choice`) and strict
tool arguments need the `structured` extra (llguidance): `uv sync --extra cpu --extra serve --extra structured`.
With it, a request that offers tools also gets its calls checked by a grammar while the rest of the answer stays free.

## Development

Requires [uv](https://docs.astral.sh/uv/).

```sh
git clone https://github.com/gityuko-hb/ayaka.git
cd ayaka
uv sync --extra cpu        # or --extra cu130 with an NVIDIA GPU
uv run pytest -m "not gpu and not integration"
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for the checks, the title convention and what a good pull request
contains, and [AGENTS.md](AGENTS.md) for the layout.

## License

[MIT](LICENSE)
