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

Before loading anything, `ayaka plan MODEL` prints whether the model fits this machine and how the memory would be
divided (the same planner a start-up runs; nothing is read but `config.json`, nothing is downloaded), and exits 5 when
it does not fit. `ayaka info` prints the version, devices, backends, installed extras and directories; `ayaka config
show` prints the server settings `ayaka serve` would use and where each comes from. Without `--config`, `serve` reads
the user's configuration file when it exists (`ayaka info` shows its path).

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
