# Ayaka

High-performance LLM inference engine with CUDA/C++ native extensions.

> **Status:** early development. The API will change.

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
