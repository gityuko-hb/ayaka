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

### Benchmark

`ayaka bench MODEL` loads the model in this process (the same flags as `serve`) and measures:

- prompt read (cold) and generation, for three fixed prompt groups at 1K, 4K, 16K tokens and the longest prompt the
  machine holds;
- the first token, with a cold and a warm system prompt;
- a short agent session;
- peak memory.

It also runs two sanity checks: a forced tool call and a JSON answer, each checked against its schema. Every raw run
goes to `results.json` (`benchmarks/results/<date>-<machine>/` unless `--out` says otherwise), together with the
machine, the versions, the power source and the GPU temperature. One summary line goes to `benchmarks/history/` when
that folder exists. A failed sanity check marks the result invalid and exits 2; the speed numbers stay, marked.
`--quick` measures 1K and 4K only.

`--record DIR` on `bench` or `serve` writes every request's prompt and output token ids, sampling parameters, times and
cache use to a versioned session file in `DIR`. It is off by default and the file is owner-only: it holds the
session's content in token form.

### Coding tools

`ayaka connect TOOL` (`claude-code`, `codex` or `opencode`) prints what the tool needs to use a running server;
`--write` puts it in the tool's configuration file after a diff, a timestamped backup and a confirmation (`--yes`
without a terminal). `--check` runs seven steps against the server in the tool's own protocol (ready, model names,
text, streaming, a forced tool call, the prefix cache, the context) and prints one fix per failure; `--output json`
gives the same report, and the exit code is 0 (no step failed), 1 (a step failed) or 2 (the check could not run).
`--observe --tokenizer MODEL` stands between the tool and the server and counts the tokens of the tool's prompt by
part, keeping only the counts. `ayaka plan MODEL --for TOOL` says whether one request on this machine holds the tool's
measured prompt.

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
