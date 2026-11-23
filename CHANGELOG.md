# Changelog

## 0.1.0 (not published on PyPI)

On an NVIDIA L4 24 GB, Ayaka reads a 4,096-token prompt of Llama-3.1-8B-Instruct in fp16 at 2,393 tok/s and
generates 15.1 tok/s after it, 0.93× the generation speed of llama.cpp on the same card and model in f16
([bench result](benchmarks/results/2026-11-23-nvidia-l4/llama-3.1-8b-instruct/results.json),
[comparison](benchmarks/results/2026-11-23-nvidia-l4/compare-llama-cpp-f16-llama-3.1-8b.json); both measured at commit
`46437b4`, before the SentencePiece grammar fix of #106, which leaves byte-level tokenizers such as Llama 3's unchanged).

The distribution is named `ayaka-llm`, because the name `ayaka` is taken on PyPI. The package is still imported as
`ayaka` and the command is still `ayaka`. This version is installed from a checkout ([Install](docs/INSTALL.md)).

### What 0.1.0 does

- Runs Llama-architecture checkpoints (and GPT-2 on the CPU) in fp16 or bf16 on an NVIDIA GPU through flashinfer and
  CUDA graphs, with a hash-chain prefix cache, chunked prefill and several sequences per request.
- `ayaka serve` answers OpenAI (Chat Completions, Completions, Responses) and Anthropic (Messages) clients, with
  streaming, tool calls (`hermes` and `llama3_json`), reasoning sections and, with the `structured` extra, JSON answers
  and forced tool calls checked by a grammar.
- `ayaka plan` prints whether a model fits before anything is loaded; `ayaka info` and `ayaka config show` print the
  machine and the settings.
- `ayaka connect` writes, checks and observes the configuration of Claude Code, Codex and OpenCode.
- `ayaka bench` measures prompt read, generation, the first token, a short agent session and peak memory, with two
  sanity checks, and writes every raw run to `results.json`.

What runs where, and the result file behind every number, is in [Capabilities](docs/CAPABILITIES.md).

### Limits

- Only Llama-architecture models run on the GPU. Other families are not supported yet.
- Linux and WSL2 only; there is no native Windows build.
- There are no native (C++/CUDA) extensions of Ayaka's own yet; the GPU kernels come from flashinfer and PyTorch.
- There is no preemption yet. A prompt longer than the KV cache is refused; when running requests together run out of
  KV cache memory, the lowest-ranked one ends early with `length` and a note that memory was the limit.
