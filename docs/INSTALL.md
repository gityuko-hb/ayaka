# Installing Ayaka

Ayaka is in early development and has no release on PyPI yet (the name `ayaka` is taken there), so it is installed
from a checkout with [uv](https://docs.astral.sh/uv/). Every command below runs from the checkout's root.

## 1. Get the code and uv

```sh
git clone https://github.com/gityuko-hb/ayaka.git
cd ayaka
```

Install uv by following its own instructions if `uv --version` does not answer. Do not call `pip` or activate the
virtual environment by hand: `uv sync` creates `.venv` and `uv run` uses it.

## 2. Pick the extras

PyTorch comes from an extra, never from the base install; pick **one** of `cpu` and `cu130`.
The extras fall into three groups — one extra per real install decision.

**Accelerator** — pick exactly one of the torch extras; `flashinfer` is Linux/WSL2 only:

| Extra | What it adds | When |
| --- | --- | --- |
| `cpu` | PyTorch for the CPU | no NVIDIA GPU, or CI |
| `cu130` | PyTorch built for CUDA 13.0 | an NVIDIA GPU (not with `cpu`) |
| `flashinfer` | the flashinfer attention backend | Linux or WSL2 with `cu130`; needs `nvcc` on `PATH` |

**Tokenizers** — either or both; `import ayaka` and the tests need neither:

| Extra | What it adds | When |
| --- | --- | --- |
| `hf` | transformers, tokenizers, huggingface-hub | Hugging Face tokenizers and downloads from the Hub |
| `tiktoken` | tiktoken and jinja2 | tiktoken-based tokenizers |

**Server** — either or both; `structured` is engine-level and works from `ayaka run` too:

| Extra | What it adds | When |
| --- | --- | --- |
| `serve` | aiohttp, for `ayaka serve` | serving HTTP clients |
| `structured` | llguidance, for JSON answers and forced tool calls | `response_format`, `tool_choice`, strict tools |

A machine with an NVIDIA GPU under Linux or WSL2, serving coding tools:

```sh
uv sync --extra cu130 --extra flashinfer --extra serve --extra structured --extra hf
```

A machine without a GPU:

```sh
uv sync --extra cpu --extra serve --extra structured --extra hf
```

## 3. WSL2

- Install the NVIDIA driver on Windows; WSL2 uses it, and nothing NVIDIA is installed inside Linux except the CUDA
  toolkit when `flashinfer` has to compile (`nvcc` on `PATH`).
- Keep the checkout and its `.venv` in the Linux file system (for example `~/ayaka`), never under `/mnt/c` or
  `/mnt/d`: a virtual environment shared with Windows breaks, and file access across the boundary is slow.
- The Hugging Face cache defaults to `~/.cache/huggingface` inside Linux; keep it there for the same reason.
- WSL2 gives Linux a share of the machine's memory; `free -g` shows it, and a model that needs more RAM than that
  will not load (see [TROUBLESHOOTING.md](TROUBLESHOOTING.md)).

## 4. Check the installation

```sh
uv run ayaka --version
uv run ayaka info
```

`ayaka info` lists the PyTorch build, the devices it sees, the backends, which extras are installed and the
directories Ayaka uses. A GPU missing from its `device` line will not be used.

## 5. First run

Before loading a model, ask whether it fits:

```sh
uv run ayaka plan HuggingFaceTB/SmolLM2-135M --device cuda --dtype fp16
```

`plan` reads `config.json` only and never downloads (a repo id must already be in the Hub cache; otherwise point it
at a local directory). Exit code 5 means it does not fit, with the shortfall and ways out.

Then generate a first answer (this downloads the model unless `--offline` is given):

```sh
uv run ayaka run HuggingFaceTB/SmolLM2-135M --device cuda --dtype fp16 -p "Say hello."
```

and serve it to OpenAI and Anthropic clients on `127.0.0.1:8000`:

```sh
uv run ayaka serve HuggingFaceTB/SmolLM2-135M --device cuda --dtype fp16
```

For a coding tool, continue with `ayaka connect` ([AI_SETUP.md](AI_SETUP.md) has the whole sequence).
