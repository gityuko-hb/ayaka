# AGENTS.md

Ayaka is a high-performance LLM inference engine with CUDA/C++ native extensions.
The Python package is **`ayaka`** (`import ayaka`). That name is already taken on PyPI by another project, so a
different distribution name is needed before publishing.
The project is in early development: do not describe features that are not in the code.

Contribution rules are in **[CONTRIBUTING.md](CONTRIBUTING.md)**. Read them before opening an issue or a pull
request, and follow them whether a person or an agent writes the text.

## Layout

```text
python/ayaka/     the package (ships py.typed); python/ayaka/__version__.py is the only place the version is set
tests/            pytest tests, outside the package
.github/          issue forms, PR template, CI, dependabot, SECURITY.md
pyproject.toml    metadata, build backend and all tool configuration (ruff, pyright, pytest, coverage)
uv.lock           committed; CI installs with `uv sync --locked`
```

## Commands

Use `uv` for everything; do not call `pip` or activate the venv by hand.

```sh
uv sync --extra cpu            # install the project, dev tools and CPU PyTorch (what CI uses)
uv sync --extra cu130          # ...or CUDA 13.0 PyTorch on a machine with an NVIDIA GPU (not both)
uv sync --extra cpu --extra hf --extra tiktoken   # add the tokenizer adapters (optional; the tests do not need them)
uv run ruff check .            # lint
uv run ruff format .           # format
uv run pyright                 # types: strict in python/, standard in tests/
uv run pytest -m "not gpu"     # what CI runs
uv build                       # build the sdist and wheel
```

CI runs the same commands on Ubuntu only for now. All of them must pass before review.

## Conventions

- Python 3.13, fully typed. `python/` is checked by pyright in strict mode; fix the type instead of adding
  `# type: ignore`, and give any unavoidable ignore a specific rule and a reason.
- Mark every test with `unit`, `integration` or `gpu`. A test that needs a GPU is always `gpu`, so hosted CI can
  skip it. Do not make a test pass by skipping it silently.
- Dependencies: runtime dependencies go in `[project].dependencies`, tools in `[dependency-groups]`
  (`uv add <pkg>` or `uv add --dev <pkg>`). Keep the runtime list small and justify each new entry in the PR.
- PyTorch is chosen by an extra, never by editing dependencies: `cpu` or `cu130` in `pyproject.toml`, each wired to
  its own PyTorch index through `[tool.uv.sources]`. The two cannot be installed together, and a plain `uv sync`
  installs no torch. Do not add `torch` to `[project].dependencies`: that would make `uv.lock` pull the CUDA
  build (and ~15 `nvidia-*` packages) into hosted Linux CI. Keep both extras on the same torch version range.
- Tokenizer adapters are optional extras, not runtime dependencies: `hf` (transformers, tokenizers,
  huggingface-hub) and `tiktoken` (tiktoken, jinja2). The adapters import them lazily through
  `ayaka.utils.import_utils.require_module`, so never import them at module top level.
- Titles of PRs follow Conventional Commits (`type(scope): description`); see CONTRIBUTING.md.
- Never report a measurement, test result or log you did not produce. Separate what you ran from what you did
  not run, and say so for GPU and CUDA checks in particular.

## Native code (planned, not started)

There is no C++ or CUDA code yet, so the build backend is hatchling. When the first extension lands:

1. Agree the approach in an issue (bindings library, CUDA architectures, minimum CUDA version).
2. Switch `[build-system]` to `scikit-build-core`, add a root `CMakeLists.txt` and keep C++/CUDA sources outside
   `python/` (for example `csrc/`). The Python package layout does not change.
3. Keep a CPU fallback or a clear import-time error, so `import ayaka` works on machines without CUDA.
4. Build every backend a change touches before review, and report GPU runs separately from compilation.
5. Build release wheels with cibuildwheel; hosted CI can compile CUDA code but cannot run it.
