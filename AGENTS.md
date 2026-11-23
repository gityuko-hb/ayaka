# AGENTS.md

Ayaka is a high-performance LLM inference engine with CUDA/C++ native extensions.
The Python package is **`ayaka`** (`import ayaka`). That name is already taken on PyPI by another project, so the
distribution is named **`ayaka-llm`** (`pyproject.toml`, `importlib.metadata.version("ayaka-llm")`).
The project is in early development: do not describe features that are not in the code.

Contribution rules are in **[CONTRIBUTING.md](CONTRIBUTING.md)**. Read them before opening an issue or a pull
request, and follow them whether a person or an agent writes the text.

## Layout

```text
python/ayaka/     the package (ships py.typed); python/ayaka/__version__.py is the only place the version is set
tests/            pytest tests, outside the package
docs/             the public pages (INSTALL, TROUBLESHOOTING, CAPABILITIES, SAFETY, AI_SETUP, AI_REFERENCE) and
                  EXPERIMENTS; .gitignore tracks them by name, everything else under docs/ is internal and ignored
benchmarks/       benchmark scripts, results/ (one folder per run) and history/
tools/            generate_capabilities.py writes docs/CAPABILITIES.md from docs/capabilities.toml
.github/          issue forms, PR template, CI, SECURITY.md
pyproject.toml    metadata, build backend and all tool configuration (ruff, pyright, pytest, coverage)
uv.lock           committed; CI installs with `uv sync --locked`
```

## Commands

Use `uv` for everything; do not call `pip` or activate the venv by hand.

```sh
uv sync --extra cpu            # install the project, dev tools and CPU PyTorch (what CI uses)
uv sync --extra cu130          # ...or CUDA 13.0 PyTorch on a machine with an NVIDIA GPU (not both)
uv sync --extra cu130 --extra flashinfer   # Linux/WSL2 GPU: the flashinfer attention backend (needs nvcc on PATH)
uv sync --extra cpu --extra hf --extra tiktoken   # add the tokenizer adapters (optional; the tests do not need them)
uv sync --extra cpu --extra serve   # add the HTTP server (aiohttp); CI's lint and test jobs install it
uv sync --extra cpu --extra serve --extra structured   # plus structured output (llguidance), as CI does
uv run ruff check .            # lint
uv run ruff format .           # format
uv run pyright                 # types: strict in python/, standard in tests/
uv run pytest -m "not gpu and not integration"   # the unit tests: what the `test` job runs
uv run pytest -m integration  # real tokenizers + worker processes; needs the hf and tiktoken extras (CI job `adapters`)
uv run pytest -m gpu           # on the dev GPU only; AYAKA_REQUIRE_GPU=1 turns a missing GPU into a failure
uv sync --extra cpu --extra serve --extra structured --group sdk   # the server against the official SDKs (CI job `serving`)
uv run pytest -m integration tests/e2e/test_serving.py   # the signal tests need POSIX (Linux, WSL2)
uv build                       # build the sdist and wheel
```

Under WSL2, keep a separate checkout with its own `.venv` in the Linux file system and run `uv` there, never
in the Windows checkout.

CI runs the same commands on Ubuntu only for now. All of them must pass before review.

## Documentation

The public pages under `docs/` describe what the code does, nothing it does not. `docs/CAPABILITIES.md` is generated:
edit `docs/capabilities.toml` (a row is `runs` only with a result file, `does-not-run` only with a result file and a
reason, otherwise `untested`) and run `uv run python tools/generate_capabilities.py`; `tests/test_docs.py` fails when
the page drifts, when a link breaks or when a page names an `ayaka` command or flag that does not exist. The
"Request fields" section of `docs/AI_REFERENCE.md` is generated too: change the table in
`python/ayaka/entrypoints/wire/policy.py` and run `uv run python tools/generate_wire_reference.py`. A new public
page is added to the allowlist in `.gitignore` by name.

## Layering

Every package `__init__.py`, including `ayaka.core`, exposes its public names through
`build_lazy_exports`: keep the literal `__all__` and `_EXPORTS` table in sync, and add a `TYPE_CHECKING`
block for the real types. Do not replace lazy exports with eager imports, so importing a package never
loads torch, NVML, HuggingFace Hub or the tokenizer adapters.

### One home per name

Every public name has exactly one defining module, and that module owns the name. `__all__` lists
what a module defines, never what it imports. A module may import a name to raise it, annotate
with it or check it, but the imported name stays out of `__all__` and out of the module's documented
public API. For example, `ayaka.utils.cancellation` raises `ayaka.core.errors.CancelledError` but
does not export it.

- Import every name from its defining module, in `python/` and in `tests/` alike:
  `from ayaka.core.errors import CancelledError`, never from `ayaka.utils.cancellation`. A test that
  follows the real owner fails in the right place.
- A package `__init__.py` exposes names only from modules inside that package (a nested subpackage
  counts), and `_EXPORTS` must name the module that defines the name, never one that re-exports it.
  A facade does not reach into another package: a shared type belongs in `ayaka.core`, the lowest
  contract layer.
- `tests/test_public_api.py` enforces this over `python/ayaka` and `tests/` with an AST scan. Do not
  weaken it with an allowlist or a skip; fix the import instead.

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
  The `flashinfer` extra (Linux only) also takes torch from the cu130 index and conflicts with `cpu`; its
  version is pinned exactly and upgraded in a PR of its own with the `gpu` parity and end-to-end results.
- Structured output is the optional `structured` extra (llguidance, no dependencies of its own). It is imported
  only when a request asks for a constraint (`ayaka.sampling.structured.compiler`); never at module top level.
- Tokenizer adapters are optional extras, not runtime dependencies: `hf` (transformers, tokenizers,
  huggingface-hub) and `tiktoken` (tiktoken, jinja2). The adapters import them lazily through
  `ayaka.utils.import_utils.require_module`, so never import them at module top level.
- Titles of PRs follow Conventional Commits with an icon in front (`<icon> type(scope): description`); see
  CONTRIBUTING.md.
- Actions in the workflows stay pinned to a full commit SHA with the version in a comment. There is no dependabot:
  a maintainer bumps the pins by hand (a tool such as `pinact` or `ratchet` helps), reads the diff and commits it.
- Search past issues and PRs before writing a new one, and update what you find. Run
  `gh issue list --state all --search "<keywords>"` and `gh pr list --state all --search "<keywords>"`, closed ones
  included. If a thread already covers the work, add the new evidence or scope to it (a comment or an edit of its
  checklist) instead of opening a second one. When earlier work is the starting point, link it by number in the
  text (for example "M0 is done, see #10") rather than restating it. Say in the PR or issue which threads you
  searched. Reading with `gh` is fine; creating or editing issues, PRs and labels needs the user's request.
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
