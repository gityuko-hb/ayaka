# Contributing to Ayaka

**AI assistance is welcome. The contributor is responsible for understanding the change and supplying real
evidence.** These rules apply to issues, pull requests and the agents helping prepare them. You should be able
to explain every changed line and answer review questions about it.

## Before starting

1. Search [issues](https://github.com/gityuko-hb/ayaka/issues) and
   [pull requests](https://github.com/gityuko-hb/ayaka/pulls), including closed ones. Add new evidence to an
   existing thread when it covers your problem.
2. Read [AGENTS.md](AGENTS.md) for the layout and commands.
3. Get agreement on scope in an issue before a large feature, a refactor, an architecture change or a change to a
   default. Agreement on scope does not guarantee a merge. Focused fixes can go straight to a PR.
4. Keep one problem per PR. Update that PR after feedback; do not open competing attempts for the same change.

## Set up

You need [uv](https://docs.astral.sh/uv/). It installs the Python version in `.python-version` for you.

```sh
uv sync --extra cpu                  # creates .venv: the project, the dev tools and the CPU build of PyTorch
uv sync --extra cu130                # ...or the CUDA 13.0 build, on Linux/Windows with an NVIDIA GPU
uv run pre-commit install            # optional: run the checks below on every commit

uv run ruff check .                  # lint
uv run ruff format --check .         # formatting (drop --check to apply it)
uv run pyright                       # types (strict for python/, standard for tests/)
uv run pytest -m "not gpu"           # tests that run anywhere
uv run pytest -m gpu                 # tests that need a CUDA GPU (CI does not run these)
```

Pick the PyTorch build once, with `uv sync`. `uv run` keeps whatever is installed, so you do not repeat the extra on
every command. A later plain `uv sync` removes torch again, and `cpu` and `cu130` cannot be installed together:
switching from one to the other is just `uv sync --extra <other>`.

Commit `uv.lock` together with any change to dependencies (`uv add`, `uv remove`, `uv lock --upgrade-package`).

## Titles

Issue and pull request titles follow [Conventional Commits](https://www.conventionalcommits.org/):
`type(scope): description`. The scope is optional.

| Type | Use for |
| --- | --- |
| `feat` | a new feature |
| `fix` | a bug fix |
| `refactor` | a change that is neither a fix nor a feature |
| `perf` | a performance improvement |
| `docs` | documentation only |
| `test` | adding or correcting tests |
| `build` | build system or dependencies |
| `ci` | CI configuration and scripts |
| `chore` | anything else that does not touch `python/` or `tests/` |

Examples: `fix(cli): keep the config file when the input is invalid`, `perf(kernels): fuse the softmax pass`.
Add `!` before the colon for a breaking change (`feat(api)!: rename generate to run`). A check on GitHub fails a
pull request whose title does not match. Pull requests are squash-merged, so the title becomes the commit message.

## Report a problem people can investigate

Use the matching [issue form](https://github.com/gityuko-hb/ayaka/issues/new/choose). A bug report should
include:

- The smallest snippet or command that reproduces it, and the expected and actual behavior.
- The ayaka version or commit, Python version, OS, GPU and driver, and the model and settings involved.
- The exact error and relevant logs, with credentials and private content removed.
- Whether you reproduced it yourself. Label a finding based only on reading code or a scanner as
  **unconfirmed static analysis**, and explain the suspected path without presenting it as an observed failure.

Do not invent reproduction steps, errors, affected hardware or severity. Report security problems privately: see
[SECURITY.md](.github/SECURITY.md).

## Prepare a focused pull request

Lead with the problem and the resulting behavior. Explain why the approach is needed, the important limits and
any compatibility effects. A file-by-file narration or an AI-generated summary of the diff is not a substitute.

- Keep unrelated formatting, cleanup, dependency changes and speculative abstractions out of the PR.
- Add a small test that fails without your change. Do not add tests that only restate the implementation.
- Add documentation where it changes what a user or developer needs to know.
- Read the complete diff, including generated files, before submitting. Remove debugging output, secrets,
  private paths and accidental files.
- New features are opt-in. The default behavior stays the same unless the PR is about changing it.

## Show the validation you actually performed

List the exact commands, the revision and environment tested, and what you observed. Keep **passed**, **failed**
and **not run** separate (the PR template has the three headings). A command you suggest running is not
evidence that it passed, and a CI run on a different revision is not validation of the submitted code.

Hosted CI has no GPU. Report GPU runs and CUDA builds separately from CPU checks, and say which GPU, driver and
CUDA version you used. If you lack a required environment, keep the PR as a draft, name the missing checks and ask
for help in the same thread. Do not claim compatibility you have not tested.

Performance claims need before/after runs on the same hardware and workload, with every changed setting
identified. Keep the individual runs and the variance. One fast run does not justify changing a default or
claiming a speedup for other machines.

## Rules for AI-assisted work

Use agents to help investigate, implement and explain work that you can review. Verify their claims against the
source, actual command output or a cited primary reference before submitting.

Never submit fabricated measurements, invented test results, fake logs or claims you have not checked. Do not
turn a repository scan into a batch of speculative issues or PRs: consolidate related findings, establish their
effect and submit the useful evidence. An agent's confidence, or another model's approval, is not validation.

Maintainers may request a smaller change, return a PR to draft, or close a submission that lacks evidence,
duplicates existing work or ignores agreed scope. A smaller, understandable contribution with honest limits is
easier to review and maintain.
