<!--
The PR title must follow Conventional Commits: `type(scope): description`.
Example: `fix(cli): keep the config file when the input is invalid`.
Read CONTRIBUTING.md first. Lead with the problem and the result, not a file-by-file list.
Placeholders below are not results: replace them with what you observed, or write "none".
-->

## Type of change

- [ ] **feat**: a new feature
- [ ] **fix**: a bug fix
- [ ] **refactor**: neither a fix nor a feature
- [ ] **perf**: a performance improvement
- [ ] **docs**: documentation only
- [ ] **test**: adding or correcting tests
- [ ] **build**: build system or dependencies
- [ ] **ci**: CI configuration and scripts
- [ ] **chore**: anything else that does not touch `python/` or `tests/`

## Problem and result
<!-- What triggered this change. How it behaves before and after. -->

## Related issue
<!-- `Fixes #123`, plus the maintainer's agreement on scope when the change needs one (features, refactors, defaults). Write "none" if not applicable. -->

## Approach
<!-- Why this fix. Compatibility effects. Keep unrelated cleanup out of the PR. -->

## Breaking changes

- [ ] No
- [ ] Yes: describe the impact and the migration path below

## Validation
<!-- Exact commands, the revision and the environment tested, what you saw. Keep the three lists apart. -->

**Passed:**
<!-- command, revision/environment, observed result -->

**Failed:**
<!-- command and failure, or "none observed" -->

**Not run:**
<!-- check, why, and what stays unverified (for example CUDA builds or GPU runs you could not do) -->

## Checklist

- [ ] I read my full diff, including generated files, and can explain every changed line.
- [ ] `uv run ruff check .` and `uv run ruff format --check .` pass.
- [ ] `uv run pyright` passes.
- [ ] `uv run pytest -m "not gpu and not integration"` passes, and I added a test that fails without this change.
- [ ] If the change touches the tokenizer adapters, the process pool or CI: `uv sync --extra hf --extra tiktoken`, then `uv run pytest -m integration` passes.
- [ ] I updated the documentation where it changes what a user or developer needs to know.

## Limits
<!-- Known limitations or remaining work. -->
