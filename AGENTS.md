# Repository Guidelines

## Project Structure & Module Organization

Ayaka is a Rust 2024 workspace. Crates live under `crates/*`, with the public facade in `crates/ayaka`, shared types in `crates/ayaka-core`, utilities in `crates/ayaka-utils`, tokenization in `crates/ayaka-tokenizer`, and sampling in `crates/ayaka-sampling`. Engine boundaries such as `ayaka-kv`, `ayaka-scheduler`, `ayaka-executor`, `ayaka-server`, and `ayaka-kernel-api` are early-stage scaffolds. Crate-level integration tests live beside crates, for example `crates/ayaka-tokenizer/tests/public_api.rs`. Documentation lives in `docs/`; public static assets are in `public/`.

## Build, Test, and Development Commands

Always prefix shell commands with `rtk`.

- `rtk cargo check --workspace`: type-check the full workspace.
- `rtk cargo build --workspace`: build all workspace crates.
- `rtk cargo test --workspace`: run all Rust tests.
- `rtk cargo test -p ayaka-tokenizer --test public_api`: run tokenizer public API tests.
- `rtk cargo fmt --all`: format Rust code using `rustfmt.toml`.
- `rtk cargo clippy --workspace --all-targets`: run lint checks when available.

## Coding Style & Naming Conventions

Use Rust 2024 idioms and keep formatting aligned with `rustfmt.toml`: 4-space indentation, field init shorthand, try shorthand, and trailing commas in match blocks. Crate and package names use kebab-case (`ayaka-tokenizer`); modules and functions use `snake_case`; types and traits use `UpperCamelCase`; constants use `SCREAMING_SNAKE_CASE`. Keep crate boundaries focused and avoid adding cross-crate dependencies without a clear ownership reason.

## Testing Guidelines

Prefer focused tests near the crate that owns the behavior. Name tests after the behavior under test, for example `streaming_detokenizer_buffers_partial_utf8`. For new public APIs, include coverage in crate tests or integration tests. Run the narrowest useful command first, then `rtk cargo test --workspace` before handing off broad changes.

## Commit & Pull Request Guidelines

Project history uses concise Conventional Commit-style messages, often emoji-prefixed, such as `feat(tokenizer): ...` and `fix: ...`. Keep commit subjects imperative and scoped when possible. Pull requests should describe the change, list verification commands, link related issues or plans, and call out docs, tests, or roadmap impacts. Include screenshots only for visual changes.

## Agent-Specific Instructions

Do not claim target architecture features are implemented unless source, tests, and docs verify them. Preserve the current-state versus target-roadmap distinction in active docs. Do not modify unrelated files, and never revert user changes without explicit instruction.
