# AGENTS.md

Project instructions for coding agents working on **Ayaka**: a Rust-first LLM inference engine using Candle as the tensor/model substrate and custom C++/CUDA kernels behind a stable C ABI.

## Project Direction

- Rust is the source of truth for runtime, scheduler, memory ownership, request state, and public APIs.
- Candle is the tensor/model substrate. Do not design Ayaka as a PyTorch extension.
- Custom C++/CUDA code is an accelerator layer only.
- Python is optional and lightweight through PyO3.
- Do not add PyTorch, ATen, torch bindings, pybind11, or Triton unless explicitly requested.

Target flow:

```text
Rust runtime/scheduler/KV ownership
  -> ayaka-candle adapter
  -> ayaka-kernel-api FFI
  -> native C ABI
  -> C++ launchers
  -> CUDA kernels
```

Kernel rule: kernels must not know RadixTree, RequestManager, Candle Tensor, or scheduler internals. Kernels receive flat descriptors: pointers, shapes, strides, dtype, offsets, page tables, and stream.

## Current Phase

Phase 1 is foundation-only:

```text
crates/ayaka-core   # IDs, dtype, device, shape, stride, layout, tensor metadata
crates/ayaka-error  # StatusCode, ErrorKind, AyakaError, Result<T>, FFI status
```

Do not implement scheduler, KV allocator, prefix/radix cache, Candle execution, CUDA kernels, Python bindings, or server APIs in Phase 1 unless explicitly requested.

## Agent Work Style

Before editing:
- Restate the goal in one short sentence.
- Identify the smallest file set needed.
- State assumptions that affect architecture or public API.
- Use a brief plan for multi-file changes.

While editing:
- Make surgical changes only.
- Match existing style and naming.
- Do not refactor unrelated code.
- Do not add speculative abstractions.
- Do not add dependencies without a clear reason.
- Remove only dead code created by your own change.

After editing:
- Run targeted verification first.
- Then run broader checks if shared APIs or workspace config changed.
- Report what passed, what failed, and why.

## Verification Commands

Targeted checks:

```bash
cargo check -p ayaka-core
cargo check -p ayaka-error
cargo test -p ayaka-core
cargo test -p ayaka-error
```

Workspace checks:

```bash
cargo fmt --all --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
```

If a command is unavailable because the repo is not initialized yet, say so and run the closest valid command.

## Expected Workspace Shape

```text
ayaka/
├── Cargo.toml
├── README.md
├── ROADMAP.md
├── TREE.md
├── CLAUDE.md
├── AGENTS.md
├── crates/
│   ├── ayaka-core/
│   └── ayaka-error/
├── native/
├── third_party/
├── docs/
├── examples/
├── benches/
├── tests/
└── tools/
```

Dependency direction:

```text
ayaka-core <- ayaka-error <- all other crates
```

Rules:
- `ayaka-core` must not depend on Candle, CUDA, native code, scheduler, runtime, or `ayaka-error`.
- `ayaka-error` must stay lightweight.
- Public IDs must use newtypes, not raw `u32`/`u64` in public APIs.
- FFI-facing structs must use `#[repr(C)]` or `#[repr(transparent)]` where appropriate.
- Avoid foundation crate renames once types become public.

## Rust Rules

Use:
- Rust 2021 edition unless workspace config says otherwise.
- `serde` only behind feature flags for metadata.
- `bitflags` for flag sets.
- `static_assertions` for ABI/layout checks when useful.

Avoid:
- `anyhow` in public library APIs.
- panics in library code except impossible test-only invariants.
- global mutable state.
- async in foundation crates.
- macros unless they remove clear repetition.
- `unsafe` in Phase 1. If unavoidable, isolate it, document invariants, and test it.

## ayaka-core Scope

`ayaka-core` owns generic primitive types only:
- IDs: `RequestId`, `SequenceId`, `BatchId`, `NodeId`, `PageId`, `LayerId`, `HeadId`, `DeviceId`.
- Aliases: `TokenId`, `Position`, `Epoch`, `Generation`.
- Types: `DType`, `DeviceKind`, `Device`, `DeviceInfo`.
- Metadata: `Shape`, `Strides`, `TensorLayout`, `TensorMeta`.
- Inference layout metadata: `KVCacheLayout`, `KVCacheFormat`, `AttentionShape`.
- Memory, architecture, scalar, time, and flag metadata.

`ayaka-core` must not own real tensor storage, Candle wrappers, CUDA streams, request lifecycle, scheduler policy, KV page allocation, or Radix tree logic.

ID style:

```rust
#[repr(transparent)]
pub struct PageId(pub u32);
```

Every ID type should provide `new`, `raw`, `INVALID`, `is_valid`, and `is_invalid`.

## ayaka-error Scope

`ayaka-error` owns shared error handling:
- `StatusCode` for stable ABI codes.
- `ErrorKind` for Rust domain errors.
- `AyakaError` with structured message and context.
- `Result<T>` alias.
- `ResultExt` and `OptionExt`.
- `AyakaStatus` for C ABI boundaries.
- `AyakaBail` and `AyakaEnsure`.

Rules:
- `StatusCode` values must be stable once public.
- Rust internals use `Result<T, AyakaError>`, not FFI status codes.
- FFI boundaries convert to/from `AyakaStatus`.
- Error messages should include invalid values when useful.
- Preserve domain context: prefer `Shape`, `DType`, `Cuda`, `KernelLaunch`, `KVCache`, etc. over generic `Internal`.

## Candle Integration Rules

Candle integration belongs in a later `ayaka-candle` crate.

Do not import `candle_core` in `ayaka-core` or `ayaka-error`.

Future bridge:

```text
candle_core::Tensor
  -> ayaka-candle TensorView adapter
  -> ayaka-kernel-api descriptor
  -> native C ABI
```

If a type is Candle-specific, keep it out of Phase 1 unless it is truly generic tensor metadata.

## Native / CUDA Rules

Native code belongs under `native/`, not in Phase 1 unless explicitly requested.

Future native layout:
- `native/include/ayaka/` contains stable C ABI headers.
- `native/src/` contains C++ host launchers and descriptor validation.
- `native/csrc/` contains CUDA kernels.
- `third_party/cutlass` is allowed for GEMM-heavy kernels.
- Avoid PyTorch/ATen/torch headers entirely.

CUDA kernels must be descriptor-driven. Never pass Rust/Candle-specific structures into kernels.

## Testing Rules

Every new public type should have focused tests for constructor behavior, invalid sentinel behavior, byte/size calculations, layout invariants, feature-gated serde behavior, and ABI layout if FFI-facing.

For foundation crates, prefer small unit tests over large integration tests.

Never claim a command passed unless it was actually run.

## Documentation Rules

When adding public types:
- Add short doc comments explaining what owns the concept.
- Include units for byte, token, page, head, layer, and device fields.
- Keep architecture essays in `docs/` or `TREE.md`, not in code comments.

## Safety Rules

Do not commit secrets, local machine paths, generated binaries, or network calls in build scripts.

Ask before deleting files, broadly renaming public APIs, changing workspace layout, adding dependencies, or introducing unsafe code.

## Response Format for Coding Tasks

Use this shape for non-trivial tasks:

```text
Goal: ...
Plan:
1. ...
2. ...
Verification:
- ...
Result:
- ...
```

For trivial one-file changes, keep the response shorter.
