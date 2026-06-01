# Ayaka

Ayaka is a Rust-first, native-accelerated LLM inference engine focused on memory-safe KV cache management, prefix-aware scheduling, and research-driven performance optimization.

Ayaka is currently a Rust workspace that establishes the core crate boundaries for an inference engine. The target architecture adds a native C++/CUDA backend, paged KV cache execution, prefix-aware scheduling, OpenAI-compatible serving, Python bindings, benchmarks, and packaging.

## What Exists Today

- Rust workspace with `ayaka-*` crates.
- Tokenizer boundary with local tokenizer loading, chat template rendering, vocabulary metadata, and streaming detokenization.
- Modular sampling processors for temperature, top-k, top-p, min-p, repetition penalty, grammar masking, and RNG strategy.
- Core IDs, flags, and error types.
- Utility modules for environment handling, logging, memory calculations, progress reporting, synchronization, and system inspection.
- Early-stage scaffolding for KV cache, prefix cache, scheduler, executor, kernel API, loader, server, CLI, telemetry, quantization, registry, pipeline, memory, and benchmarks.
- Research notes, ADRs, design wiki pages, specs, and implementation plans.

## Target Direction

Ayaka is building toward:

- a Rust control plane for request lifecycle, scheduler policy, KV ownership, prefix reuse, server APIs, and observability;
- a native execution plane for C++/CUDA kernels and backend-specific memory/runtime management;
- memory-safe paged KV cache management with explicit ownership and lease boundaries;
- prefix-aware scheduling backed by a Radix-style prefix cache;
- benchmark-driven optimization across prefill, decode, attention, KV cache, sampling, and end-to-end serving;
- future support for quantization, speculative decoding, MoE, multi-GPU execution, Python packaging, and production deployment.

## Workspace

| Crate | Role |
|---|---|
| `ayaka` | Public facade crate. |
| `ayaka-core` | Shared IDs, flags, and error model. |
| `ayaka-utils` | Environment, logging, memory, progress, sync, and system helpers. |
| `ayaka-tokenizer` | Tokenization, chat templates, vocab metadata, and streaming detokenization. |
| `ayaka-sampling` | Logits processors and sampling algorithms. |
| `ayaka-kv` | KV cache management boundary. |
| `ayaka-prefix-cache` | Prefix cache boundary. |
| `ayaka-scheduler` | Scheduling boundary. |
| `ayaka-executor` | Execution engine boundary. |
| `ayaka-kernel-api` | Kernel API boundary. |
| `ayaka-memory` | Memory management boundary. |
| `ayaka-loader` | Model and weight loading boundary. |
| `ayaka-pipeline` | Pipeline orchestration boundary. |
| `ayaka-registry` | Registry boundary. |
| `ayaka-quant` | Quantization boundary. |
| `ayaka-server` | Serving boundary. |
| `ayaka-cli` | Command-line interface boundary. |
| `ayaka-telemetry` | Metrics, tracing, and profiling boundary. |
| `ayaka-bench` | Benchmark crate boundary. |

Some crates are intentionally early-stage. See [Current State](docs/architecture/current-state.md) for the implementation snapshot and [Target Architecture](docs/architecture/target-architecture.md) for the completed design.

## Important Libraries

The workspace currently depends on Rust ecosystem libraries for serialization, errors, tracing, async runtime, tokenization, memory mapping, Hugging Face Hub access, CUDA/Metal bindings, Python extension direction, random sampling, progress reporting, and system inspection.

Selected dependencies include `serde`, `thiserror`, `anyhow`, `tracing`, `tokio`, `tokenizers`, `minijinja`, `memmap2`, `safetensors`, `hf-hub`, `candle-core`, `candle-nn`, `pyo3`, `cudarc`, `cuda`, `metal`, `rayon`, `rand`, `half`, `bitflags`, and `sysinfo`.

## Development

Useful local commands:

```powershell
rtk cargo check --workspace
rtk cargo test --workspace
rtk cargo test -p ayaka-tokenizer --test public_api
```

The `rtk` prefix is part of the local development environment.

## Documentation

- [Documentation Index](docs/README.md)
- [Current State](docs/architecture/current-state.md)
- [Target Architecture](docs/architecture/target-architecture.md)
- [Roadmap](docs/roadmap.md)
- [Design Wiki](docs/wiki/07-helix-design/00-index.md)

## Naming Note

Ayaka is the active project name. Some older docs, ADRs, plans, and research notes use Helix naming because they were written before the project rename. Active entry-point docs use Ayaka terminology and mark target architecture separately from current implementation.
