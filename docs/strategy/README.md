# Ayaka Strategy

Internal strategy documentation for **Ayaka** (`ayaka-llm` distribution, `ayaka` Python package).
These files live under `docs/strategy/` and remain untracked by `.gitignore`; public claims live
exclusively in the allowlisted pages under `docs/` and must always be backed by committed
measurements in `docs/EXPERIMENTS.md` and `docs/CAPABILITIES.md`.

## North Star

> **A developer on a small-to-mid NVIDIA GPU points their coding agent (Claude Code, Codex CLI,
> OpenCode) at Ayaka with two commands — and gets predictable memory planning, schema-reliable
> tool calling, exact-output acceleration, and honest, measured performance.**

## Core Thesis

Most LLM inference engines optimize for one of two extremes:

1. **Datacenter throughput engines** (`vLLM`, `SGLang`, `TensorRT-LLM`) assume multi-GPU H100/B200
   nodes, heavy dependency stacks, and high-concurrency batching. On a 4–24 GB single GPU running
   an interactive coding agent, their operational overhead, memory unpredictability, and client
   friction get in the way.
2. **General-purpose local runners** (`llama.cpp` and wrappers) excel at single-user raw C++ speed
   across consumer hardware, but treat coding-agent serving concerns — multi-protocol wire fidelity
   (OpenAI Chat/Completions/Responses + Anthropic Messages), client prompt-churn normalization,
   prefix-cache preservation, grammar-constrained tool calls inside CUDA graphs, and reproducible
   pre-registered engineering gates — as secondary.

Ayaka occupies the gap: **a coding-agent-first local inference engine built for 4–24 GB NVIDIA GPUs
(scaling up to 32–64 GB+ Frontier MoE), engineered with datacenter-grade invariants and strict
measurement honesty.**

## Strategic Priorities & Long-Horizon Arc (2026-11)

1. **`0.2.0` GA (M4 – M11) — Complete Single-Machine Coding-Agent & Frontier MoE Engine**:
   - Track internal progress via pre-release checkpoints (`v0.2.0-a1` after M4 through `v0.2.0-a7`
     after M10), launching `ayaka-llm 0.2.0` publicly on PyPI once **M11 (Frontier MoE)** passes.
   - Delivers exact-output speculation (M4), **Serving Core v2** with dedicated prompt rendering,
     request-scoped output interpretation (`none` & `thinking`), multi-family reasoning/tool
     parsers, and Engine + Render gRPC (M5), crash-proof memory & host-RAM KV tiering (M6), fast
     FP8 & W4A16 quantized GEMM via FlashInfer/Triton/Marlin (M7), hybrid GDN/MLA/vision (M8),
     RAM-offloaded MoE verified vs Strata & FreeToken (M9), agent platform & MCP (M10), and
     Frontier MoE with sparse attention, Engram, RAM+SSD hierarchy, and FP4 (M11).
2. **`0.3.0` (M12 – M13) — Scale-Out & Native Kernel Engine (`kops` / Windows Native)**:
   - Multi-GPU TP/PP/EP, P/D disaggregation, and `Kimi K3`-class scale-out (M12), paired with the
     native C++/CUDA kernel engine (`csrc/`, Blackwell FP4/FP8 fused kernels, MoE offload DMA) and
     native Windows wheels without WSL2 (M13 — de-risked early by spike `M13-N00` right after M7).
3. **`0.4.0` (M14) — Multi-Agent & Distributed Prefix Fabric**:
   - Zero-copy shared prefix/KV/state trees for coding-agent swarms, cross-agent speculative KV
     prewarm, multi-tenant interactive QoS priority lanes, and opt-in persistent session/KV
     snapshots across server restarts.
4. **`1.0.0` GA (M15) — Omni-Modal Computer-Use & Stable ABI**:
   - Real-time screen/video/audio streaming with incremental frame KV reuse for computer-use coding
     agents, frozen public Python/Wire/Plugin ABI with SemVer guarantees, and full security/privacy
     audit.

## Non-Goals (For Now)

- **Not a model training or fine-tuning framework.** Inference and serving only.
- **Not a GUI chat app or wrapper ecosystem.** Ayaka is an engine, CLI, and protocol server; coding
  agents and editors are the UI.
- **Not chasing unverified leaderboard claims.** Every headline number is measured on named
  hardware with raw `results.json` and pre-registered thresholds.
- **Not supporting unsafe checkpoint formats.** Pickle weights (`.bin`, `.pt`) and implicit remote
  code execution stay rejected by default.
- **Not rewriting every kernel in C++/CUDA prematurely.** Use PyTorch, FlashInfer, Triton, and
  proven quantized GEMM kernels first; invest in `csrc/` only where existing kernels cannot meet a
  measured gate or where Windows native requires it.

## Document Index

| Document | Contents |
| --- | --- |
| [positioning.md](positioning.md) | Target users, 3-tier hardware ladder, competitive landscape, and 4 differentiators |
| [principles.md](principles.md) | Measurement gates, engineering invariants, safety/privacy posture, and release strategy |
| [architecture_bets.md](architecture_bets.md) | Key technical bets, trade-offs, and trigger conditions for native C++/CUDA kernels |
| [../ROADMAP/README.md](../ROADMAP/README.md) | Execution roadmap, milestone map, dependency graph, and sync with GitHub issue #62 |
