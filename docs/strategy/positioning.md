# Positioning, Hardware Tiers, and Competitive Landscape

## Target Users & Primary Workload

Ayaka's first-class workload is **interactive coding agents** — specifically **Claude Code**,
**Codex CLI**, and **OpenCode** (expanding to 6 harnesses in M10) — running against a local or
single-node GPU server.

Coding-agent traffic has distinct properties that drive every engineering choice in Ayaka:

- **Massive, mostly static prefixes**: 10K–50K+ tokens of system instructions, repo context, and
  14–25+ tool schemas sent on every turn. Small client churn (e.g., Claude Code's per-session
  billing header) destroys naive prefix caches; normalizing it recovered `84.5%` vs `0.2%` cached
  tokens on real Llama-3.2-1B sessions (`docs/EXPERIMENTS.md`, #57).
- **Strict structured tool calls**: agents break immediately when a model emits malformed JSON,
  wrong argument types (e.g., `"tty": "true"` instead of boolean in Codex, fixed in #110), or
  unclosed reasoning blocks. Grammar-constrained decoding must run fast inside CUDA graphs (#73).
- **High edit/copy repetition**: agents read files and rewrite them with localized edits, making
  exact-output speculative decoding (lookup, predicted outputs, MTP, DFlash) disproportionately
  effective when low-batch decode is memory-bandwidth bound.
- **Low concurrency, latency-sensitive**: 1–4 active streams on a single developer machine where
  TTFT, inter-token latency (ITL), and burst jitter matter more than 256-stream datacenter
  throughput.
- **Hard VRAM ceilings**: a single context explosion or concurrent burst must never crash the
  server or silently drop an active coding session.

## Three-Tier Hardware Ladder

Rather than optimizing for only one card, Ayaka positions explicitly across three NVIDIA GPU tiers:

| Tier | Representative Hardware | Primary Role in Strategy | Target Models & Precision |
| --- | --- | --- | --- |
| **Tier 1: Entry / Floor (4–8 GB VRAM)** | RTX 3050/4050/4060 Laptop (WSL2 & Linux) | Proves memory discipline, `ayaka plan`, W4A16 execution, and host offload; keeps small models genuinely runnable | 1B–3B in fp16/bf16; 7B–8B in W4A16 (M7); small MoE with RAM offload (M9) |
| **Tier 2: Primary Sweet Spot (12–24 GB VRAM)** | NVIDIA L4 24 GB, RTX 3090/4090/5070 Ti/5080 | **Flagship daily-driver coding-agent tier** for `0.2.0`+ PyPI adoption and head-to-head benchmarks | 7B–9B in bf16/FP8 (with DFlash/MTP); 14B (Phi-4) in fast FP8; 27B–30B MoE (Qwen3-30B-A3B) in FP8/W4 with RAM offload |
| **Tier 3: Frontier Local (32–64 GB+ VRAM)** | RTX 5090, dual-24 GB, 48–64 GB workstation / multi-GPU | High-context Frontier MoE, multi-GPU TP/PP, P/D disaggregation | DeepSeek/Qwen/GLM/MiniMax Frontier MoE (M11), Kimi K3 (M12) |

> **Honest floor rule**: If a coding tool's full prompt or a model's full context does not fit a
> Tier 1 (4 GB) card, `ayaka plan --for <tool>` and `docs/CAPABILITIES.md` state that plainly with
> exact token limits and remedies rather than hiding behind paging or OOM crashes.

## Competitive Landscape

Every competitor comparison in Ayaka is measured on the **same machine, model, and comparable
settings** (never copied from third-party H100/B200 blog posts):

| Engine | Core Strength | Where Ayaka Differentiates | Benchmark Touchpoints |
| --- | --- | --- | --- |
| **llama.cpp** | Fast C++/CUDA GGUF execution, huge hardware/OS reach, minimal setup | Coding-agent native UX (`ayaka connect`, prompt-churn normalization, Anthropic Messages + OpenAI Responses wire fidelity, request-scoped thinking & multi-family parsers), grammar-in-CUDA-graph, multi-request prefix cache & host KV tiering, reproducible Python/CUDA extensibility | Measured on RTX 3050 (#59) and NVIDIA L4 (#105: Llama-3.1-8B fp16 decode `0.93×`, 1K prefill `1.27×`, 4K prefill `1.02×`). Re-benchmarked after M4 (speculation) and M7 (fast FP8/W4A16). |
| **vLLM** | Datacenter SOTA throughput, broad model catalog, huge ecosystem | Lightweight single-GPU footprint, lazy imports (`import ayaka` loads no torch), upfront `ayaka plan` without downloading weights, coding-tool auto-config & 7-step `--check`, strict exact-output determinism, small-machine RAM/SSD MoE offload | Reference for protocol semantics, `UnifiedParser`/`tool_call`/gRPC patterns (M5), DFlash/MTP design (#127–#129), and L4 single-GPU comparisons. |
| **SGLang** | RadixAttention prefix reuse, fast structured output, strong datacenter perf | Consumer/small-GPU memory safety (M6 host KV tier & OOM recovery), zero-telemetry local-first UX, WSL2/Windows path, coding-agent normalization & session tooling | Reference for `render`/Jinja `ThinkingPolicy`, request-scoped `ChatResponseProcessor`, gRPC decoupling (M5), and speculative/MTP/DFlash mechanics (#127–#129). |
| **TensorRT-LLM** | Peak NVIDIA datacenter latency/throughput via compiled engines | Zero engine-build step, fast iteration in Python, strict exact-output rules (no relaxed-acceptance output drift), 4–24 GB consumer focus | Reference for DFlash/MTP configs (#127–#129). |
| **Strata & FreeToken** | Small-GPU MoE expert offload and local speculative decoding (MTP/DFlash2) | Full multi-protocol coding-agent server + prefix cache + grammar-in-graph + memory planner, head-to-head measured comparisons on identical hardware | Direct benchmark targets for M4 (DFlash/MTP) and M9 (MoE on small machines). |

## Four Core Differentiators (Moats)

1. **Coding-Agent Native End-to-End**
   - Native OpenAI (Chat Completions, Completions, Responses) and Anthropic (Messages) adapters locked
     by golden-byte tests (#81, #92–#94).
   - Two-command onboarding: `ayaka serve` + `ayaka connect <tool> --write` / `--check` / `--observe`.
   - Client-aware prompt normalization and schema-strict tool-call grammars inside CUDA graphs.
2. **Honest Memory & Capability Engineering**
   - `ayaka plan MODEL [--for TOOL]` predicts exact VRAM fit and KV capacity from `config.json` /
     GGUF headers without downloading weights.
   - `docs/CAPABILITIES.md` is generated from `capabilities.toml` where every `runs` entry requires a
     real `results.json` file with both sanity checks (forced tool call + JSON schema) passing.
3. **Exact-Output Acceleration for Small GPUs**
   - Position-keyed counter-based RNG guarantees identical greedy and one-point sampled outputs
     whether speculation is on or off.
   - Combines model-free lookup, native MTP, DFlash block drafters, fast quantized GEMM, and
     (upcoming) RAM/SSD MoE offload without silently degrading quality.
4. **Clean, Inspectable, Privacy-Respecting Architecture**
   - Python 3.13 strict-typed core, one-home-per-name module layering enforced by AST tests, lazy
     imports, zero telemetry, loopback-by-default, safetensors-only strict trust policy.
