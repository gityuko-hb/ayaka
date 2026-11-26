# Architecture Bets & Trade-offs

This document records Ayaka's major technical bets, why each was chosen, what risks it carries, and
what evidence triggers a course correction.

## Bet 1: Python-First Engine + Modular `ops` Backend Contract (Defer `csrc/` Until Necessary)

- **Choice**: Keep the engine core, scheduler, KV manager, models, and HTTP server in strictly typed
  Python 3.13, delegating heavy GPU ops through `ayaka.ops` backend contracts (`torch`,
  `flashinfer`, and upcoming `triton` / quantized GEMM backends).
- **Critical update from M3 (#121)**: PyTorch slice-dequantization makes FP8 decode `0.11–0.12×`
  of bf16 on an NVIDIA L4 (`1.78 tok/s` vs `15.6 tok/s`). Waiting until the late "Windows native"
  milestone to fix this would block 14B+ and 4–8 GB usability for months.
- **Course correction (M7)**: Unlock fast **FP8 GEMM** and **W4A16** using **existing GPU kernel
  libraries** (`torch._scaled_mm`, `FlashInfer` FP8/GEMM ops, `Triton` block-FP8 kernels, and/or
  `Marlin` W4A16 kernels) behind the `ayaka.ops` quantization contract — keeping hatchling as the
  build backend until `csrc/` is genuinely needed.
- **When `csrc/` (`scikit-build-core` + `CMakeLists.txt`) triggers**:
  1. Run a small **binding ADR + 2-kernel CUDA graph spike (`M13-N00`)** right after **M7
     (`0.2.0-a4`)**, in parallel with M8/M9, to de-risk C++/CUDA integration early (especially if
     M9 MoE offload needs custom DMA kernels).
  2. Activate full native `csrc/` development (**M13**, targeting `0.3.0`) when a measured
     bottleneck cannot be solved by FlashInfer/Triton (e.g., fused MoE RAM offload DMA/kernels,
     Blackwell FP4/FP8 fused ops, fused sampling/KV codecs) or for native **Windows without WSL2**.
- **Non-negotiable invariant**: `import ayaka` must always succeed on machines without CUDA (CPU
  fallback or clean capability error).

## Bet 2: Exact-Output Determinism & Position-Keyed RNG

- **Choice**: Counter-based RNG keyed by `(request_seed, token_position, DrawStream)` so every
  token position draws identically regardless of batch composition or speculative window width.
- **Why**: Enables mathematically verifiable **zero-output-change** speculative decoding (Lookup,
  MTP, DFlash) for both greedy and one-point sampled decoding, and makes debugging warm-cache /
  offload / preemption bugs reproducible.
- **Trade-off**: Keeps sampling on `torch` (FlashInfer 0.7.4's sampling APIs do not accept external
  uniform draws `u`, verified in #99). Accepted because sampling is a small fraction of decode time
  inside CUDA graphs, and determinism is core to Ayaka's trust model.

## Bet 3: Upfront Memory Planning & Zero-Silent-Paging

- **Choice**: `ayaka plan` computes weights, KV cache blocks, runtime/CUDA context gap, workspace,
  and graph overhead *before* loading weights, and startup reconciles the real allocator against the
  plan.
- **Why**: On 4–24 GB consumer GPUs (especially under WSL2/WDDM, where oversubscription silently
  pages to host RAM and halves throughput — observed in #59 and #73), guessing memory budgets causes
  either hard OOM crashes or silent 2×–5× slowdowns.
- **Next evolution (M6)**: Pair upfront planning with runtime resilience — pressure levels,
  preemption, step-level OOM recovery, atomic KV pool resize, and a host-RAM KV tier — so the server
  gracefully degrades latency instead of dying.

## Bet 4: Prefix-Cache Preservation as a First-Class Serving Primitive

- **Choice**: Hash-chain paged prefix cache (#52) combined with explicit client prompt-churn
  normalization (#57) and schema-preserving prompt layout (`tool_choice="none"` keeping tools).
- **Why**: Coding agents resend 10K–50K tokens of identical context on every turn. Preserving prefix
  cache hits turns multi-second cold prefills into ~40–90 ms warm TTFT.
- **Constraint**: Normalization rules must never alter semantics that tools depend on (e.g., Codex
  skill paths were explicitly *not* normalized in #57 because rewriting them would break tool calls).

## Bet 5: Pluggable Structured Output Inside CUDA Graphs (`llguidance` + `xgrammar`)

- **Choice**: Keep grammar compilation/masking lazily loaded on the host behind a pluggable
  `StructuredOutputBackend` / `StructuredOutputGrammar` contract (`llguidance` default for zero-dep
  cross-OS support + optional `--extra xgrammar` in M5), and apply the byte-wise token bitmask
  **inside** captured decode CUDA graphs (#73/#79).
- **Why**: Coding agents rely on strict JSON, regex/EBNF, and XML/JSON tool-call schemas on nearly
  every turn. Falling back to eager execution cost `1.4×–6.8×` per step; keeping the mask inside
  the graph brought overhead down to `0%–8%` (`docs/EXPERIMENTS.md`). Adding `STRUCTURAL_TAG` +
  `ToolStrictLevel` + `validate_tokens`/`rollback(k)` in M5 extends CUDA-graph constraints to XML
  tool families (`qwen3_coder`, `mimo`, `glm_xml`, `minimax`, `deepseek_dsml`) and unlocks **Guided
  Speculative Decoding** alongside M4.
- **Trade-off**: Doubles captured graphs (masked + unmasked per bucket, ~20 MiB per graph), managed
  through explicit bucket sizing (`1, 2, 4, 8` default) and memory-plan accounting. XGrammar uses
  dual compilers (`json` vs `full_vocab` for `STRUCTURAL_TAG`) to isolate special-token behavior.

## Bet 6: Multi-Stage Speculative Decoding Tailored to Bandwidth-Bound GPUs

- **Choice**: Build one shared verification/controller/feature-sink substrate (#127) supporting three
  complementary drafters:
  1. **Lookup / predicted outputs** (0 extra VRAM; strongest on file edits/copying).
  2. **Native MTP** (MiMo-7B; uses checkpoint's own MTP layer, ~0.42 GB + 1/36 KV).
  3. **DFlash block drafter** (Qwen3-8B / Llama-3.1-8B; single-pass block drafting conditioned on
     target hidden-state taps, accounting honestly for the 1–2 GB VRAM displaced from KV).
- **Why**: On small/mid GPUs (L4 ~250–300 GB/s), low-batch decode is severely memory-bandwidth
  bound. A cost-aware controller disables speculation automatically when batch concurrency rises or
  acceptance drops.

## Bet 7: Decoupled Render, Request-Scoped Parsers & gRPC Boundary (M5)

- **Choice**: Keep the Render + Request-Scoped Parser + gRPC layer in **pure Python 3.13**
  (optional `grpc` extra, lazy imports, hatchling build) while adopting the clean architectural
  boundaries proven by SGLang (`sglang-processor`, `sglang-grpc`) and vLLM (`rust/src/parser`,
  `rust/src/server/src/grpc`):
  1. Dedicated **Render** layer inspecting Jinja `ThinkingPolicy` and returning `(prompt, token_ids,
     thinking_enabled, starts_inside_reasoning, render_fingerprint)` per request.
  2. **Request-scoped** `ChatResponseProcessor` (`force_reasoning: None | True | False`, with
     `False` cleanly disabling reasoning interpretation per request) + synchronized `llguidance`
     `constraint_start()` + `UnifiedParser`/`CombinedParser` + `ToolSchemas` argument coercion.
  3. **Two gRPC contracts**: `Inference` gRPC wrapping the frozen `EngineClient` seam (decoupling
     HTTP gateway from GPU worker) and engine-free `Render`/`Parse` gRPC (CPU-only, no `torch`).
- **Why Python first (not Rust immediately)**: Measurements in #77/#90 showed prompt render + encode
  + SSE formatting consume `< 1%` of cold TTFT on single-GPU setups. Defining clean `.proto` and
  semantic-event contracts in Python lets Ayaka ship full multi-family parser coverage and gRPC
  decoupling immediately without adding a Rust/PyO3 toolchain, while leaving a clean drop-in seam if
  high-concurrency profiling ever justifies a Rust processor later.

## Bet 8: Quantized MoE + Host-RAM/SSD Expert Offload (M9 & M11)

- **Choice**: Run M7 (FP8 & W4A16) *before* M9 (MoE on small machines) so large sparse models
  (e.g., `Qwen3-30B-A3B`: 61 GB in bf16 -> ~30 GB in FP8 / ~16 GB in W4) fit cleanly inside
  standard 32–64 GB host RAM, backed by a measured miss-cost model and profile-guided expert cache.
- **Why**: Sparse MoE architectures offer Frontier coding intelligence at 2B–4B active parameters
  per token — the single highest-leverage way to bring frontier coding quality to 8–24 GB GPUs.
