# Master Execution Graph (86 Steps: Serving Core v2 -> 1.0.0 GA)

Single operational queue across all remaining milestones, in the round order decided on `2026-11-27`:
Serving Core v2, memory, fast quantized kernels, speculative decoding, then the later milestones. Pick the first
`READY` step, open its milestone file for exact `Touches`, `Deliverable`, and `Exit Check`, finish it in one focused
PR, and advance its status.

Step IDs (`Serving 1`, `Memory 1`, `Speculative 1`, ...) name the step within its milestone. Each milestone file
has the full step names. Step numbers that are not used (for example speculative step 5) stay empty so that other
files keep their references.

## Status Legend

- `OPEN_PR` — implementation already open on GitHub, awaiting merge/benchmark sign-off
- `READY` — all prerequisites met; can start immediately
- `BLOCKED` — waiting on prerequisite step(s) in `Depends On`
- `GATED` — conditional step; requires an RFC + measurement go/no-go before implementation
- `DONE` — merged and exit check verified

---

## Stage 1 — Road to `0.2.0` Public PyPI GA (63 steps)

### Serving Core v2 (`v0.2.0-a1`) · [M05-serving-v2.md](M05-serving-v2.md)

| Step | Status | Depends On | Hardware | Summary |
| --- | --- | --- | --- | --- |
| Serving 1 | `READY` | `main` (models and quantization done) | CPU | `ayaka.entrypoints.render`: Jinja `ThinkingPolicy` detection, model render hooks, `RenderedChat` |
| Serving 2 | `BLOCKED` | Serving 1 | CPU | OpenAI & Anthropic wire upgrade: `reasoning_effort`, `thinking`, `chat_template_kwargs`, prefill, `developer` |
| Serving 3 | `BLOCKED` | Serving 1, Serving 2 | CPU + GPU-L4 | Request-scoped `ChatResponseProcessor` (`none` + `thinking`), `UnifiedParser`, synchronized constraint start |
| Serving 4 | `BLOCKED` | Serving 3 | CPU | Enhanced `reasoning_parser` / `think` (`tool_start` recovery, `deepseek_v3`, `glm45`, `minimax_m3`) |
| Serving 5 | `BLOCKED` | Serving 3 | CPU | `ToolSchemas` type coercion, `output_grammar` spec (`ToolStrictLevel`), JSON family (`hermes`, `llama3_json`, `qwen3_json`, `deepseek_json`) |
| Serving 6 | `BLOCKED` | Serving 4, Serving 5 | CPU + GPU-L4 | XML/tag tool parsers (`qwen3_coder`, `mimo`, `glm_xml`, `minimax_m2/m3`, `deepseek_dsml`) + L4 check |
| Serving 7 | `BLOCKED` | Serving 5 | CPU + GPU-L4 | Structured Output v2: 6 `ConstraintKind` (`REGEX`/`CHOICE`/`GRAMMAR`/`STRUCTURAL_TAG`), `StructuredOutputBackend`, `validate_tokens` & `rollback(k)` grammar contract |
| Serving 8 | `BLOCKED` | Serving 6, Serving 7 | CPU + GPU-L4 | `XGrammarBackend` (`--extra xgrammar`, dual-compiler JSON vs `STRUCTURAL_TAG`), XML tool grammar in CUDA graphs |
| Serving 9 | `BLOCKED` | Serving 3, Serving 7 | CPU | Versioned `proto/ayaka/v1/{inference,render}.proto` (64 MiB limit) & lazy optional `grpc` extra |
| Serving 10 | `BLOCKED` | Serving 9 | CPU + GPU-L4 | Engine/Inference gRPC server & remote `GrpcEngineClient` (`ayaka serve --engine-grpc`) |
| Serving 11 | `BLOCKED` | Serving 9, Serving 6 | CPU | Engine-free `Render` & `Parse` gRPC service (`ayaka render-server`, zero-GPU / zero-torch) |
| Serving 12 | `BLOCKED` | Serving 8, Serving 10, Serving 11 | GPU-L4 | Benchmarks (`XGrammar` vs `LLGuidance` & gRPC overhead), live OpenAI/Claude round-trips, tag `v0.2.0-a1` |
| Serving 13 | `READY` | — | CPU | Keep tools in the prompt for `tool_choice="none"` (PR #134, moved from speculative decoding) |

### Memory (`v0.2.0-a2`) · [M06-memory.md](M06-memory.md)

| Step | Status | Depends On | Hardware | Summary |
| --- | --- | --- | --- | --- |
| Memory 1 | `BLOCKED` | Serving 12 | CPU + GPU | `ayaka doctor` CLI (CUDA, NVML/WSL2 gap, pin probe, `nvcc`, cache, plan sanity) |
| Memory 2 | `BLOCKED` | Memory 1 | CPU | Memory wire error codes (`out_of_memory`, `preempted_timeout`, `limit="memory"`) & over-long policy |
| Memory 3 | `BLOCKED` | Memory 2 | CPU + GPU | Pressure watermarks (`normal`/`high`/`critical`) & scheduler recompute preemption |
| Memory 4 | `BLOCKED` | Memory 3 | GPU | Step-level CUDA OOM catch, cache clear & graceful batch recovery |
| Memory 5 | `BLOCKED` | Memory 4 | CPU + GPU | Wave 1 fault-injection suite (3 long reqs on tiny KV pool, injected OOM, integrity checks) |
| Memory 6 | `BLOCKED` | Memory 1 | GPU-L4 + WSL2 | `HostMemoryConfig` & Linux/WSL2 `cudaHostAlloc` pinning ceiling probe |
| Memory 7 | `BLOCKED` | Memory 3, Memory 6 | GPU | Host-RAM KV store, async write-behind, promotion & shared-prefix VRAM pinning |
| Memory 8 | `BLOCKED` | Memory 7 | CPU + GPU | Wait for in-flight duplicate cold prefixes & optional prewarm hook |
| Memory 9 | `BLOCKED` | Memory 7 | GPU | Atomic runtime KV pool resize with clean rollback on failure |
| Memory 10 | `BLOCKED` | Memory 5, Memory 8, Memory 9 | GPU-L4 + 4GB | Session replay (`entrypoints/replay.py`), host-tier `< 0.5×` cold prefill gate, tag `v0.2.0-a2` |

### Fast Quantized Kernels: FP8 & W4A16 (`v0.2.0-a3`) · [M07-fast-quant.md](M07-fast-quant.md)

| Step | Status | Depends On | Hardware | Summary |
| --- | --- | --- | --- | --- |
| Fast quantized kernels 1 | `BLOCKED` | Memory 10 | CPU | Quantized GEMM op contracts & `sm_arch` backend dispatch (preserve `sm < 89` fallback) |
| Fast quantized kernels 2 | `BLOCKED` | Fast quantized kernels 1 | GPU-L4 | Fast per-tensor / per-row FP8 GEMM (`torch._scaled_mm` / FlashInfer) inside CUDA graphs; A100 runs the weight-only FP8 path |
| Fast quantized kernels 3 | `BLOCKED` | Fast quantized kernels 1 | GPU-L4 | Fast 128×128 block-FP8 GEMM (Triton / FlashInfer, optional `triton` extra) in CUDA graphs |
| Fast quantized kernels 4 | `BLOCKED` | Fast quantized kernels 1 | GPU-L4 + 4GB | Fast safetensors W4A16 (INT4/AWQ/GPTQ Marlin) kernel, loader & `ayaka plan` sizing |
| Fast quantized kernels 5 | `BLOCKED` | Fast quantized kernels 2, 3, 4 | GPU-L4 + 4GB | Quantized KL & speed gates (`FP8 ≥ 1.2×`, `Phi-4 ≥ 10 tok/s`, `W4A16 ≥ 1.8×`), vs `llama.cpp`, tag `v0.2.0-a3` |

> **Parallel Early Spike triggered right after fast quantized kernels step 5:**
> native kernel engine step 0 *(Binding ADR + 2-kernel CUDA graph spike)* — see [M13-native-kernels-windows.md](M13-native-kernels-windows.md).

### Speculative Decoding (`v0.2.0-a4`) · [M04-speculative.md](M04-speculative.md)

| Step | Status | Depends On | Hardware | Summary |
| --- | --- | --- | --- | --- |
| Speculative 1 | `OPEN_PR` (#131) | Fast quantized kernels 5 (measurements are repeated after it) | GPU-L4 | `ayaka.speculative`, lookup drafter, verify CUDA graph, cost controller, `prediction`, L4 bench |
| Speculative 2 | `OPEN_PR` (#132) | Speculative 1 | GPU-L4 | `ForwardContext` feature sink + MiMo-7B MTP drafter + weight/KV plan + L4 bench |
| Speculative 3 | `OPEN_PR` (#133) | Speculative 2 | GPU-L4 | Multi-layer hidden taps + DFlash block drafter (`Qwen3-8B`, `Llama-3.1-8B`) + L4 bench |
| Speculative 4 | `BLOCKED` | Speculative 1 | GPU-L4 + 4GB | Enable `cuda_graph=True` default on CUDA + memory reconciliation check |
| Speculative 5 | — | — | — | Empty. Moved to Serving 13 (`tool_choice="none"`, PR #134) |
| Speculative 6 | `BLOCKED` | Speculative 1 to 4, Fast quantized kernels 5 | GPU-L4 | Record lookup default decision against the pre-registered gate, close #127–#129, tag `v0.2.0-a4` |

### Hybrid Dense, State & Vision (`v0.2.0-a5`) · [M08-hybrid-vision.md](M08-hybrid-vision.md)

| Step | Status | Depends On | Hardware | Summary |
| --- | --- | --- | --- | --- |
| Hybrid 1 | `BLOCKED` | Fast quantized kernels 5 | CPU | Hybrid cache spec: recurrent state slots (GDN/KDA) & latent KV pool (MLA) |
| Hybrid 2 | `BLOCKED` | Hybrid 1 | GPU-L4 | GDN/linear-attention & MLA op contracts + GPU kernels (MLA `≥ 50%` KV savings) |
| Hybrid 3 | `BLOCKED` | Hybrid 2 | GPU-L4 | State checkpointing & suffix replay (1-chunk == N-chunk; state prefix hit exact) |
| Hybrid 4 | `BLOCKED` | Hybrid 2, Hybrid 3 | GPU-L4 | Representative hybrid models (`Qwen3.5`/`3.8`, `Gemma 4` + UnifiedParser, `gpt-oss` class) parity |
| Hybrid 5 | `BLOCKED` | Hybrid 1 | CPU | Image preprocessing (processor-exact), token expansion & OpenAI/Anthropic wire payloads |
| Hybrid 6 | `BLOCKED` | Hybrid 4, Hybrid 5 | GPU-L4 | Vision encoder & projector + `1024×768` screenshot e2e through `ayaka serve` |
| Hybrid 7 | `BLOCKED` | Hybrid 1 | GPU-L4 | Opt-in KV FP8 codec (`float8_e4m3fn`) capacity mode & pre-registered quality gate |
| Hybrid 8 | `BLOCKED` | Hybrid 4, Hybrid 6, Hybrid 7 | GPU-L4 | `ayaka plan` (state/KV/vision) + `ayaka bench` (image & long-prompt), tag `v0.2.0-a5` |

### MoE on Small Machines (`v0.2.0-a6`) · [M09-moe-small-machines.md](M09-moe-small-machines.md)

| Step | Status | Depends On | Hardware | Summary |
| --- | --- | --- | --- | --- |
| MoE 1 | `BLOCKED` | Hybrid 8 | GPU-L4 + WSL2 | PCIe bandwidth & large-region pinning probe; target MoE selection (`Qwen3-30B-A3B` FP8/W4) |
| MoE 2 | `BLOCKED` | MoE 1 | CPU + GPU | Host expert store with 3 residency modes (`fused`, `offload`, streaming/fallback) |
| MoE 3 | `BLOCKED` | MoE 2 | GPU-L4 + 4GB | Dynamic GPU expert cache & offload decode (`offload` == `fused` exact token parity) |
| MoE 4 | `BLOCKED` | MoE 3 | GPU-L4 | Prefill staging buffers & overlapped H2D expert copies |
| MoE 5 | `BLOCKED` | MoE 4 | GPU-L4 | Routing trace, profile+decay vs LRU, calibrated `p/q` miss-cost model (3 cache sizes) & `plan` |
| MoE 6 | `BLOCKED` | MoE 4, Speculative 2 | GPU-L4 + 4GB | Feature sink & MTP drafting over offloaded MoE + payoff measurement |
| MoE 7 | `BLOCKED` | MoE 5, MoE 6 | GPU-L4 | Head-to-head benchmark vs Strata & FreeToken on same card, tag `v0.2.0-a6` |

### Agent Platform (`v0.2.0-a7`) · [M10-agent-platform.md](M10-agent-platform.md)

| Step | Status | Depends On | Hardware | Summary |
| --- | --- | --- | --- | --- |
| Agent 1 | `BLOCKED` | MoE 7 | CPU + GPU | Session store & stateful `previous_response_id` (100-turn `cached_tokens > 0`) |
| Agent 2 | `BLOCKED` | Agent 1 | CPU | Turn-safe context compaction (never splits/orphans `tool_use`/`tool_result` pairs) |
| Agent 3 | `BLOCKED` | Agent 1, Serving 3 | CPU + GPU | Reasoning budget enforcer (`low`/`medium`/`high`/`max`) forcing clean `</think>` close atop Serving Core v2 |
| Agent 4 | `BLOCKED` | Memory 1 | CPU | Built-in MCP server (`plan`, `bench`, `doctor`, `connect`, `status`, `stop`) + auth/loopback |
| Agent 5 | `BLOCKED` | Agent 1 to 4 | GPU-L4 | Expand `ayaka connect` to 6 harnesses (≥ 4 passing 7/7) + 100-turn agent eval suite |
| Agent 6 | `GATED` | Agent 5 | GPU-L4 | Extended: Embeddings/Rerank (`recall@k`) & opt-in OpenTelemetry (`< 1%` overhead) |
| Agent 7 | `GATED` | Agent 5 | GPU-L4 | Extended: LoRA/Multi-LoRA (exact merge parity) & prefix-safe Model Router, tag `v0.2.0-a7` |

### Frontier MoE & `0.2.0` Public PyPI GA · [M11-frontier-moe.md](M11-frontier-moe.md)

| Step | Status | Depends On | Hardware | Summary |
| --- | --- | --- | --- | --- |
| Frontier 1 | `BLOCKED` | MoE 7, Hybrid 2 | GPU-L4 | Sparse attention index & QSA/DSA kernels + Needle/RULER quality gates |
| Frontier 2 | `BLOCKED` | Frontier 1 | GPU-L4 | Out-of-VRAM n-gram & Engram lookup tables (VRAM/RAM tiered, exact parity) |
| Frontier 3 | `BLOCKED` | MoE 5 | GPU + NVMe | Hierarchical 3-tier `VRAM -> pinned RAM -> NVMe SSD` expert store (`== fused`) |
| Frontier 4 | `BLOCKED` | Fast quantized kernels 4 | GPU-L4 / sm100 | MXFP4 / NVFP4 dense & MoE expert quantization (Blackwell native + fallback) |
| Frontier 5 | `BLOCKED` | Speculative 3, Frontier 3 | GPU-L4 | DFlash / DSpark + suffix drafting on Frontier & offloaded MoE |
| Frontier 6 | `BLOCKED` | Frontier 1 to 4 | GPU 8–64GB | Representative Frontier MoE families (`Qwen3.8-Flash-Next`, `DeepSeek-V4.1-Flash`, `GLM-5.3-Flash`, `MiniMax-M2.5` class) |
| Frontier 7 | `BLOCKED` | Frontier 6 | GPU 8–64GB | Tiered `ayaka plan` (VRAM+RAM+SSD+tok/s) & frontier long-context e2e/bench |
| Frontier 8 | `BLOCKED` | Frontier 5, Frontier 7, Agent 5 | CPU + GPU | **Release Gate `ayaka-llm 0.2.0` (PyPI)**: benchmark intake form, `uv build`, docs audit, PyPI publish |

---

## Stage 2 — Scale-Out & Native Kernel Engine: `0.3.0` (13 steps)

### Scale-Out (`v0.3.0-a1`) · [M12-scale-out.md](M12-scale-out.md)

| Step | Status | Depends On | Hardware | Summary |
| --- | --- | --- | --- | --- |
| Scale-out 1 | `BLOCKED` | Frontier 8 | CPU + Multi-GPU | Parallel planner (`ayaka plan --distributed`) & NCCL/gRPC distributed comms contract |
| Scale-out 2 | `BLOCKED` | Scale-out 1 | 2+ GPU | Intra-node Tensor Parallelism (2-GPU TP == 1-GPU greedy tokens) |
| Scale-out 3 | `BLOCKED` | Scale-out 2 | 2+ GPU | KV transfer protocol & Prefill/Decode (P/D) worker disaggregation |
| Scale-out 4 | `BLOCKED` | Scale-out 2 | 2+ GPU | Expert Parallelism (EP) & MoE load balancing |
| Scale-out 5 | `BLOCKED` | Scale-out 3, Scale-out 4 | Multi-Node / 8-GPU | 2-machine benchmark & `Kimi K3`-class (+ UnifiedParser) run (8 GPU or 128 GB + SSD) |
| Scale-out 6 | `GATED` | Scale-out 5 | Conditional | Conditional Vulkan/Metal evaluation (only if real hardware exists), tag `v0.3.0-a1` |

### Native Kernel Engine (`kops`) & Windows Native (`0.3.0`) · [M13-native-kernels-windows.md](M13-native-kernels-windows.md)

| Step | Status | Depends On | Hardware | Summary |
| --- | --- | --- | --- | --- |
| Native kernel 0 | `BLOCKED` | Fast quantized kernels 5 | Linux + Win GPU | **Early Spike**: Binding ADR + 2-kernel CUDA graph capture validation |
| Native kernel 1 | `BLOCKED` | Native kernel 0, Frontier 8 | Linux + Win | Build migration: `scikit-build-core` + `CMakeLists.txt` + `csrc/` (keep zero-CUDA `import ayaka`) |
| Native kernel 2 | `BLOCKED` | Native kernel 1 | Linux + Win GPU | Native paged attention & KV kernels (`kops`, full parity vs `torch`) |
| Native kernel 3 | `BLOCKED` | Native kernel 1 | Linux + Win GPU | Native fused deterministic sampler & fast FP8/INT8 KV codecs |
| Native kernel 4 | `BLOCKED` | Native kernel 1 | Linux + Win GPU | Native quantized GEMM (FP8/W4A16/Blackwell FP4) & fused MoE offload DMA |
| Native kernel 5 | `BLOCKED` | Native kernel 2 to 4 | Win GPU native | Full native Windows e2e, CUDA graphs & 7/7 `ayaka connect --check` (no WSL2) |
| Native kernel 6 | `BLOCKED` | Native kernel 5, Scale-out 6 | Linux + Win CI | `cibuildwheel` release wheels (Linux + Windows) & publish **`ayaka-llm 0.3.0`** |

---

## Stage 3 — Agent Swarm OS & `1.0.0` GA: `0.4.0` -> `1.0.0` (10 steps)

### Multi-Agent & Distributed Prefix Fabric (`0.4.0`) · [M14-multi-agent-fabric.md](M14-multi-agent-fabric.md)

| Step | Status | Depends On | Hardware | Summary |
| --- | --- | --- | --- | --- |
| Fabric 1 | `BLOCKED` | Native kernel 6, Agent 5 | GPU / Multi-GPU | Zero-copy shared prefix/KV/state fabric across concurrent coding subagents (using Serving Core v2 render fingerprints) |
| Fabric 2 | `BLOCKED` | Fabric 1 | GPU | Cross-agent speculative KV prewarm & branch-aware subagent tree scheduling |
| Fabric 3 | `BLOCKED` | Fabric 1 | CPU + GPU | Multi-tenant QoS, foreground interactive priority lanes & fairness scheduler |
| Fabric 4 | `BLOCKED` | Agent 1, Fabric 1 | CPU + NVMe | Opt-in owner-only/encrypted persistent KV & session snapshot store across restarts |
| Fabric 5 | `BLOCKED` | Fabric 2 to 4 | GPU / Multi-GPU | 8-agent coding-swarm benchmark (cache share, foreground ITL SLO) & publish **`ayaka-llm 0.4.0`** |

### Omni-Modal Computer-Use & `1.0.0` GA · [M15-omni-v1.md](M15-omni-v1.md)

| Step | Status | Depends On | Hardware | Summary |
| --- | --- | --- | --- | --- |
| Omni 1 | `BLOCKED` | Hybrid 6, Fabric 5 | GPU | Screen/video & audio streaming inputs with incremental frame diffing & visual KV reuse |
| Omni 2 | `BLOCKED` | Omni 1 | GPU | Real-time multimodal computer-use action loop (screenshot stream -> structured GUI/tool action) |
| Omni 3 | `BLOCKED` | Fabric 5 | CPU | Freeze public Python API, Wire/gRPC contracts & Backend/Model Plugin ABI (SemVer contract) |
| Omni 4 | `BLOCKED` | Omni 3 | CPU + GPU | Full security, privacy & supply-chain audit (MCP, disk sessions, `csrc/`, parsers, signed wheels) |
| Omni 5 | `BLOCKED` | Omni 2, Omni 4 | All Tiers | Multi-tier verified `CAPABILITIES.md` & publish **`ayaka-llm 1.0.0` GA** |
