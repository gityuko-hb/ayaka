# Master Execution Graph (86 Nodes: M4 -> 1.0.0 GA)

Single operational queue across all remaining milestones (`M04`–`M15`). Pick the first `READY`
Node, open its milestone file for exact `Touches`, `Deliverable`, and `Exit Check`, finish it in
one focused PR, and advance its status.

## Status Legend

- `OPEN_PR` — implementation already open on GitHub, awaiting merge/benchmark sign-off
- `READY` — all prerequisites met; can start immediately
- `BLOCKED` — waiting on prerequisite Node(s) in `Depends On`
- `GATED` — conditional Node; requires an RFC + measurement go/no-go before implementation
- `DONE` — merged and exit check verified

---

## Stage 1 — Road to `0.2.0` Public PyPI GA (M4 – M11: 63 Nodes)

### M4 — Speculative Decoding (`v0.2.0-a1`) · [M04-speculative.md](M04-speculative.md)

| Node | Status | Depends On | Hardware | Summary |
| --- | --- | --- | --- | --- |
| `M4-N01` | `OPEN_PR` (#131) | M3 | GPU-L4 | `ayaka.speculative`, lookup drafter, verify CUDA graph, cost controller, `prediction`, L4 bench |
| `M4-N02` | `OPEN_PR` (#132) | `M4-N01` | GPU-L4 | `ForwardContext` feature sink + MiMo-7B MTP drafter + weight/KV plan + L4 bench |
| `M4-N03` | `OPEN_PR` (#133) | `M4-N02` | GPU-L4 | Multi-layer hidden taps + DFlash block drafter (`Qwen3-8B`, `Llama-3.1-8B`) + L4 bench |
| `M4-N04` | `BLOCKED` | `M4-N01` | GPU-L4 + 4GB | Enable `cuda_graph=True` default on CUDA + memory reconciliation check |
| `M4-N05` | `READY` | — | CPU | Lock `tool_choice="none"` keeping tools in prompt to preserve prefix cache |
| `M4-N06` | `BLOCKED` | `M4-N01..N05` | GPU-L4 | M4 exit audit, record lookup default decision, tag `v0.2.0-a1` |

### M5 — Serving Core v2: Render, Parsers, Structured Output (XGrammar) & gRPC (`v0.2.0-a2`) · [M05-serving-v2.md](M05-serving-v2.md)

| Node | Status | Depends On | Hardware | Summary |
| --- | --- | --- | --- | --- |
| `M5-N01` | `BLOCKED` | `M4-N06` | CPU | `ayaka.entrypoints.render`: Jinja `ThinkingPolicy` detection, model render hooks, `RenderedChat` |
| `M5-N02` | `BLOCKED` | `M5-N01` | CPU | OpenAI & Anthropic wire upgrade: `reasoning_effort`, `thinking`, `chat_template_kwargs`, prefill, `developer` |
| `M5-N03` | `BLOCKED` | `M5-N01`, `M5-N02` | CPU + GPU-L4 | Request-scoped `ChatResponseProcessor` (`none` + `thinking`), `UnifiedParser`, synchronized constraint start |
| `M5-N04` | `BLOCKED` | `M5-N03` | CPU | Enhanced `reasoning_parser` / `think` (`tool_start` recovery, `deepseek_v3`, `glm45`, `minimax_m3`) |
| `M5-N05` | `BLOCKED` | `M5-N03` | CPU | `ToolSchemas` type coercion, `output_grammar` spec (`ToolStrictLevel`), JSON family (`hermes`, `llama3_json`, `qwen3_json`, `deepseek_json`) |
| `M5-N06` | `BLOCKED` | `M5-N04`, `M5-N05` | CPU + GPU-L4 | XML/tag tool parsers (`qwen3_coder`, `mimo`, `glm_xml`, `minimax_m2/m3`, `deepseek_dsml`) + L4 check |
| `M5-N07` | `BLOCKED` | `M5-N05` | CPU + GPU-L4 | Structured Output v2: 6 `ConstraintKind` (`REGEX`/`CHOICE`/`GRAMMAR`/`STRUCTURAL_TAG`), `StructuredOutputBackend`, `rollback` for Guided Speculation |
| `M5-N08` | `BLOCKED` | `M5-N06`, `M5-N07` | CPU + GPU-L4 | `XGrammarBackend` (`--extra xgrammar`, dual-compiler JSON vs `STRUCTURAL_TAG`), XML tool grammar in CUDA graphs |
| `M5-N09` | `BLOCKED` | `M5-N03`, `M5-N07` | CPU | Versioned `proto/ayaka/v1/{inference,render}.proto` (64 MiB limit) & lazy optional `grpc` extra |
| `M5-N10` | `BLOCKED` | `M5-N09` | CPU + GPU-L4 | Engine/Inference gRPC server & remote `GrpcEngineClient` (`ayaka serve --engine-grpc`) |
| `M5-N11` | `BLOCKED` | `M5-N09`, `M5-N06` | CPU | Engine-free `Render` & `Parse` gRPC service (`ayaka render-server`, zero-GPU / zero-torch) |
| `M5-N12` | `BLOCKED` | `M5-N08`, `M5-N10`, `M5-N11` | GPU-L4 | Benchmarks (`XGrammar` vs `LLGuidance` & gRPC overhead), live OpenAI/Claude round-trips, tag `v0.2.0-a2` |

### M6 — Memory (`v0.2.0-a3`) · [M06-memory.md](M06-memory.md)

| Node | Status | Depends On | Hardware | Summary |
| --- | --- | --- | --- | --- |
| `M6-N01` | `BLOCKED` | `M5-N12` | CPU + GPU | `ayaka doctor` CLI (CUDA, NVML/WSL2 gap, pin probe, `nvcc`, cache, plan sanity) |
| `M6-N02` | `BLOCKED` | `M6-N01` | CPU | Memory wire error codes (`out_of_memory`, `preempted_timeout`, `limit="memory"`) & over-long policy |
| `M6-N03` | `BLOCKED` | `M6-N02` | CPU + GPU | Pressure watermarks (`normal`/`high`/`critical`) & scheduler recompute preemption |
| `M6-N04` | `BLOCKED` | `M6-N03` | GPU | Step-level CUDA OOM catch, cache clear & graceful batch recovery |
| `M6-N05` | `BLOCKED` | `M6-N04` | CPU + GPU | Wave 1 fault-injection suite (3 long reqs on tiny KV pool, injected OOM, integrity checks) |
| `M6-N06` | `BLOCKED` | `M6-N01` | GPU-L4 + WSL2 | `HostMemoryConfig` & Linux/WSL2 `cudaHostAlloc` pinning ceiling probe |
| `M6-N07` | `BLOCKED` | `M6-N03`, `M6-N06` | GPU | Host-RAM KV store, async write-behind, promotion & shared-prefix VRAM pinning |
| `M6-N08` | `BLOCKED` | `M6-N07` | CPU + GPU | Wait for in-flight duplicate cold prefixes & optional prewarm hook |
| `M6-N09` | `BLOCKED` | `M6-N07` | GPU | Atomic runtime KV pool resize with clean rollback on failure |
| `M6-N10` | `BLOCKED` | `M6-N05`, `M6-N08`, `M6-N09` | GPU-L4 + 4GB | Session replay (`entrypoints/replay.py`), host-tier `< 0.5×` cold prefill gate, tag `v0.2.0-a3` |

### M7 — Fast Quantized Kernels (`v0.2.0-a4`) · [M07-fast-quant.md](M07-fast-quant.md)

| Node | Status | Depends On | Hardware | Summary |
| --- | --- | --- | --- | --- |
| `M7-N01` | `BLOCKED` | `M6-N10` | CPU | Quantized GEMM op contracts & `sm_arch` backend dispatch (preserve `sm < 89` fallback) |
| `M7-N02` | `BLOCKED` | `M7-N01` | GPU-L4 | Fast per-tensor / per-row FP8 GEMM (`torch._scaled_mm` / FlashInfer) inside CUDA graphs |
| `M7-N03` | `BLOCKED` | `M7-N01` | GPU-L4 | Fast 128×128 block-FP8 GEMM (Triton / FlashInfer, optional `triton` extra) in CUDA graphs |
| `M7-N04` | `BLOCKED` | `M7-N01` | GPU-L4 + 4GB | Fast safetensors W4A16 (INT4/AWQ/GPTQ Marlin) kernel, loader & `ayaka plan` sizing |
| `M7-N05` | `BLOCKED` | `M7-N02..N04` | GPU-L4 + 4GB | Quantized KL & speed gates (`FP8 ≥ 1.2×`, `Phi-4 ≥ 10 tok/s`, `W4A16 ≥ 1.8×`), vs `llama.cpp`, tag `v0.2.0-a4` |

> **Parallel Early Spike triggered right after `M7-N05`:**
> `M13-N00` *(Binding ADR + 2-kernel CUDA graph spike)* — see [M13-native-kernels-windows.md](M13-native-kernels-windows.md).

### M8 — Hybrid Dense, State & Vision (`v0.2.0-a5`) · [M08-hybrid-vision.md](M08-hybrid-vision.md)

| Node | Status | Depends On | Hardware | Summary |
| --- | --- | --- | --- | --- |
| `M8-N01` | `BLOCKED` | `M7-N05` | CPU | Hybrid cache spec: recurrent state slots (GDN/KDA) & latent KV pool (MLA) |
| `M8-N02` | `BLOCKED` | `M8-N01` | GPU-L4 | GDN/linear-attention & MLA op contracts + GPU kernels (MLA `≥ 50%` KV savings) |
| `M8-N03` | `BLOCKED` | `M8-N02` | GPU-L4 | State checkpointing & suffix replay (1-chunk == N-chunk; state prefix hit exact) |
| `M8-N04` | `BLOCKED` | `M8-N02`, `M8-N03` | GPU-L4 | Representative hybrid models (`Qwen3.5`/`3.8`, `Gemma 4` + UnifiedParser, `gpt-oss` class) parity |
| `M8-N05` | `BLOCKED` | `M8-N01` | CPU | Image preprocessing (processor-exact), token expansion & OpenAI/Anthropic wire payloads |
| `M8-N06` | `BLOCKED` | `M8-N04`, `M8-N05` | GPU-L4 | Vision encoder & projector + `1024×768` screenshot e2e through `ayaka serve` |
| `M8-N07` | `BLOCKED` | `M8-N01` | GPU-L4 | Opt-in KV FP8 codec (`float8_e4m3fn`) capacity mode & pre-registered quality gate |
| `M8-N08` | `BLOCKED` | `M8-N04`, `M8-N06`, `M8-N07` | GPU-L4 | `ayaka plan` (state/KV/vision) + `ayaka bench` (image & long-prompt), tag `v0.2.0-a5` |

### M9 — MoE on Small Machines (`v0.2.0-a6`) · [M09-moe-small-machines.md](M09-moe-small-machines.md)

| Node | Status | Depends On | Hardware | Summary |
| --- | --- | --- | --- | --- |
| `M9-N01` | `BLOCKED` | `M8-N08` | GPU-L4 + WSL2 | PCIe bandwidth & large-region pinning probe; target MoE selection (`Qwen3-30B-A3B` FP8/W4) |
| `M9-N02` | `BLOCKED` | `M9-N01` | CPU + GPU | Host expert store with 3 residency modes (`fused`, `offload`, streaming/fallback) |
| `M9-N03` | `BLOCKED` | `M9-N02` | GPU-L4 + 4GB | Dynamic GPU expert cache & offload decode (`offload` == `fused` exact token parity) |
| `M9-N04` | `BLOCKED` | `M9-N03` | GPU-L4 | Prefill staging buffers & overlapped H2D expert copies |
| `M9-N05` | `BLOCKED` | `M9-N04` | GPU-L4 | Routing trace, profile+decay vs LRU, calibrated `p/q` miss-cost model (3 cache sizes) & `plan` |
| `M9-N06` | `BLOCKED` | `M9-N04`, `M4-N02` | GPU-L4 + 4GB | Feature sink & MTP drafting over offloaded MoE + payoff measurement |
| `M9-N07` | `BLOCKED` | `M9-N05`, `M9-N06` | GPU-L4 | Head-to-head benchmark vs Strata & FreeToken on same card, tag `v0.2.0-a6` |

### M10 — Agent Platform (`v0.2.0-a7`) · [M10-agent-platform.md](M10-agent-platform.md)

| Node | Status | Depends On | Hardware | Summary |
| --- | --- | --- | --- | --- |
| `M10-N01` | `BLOCKED` | `M9-N07` | CPU + GPU | Session store & stateful `previous_response_id` (100-turn `cached_tokens > 0`) |
| `M10-N02` | `BLOCKED` | `M10-N01` | CPU | Turn-safe context compaction (never splits/orphans `tool_use`/`tool_result` pairs) |
| `M10-N03` | `BLOCKED` | `M10-N01`, `M5-N03` | CPU + GPU | Reasoning budget enforcer (`low`/`medium`/`high`/`max`) forcing clean `</think>` close atop M5 |
| `M10-N04` | `BLOCKED` | `M6-N01` | CPU | Built-in MCP server (`plan`, `bench`, `doctor`, `connect`, `status`, `stop`) + auth/loopback |
| `M10-N05` | `BLOCKED` | `M10-N01..N04` | GPU-L4 | Expand `ayaka connect` to 6 harnesses (≥ 4 passing 7/7) + 100-turn agent eval suite |
| `M10-N06` | `GATED` | `M10-N05` | GPU-L4 | Extended: Embeddings/Rerank (`recall@k`) & opt-in OpenTelemetry (`< 1%` overhead) |
| `M10-N07` | `GATED` | `M10-N05` | GPU-L4 | Extended: LoRA/Multi-LoRA (exact merge parity) & prefix-safe Model Router, tag `v0.2.0-a7` |

### M11 — Frontier MoE & `0.2.0` Public PyPI GA · [M11-frontier-moe.md](M11-frontier-moe.md)

| Node | Status | Depends On | Hardware | Summary |
| --- | --- | --- | --- | --- |
| `M11-N01` | `BLOCKED` | `M9-N07`, `M8-N02` | GPU-L4 | Sparse attention index & QSA/DSA kernels + Needle/RULER quality gates |
| `M11-N02` | `BLOCKED` | `M11-N01` | GPU-L4 | Out-of-VRAM n-gram & Engram lookup tables (VRAM/RAM tiered, exact parity) |
| `M11-N03` | `BLOCKED` | `M9-N05` | GPU + NVMe | Hierarchical 3-tier `VRAM -> pinned RAM -> NVMe SSD` expert store (`== fused`) |
| `M11-N04` | `BLOCKED` | `M7-N04` | GPU-L4 / sm100 | MXFP4 / NVFP4 dense & MoE expert quantization (Blackwell native + fallback) |
| `M11-N05` | `BLOCKED` | `M4-N03`, `M11-N03` | GPU-L4 | DFlash / DSpark + suffix drafting on Frontier & offloaded MoE |
| `M11-N06` | `BLOCKED` | `M11-N01..N04` | GPU 8–64GB | Representative Frontier MoE families (`Qwen3.8-Flash-Next`, `DeepSeek-V4.1-Flash`, `GLM-5.3-Flash`, `MiniMax-M2.5` class) |
| `M11-N07` | `BLOCKED` | `M11-N06` | GPU 8–64GB | Tiered `ayaka plan` (VRAM+RAM+SSD+tok/s) & frontier long-context e2e/bench |
| `M11-N08` | `BLOCKED` | `M11-N05`, `M11-N07`, `M10-N05` | CPU + GPU | **Release Gate `ayaka-llm 0.2.0` (PyPI)**: benchmark intake form, `uv build`, docs audit, PyPI publish |

---

## Stage 2 — Scale-Out & Native Kernel Engine: `0.3.0` (M12 – M13: 13 Nodes)

### M12 — Scale-Out (`v0.3.0-a1`) · [M12-scale-out.md](M12-scale-out.md)

| Node | Status | Depends On | Hardware | Summary |
| --- | --- | --- | --- | --- |
| `M12-N01` | `BLOCKED` | `M11-N08` | CPU + Multi-GPU | Parallel planner (`ayaka plan --distributed`) & NCCL/gRPC distributed comms contract |
| `M12-N02` | `BLOCKED` | `M12-N01` | 2+ GPU | Intra-node Tensor Parallelism (2-GPU TP == 1-GPU greedy tokens) |
| `M12-N03` | `BLOCKED` | `M12-N02` | 2+ GPU | KV transfer protocol & Prefill/Decode (P/D) worker disaggregation |
| `M12-N04` | `BLOCKED` | `M12-N02` | 2+ GPU | Expert Parallelism (EP) & MoE load balancing |
| `M12-N05` | `BLOCKED` | `M12-N03`, `M12-N04` | Multi-Node / 8-GPU | 2-machine benchmark & `Kimi K3`-class (+ UnifiedParser) run (8 GPU or 128 GB + SSD) |
| `M12-N06` | `GATED` | `M12-N05` | Conditional | Conditional Vulkan/Metal evaluation (only if real hardware exists), tag `v0.3.0-a1` |

### M13 — Native Kernel Engine (`kops`) & Windows Native (`0.3.0`) · [M13-native-kernels-windows.md](M13-native-kernels-windows.md)

| Node | Status | Depends On | Hardware | Summary |
| --- | --- | --- | --- | --- |
| `M13-N00` | `BLOCKED` | `M7-N05` | Linux + Win GPU | **Early Spike**: Binding ADR + 2-kernel CUDA graph capture validation |
| `M13-N01` | `BLOCKED` | `M13-N00`, `M11-N08` | Linux + Win | Build migration: `scikit-build-core` + `CMakeLists.txt` + `csrc/` (keep zero-CUDA `import ayaka`) |
| `M13-N02` | `BLOCKED` | `M13-N01` | Linux + Win GPU | Native paged attention & KV kernels (`kops`, full parity vs `torch`) |
| `M13-N03` | `BLOCKED` | `M13-N01` | Linux + Win GPU | Native fused deterministic sampler & fast FP8/INT8 KV codecs |
| `M13-N04` | `BLOCKED` | `M13-N01` | Linux + Win GPU | Native quantized GEMM (FP8/W4A16/Blackwell FP4) & fused MoE offload DMA |
| `M13-N05` | `BLOCKED` | `M13-N02..N04` | Win GPU native | Full native Windows e2e, CUDA graphs & 7/7 `ayaka connect --check` (no WSL2) |
| `M13-N06` | `BLOCKED` | `M13-N05`, `M12-N06` | Linux + Win CI | `cibuildwheel` release wheels (Linux + Windows) & publish **`ayaka-llm 0.3.0`** |

---

## Stage 3 — Agent Swarm OS & `1.0.0` GA: `0.4.0` -> `1.0.0` (M14 – M15: 10 Nodes)

### M14 — Multi-Agent & Distributed Prefix Fabric (`0.4.0`) · [M14-multi-agent-fabric.md](M14-multi-agent-fabric.md)

| Node | Status | Depends On | Hardware | Summary |
| --- | --- | --- | --- | --- |
| `M14-N01` | `BLOCKED` | `M13-N06`, `M10-N05` | GPU / Multi-GPU | Zero-copy shared prefix/KV/state fabric across concurrent coding subagents (using M5 render fingerprints) |
| `M14-N02` | `BLOCKED` | `M14-N01` | GPU | Cross-agent speculative KV prewarm & branch-aware subagent tree scheduling |
| `M14-N03` | `BLOCKED` | `M14-N01` | CPU + GPU | Multi-tenant QoS, foreground interactive priority lanes & fairness scheduler |
| `M14-N04` | `BLOCKED` | `M10-N01`, `M14-N01` | CPU + NVMe | Opt-in owner-only/encrypted persistent KV & session snapshot store across restarts |
| `M14-N05` | `BLOCKED` | `M14-N02..N04` | GPU / Multi-GPU | 8-agent coding-swarm benchmark (cache share, foreground ITL SLO) & publish **`ayaka-llm 0.4.0`** |

### M15 — Omni-Modal Computer-Use & `1.0.0` GA · [M15-omni-v1.md](M15-omni-v1.md)

| Node | Status | Depends On | Hardware | Summary |
| --- | --- | --- | --- | --- |
| `M15-N01` | `BLOCKED` | `M8-N06`, `M14-N05` | GPU | Screen/video & audio streaming inputs with incremental frame diffing & visual KV reuse |
| `M15-N02` | `BLOCKED` | `M15-N01` | GPU | Real-time multimodal computer-use action loop (screenshot stream -> structured GUI/tool action) |
| `M15-N03` | `BLOCKED` | `M14-N05` | CPU | Freeze public Python API, Wire/gRPC contracts & Backend/Model Plugin ABI (SemVer contract) |
| `M15-N04` | `BLOCKED` | `M15-N03` | CPU + GPU | Full security, privacy & supply-chain audit (MCP, disk sessions, `csrc/`, parsers, signed wheels) |
| `M15-N05` | `BLOCKED` | `M15-N02`, `M15-N04` | All Tiers | Multi-tier verified `CAPABILITIES.md` & publish **`ayaka-llm 1.0.0` GA** |
