# Ayaka Roadmap

Internal source of truth for the Ayaka capability roadmap and sequential **Execution Nodes**.
Public tracking index: [gityuko-hb/ayaka#62](https://github.com/gityuko-hb/ayaka/issues/62).

**Last updated:** `2026-11-26` (synced against `main` @ `3292d09`, open PRs #131–#133, and the
insertion of **M5: Serving Core v2 — Render, Request-Scoped Parsers & gRPC**).

## How This Roadmap Works

- A milestone is a **capability**, judged by runnable **exit conditions** — not by PR count.
- Every active and future milestone (`M04`–`M15`) lives in its own file containing its goals,
  exit conditions, pre-registered measurement gates, and sequential **Execution Nodes**
  (`M4-N01` .. `M15-N05`, **86 Nodes total**).
- Use [EXECUTION_GRAPH.md](EXECUTION_GRAPH.md) to pick the next `READY` Node in dependency order.
- Every open technical decision has a pre-registered measurement and threshold written *before* the
  run; new features stay **opt-in** until a before/after run on the same hardware clears its gate.
- Strategy rationale, hardware tiers, and competitive positioning live in
  [../strategy/README.md](../strategy/README.md).

## Release Ladder

| Release | Milestones Included | Meaning |
| --- | --- | --- |
| **`0.1.0`** *(done, preview)* | M0 – M3 | Foundation, GPU execution, Serving + hardening, Dense/MoE & sliced FP8 (checkout / GitHub preview) |
| **`0.2.0-a1` .. `a7`** *(internal tags)* | M4, M5, M6, M7, M8, M9, M10 | Per-milestone regression & benchmark checkpoints on the road to `0.2.0` |
| **`0.2.0` GA** *(First Public PyPI)* | **M4 – M11** | Full single-machine coding-agent & Frontier MoE engine on 4–64 GB GPUs |
| **`0.3.0`** | **M12 + M13** | Multi-GPU Scale-out (TP/PP/EP, P/D split, `Kimi K3`) + Native Kernel Engine (`kops`/`csrc/`) & Windows Native wheels |
| **`0.4.0`** | **M14** | Multi-Agent & Distributed Prefix Fabric (swarm zero-copy KV/state trees, QoS lanes, opt-in persistent KV) |
| **`1.0.0` GA** | **M15** | Omni-Modal Computer-Use (screen/video/audio), frozen Python/Wire/Plugin ABI, security audit |

## Milestone Map

| ID | Milestone | Target Tag | Nodes | Status | File |
| --- | --- | --- | ---: | --- | --- |
| **M0** | **Foundation — CPU, torch backend** | `0.0.1` | — | **done** (#10) | [01-completed.md](01-completed.md) |
| **M1** | **GPU execution** | `0.0.1` | — | **done** (#26, #99) | [01-completed.md](01-completed.md) |
| **M2** | **Serving & hardening** | `0.1.0` | — | **done** (#43–#110) | [01-completed.md](01-completed.md) |
| **M3** | **Models and quantization** | `0.1.0` | — | **done** (#111–#121, #130) | [01-completed.md](01-completed.md) |
| **M4** | **Speculative decoding** | `0.2.0-a1` | 6 | **in progress** (#127–#129, PRs #131–#133) | [M04-speculative.md](M04-speculative.md) |
| **M5** | **Serving Core v2: Render, Parsers, Structured Output (XGrammar) & gRPC** | `0.2.0-a2` | 12 | **next** (not filed) | [M05-serving-v2.md](M05-serving-v2.md) |
| **M6** | **Memory** | `0.2.0-a3` | 10 | **planned** (not filed) | [M06-memory.md](M06-memory.md) |
| **M7** | **Fast Quantized Kernels (FP8 & W4A16)** | `0.2.0-a4` | 5 | **planned** (not filed) | [M07-fast-quant.md](M07-fast-quant.md) |
| **M8** | **Hybrid dense, state and vision** | `0.2.0-a5` | 8 | **planned** (not filed) | [M08-hybrid-vision.md](M08-hybrid-vision.md) |
| **M9** | **MoE on small machines** | `0.2.0-a6` | 7 | **planned** (not filed) | [M09-moe-small-machines.md](M09-moe-small-machines.md) |
| **M10** | **Agent platform** | `0.2.0-a7` | 7 | **planned** (not filed) | [M10-agent-platform.md](M10-agent-platform.md) |
| **M11** | **Frontier MoE & `0.2.0` Public PyPI GA** | **`0.2.0`** | 8 | **planned** (not filed) | [M11-frontier-moe.md](M11-frontier-moe.md) |
| **M12** | **Scale-out** | `0.3.0-a1` | 6 | **planned** (not filed) | [M12-scale-out.md](M12-scale-out.md) |
| **M13** | **Native Kernel Engine (`kops`) & Windows Native** | **`0.3.0`** | 7 | **spike `M13-N00` after M7** | [M13-native-kernels-windows.md](M13-native-kernels-windows.md) |
| **M14** | **Multi-Agent & Distributed Prefix Fabric** | **`0.4.0`** | 5 | **planned** (not filed) | [M14-multi-agent-fabric.md](M14-multi-agent-fabric.md) |
| **M15** | **Omni-Modal Computer-Use & `1.0.0` GA** | **`1.0.0`** | 5 | **planned** (not filed) | [M15-omni-v1.md](M15-omni-v1.md) |

## Dependency Graph

```text
M0 Foundation -> M1 GPU -> M2 Serving (0.1.0) -> M3 Models & Quantization (done)
                                                        |
                                                        v
                                        M4 Speculative Decoding [0.2.0-a1] (in progress)
                                                        |
                                                        v
                         M5 Serving Core v2: Render, Request-Scoped Parsers & gRPC [0.2.0-a2]
                                                        |
                                                        v
                                              M6 Memory [0.2.0-a3]
                                                        |
                                                        v
                                    M7 Fast Quantized Kernels [0.2.0-a4]
                                        |                           |
                                        |                           +--> M13-N00 Early Spike
                                        v                                (Binding ADR + 2-kernel graph)
                                 M8 Hybrid, State & Vision [0.2.0-a5]
                                        |
                                        v
                                M9 MoE on Small Machines [0.2.0-a6]
                                        |
                                        v
                                  M10 Agent Platform [0.2.0-a7]
                                        |
                                        v
                                   M11 Frontier MoE
                                        |
                          =======[ Public PyPI 0.2.0 GA ]=======
                                        |
                                        v
                               M12 Scale-out [0.3.0-a1]
                                        |
                                        v
                   M13 Native Kernel Engine & Windows Native [0.3.0]
                                        |
                                        v
                    M14 Multi-Agent & Distributed Prefix Fabric [0.4.0]
                                        |
                                        v
                      M15 Omni-Modal Computer-Use & ABI [1.0.0 GA]
```

## Risk Register

| Risk | Impact | Mitigation |
| --- | --- | --- |
| Long `0.2.0` horizon (M4–M11) causes drift or regressions | Broken invariants before PyPI launch | Mandatory internal tags `0.2.0-a1`..`a7` with exit audits and `benchmarks/history/` after every M |
| Request-scoped thinking/parser mismatch deadlocks `llguidance` | Hung or corrupted tool calls | Unified `RenderedChat` -> `ChatResponseProcessor` -> `constraint_start()` contract in M5 Wave 1 |
| `grpcio` event-loop contention slows token streaming | Latency regression on remote gateway | Pre-registered L4 overhead gate (`≤ 0.3 ms` p50/token, `≤ 2%` tok/s) & UDS transport in `M5-N12` |
| XGrammar special-token / rollback bugs with XML `STRUCTURAL_TAG` or M4 speculation | Invalid tool calls or grammar desync | Dual-compiler isolation (`json` vs `full_vocab`), exact guided-speculative equivalence tests & head-to-head L4 gate (`M5-N08`, `M5-N12`) |
| 4 GB VRAM too tight for full coding-tool system prompt or strong agent model | Tier 1 usability ceiling | Honest `ayaka plan --for <tool>` + W4A16 (`M7-N04`) + Host KV (`M6-N07`) + RAM MoE offload (`M9`) |
| WSL2 WDDM oversubscription / pinning limits | Silent 2×–5× paging slowdown or capped host tier | Startup reconciliation, `ayaka doctor` (`M6-N01`), Linux/WSL2 pin probes (`M6-N06`, `M9-N01`) |
| FP8 / W4A16 kernel incompatibility with CUDA graphs or `sm_86`/`sm_89` | M7 gate missed | Pre-registered per-backend parity & graph-capture tests on both L4 (`sm_89`) and RTX 3050 (`sm_86`) |
| Speculative decoding regresses busy multi-stream servers | Hurts concurrent serving | Cost-aware controller + mandatory multi-concurrency gate (`M4-N01`, `M4-N06`); opt-in default |
| Warm KV / hybrid state / offloaded MoE corruption | Silent wrong tokens across turns | Exact token-ID equivalence tests (cold vs warm, 1-chunk vs N-chunk, offload vs fused) |
| Frontier model definitions shift rapidly | Wasted modeling work | M8/M11/M12 model names are representative targets; exact checkpoints pinned only when each M opens |
| Native `csrc/` ABI/graph surprises delaying `0.3.0` | Stalls Windows native & custom DMA | Early low-risk spike `M13-N00` right after M7 (`0.2.0-a4`) |

## Index

- [EXECUTION_GRAPH.md](EXECUTION_GRAPH.md) — Master sequential queue & cross-milestone DAG across all **86 Execution Nodes**
- [decisions.md](decisions.md) — Complete settled & open decision log
- [01-completed.md](01-completed.md) — Completed milestones (M0–M3)
- Milestone Execution Files:
  [M04](M04-speculative.md) · [M05](M05-serving-v2.md) · [M06](M06-memory.md) ·
  [M07](M07-fast-quant.md) · [M08](M08-hybrid-vision.md) · [M09](M09-moe-small-machines.md) ·
  [M10](M10-agent-platform.md) · [M11](M11-frontier-moe.md) · [M12](M12-scale-out.md) ·
  [M13](M13-native-kernels-windows.md) · [M14](M14-multi-agent-fabric.md) · [M15](M15-omni-v1.md)
