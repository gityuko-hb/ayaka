# Ayaka Roadmap

Internal source of truth for the Ayaka capability roadmap and sequential **Execution Nodes**.
Public tracking index: [gityuko-hb/ayaka#62](https://github.com/gityuko-hb/ayaka/issues/62).

**Last updated:** `2026-11-27` (synced against `main` @ `e4d0524`, the milestone order decided on `2026-11-27`,
and the open pull requests and issues of each milestone listed below).

## How This Roadmap Works

- A milestone is a **capability**, judged by runnable **exit conditions** — not by PR count.
- Every active and future milestone lives in its own file containing its goals, exit conditions,
  pre-registered measurement gates, and sequential **Execution Nodes** (called steps in prose, for example
  "memory step 3"). There are 86 steps in total.
- Use [EXECUTION_GRAPH.md](EXECUTION_GRAPH.md) to pick the next `READY` step in dependency order.
- Every open technical decision has a pre-registered measurement and threshold written *before* the
  run; new features stay **opt-in** until a before/after run on the same hardware clears its gate.
- Strategy rationale, hardware tiers, and competitive positioning live in
  [../strategy/README.md](../strategy/README.md).

## Milestone Order (decided 2026-11-27)

The round runs the four milestones below in this order. Each one depends on the exit audit of the one before it.

1. **Serving Core v2** — internal tag `v0.2.0-a1`. Starts on `main`, where the models and quantization milestone is
   done. Absorbs PR #134 (keep tools in the prompt for `tool_choice="none"`). Open PRs #137, #138, #139 and issue #136.
2. **Memory** — internal tag `v0.2.0-a2`. Depends on the Serving Core v2 exit audit. Open PRs #141, #142 and issue #140.
3. **Fast quantized kernels (FP8 and W4A16)** — internal tag `v0.2.0-a3`. Depends on the memory exit audit. Not filed yet.
4. **Speculative decoding** — internal tag `v0.2.0-a4`. Depends on the fast quantized kernels exit audit. Open PRs
   #131, #132, #133, #135 and issues #127–#129.

Reason: speculative decoding builds on the surfaces the other three define. Its verify windows must roll back grammar
state (Serving Core v2 structured output), its drafts take KV blocks that preemption and the host tier must account
for (memory), and its cost-aware controller and benchmarks depend on the per-token decode cost, which the FP8 and
4-bit kernels change. Speculative numbers measured before those land would have to be measured again.

## Hardware and Reference Models This Round

- Development GPU: NVIDIA A100-SXM4-80GB (sm_80).
- Gates written for the NVIDIA L4 (sm_89) stay **unverified** until they run on an NVIDIA L4. The milestone files
  say so where a gate applies. No L4 result is claimed in this round yet.
- sm_80 has no FP8 tensor cores. On A100, FP8 checkpoints run as weight-only FP8 (W8A16) kernels. Native FP8 GEMM is
  an sm_89 and newer path.
- Reference models: `Qwen/Qwen3-8B` (bf16); `Qwen/Qwen3-8B-FP8` and `Qwen/Qwen3-8B-AWQ` for quantized kernels;
  `z-lab/Qwen3-8B-DFlash-b16` for speculative decoding; `Qwen/Qwen3-0.6B` for fast tests.

## Release Ladder

| Release | Milestones Included | Meaning |
| --- | --- | --- |
| **`0.1.0`** *(done, preview)* | Foundation through models and quantization | Foundation, GPU execution, Serving + hardening, Dense/MoE & sliced FP8 (checkout / GitHub preview) |
| **`0.2.0-a1` .. `a4`** *(internal tags)* | Serving Core v2, memory, fast quantized kernels, speculative decoding | Per-milestone regression & benchmark checkpoints on the road to `0.2.0` |
| **`0.2.0-a5` .. `a7`** *(internal tags)* | Hybrid, state and vision; MoE on small machines; agent platform | Continuation of the same checkpoints |
| **`0.2.0` GA** *(First Public PyPI)* | **Serving Core v2 through frontier MoE** | Full single-machine coding-agent & Frontier MoE engine on 4–64 GB GPUs |
| **`0.3.0`** | **Scale-out + native kernel engine** | Multi-GPU Scale-out (TP/PP/EP, P/D split, `Kimi K3`) + Native Kernel Engine (`kops`/`csrc/`) & Windows Native wheels |
| **`0.4.0`** | **Multi-agent fabric** | Multi-Agent & Distributed Prefix Fabric (swarm zero-copy KV/state trees, QoS lanes, opt-in persistent KV) |
| **`1.0.0` GA** | **Omni-modal** | Omni-Modal Computer-Use (screen/video/audio), frozen Python/Wire/Plugin ABI, security audit |

## Milestone Map

| Order | Milestone | Target Tag | Steps | Status | File |
| --- | --- | --- | ---: | --- | --- |
| done | **Foundation — CPU, torch backend** | `0.0.1` | — | **done** (#10) | [01-completed.md](01-completed.md) |
| done | **GPU execution** | `0.0.1` | — | **done** (#26, #99) | [01-completed.md](01-completed.md) |
| done | **Serving & hardening** | `0.1.0` | — | **done** (#43–#110) | [01-completed.md](01-completed.md) |
| done | **Models and quantization** | `0.1.0` | — | **done** (#111–#121, #130) | [01-completed.md](01-completed.md) |
| 1 | **Serving Core v2: Render, Parsers, Structured Output (XGrammar) & gRPC** | `0.2.0-a1` | 13 | **next** (not filed; open PRs #137, #138, #139, issue #136) | [M05-serving-v2.md](M05-serving-v2.md) |
| 2 | **Memory** | `0.2.0-a2` | 10 | **planned** (not filed; open PRs #141, #142, issue #140) | [M06-memory.md](M06-memory.md) |
| 3 | **Fast Quantized Kernels (FP8 & W4A16)** | `0.2.0-a3` | 5 | **planned** (not filed) | [M07-fast-quant.md](M07-fast-quant.md) |
| 4 | **Speculative decoding** | `0.2.0-a4` | 5 | **in progress** (#127–#129, PRs #131–#133, #135; PR #134 moved to step 13 of Serving Core v2) | [M04-speculative.md](M04-speculative.md) |
| 5 | **Hybrid dense, state and vision** | `0.2.0-a5` | 8 | **planned** (not filed) | [M08-hybrid-vision.md](M08-hybrid-vision.md) |
| 6 | **MoE on small machines** | `0.2.0-a6` | 7 | **planned** (not filed) | [M09-moe-small-machines.md](M09-moe-small-machines.md) |
| 7 | **Agent platform** | `0.2.0-a7` | 7 | **planned** (not filed) | [M10-agent-platform.md](M10-agent-platform.md) |
| 8 | **Frontier MoE & `0.2.0` Public PyPI GA** | **`0.2.0`** | 8 | **planned** (not filed) | [M11-frontier-moe.md](M11-frontier-moe.md) |
| 9 | **Scale-out** | `0.3.0-a1` | 6 | **planned** (not filed) | [M12-scale-out.md](M12-scale-out.md) |
| 10 | **Native Kernel Engine (`kops`) & Windows Native** | **`0.3.0`** | 7 | **spike (step 0) after fast quantized kernels milestone** | [M13-native-kernels-windows.md](M13-native-kernels-windows.md) |
| 11 | **Multi-Agent & Distributed Prefix Fabric** | **`0.4.0`** | 5 | **planned** (not filed) | [M14-multi-agent-fabric.md](M14-multi-agent-fabric.md) |
| 12 | **Omni-Modal Computer-Use & `1.0.0` GA** | **`1.0.0`** | 5 | **planned** (not filed) | [M15-omni-v1.md](M15-omni-v1.md) |

## Dependency Graph

```text
Foundation -> GPU execution -> Serving & hardening (0.1.0) -> Models & Quantization (done)
                                                                      |
                                                                      v
                                 Serving Core v2: Render, Request-Scoped Parsers & gRPC [0.2.0-a1]
                                                                      |
                                                                      v
                                                        Memory [0.2.0-a2]
                                                                      |
                                                                      v
                                              Fast Quantized Kernels [0.2.0-a3]
                                                |                           |
                                                |                           +--> Native kernel engine step 0 (Early Spike)
                                                |                                (Binding ADR + 2-kernel graph)
                                                v
                                        Speculative Decoding [0.2.0-a4]
                                                |
                                                v
                                  Hybrid, State & Vision [0.2.0-a5]
                                                |
                                                v
                                  MoE on Small Machines [0.2.0-a6]
                                                |
                                                v
                                    Agent Platform [0.2.0-a7]
                                                |
                                                v
                                       Frontier MoE
                                                |
                          =======[ Public PyPI 0.2.0 GA ]=======
                                                |
                                                v
                                  Scale-out [0.3.0-a1]
                                                |
                                                v
                     Native Kernel Engine & Windows Native [0.3.0]
                                                |
                                                v
                      Multi-Agent & Distributed Prefix Fabric [0.4.0]
                                                |
                                                v
                        Omni-Modal Computer-Use & ABI [1.0.0 GA]
```

## Risk Register

| Risk | Impact | Mitigation |
| --- | --- | --- |
| Long `0.2.0` horizon causes drift or regressions | Broken invariants before PyPI launch | Mandatory internal tags `0.2.0-a1`..`a7` with exit audits and `benchmarks/history/` after every milestone |
| Request-scoped thinking/parser mismatch deadlocks `llguidance` | Hung or corrupted tool calls | Unified `RenderedChat` -> `ChatResponseProcessor` -> `constraint_start()` contract in Serving Core v2 step 3 |
| `grpcio` event-loop contention slows token streaming | Latency regression on remote gateway | Pre-registered L4 overhead gate (`≤ 0.3 ms` p50/token, `≤ 2%` tok/s) & UDS transport in Serving Core v2 step 12 |
| XGrammar special-token / rollback bugs with XML `STRUCTURAL_TAG` or speculative decoding | Invalid tool calls or grammar desync | Dual-compiler isolation (`json` vs `full_vocab`), `rollback(k)` replay tests in Serving Core v2 step 7, and guided-speculation equivalence tests in speculative decoding step 6 |
| 4 GB VRAM too tight for full coding-tool system prompt or strong agent model | Tier 1 usability ceiling | Honest `ayaka plan --for <tool>` + W4A16 (fast quantized kernels step 4) + Host KV (memory step 7) + RAM MoE offload (MoE on small machines milestone) |
| WSL2 WDDM oversubscription / pinning limits | Silent 2×–5× paging slowdown or capped host tier | Startup reconciliation, `ayaka doctor` (memory step 1), Linux/WSL2 pin probes (memory step 6, MoE on small machines step 1) |
| FP8 / W4A16 kernel incompatibility with CUDA graphs or `sm_86`/`sm_89`/`sm_80` | Fast quantized kernels gate missed | Pre-registered per-backend parity & graph-capture tests on NVIDIA L4 (`sm_89`) and RTX 3050 (`sm_86`); A100 (`sm_80`) runs use the weight-only FP8 path and are reported separately |
| Speculative decoding regresses busy multi-stream servers | Hurts concurrent serving | Cost-aware controller + mandatory multi-concurrency gate (speculative decoding steps 1 and 6); opt-in default |
| Warm KV / hybrid state / offloaded MoE corruption | Silent wrong tokens across turns | Exact token-ID equivalence tests (cold vs warm, 1-chunk vs N-chunk, offload vs fused) |
| Frontier model definitions shift rapidly | Wasted modeling work | Model names in the hybrid, MoE, frontier MoE and scale-out milestones are representative targets; exact checkpoints pinned only when each milestone opens |
| Native `csrc/` ABI/graph surprises delaying `0.3.0` | Stalls Windows native & custom DMA | Early low-risk spike (native kernel engine step 0) right after the fast quantized kernels milestone (`0.2.0-a3`) |

## Index

- [EXECUTION_GRAPH.md](EXECUTION_GRAPH.md) — Master sequential queue & cross-milestone DAG across all **86 steps**
- [decisions.md](decisions.md) — Complete settled & open decision log
- [01-completed.md](01-completed.md) — Completed milestones (foundation through models and quantization)
- Milestone Execution Files:
  [Serving Core v2](M05-serving-v2.md) · [Memory](M06-memory.md) ·
  [Fast quantized kernels](M07-fast-quant.md) · [Speculative decoding](M04-speculative.md) ·
  [Hybrid, state and vision](M08-hybrid-vision.md) · [MoE on small machines](M09-moe-small-machines.md) ·
  [Agent platform](M10-agent-platform.md) · [Frontier MoE](M11-frontier-moe.md) · [Scale-out](M12-scale-out.md) ·
  [Native kernel engine](M13-native-kernels-windows.md) · [Multi-agent fabric](M14-multi-agent-fabric.md) ·
  [Omni-modal](M15-omni-v1.md)
