# Decision Log (Settled & Open)

Tracks every architectural, product, and default-setting decision from issue
[#62](https://github.com/gityuko-hb/ayaka/issues/62), experimental runs in
`docs/EXPERIMENTS.md`, and the `2026-11-25` / `2026-11-26` strategy & roadmap rebuilds.

Rule: Every default change or open performance decision requires a pre-registered measurement and
threshold before flipping the default.

---

## Settled Decisions

| Decision | Resolution | Evidence / Reference |
| --- | --- | --- |
| **PyPI distribution name** | `ayaka-llm` (imported as `ayaka`, CLI `ayaka`) | [#102](https://github.com/gityuko-hb/ayaka/issues/102), [#107](https://github.com/gityuko-hb/ayaka/pull/107) |
| **First coding tools for `ayaka connect`** | Claude Code, Codex CLI, and OpenCode (all 3 passing 7/7 `--check` & real tool replays) | [#56](https://github.com/gityuko-hb/ayaka/issues/56), [#101](https://github.com/gityuko-hb/ayaka/pull/101), [#104](https://github.com/gityuko-hb/ayaka/pull/104), [#109](https://github.com/gityuko-hb/ayaka/pull/109) |
| **Public docs vs internal planning in `docs/`** | Public pages allowlisted individually in `.gitignore`; `docs/ROADMAP/` and `docs/strategy/` stay untracked local internal docs | [#60](https://github.com/gityuko-hb/ayaka/issues/60), [#71](https://github.com/gityuko-hb/ayaka/pull/71), `.gitignore` |
| **8-bit weights timing vs Serving** | Serving completed on 16-bit first; FP8 landed in M3 Models & Quantization | [#112](https://github.com/gityuko-hb/ayaka/issues/112), [#117](https://github.com/gityuko-hb/ayaka/pull/117) |
| **Speculative decoding vs Models & Quantization** | Models & Quantization (M3) completed first; Speculative decoding (M4) follows immediately | [#111](https://github.com/gityuko-hb/ayaka/issues/111)–[#113](https://github.com/gityuko-hb/ayaka/issues/113), [#127](https://github.com/gityuko-hb/ayaka/issues/127)–[#129](https://github.com/gityuko-hb/ayaka/issues/129) |
| **Server-engine seam placement** | Frozen `EngineClient` seam established in M2 hardening (`python/ayaka/engine/seam.py`); wrapped over gRPC in M5 (`M5-N10`) | [#78](https://github.com/gityuko-hb/ayaka/issues/78), [#89](https://github.com/gityuko-hb/ayaka/pull/89) |
| **SSE frame coalescing (`sse_coalesce_ms`)** | Keep `sse_coalesce_ms = 0` (SSE encoding is `0.21%` of step time, far below `5%` gate) | [#77](https://github.com/gityuko-hb/ayaka/issues/77), [#90](https://github.com/gityuko-hb/ayaka/pull/90), `docs/EXPERIMENTS.md` |
| **Segmented prompt tokenize cache** | Deferred (`render + encode` is `0.80–0.98%` of cold TTFT, well below `5%` gate) | [#77](https://github.com/gityuko-hb/ayaka/issues/77), [#90](https://github.com/gityuko-hb/ayaka/pull/90), `docs/EXPERIMENTS.md` |
| **Claude Code billing-header normalization** | Keep enabled (`+84.3` percentage points cross-session cache hit, `20/20` identical answers); Codex skill paths & OpenCode explicitly not normalized | [#57](https://github.com/gityuko-hb/ayaka/issues/57), `docs/EXPERIMENTS.md` |
| **Sampling backend (FlashInfer vs torch)** | Keep sampling on `torch` (FlashInfer 0.7.4 lacks external uniform draw `u` support required by position-keyed RNG) | [#99](https://github.com/gityuko-hb/ayaka/pull/99), `docs/EXPERIMENTS.md` |
| **Default compute dtype (`fp16` vs `bf16`)** | Keep `fp16` default (`bf16` gain was `≤ 2%`, below `5%` switch gate) | [#99](https://github.com/gityuko-hb/ayaka/pull/99), `docs/EXPERIMENTS.md` |
| **FP8 default & status in M3** | Keep 16-bit default; document sliced FP8 as *"fits, but slow"* (`0.11–0.12×` bf16 on L4, below `0.5×` gate) until M7 fast kernels land | [#121](https://github.com/gityuko-hb/ayaka/pull/121), `docs/EXPERIMENTS.md`, `docs/TROUBLESHOOTING.md` |
| **Target hardware positioning** | 3-tier ladder: 4–8 GB (floor/offload), 12–24 GB (primary coding-agent sweet spot), 32–64 GB+ (Frontier MoE) | Strategy rebuild (`2026-11-25`) |
| **Serving Core v2 insertion (M5)** | Insert **M5** right after M4 Speculative: pure Python 3.13 Render Engine, request-scoped output interpretation (`none` & `thinking`), multi-family reasoning/XML/JSON tool parsers, Structured Output v2 (`llguidance` + optional `--extra xgrammar`, 6 `ConstraintKind` + `rollback`), and dual gRPC (`Inference` + engine-free `Render`) | Serving & XGrammar research (`2026-11-26`), [M05](M05-serving-v2.md) |
| **Fast quantized kernel path (M7)** | Dedicated M7 milestone (FP8 GEMM + safetensors INT4/AWQ/GPTQ Marlin W4A16) via FlashInfer/`torch._scaled_mm`/Triton/Marlin; GGUF Q4_K_M execution deferred/opt-in | Strategy rebuild (`2026-11-25`) |
| **PyPI public launch timing (`0.2.0`)** | Keep `0.1.0` as preview; internal tags `0.2.0-a1`..`a7` after M4–M10; first public PyPI launch **`0.2.0` GA** once **M11 (Frontier MoE)** completes | Roadmap expansion (`2026-11-25/26`) |
| **Post-`0.2.0` long-horizon ladder** | `0.3.0` = M12 (Scale-out) + M13 (Native Kernel Engine & Windows Native); `0.4.0` = M14 (Multi-Agent & Distributed Prefix Fabric); `1.0.0` GA = M15 (Omni-Modal Computer-Use & Stable ABI) | Roadmap expansion (`2026-11-25/26`) |
| **Early native C++/CUDA spike (`M13-N00`)** | Run Binding ADR + 2-kernel CUDA graph spike immediately after M7 (`0.2.0-a4`), in parallel with M8/M9 | Roadmap expansion (`2026-11-25/26`) |
| **Mid-term ordering (Hybrid/Vision vs MoE offload)** | M8 (Hybrid dense, state and vision) before M9 (MoE on small machines), with M7 already unlocking FP8/W4A16 for M9 | Roadmap rebuild (`2026-11-25`) |
| **M10 Agent platform scoping** | Split into Tier 1 Core (`M10-N01..N05`) and Tier 2 Extended/Gated (`M10-N06..N07`) | Roadmap rebuild (`2026-11-25`) |
| **`cuda_graph=True` default on CUDA** | Approved for dedicated default-change PR `M4-N04` (`2.13×–9.09×` over eager on L4, above `1.5×` gate; confirm 4 GB reconciliation) | [#99](https://github.com/gityuko-hb/ayaka/pull/99), `M4-N04` |
| **`tool_choice="none"` prompt layout** | Preserve tools in prompt to protect prefix-cache hits (locked by test in `M4-N05`) | `M4-N05` |

---

## Open Decisions (Gated by Future Milestones)

| Milestone | Open Decision | Pre-Registered Gate / Criteria |
| --- | --- | --- |
| **M4** | Enable lookup speculation by default? | Keep opt-in until `≥ 1.15×` on code-edit prompts AND `≥ 0.98×` at concurrency `≥ 4` on L4 |
| **M4** | Cost-aware controller vs fixed `k`; candidate-reuse module? | Decided from PR #131 multi-prompt & multi-concurrency measurements |
| **M5** | Default structured-output backend on Linux CUDA (`llguidance` vs `xgrammar`) | Keep `llguidance` default unless `xgrammar` (`--extra xgrammar`) wins `≥ 15%` on CUDA-graph mask / guided-speculative throughput (or uniquely unlocks guided speculation) with no cold-compile regression (`M5-N12`) |
| **M5** | Should Render/Parsers ever port to Rust/PyO3? | Keep pure Python 3.13 unless concurrent serving benchmarks show CPU parse/render `> 5%` of GPU step time |
| **M5** | gRPC transport overhead (`GrpcEngineClient` vs in-process) | Must stay `≤ 0.3 ms` p50/token added latency and `≤ 2%` throughput delta on L4 (`M5-N12`) |
| **M6** | Exact wire error codes (`out_of_memory`, `preempted_timeout`, `FinishEvent.limit="memory"`) | Settled in #140 / PR #141 against OpenAI & Anthropic wire compatibility |
| **M6** | Host KV tier on by default & default `ram_keep`? | Opt-in (`enabled=False`, `--host-kv`, `--ram-keep 0.20`) implemented; keep opt-in until WSL2/Linux pinned-RAM ceiling measured, promotion `< 0.5×` cold prefill (`≥ 4K`), and write-behind overhead `≤ 2%` |
| **M6** | Time-budgeted chunking & overlapped scheduling? | Decided from M6 replay/concurrency measurements |
| **M7** | GGUF `Q4_K_M` direct execution timing (after safetensors AWQ/GPTQ Marlin in `M7-N04`) | Evaluated as opt-in follow-up if user demand exceeds safetensors INT4 availability |
| **M8** | State checkpoint density, KV FP8 default, vision lazy vs eager load, flagship hybrid model | Measured when M8 opens (note: #99 already showed FlashInfer 0.7.4 FP8 KV is `1.28–1.50×` slower at batch 1) |
| **M9** | WSL2 large expert pinning limit, profile+decay vs LRU, dynamic expert/KV VRAM split, MTP on 4 GB | Measured in `M9-N01` / `M9-N05` / `M9-N06` benchmarks |
| **M10** | MCP on by default vs `--mcp`; session disk persistence policy; Tier 2 (`M10-N06..N07`) go/no-go gates | Security review + value/overhead measurement gates in M10 |
| **M11** | RAM vs SSD crossover, QSA vs DSA order, Engram VRAM vs RAM, FP4 quality, DFlash vs DSpark on MoE | Measured across `M11-N01..N05` |
| **M12** | TP vs PP first on consumer dual-GPU; P/D payoff on 2 GPUs; Vulkan/Metal hardware availability | Measured in `M12-N02`, `M12-N03`, `M12-N06` |
| **M13** | C++/CUDA binding library, minimum CUDA version, target `sm_` list, full `csrc/` trigger date | Settled in early spike `M13-N00` right after M7 (`0.2.0-a4`) |
| **M14** | Local vs networked swarm prefix fabric protocol; persistent KV encryption & eviction defaults | Measured in `M14-N01`, `M14-N04`, `M14-N05` |
| **M15** | Video/screen frame-diffing architecture & public ABI deprecation window | Decided in `M15-N01` & `M15-N03` RFCs |
