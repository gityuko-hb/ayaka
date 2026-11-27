# Decision Log (Settled & Open)

Tracks every architectural, product, and default-setting decision from issue
[#62](https://github.com/gityuko-hb/ayaka/issues/62), experimental runs in
`docs/EXPERIMENTS.md`, and the `2026-11-25` / `2026-11-26` strategy & roadmap rebuilds, and the
`2026-11-27` milestone reorder.

Rule: Every default change or open performance decision requires a pre-registered measurement and
threshold before flipping the default.

---

## Settled Decisions

| Decision | Resolution | Evidence / Reference |
| --- | --- | --- |
| **Milestone order for this round (`2026-11-27`)** | Serving Core v2 (`v0.2.0-a1`), then memory (`v0.2.0-a2`), then fast quantized kernels FP8 & W4A16 (`v0.2.0-a3`), then speculative decoding (`v0.2.0-a4`). Each milestone starts after the exit audit of the one before it. Speculative decoding moved last because its verify windows roll back grammar state (Serving Core v2), its drafts take KV blocks that preemption and the host tier account for (memory), and its controller and benchmarks depend on the per-token decode cost that the FP8 and 4-bit kernels change. Speculative numbers measured before those land would be measured again. | [README](README.md), [M05](M05-serving-v2.md), [M06](M06-memory.md), [M07](M07-fast-quant.md), [M04](M04-speculative.md) |
| **Reference models and hardware for this round (`2026-11-27`)** | Qwen/Qwen3-8B (bf16); Qwen/Qwen3-8B-FP8 and Qwen/Qwen3-8B-AWQ for quantized kernels; z-lab/Qwen3-8B-DFlash-b16 for speculative decoding; Qwen/Qwen3-0.6B for fast tests. Development GPU is an NVIDIA A100-SXM4-80GB (sm_80). Gates written for the NVIDIA L4 stay unverified until they run on an L4. | [README](README.md), [M07](M07-fast-quant.md) |
| **PyPI distribution name** | `ayaka-llm` (imported as `ayaka`, CLI `ayaka`) | [#102](https://github.com/gityuko-hb/ayaka/issues/102), [#107](https://github.com/gityuko-hb/ayaka/pull/107) |
| **First coding tools for `ayaka connect`** | Claude Code, Codex CLI, and OpenCode (all 3 passing 7/7 `--check` & real tool replays) | [#56](https://github.com/gityuko-hb/ayaka/issues/56), [#101](https://github.com/gityuko-hb/ayaka/pull/101), [#104](https://github.com/gityuko-hb/ayaka/pull/104), [#109](https://github.com/gityuko-hb/ayaka/pull/109) |
| **Public docs vs internal planning in `docs/`** | Public pages allowlisted individually in `.gitignore`; `docs/ROADMAP/` and `docs/strategy/` stay untracked local internal docs | [#60](https://github.com/gityuko-hb/ayaka/issues/60), [#71](https://github.com/gityuko-hb/ayaka/pull/71), `.gitignore` |
| **8-bit weights timing vs Serving** | Serving completed on 16-bit first; FP8 landed in the models and quantization milestone | [#112](https://github.com/gityuko-hb/ayaka/issues/112), [#117](https://github.com/gityuko-hb/ayaka/pull/117) |
| **Speculative decoding vs Models & Quantization** | Models & Quantization completed first. Speculative decoding is the last of the four round milestones in the `2026-11-27` order | [#111](https://github.com/gityuko-hb/ayaka/issues/111)–[#113](https://github.com/gityuko-hb/ayaka/issues/113), [#127](https://github.com/gityuko-hb/ayaka/issues/127)–[#129](https://github.com/gityuko-hb/ayaka/issues/129) |
| **Server-engine seam placement** | Frozen `EngineClient` seam established in serving hardening (`python/ayaka/engine/seam.py`); wrapped over gRPC in Serving Core v2 step 10 | [#78](https://github.com/gityuko-hb/ayaka/issues/78), [#89](https://github.com/gityuko-hb/ayaka/pull/89) |
| **SSE frame coalescing (`sse_coalesce_ms`)** | Keep `sse_coalesce_ms = 0` (SSE encoding is `0.21%` of step time, far below `5%` gate) | [#77](https://github.com/gityuko-hb/ayaka/issues/77), [#90](https://github.com/gityuko-hb/ayaka/pull/90), `docs/EXPERIMENTS.md` |
| **Segmented prompt tokenize cache** | Deferred (`render + encode` is `0.80–0.98%` of cold TTFT, well below `5%` gate) | [#77](https://github.com/gityuko-hb/ayaka/issues/77), [#90](https://github.com/gityuko-hb/ayaka/pull/90), `docs/EXPERIMENTS.md` |
| **Claude Code billing-header normalization** | Keep enabled (`+84.3` percentage points cross-session cache hit, `20/20` identical answers); Codex skill paths & OpenCode explicitly not normalized | [#57](https://github.com/gityuko-hb/ayaka/issues/57), `docs/EXPERIMENTS.md` |
| **Sampling backend (FlashInfer vs torch)** | Keep sampling on `torch` (FlashInfer 0.7.4 lacks external uniform draw `u` support required by position-keyed RNG) | [#99](https://github.com/gityuko-hb/ayaka/pull/99), `docs/EXPERIMENTS.md` |
| **Default compute dtype (`fp16` vs `bf16`)** | Keep `fp16` default (`bf16` gain was `≤ 2%`, below `5%` switch gate) | [#99](https://github.com/gityuko-hb/ayaka/pull/99), `docs/EXPERIMENTS.md` |
| **FP8 default & status in models and quantization** | Keep 16-bit default; document sliced FP8 as *"fits, but slow"* (`0.11–0.12×` bf16 on L4, below `0.5×` gate) until the fast quantized kernels milestone lands | [#121](https://github.com/gityuko-hb/ayaka/pull/121), `docs/EXPERIMENTS.md`, `docs/TROUBLESHOOTING.md` |
| **Target hardware positioning** | 3-tier ladder: 4–8 GB (floor/offload), 12–24 GB (primary coding-agent sweet spot), 32–64 GB+ (Frontier MoE) | Strategy rebuild (`2026-11-25`) |
| **Serving Core v2 placement** | Serving Core v2 is the first milestone of the `2026-11-27` order (`v0.2.0-a1`), starting on `main`: pure Python 3.13 Render Engine, request-scoped output interpretation (`none` & `thinking`), multi-family reasoning/XML/JSON tool parsers, Structured Output v2 (`llguidance` + optional `--extra xgrammar`, 6 `ConstraintKind`, and the `validate_tokens` / `rollback(k)` grammar contract), and dual gRPC (`Inference` + engine-free `Render`) | Serving & XGrammar research (`2026-11-26`), [M05](M05-serving-v2.md) |
| **Guided speculation split (`2026-11-27`)** | Serving Core v2 delivers the grammar contract (`validate_tokens`, `rollback(k)`) with its own tests. Wiring it into speculative verify windows, with exact-output equivalence tests, moves to the speculative decoding milestone | [M05](M05-serving-v2.md), [M04](M04-speculative.md) |
| **`tool_choice="none"` prompt layout (`2026-11-27`)** | Keep tools in the prompt to protect prefix-cache hits. The step (PR #134) moved from the speculative decoding milestone into Serving Core v2 step 13 | [M05](M05-serving-v2.md), [M04](M04-speculative.md) |
| **Fast quantized kernel path** | Dedicated milestone (FP8 GEMM + safetensors INT4/AWQ/GPTQ Marlin W4A16) via FlashInfer/`torch._scaled_mm`/Triton/Marlin; GGUF Q4_K_M execution deferred/opt-in. sm_80 has no FP8 tensor cores, so A100 runs FP8 checkpoints as weight-only W8A16; native FP8 GEMM is an sm_89+ path | Strategy rebuild (`2026-11-25`), [M07](M07-fast-quant.md) |
| **PyPI public launch timing (`0.2.0`)** | Keep `0.1.0` as preview; internal tags `0.2.0-a1`..`a7` for the seven milestones after models and quantization, in the round order; first public PyPI launch **`0.2.0` GA** once frontier MoE completes | Roadmap expansion (`2026-11-25/26`), milestone order (`2026-11-27`) |
| **Post-`0.2.0` long-horizon ladder** | `0.3.0` = scale-out + native kernel engine; `0.4.0` = multi-agent & distributed prefix fabric; `1.0.0` GA = omni-modal computer-use & stable ABI | Roadmap expansion (`2026-11-25/26`) |
| **Early native C++/CUDA spike** | Run Binding ADR + 2-kernel CUDA graph spike (native kernel engine step 0) immediately after the fast quantized kernels milestone (`0.2.0-a3`), in parallel with hybrid and MoE work | Roadmap expansion (`2026-11-25/26`), milestone order (`2026-11-27`) |
| **Mid-term ordering (Hybrid/Vision vs MoE offload)** | Hybrid, state and vision before MoE on small machines, with the fast quantized kernels milestone already unlocking FP8/W4A16 for MoE | Roadmap rebuild (`2026-11-25`) |
| **Agent platform scoping** | Split into Tier 1 Core (steps 1 to 5) and Tier 2 Extended/Gated (steps 6 and 7) | Roadmap rebuild (`2026-11-25`) |
| **`cuda_graph=True` default on CUDA** | Approved for dedicated default-change PR speculative decoding step 4 (`2.13×–9.09×` over eager on L4, above `1.5×` gate; confirm 4 GB reconciliation) | [#99](https://github.com/gityuko-hb/ayaka/pull/99), [M04](M04-speculative.md) |

---

## Open Decisions (Gated by Future Milestones)

| Milestone | Open Decision | Pre-Registered Gate / Criteria |
| --- | --- | --- |
| **Speculative decoding** | Enable lookup speculation by default? | Keep opt-in until `≥ 1.15×` on code-edit prompts AND `≥ 0.98×` at concurrency `≥ 4` on L4 |
| **Speculative decoding** | Cost-aware controller vs fixed `k`; candidate-reuse module? | Decided from PR #131 multi-prompt & multi-concurrency measurements, repeated after the fast quantized kernels and memory milestones |
| **Serving Core v2** | Default structured-output backend on Linux CUDA (`llguidance` vs `xgrammar`) | Keep `llguidance` default unless `xgrammar` (`--extra xgrammar`) wins `≥ 15%` on CUDA-graph mask / constrained throughput (or uniquely supports the `rollback(k)` contract) with no cold-compile regression (Serving Core v2 step 12) |
| **Serving Core v2** | Should Render/Parsers ever port to Rust/PyO3? | Keep pure Python 3.13 unless concurrent serving benchmarks show CPU parse/render `> 5%` of GPU step time |
| **Serving Core v2** | gRPC transport overhead (`GrpcEngineClient` vs in-process) | Must stay `≤ 0.3 ms` p50/token added latency and `≤ 2%` throughput delta on L4 (Serving Core v2 step 12) |
| **Memory** | Exact wire error codes (`out_of_memory`, `preempted_timeout`, `FinishEvent.limit="memory"`) | Finalized in memory step 2 RFC / PR against OpenAI & Anthropic wire compatibility |
| **Memory** | Host KV tier on by default & default `ram_keep`? | Opt-in until WSL2/Linux pinned-RAM ceiling measured, promotion `< 0.5×` cold prefill (`≥ 4K`), and write-behind overhead `≤ 2%` |
| **Memory** | Time-budgeted chunking & overlapped scheduling? | Decided from memory replay/concurrency measurements |
| **Fast quantized kernels** | GGUF `Q4_K_M` direct execution timing (after safetensors AWQ/GPTQ Marlin in fast quantized kernels step 4) | Evaluated as opt-in follow-up if user demand exceeds safetensors INT4 availability |
| **Hybrid, state and vision** | State checkpoint density, KV FP8 default, vision lazy vs eager load, flagship hybrid model | Measured when the milestone opens (note: #99 already showed FlashInfer 0.7.4 FP8 KV is `1.28–1.50×` slower at batch 1) |
| **MoE on small machines** | WSL2 large expert pinning limit, profile+decay vs LRU, dynamic expert/KV VRAM split, MTP on 4 GB | Measured in MoE on small machines steps 1, 5 and 6 |
| **Agent platform** | MCP on by default vs `--mcp`; session disk persistence policy; Tier 2 (steps 6 and 7) go/no-go gates | Security review + value/overhead measurement gates in the agent platform milestone |
| **Frontier MoE** | RAM vs SSD crossover, QSA vs DSA order, Engram VRAM vs RAM, FP4 quality, DFlash vs DSpark on MoE | Measured across frontier MoE steps 1 to 5 |
| **Scale-out** | TP vs PP first on consumer dual-GPU; P/D payoff on 2 GPUs; Vulkan/Metal hardware availability | Measured in scale-out steps 2, 3 and 6 |
| **Native kernel engine** | C++/CUDA binding library, minimum CUDA version, target `sm_` list, full `csrc/` trigger date | Settled in the early spike (native kernel engine step 0) right after the fast quantized kernels milestone (`0.2.0-a3`) |
| **Multi-agent fabric** | Local vs networked swarm prefix fabric protocol; persistent KV encryption & eviction defaults | Measured in multi-agent fabric steps 1, 4 and 5 |
| **Omni-modal** | Video/screen frame-diffing architecture & public ABI deprecation window | Decided in omni-modal steps 1 and 3 RFCs |
