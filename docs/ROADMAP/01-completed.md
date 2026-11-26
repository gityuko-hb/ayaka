# Completed Milestones (M0 – M3)

Record of completed milestones, their verified exit conditions, key measured findings, and tracking
issues/PRs. Every number below comes from committed results in `docs/EXPERIMENTS.md`,
`docs/CAPABILITIES.md`, and closing comments on GitHub.

---

## M0 — Foundation (CPU, torch backend)

- **Status:** Done ([#10](https://github.com/gityuko-hb/ayaka/issues/10))
- **Goal:** A `GenerateRequest` flows through every engine layer to `RequestOutput` on CPU, with
  both a tiny test model and real GPT-2.
- **Verified Exit Conditions:**
  - E2E generation and determinism suites green on CPU CI.
  - GPT-2 greedy output matches `transformers`.
  - Layer/import-boundary tests green (`import ayaka` loads no `torch`).
  - `ayaka run` prints generated text.
- **Delivered:**
  - `ayaka.ops` contracts and `torch` reference backend; tensor specs, safetensors loader, weight
    mapping; model/device/cache/scheduler/engine configs; layers; paged KV spec/pool/table/manager;
    model registry & GPT-2; position-keyed counter RNG & sampler; scheduler & queue; input builder &
    model runner; engine core & synchronous facade; `ayaka run`; CPU CI.
  - Step contracts (`rows_per_request`, deferred release, multi-stream `DrawStream`) designed upfront
    for speculative verification in M4.

---

## M1 — GPU Execution

- **Status:** Done ([#26](https://github.com/gityuko-hb/ayaka/issues/26), PRs #28–#37; L4 decision
  gates recorded in [#99](https://github.com/gityuko-hb/ayaka/pull/99))
- **Goal:** Run the end-to-end path on CUDA through FlashInfer, decode through CUDA graphs, and add
  Llama as the second architecture.
- **Verified Exit Conditions:**
  - GPU e2e and device-sync-guard suites green on development GPUs (RTX 3050 4 GB WSL2 and NVIDIA L4
    24 GB Linux).
  - Op parity for all FlashInfer-backed families; FlashInfer `3.6×–6.9×` faster than `torch`
    attention across decode and prefill shapes on L4 (#99).
  - Small & real Llama checkpoints match `transformers` greedy output.
  - Batch-1 decode runs through CUDA graphs (`2.13×` speedup on Llama-3.2-1B, `9.09×` on SmolLM2-135M
    on L4, #99) with graph == eager equivalence.
  - Memory plan matches allocator ledger after startup (`0`-byte diff on native Linux L4, #99).
- **Key Findings Recorded (#99):**
  - FlashInfer 0.7.4 sampling does not accept external uniform draws `u` -> sampling stays on
    `torch`.
  - `bf16` vs `fp16` delta is within `2%` (below `5%` switch gate) -> default remains `fp16`.
  - FlashInfer 0.7.4 FP8 E4M3 KV runs accurately (`3.8–4.0%` rel L2 vs fp64) but is `1.28–1.50×`
    slower than fp16 KV at batch 1 up to 8K context on `sm_89` -> deferred to M7 as opt-in capacity
    mode.

---

## M2 — Serving (`0.1.0` preview)

- **Status:** Done (core #43, #50, #52–#60, #73; hardening #76–#78; follow-ups #74, #102–#108; PRs
  #44–#49, #63–#72, #75, #79–#110)
- **Goal:** A developer points a coding tool at Ayaka with two commands (`ayaka serve` +
  `ayaka connect`).
- **Verified Exit Conditions:**
  - Official OpenAI (Chat Completions, Completions, Responses) and Anthropic (Messages) SDK clients
    answered with streaming and tool calls (CI `serving` job + real GPU runs).
  - `ayaka connect <tool> --check` passes **7/7** for Claude Code, Codex CLI, and OpenCode on
    Llama-3.2-1B (#56) and Llama-3.1-8B with `llama3_json` (#101, #104, #109, #110).
  - Real tool round-trip replay fixtures committed for Claude Code (#101), Codex CLI, and OpenCode
    (#104); optional `llama3_json` argument schema enforcement reduced Codex string-typed errors
    from 16/18 to 0/21 and lifted OpenCode schema-valid calls from 6/19 to 21/21 (#110).
  - Concurrent requests finish with `cached_tokens > 0` (#52); Claude Code billing-header
    normalization lifts cross-session cache hits from `0.2%` to `84.5%` (#57).
  - `ayaka plan` matches real startup exactly on native Linux L4 (2,397 usable blocks / 38,352
    tokens cold and warm for Llama-3.1-8B, #100/#105). *Known caveat*: WSL2 WDDM cold-start gap
    documented in `docs/TROUBLESHOOTING.md` (to be addressed cleanly in M5 `doctor`/policy).
  - `ayaka bench` writes valid `results.json` with passing JSON + forced-tool sanity checks
    (SentencePiece grammar bug on TinyLlama fixed in #106).
  - Constrained decode runs inside CUDA graphs with `0%–8%` overhead vs free graphs (`1.4×–6.8×`
    faster than eager control, #73/#79).
  - Serving hardening completed across 4 waves (#76–#78, PRs #80–#98): golden-byte SSE/error locks,
    split-invariant stop matching, single wire policy table, `<700`-line entrypoint cap, frozen
    `EngineClient` seam, `/health` stall reporting, bounded shutdown drain, startup GC freeze,
    single-validated admission, bounded detokenizer seed, stream watchdog, and single preprocess
    render thread.
  - Release `0.1.0` prepared as distribution `ayaka-llm` (#102/#107; preview only, not uploaded to
    PyPI).
- **Recorded Baselines vs llama.cpp:**
  - RTX 3050 4 GB (Llama-3.2-1B fp16, #59): 4K decode `51.0` vs `58.7 tok/s` (`0.87×`); 1K prefill
    `0.97×`; 4K prefill `0.71×`.
  - NVIDIA L4 24 GB (Llama-3.1-8B fp16, #105): 4K decode `15.05` vs `16.15 tok/s` (`0.93×`); 1K
    prefill `1.27×`; 4K prefill `1.02×`; cold 4K TTFT `0.98×`.

---

## M3 — Models and Quantization

- **Status:** Done (`2026-11-24`: [#111](https://github.com/gityuko-hb/ayaka/issues/111),
  [#112](https://github.com/gityuko-hb/ayaka/issues/112),
  [#113](https://github.com/gityuko-hb/ayaka/issues/113); PRs #114–#121, #130; follow-up fixes
  #124–#126)
- **Goal:** Support common dense families, load 8-bit weights through a quantization contract, size
  mixed-quantization GGUF files in `ayaka plan`, and run the first MoE model `fused`.
- **Verified Exit Conditions (measured on NVIDIA L4 24 GB, #121):**
  - **Dense parity**: Qwen2/Qwen2.5 (#114), Qwen3 with per-head QK RMSNorm (#115), Phi-3/Phi-4 with
    attention backend fallback for unsupported head dims (#116), and MiMo-7B (`MiMoForCausalLM`,
    #130) all match `transformers` greedy outputs. Follow-up bugs surfaced during parity runs
    (#122/#124 chunked torch decode attention; #123/#125 allocator cache clear; #126 shared cuBLAS
    graph-warmup stream) were fixed on `main`.
  - **FP8 quality**: First-token KL vs 16-bit (`quant_kl.py`, 32 prompts, threshold `≤ 0.05 nats`):
    - `Meta-Llama-3.1-8B-Instruct-FP8`: mean `0.0026 nats`, `32/32` top-1 agreement.
    - `Qwen3-8B-FP8` (128×128 blocks): mean `0.0085 nats`, `32/32` top-1 agreement.
    - `phi-4-FP8-dynamic` (14B, does not fit in bf16 on 24 GB L4) runs end to end with passing
      sanity checks (20.7 GiB peak).
  - **FP8 speed finding**: Sliced dequantization yields `1.78 tok/s` vs `15.6 tok/s` (`0.11–0.12×`
    of bf16) on 8B and `0.9 tok/s` on Phi-4 14B — below the `0.5×` recommendation gate, so FP8 is
    honestly documented as *"fits, but slow"* in `docs/TROUBLESHOOTING.md`. This finding directly
    triggered adding **M6 (Fast Quantized Kernels)** before public PyPI `0.2.0`.
  - **GGUF plan**: `ayaka plan` computes exact `weights_bytes` (`4,912,898,304 B`) on mixed
    `Q4_K_M` GGUF headers (#118), cross-checked with `gguf 0.17.1`.
  - **First MoE (`fused`)**: MoE routing/grouped expert ops (#119) + OLMoE-1B-7B and Qwen3-MoE
    models (#120). OLMoE-1B-7B matches `transformers` and generates `41.5 tok/s` after 1K context
    with passing sanity checks. Sizing check confirmed `Qwen3-30B-A3B` needs `61.06 GB` in bf16
    (`57.98 GB` experts), exceeding 56.9 GB host RAM — validating the decision to ship M6
    (FP8/W4A16) before M8 (MoE offload).
