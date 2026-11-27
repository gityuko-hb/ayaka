# Speculative decoding (`v0.2.0-a4`)

- **Goal:** Speed up memory-bandwidth-bound low-batch coding-agent decoding (copying, editing,
  structured generation) without changing a single output token.
- **Status:** In progress (PRs #131, #132, #133 and #135 open; issues #127–#129)
- **Order:** Fourth milestone. It starts after the fast quantized kernels exit audit (fast quantized kernels step 5). The open PRs keep their current code; their L4 numbers are measured again after that audit.
- **Hardware this round:** NVIDIA A100-SXM4-80GB (sm_80) is the development GPU. Gates written for the NVIDIA L4 stay unverified until they run on an NVIDIA L4; no L4 result is claimed in this file.
- **Reference models this round:** `Qwen/Qwen3-8B` (bf16), with `Qwen/Qwen3-8B-FP8` and `Qwen/Qwen3-8B-AWQ` for quantized kernels and `z-lab/Qwen3-8B-DFlash-b16` for speculative decoding; `Qwen/Qwen3-0.6B` for fast tests.
- **Tracking:** [#127](https://github.com/gityuko-hb/ayaka/issues/127),
  [#128](https://github.com/gityuko-hb/ayaka/issues/128),
  [#129](https://github.com/gityuko-hb/ayaka/issues/129)
- **Target Checkpoint:** `v0.2.0-a4` (internal tag on the road to `0.2.0` GA)

## Exit Conditions

1. Greedy decoding with and without speculation emits **identical token IDs** across all window
   sizes and ragged batches.
2. Position-keyed sampled decoding (`temperature > 0`) obeys the same exact-output test for
   one-point acceptance; a lookup draft is never rejected for lacking a candidate distribution.
3. An adversarial drafter (random, always wrong, always right, out of vocabulary) cannot alter
   outputs or fail a request.
4. Lookup, MTP (`MiMo-7B-RL`), and DFlash (`Qwen3-8B`, `Llama-3.1-8B`) are opt-in, accounted in
   `ayaka plan`, and benchmarked on an NVIDIA L4 across **4 prompt classes** and **multiple
   concurrency levels**, with results committed to `docs/EXPERIMENTS.md`.
5. `cuda_graph=True` is enabled by default on CUDA (speculative decoding step 4). The tool-choice prompt layout (PR #134) is in Serving Core v2 step 13.
6. **Guided speculation** (moved here from Serving Core v2): the grammar contract (`validate_tokens`, `rollback(k)`) is wired into the verify windows, so constrained requests keep speculation on. Greedy and position-keyed sampled outputs match non-speculative constrained decoding, including adversarial drafts.

## Pre-Registered Measurement Gates

- **Lookup default policy:** Stays opt-in in #131. Promoting lookup to default requires a separate
  PR passing BOTH `≥ 1.15×` tok/s on code-edit prompts (batch 1, L4) AND `≥ 0.98×` (no regression)
  at concurrency `≥ 4`.
- **MTP & DFlash:** Always opt-in; must report net tok/s **alongside** displaced KV tokens on L4.

---

## Execution Nodes (6 Nodes)

### speculative decoding step 1 — Lookup Drafting, Verified CUDA Graphs & Cost-Aware Controller
- **Status:** `OPEN_PR` ([#131](https://github.com/gityuko-hb/ayaka/pull/131), branch `feat/speculative-foundation`)
- **Depends On:** fast quantized kernels step 5 (measurements are taken again after it)
- **Hardware:** `CPU` + `GPU-L4`
- **Touches:** `python/ayaka/speculative/`, `python/ayaka/configs/`, `python/ayaka/execution/graph.py`, `python/ayaka/execution/runner.py`, `python/ayaka/scheduler/`, `python/ayaka/entrypoints/`, `python/ayaka/metrics/`, `tests/speculative/`, `benchmarks/e2e/`
- **Deliverable:**
  - `ayaka.speculative` package with `SpeculativeConfig`, proposer contract, reference exact-output
    acceptance rules, n-gram lookup drafter, and cost-aware dynamic window controller.
  - CUDA-graph capture for multi-row verify windows (`GraphRunner.bucket_for` support).
  - OpenAI `prediction` parameter & accepted-draft `Usage`/Prometheus metrics.
  - L4 benchmark across 4 prompt classes (code-edit, prose, JSON/tools, random) & concurrency levels.
- **Exit Check:** Unit/adversarial/greedy+sampled equivalence tests green; L4 benchmark JSON and
  analysis recorded in `docs/EXPERIMENTS.md`; PR #131 merged with 9/9 CI.

### speculative decoding step 2 — Hidden-State Feature Sink & MiMo-7B MTP Drafter
- **Status:** `OPEN_PR` ([#132](https://github.com/gityuko-hb/ayaka/pull/132), branch `feat/speculative-mtp`)
- **Depends On:** speculative decoding step 1
- **Hardware:** `CPU` + `GPU-L4`
- **Touches:** `python/ayaka/speculative/`, `python/ayaka/models/mimo.py`, `python/ayaka/execution/`, `python/ayaka/memory/`, `tests/speculative/`, `benchmarks/e2e/`
- **Deliverable:**
  - Reusable `ForwardContext` hidden-state feature sink writing into fixed buffers inside captured
    CUDA graphs.
  - MiMo-7B-RL native MTP layer loader (`model.mtp_layers.0.*`), argmax drafting, KV layer 36
    accounting and weight sizing in `ayaka plan`.
  - L4 speed & acceptance benchmarks vs plain decoding and lookup.
- **Exit Check:** Exact greedy/one-point parity on `XiaomiMiMo/MiMo-7B-RL`; `ayaka plan` matches
  startup reconciliation; L4 results committed to `docs/EXPERIMENTS.md`; PR #132 merged.

### speculative decoding step 3 — Multi-Layer Tap Sink & DFlash Block Drafter
- **Status:** `OPEN_PR` ([#133](https://github.com/gityuko-hb/ayaka/pull/133), branch `feat/speculative-dflash`)
- **Depends On:** speculative decoding step 2
- **Hardware:** `CPU` + `GPU-L4`
- **Touches:** `python/ayaka/speculative/`, `python/ayaka/models/`, `python/ayaka/memory/`, `tests/speculative/`, `benchmarks/e2e/`
- **Deliverable:**
  - Multi-layer hidden-state tap sink (`target_layer_ids`), non-causal DFlash block drafter
    (`z-lab/Qwen3-8B-DFlash-b16`, `z-lab/LLaMA3.1-8B-Instruct-DFlash-UltraChat`) sharing target
    embeddings and `lm_head`.
  - Honest displaced-KV VRAM accounting in `ayaka plan`.
  - L4 benchmark vs plain decoding, lookup, and other engines with concurrency sweep.
- **Exit Check:** Exact greedy/one-point parity on Qwen3-8B & Llama-3.1-8B; displaced KV reported
  next to speedup in `docs/EXPERIMENTS.md`; PR #133 merged.

### speculative decoding step 4 — Enable `cuda_graph=True` by Default on CUDA
- **Status:** `BLOCKED` (waiting on speculative decoding step 1 graph bucket stabilization)
- **Depends On:** speculative decoding step 1
- **Hardware:** `GPU-L4` + `GPU-4GB-WSL2`
- **Touches:** `python/ayaka/configs/`, `python/ayaka/entrypoints/cli.py`, `docs/INSTALL.md`, `docs/AI_REFERENCE.md`, `tests/`
- **Deliverable:**
  - Dedicated default-change PR flipping `cuda_graph=True` on CUDA (backed by GPU execution milestone #99 L4 gate:
    `2.13×–9.09×` speedup over eager, threshold `≥ 1.5×`).
  - Explicit verification on both 24 GB L4 and 4 GB RTX 3050 (default buckets `1, 2, 4, 8`) so
    startup memory reconciliation stays green.
- **Exit Check:** `uv run pytest -m "not gpu and not integration"` + GPU startup reconciliation green
  on L4 and RTX 3050; `tests/test_docs.py` green.

### speculative decoding step 5 — Moved to Serving Core v2 step 13

The tool-choice prompt step (PR #134) moved to Serving Core v2 on 2026-11-27. Step 5 stays empty so that the step numbers used in other files do not change.

### speculative decoding step 6 — Exit Audit & `v0.2.0-a4` Checkpoint
- **Status:** `BLOCKED`
- **Depends On:** speculative decoding steps 1 to 4, fast quantized kernels step 5
- **Hardware:** `GPU-L4`
- **Touches:** `docs/EXPERIMENTS.md`, `docs/capabilities.toml`, `docs/CAPABILITIES.md`, `docs/ROADMAP/decisions.md`
- **Deliverable:**
  - Record final lookup default decision against the pre-registered `≥ 1.15×` / `≥ 0.98×` gate.
  - Regenerate `docs/CAPABILITIES.md` (`tools/generate_capabilities.py`), close #127–#129, and tag
    internal checkpoint `v0.2.0-a4`.
- **Exit Check:** All speculative decoding exit conditions verified; `tests/test_docs.py` & CI green; tag `v0.2.0-a4`
  recorded.

## Risks

- Subtle interactions between speculative verification, repetition/frequency penalties, and
  llguidance grammar masks.
- Verify-window CUDA graphs inflating startup VRAM on Tier 1 (4 GB) cards.
- Speculation regressing throughput under concurrency > 1 if controller backoff lags.
- Speculative numbers measured before the fast quantized kernels and the memory milestone land would have to be measured again, because both change the per-token decode cost and the KV blocks that drafts take.
