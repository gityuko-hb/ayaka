# M4 — Speculative Decoding (`v0.2.0-a1`)

- **Goal:** Speed up memory-bandwidth-bound low-batch coding-agent decoding (copying, editing,
  structured generation) without changing a single output token.
- **Status:** In Progress (PRs #131, #132, #133 open)
- **Tracking:** [#127](https://github.com/gityuko-hb/ayaka/issues/127),
  [#128](https://github.com/gityuko-hb/ayaka/issues/128),
  [#129](https://github.com/gityuko-hb/ayaka/issues/129)
- **Target Checkpoint:** `v0.2.0-a1` (internal tag on the road to `0.2.0` GA)

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
5. `cuda_graph=True` is enabled by default on CUDA (`M4-N04`) and `tool_choice="none"` preserves
   tools in the prompt (`M4-N05`).

## Pre-Registered Measurement Gates

- **Lookup default policy:** Stays opt-in in #131. Promoting lookup to default requires a separate
  PR passing BOTH `≥ 1.15×` tok/s on code-edit prompts (batch 1, L4) AND `≥ 0.98×` (no regression)
  at concurrency `≥ 4`.
- **MTP & DFlash:** Always opt-in; must report net tok/s **alongside** displaced KV tokens on L4.

---

## Execution Nodes (6 Nodes)

### `M4-N01` — Lookup Drafting, Verified CUDA Graphs & Cost-Aware Controller
- **Status:** `OPEN_PR` ([#131](https://github.com/gityuko-hb/ayaka/pull/131), branch `feat/speculative-foundation`)
- **Depends On:** M3 (`main` @ `3292d09`)
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

### `M4-N02` — Hidden-State Feature Sink & MiMo-7B MTP Drafter
- **Status:** `OPEN_PR` ([#132](https://github.com/gityuko-hb/ayaka/pull/132), branch `feat/speculative-mtp`)
- **Depends On:** `M4-N01`
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

### `M4-N03` — Multi-Layer Tap Sink & DFlash Block Drafter
- **Status:** `OPEN_PR` ([#133](https://github.com/gityuko-hb/ayaka/pull/133), branch `feat/speculative-dflash`)
- **Depends On:** `M4-N02`
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

### `M4-N04` — Enable `cuda_graph=True` by Default on CUDA
- **Status:** `BLOCKED` (waiting on `M4-N01` graph bucket stabilization)
- **Depends On:** `M4-N01`
- **Hardware:** `GPU-L4` + `GPU-4GB-WSL2`
- **Touches:** `python/ayaka/configs/`, `python/ayaka/entrypoints/cli.py`, `docs/INSTALL.md`, `docs/AI_REFERENCE.md`, `tests/`
- **Deliverable:**
  - Dedicated default-change PR flipping `cuda_graph=True` on CUDA (backed by M1 #99 L4 gate:
    `2.13×–9.09×` speedup over eager, threshold `≥ 1.5×`).
  - Explicit verification on both 24 GB L4 and 4 GB RTX 3050 (default buckets `1, 2, 4, 8`) so
    startup memory reconciliation stays green.
- **Exit Check:** `uv run pytest -m "not gpu and not integration"` + GPU startup reconciliation green
  on L4 and RTX 3050; `tests/test_docs.py` green.

### `M4-N05` — Preserve Tools in Prompt for `tool_choice="none"`
- **Status:** `READY`
- **Depends On:** None
- **Hardware:** `CPU`
- **Touches:** `python/ayaka/entrypoints/`, `python/ayaka/sampling/structured/`, `tests/entrypoints/`
- **Deliverable:**
  - Keep tool schemas rendered in the prompt when `tool_choice="none"` so switching tool choice
    mid-session preserves prefix-cache blocks, while disallowing tool-call emission cleanly.
  - Golden-byte wire test + prefix-cache hit unit test verifying `cached_tokens > 0` across
    `tool_choice="auto"` -> `"none"`.
- **Exit Check:** Golden-byte and prefix-cache unit tests pass; CI green.

### `M4-N06` — M4 Exit Audit & `v0.2.0-a1` Checkpoint
- **Status:** `BLOCKED`
- **Depends On:** `M4-N01`, `M4-N02`, `M4-N03`, `M4-N04`, `M4-N05`
- **Hardware:** `GPU-L4`
- **Touches:** `docs/EXPERIMENTS.md`, `docs/capabilities.toml`, `docs/CAPABILITIES.md`, `docs/ROADMAP/decisions.md`
- **Deliverable:**
  - Record final lookup default decision against the pre-registered `≥ 1.15×` / `≥ 0.98×` gate.
  - Regenerate `docs/CAPABILITIES.md` (`tools/generate_capabilities.py`), close #127–#129, and tag
    internal checkpoint `v0.2.0-a1`.
- **Exit Check:** All M4 exit conditions verified; `tests/test_docs.py` & CI green; tag `v0.2.0-a1`
  recorded.

## Risks

- Subtle interactions between speculative verification, repetition/frequency penalties, and
  llguidance grammar masks.
- Verify-window CUDA graphs inflating startup VRAM on Tier 1 (4 GB) cards.
- Speculation regressing throughput under concurrency > 1 if controller backoff lags.
