# M9 — MoE on Small Machines (`v0.2.0-a6`)

- **Goal:** Run MoE models larger than available VRAM (e.g., `Qwen3-30B-A3B` in M7 FP8/W4A16 on
  12–24 GB VRAM + 32–64 GB RAM, or `OLMoE-1B-7B` on 4 GB VRAM) with experts stored in host RAM and
  dynamically cached on GPU — measured head-to-head against **Strata** and **FreeToken** on the
  same card.
- **Status:** Planned (not filed yet)
- **Target Checkpoint:** `v0.2.0-a6`
- **Leverage from M7:** M3 (#121) showed `Qwen3-30B-A3B` in bf16 needed `61.06 GB` (`57.98 GB`
  experts), exceeding 56.9 GB host RAM. With M7 FP8 (`~30 GB`) and W4A16 (`~16 GB`), 30B-class MoE
  fits cleanly in standard host RAM.

## Exit Conditions

1. `offload` produces **identical token IDs** to `fused` on a tiny MoE and on `OLMoE-1B-7B` with a
   constrained GPU expert cache.
2. A real MoE model larger than VRAM runs end to end with an expert cache smaller than the total
   experts and passes `ayaka bench` sanity checks.
3. `ayaka bench` records tok/s, expert cache hit rate, and host-to-device bytes copied per token.
4. The analytical miss-cost model (`p/q`) matches measured latency across **3 distinct cache sizes**.
5. **Strata** and **FreeToken** are measured directly by Ayaka on the same machine, model, and
   recommended config and recorded in `docs/EXPERIMENTS.md`.

## Decisions to Measure

- Safe pinned-RAM ceiling on WSL2 vs native Linux for large expert stores.
- Hardware `p` (PCIe transfer cost) and `c` (compute cost) parameters.
- Whether two expert layers fit alongside KV on a 4 GB Tier 1 card.
- Profile-guided + decay eviction vs LRU; dynamic VRAM split between expert cache and KV pool.
- Whether MTP pays on 4–24 GB offloaded MoE; DFlash2 ordering.

---

## Execution Nodes (7 Nodes)

### `M9-N01` — PCIe Bandwidth, WSL2 Large-Pinning Probe & Target MoE Selection
- **Status:** `BLOCKED`
- **Depends On:** `M8-N08`
- **Hardware:** `GPU-L4` + `GPU-4GB-WSL2`
- **Touches:** `benchmarks/memory/`, `docs/EXPERIMENTS.md`, `docs/ROADMAP/decisions.md`
- **Deliverable:**
  - Measure sustained H2D expert-sized transfer bandwidth and maximum safe pinned host allocation on
    Linux L4 and WSL2 RTX 3050.
  - Lock target MoE checkpoints by host RAM + precision (e.g., `Qwen3-30B-A3B` FP8/W4A16 on L4;
    `OLMoE-1B-7B` W4A16/bf16 on 4 GB).
- **Exit Check:** Bandwidth & pinning report committed to `docs/EXPERIMENTS.md`.

### `M9-N02` — Host Expert Store (3 Residency Modes)
- **Status:** `BLOCKED`
- **Depends On:** `M9-N01`
- **Hardware:** `CPU` + `GPU-L4`
- **Touches:** `python/ayaka/memory/`, `python/ayaka/weights/`, `python/ayaka/layers/moe.py`, `tests/layers/`
- **Deliverable:**
  - Host expert store supporting 3 modes: `fused` (all experts resident on GPU), `offload`
    (pinned/pageable host RAM -> GPU expert cache), and streaming/fallback mode for quantized &
    16-bit experts.
- **Exit Check:** Unit tests for weight layout, memory accounting, and mode selection pass.

### `M9-N03` — Dynamic GPU Expert Cache & Offload Decode Parity
- **Status:** `BLOCKED`
- **Depends On:** `M9-N02`
- **Hardware:** `GPU-L4` + `GPU-4GB-WSL2`
- **Touches:** `python/ayaka/layers/moe.py`, `python/ayaka/execution/`, `python/ayaka/ops/`, `tests/e2e/`
- **Deliverable:**
  - GPU expert slot cache with miss-fetch path during decode; support both 16-bit and M7 FP8/W4A16
    expert tensors.
- **Exit Check:** `offload` emits **identical token IDs** to `fused` on tiny MoE and `OLMoE-1B-7B`
  across multiple cache capacities.

### `M9-N04` — Prefill Staging Buffers & Overlapped H2D Expert Copies
- **Status:** `BLOCKED`
- **Depends On:** `M9-N03`
- **Hardware:** `GPU-L4`
- **Touches:** `python/ayaka/layers/moe.py`, `python/ayaka/execution/`, `tests/e2e/`
- **Deliverable:**
  - Double-buffered copy stream overlapping next-layer expert H2D prefetch with current-layer
    compute during chunked prefill and decode (evaluating whether `M13-N00` spike insights suggest
    a custom DMA kernel later).
- **Exit Check:** Exact token parity preserved; measured overlap speedup over synchronous copy
  recorded on L4.

### `M9-N05` — Routing Trace, Profile+Decay Eviction, `p/q` Miss-Cost Model & `ayaka plan`
- **Status:** `BLOCKED`
- **Depends On:** `M9-N04`
- **Hardware:** `GPU-L4`
- **Touches:** `python/ayaka/layers/moe.py`, `python/ayaka/metrics/`, `python/ayaka/entrypoints/plan.py`, `python/ayaka/entrypoints/bench.py`, `benchmarks/e2e/`
- **Deliverable:**
  - Routing trace recorder, profile-guided + decay eviction compared against plain LRU, analytical
    `p/q` miss-cost model validated at **3 distinct cache sizes**, and `ayaka plan` / `ayaka bench`
    reporting expert cache hit rate and bytes copied per token.
- **Exit Check:** Predicted vs measured latency across 3 cache sizes documented; `ayaka plan` prints
  accurate VRAM/RAM split and cache hit-rate estimates.

### `M9-N06` — Feature Sink & MTP Drafting over Offloaded MoE
- **Status:** `BLOCKED`
- **Depends On:** `M9-N04`, `M4-N02`
- **Hardware:** `GPU-L4` + `GPU-4GB-WSL2`
- **Touches:** `python/ayaka/speculative/`, `python/ayaka/layers/moe.py`, `benchmarks/e2e/`
- **Deliverable:**
  - Wire M4's hidden-state sink + MTP drafting over offloaded MoE and measure whether amortizing
    expert PCIe transfers across multi-token verify windows wins on 4–24 GB.
- **Exit Check:** Exact token parity verified; payoff vs memory cost recorded in
  `docs/EXPERIMENTS.md`.

### `M9-N07` — Head-to-Head Strata & FreeToken Comparison + `v0.2.0-a6`
- **Status:** `BLOCKED`
- **Depends On:** `M9-N05`, `M9-N06`
- **Hardware:** `GPU-L4`
- **Touches:** `benchmarks/e2e/`, `docs/EXPERIMENTS.md`, `docs/capabilities.toml`, `docs/CAPABILITIES.md`
- **Deliverable:**
  - Run Ayaka, **Strata**, and **FreeToken** side-by-side on the same machine, model, and
    recommended config (alternating rounds), update `docs/CAPABILITIES.md`, and tag `v0.2.0-a6`.
- **Exit Check:** All 5 M9 exit conditions green; raw results committed; tag `v0.2.0-a6`.

## Risks

- PCIe Gen3/Gen4 and host-RAM bandwidth caps limiting throughput on cold/diverse routing.
- Expert cache thrashing under multi-stream coding sessions.
- Quantized expert dequantization + offload introducing subtle numerical drift.
