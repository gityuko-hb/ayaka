# M11 — Frontier MoE & Public PyPI `0.2.0` GA

- **Goal:** Run frontier-class sparse MoE models (`Qwen3.8-Flash-Next`, `DeepSeek-V4.1-Flash`,
  `GLM-5.3-Flash`, `MiniMax-M2.5` class) on **8–64 GB** GPUs via sparse attention, RAM/NVMe SSD
  expert hierarchy, FP4 quantization, out-of-VRAM n-gram/Engram tables, and Frontier MoE
  block/suffix drafting — culminating in the **first public PyPI launch: `ayaka-llm 0.2.0`**.
- **Status:** Planned (not filed yet)
- **Target Release:** **`ayaka-llm 0.2.0` (Public PyPI GA)**
- **Representative Architecture Targets** *(exact checkpoints pinned when M11 opens)*:
  `Qwen3.8-Flash-Next`, `DeepSeek-V4.1-Flash`, `GLM-5.3-Flash`, `MiniMax-M2.5` class.

## Exit Conditions

1. Tiny-model parity passes for each representative frontier MoE architecture, plus at least one
   real frontier checkpoint running end to end with expert offload on 8–64 GB.
2. DeepSeek-class model (MLA + sparse attention + Engram + FP8/FP4) matches on a tiny reference and
   completes a real long-context run.
3. GLM-class model (KDA + sparse attention + MLA + mHC) matches on a tiny reference.
4. Sparse attention stays inside pre-registered Needle-in-a-Haystack / RULER quality thresholds.
5. Out-of-VRAM n-gram / Engram lookup tables retrieve identically from RAM without corruption.
6. Hierarchical 3-tier (`VRAM -> RAM -> SSD`) experts emit identical tokens to `fused` on a tiny
   model, with hit rate and throughput measured on a real model.
7. DFlash / DSpark on frontier/offloaded MoE preserves exact tokens and speeds up at least one
   prompt class.
8. `ayaka plan` estimates VRAM, RAM, NVMe SSD, and expected tok/s; **`ayaka-llm 0.2.0`** passes
   clean-environment build/install verification and is published to PyPI with measured release
   notes.

---

## Execution Nodes (8 Nodes)

### `M11-N01` — Sparse Attention Index & QSA / DSA Kernels
- **Status:** `BLOCKED`
- **Depends On:** `M9-N07`, `M8-N02`
- **Hardware:** `GPU-L4`
- **Touches:** `python/ayaka/kvcache/`, `python/ayaka/ops/`, `python/ayaka/ops/backends/`, `benchmarks/e2e/`
- **Deliverable:**
  - Sparse KV indexing structures and QSA / DSA attention ops (`torch` ref + GPU kernel), deciding
    QSA vs DSA implementation order via pre-registered Needle/RULER benchmarks.
- **Exit Check:** Tiny parity green; long-context Needle-in-a-Haystack & RULER scores clear
  pre-registered thresholds in `docs/EXPERIMENTS.md`.

### `M11-N02` — Out-of-VRAM N-gram & Engram Lookup Tables
- **Status:** `BLOCKED`
- **Depends On:** `M11-N01`
- **Hardware:** `GPU-L4`
- **Touches:** `python/ayaka/layers/`, `python/ayaka/memory/`, `tests/layers/`
- **Deliverable:**
  - Tiered VRAM / host-RAM lookup tables for n-gram and Engram conditional embeddings when tables
    exceed VRAM, with latency measurements for VRAM vs RAM placement.
- **Exit Check:** Exact lookup equivalence when table resides in RAM vs VRAM; zero corruption under
  concurrent batches.

### `M11-N03` — Hierarchical 3-Tier (`VRAM -> Pinned RAM -> NVMe SSD`) Expert Store
- **Status:** `BLOCKED`
- **Depends On:** `M9-N05`
- **Hardware:** `GPU-L4` + `NVMe SSD`
- **Touches:** `python/ayaka/memory/`, `python/ayaka/layers/moe.py`, `python/ayaka/weights/`, `benchmarks/e2e/`
- **Deliverable:**
  - Extend M9's host expert store with a direct-I/O / async NVMe SSD cold tier (`VRAM cache ->
    pinned RAM -> NVMe SSD`), measuring RAM-vs-SSD crossover latency, hit rates, and SSD read
    amplification.
- **Exit Check:** Hierarchical SSD experts produce **identical tokens** to `fused` on a tiny MoE;
  real-model hit rate and tok/s recorded in `docs/EXPERIMENTS.md`.

### `M11-N04` — MXFP4 / NVFP4 Weight & FP4 Expert Execution
- **Status:** `BLOCKED`
- **Depends On:** `M7-N04`
- **Hardware:** `GPU-L4` / Blackwell (`sm_100`/`sm_120`)
- **Touches:** `python/ayaka/layers/quantization/`, `python/ayaka/ops/`, `python/ayaka/weights/`, `benchmarks/e2e/`
- **Deliverable:**
  - MXFP4 and NVFP4 quantization contracts, loader, and dense/MoE expert GEMM kernels (native FP4
    on Blackwell where available + dequant fallback on `sm_86`/`sm_89`).
- **Exit Check:** First-token KL vs 16-bit/FP8 and tool-call JSON sanity gates pass; recorded in
  `docs/EXPERIMENTS.md`.

### `M11-N05` — DFlash / DSpark & Suffix Drafting for Frontier / Offloaded MoE
- **Status:** `BLOCKED`
- **Depends On:** `M4-N03`, `M11-N03`
- **Hardware:** `GPU-L4`
- **Touches:** `python/ayaka/speculative/`, `benchmarks/e2e/`
- **Deliverable:**
  - Extend M4's DFlash block drafter to Frontier & offloaded MoE architectures, plus evaluate
    DSpark and suffix-tree combinations to amortize expert offload fetches.
- **Exit Check:** Exact token parity preserved; measurable speedup on at least one prompt class
  recorded in `docs/EXPERIMENTS.md`.

### `M11-N06` — Representative Frontier MoE Model Families
- **Status:** `BLOCKED`
- **Depends On:** `M11-N01`, `M11-N02`, `M11-N03`, `M11-N04`
- **Hardware:** `GPU 8–64 GB`
- **Touches:** `python/ayaka/models/`, `python/ayaka/weights/`, `tests/e2e/`
- **Deliverable:**
  - Architecture implementations and tiny-reference parity suites for `Qwen3.8-Flash-Next`,
    `DeepSeek-V4.1-Flash` (MLA + sparse + Engram + FP8/FP4), `GLM-5.3-Flash` (KDA + sparse + MLA +
    mHC), and `MiniMax-M2.5` class, plus real offloaded long-context runs on 8–64 GB GPUs.
- **Exit Check:** All 4 tiny-model parity suites green; at least one real frontier checkpoint runs
  end to end with offload and passing sanity checks.

### `M11-N07` — Tiered `ayaka plan` (VRAM + RAM + SSD + tok/s) & Frontier E2E Bench
- **Status:** `BLOCKED`
- **Depends On:** `M11-N06`
- **Hardware:** `GPU 8–64 GB`
- **Touches:** `python/ayaka/entrypoints/plan.py`, `python/ayaka/entrypoints/bench.py`, `benchmarks/e2e/`, `docs/EXPERIMENTS.md`, `docs/capabilities.toml`, `docs/CAPABILITIES.md`
- **Deliverable:**
  - `ayaka plan` outputs full 3-tier memory budget (`VRAM`, `RAM`, `SSD`) and predicted tok/s
    estimates; refreshed benchmark suite against `llama.cpp`, Strata, and FreeToken.
- **Exit Check:** `ayaka plan` estimates match observed startup/offload telemetry; all frontier
  numbers committed with raw `results.json`.

### `M11-N08` — Public PyPI Release Gate: `ayaka-llm 0.2.0` GA
- **Status:** `BLOCKED`
- **Depends On:** `M11-N05`, `M11-N07`, `M10-N05`
- **Hardware:** `CPU` + `GPU-L4` + `GPU-4GB-WSL2`
- **Touches:** `python/ayaka/__version__.py`, `.github/ISSUE_TEMPLATE/`, `README.md`, `docs/INSTALL.md`, `docs/TROUBLESHOOTING.md`, `docs/CAPABILITIES.md`, `CHANGELOG.md`
- **Deliverable:**
  - Add community benchmark issue form, reproducible test-request template, and intake workflow.
  - Bump version to `0.2.0`, verify clean-environment install of built `sdist` & `wheel`
    (`uv build`) with CPU and CUDA extras, audit every public doc (`tests/test_docs.py` green).
  - Publish **`ayaka-llm 0.2.0`** to PyPI and GitHub Releases with measured release notes naming
    machine, model, and version.
- **Exit Check:** `pip`/`uv add ayaka-llm==0.2.0` installs and passes `ayaka info`, `ayaka plan`,
  `ayaka serve`, and `ayaka connect --check` cleanly; all M4–M11 exit conditions verified.

## Risks

- Sparse attention silently dropping essential symbol definitions in long coding contexts.
- NVMe SSD wear and tail-latency spikes under poor expert locality.
- FP4 quantization degrading strict JSON/tool-call schema adherence.
- Complexity of coordinating block speculation over 3-tier offloaded experts.
