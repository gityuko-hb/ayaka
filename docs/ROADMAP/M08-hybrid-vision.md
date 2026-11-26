# M8 — Hybrid Dense, State, and Vision (`v0.2.0-a5`)

- **Goal:** Support hybrid linear-attention / recurrent-state architectures (GDN/KDA), Multi-Head
  Latent Attention (MLA), multimodal image inputs, and an opt-in quality-gated FP8 KV cache on
  4–24 GB GPUs.
- **Status:** Planned (not filed yet)
- **Target Checkpoint:** `v0.2.0-a5`
- **Representative Architecture Targets** *(exact Hugging Face checkpoints pinned when M8 opens)*:
  - Hybrid linear-attention + dense: `Qwen3.5-9B`, `Qwen3.8-27B` class (GDN/state layers).
  - Multimodal / sliding-window dense: `Gemma 4` class.
  - Attention-sink / open-weight coding dense/small-MoE: `gpt-oss` class.

## Exit Conditions

1. Representative hybrid models (`Qwen3.5`/`Qwen3.8` class) match `transformers` greedy outputs on
   the same weights and dtype.
2. One GDN prefill chunk produces the exact same tokens as multi-chunk prefill, and a **state
   prefix-cache hit** produces the exact same tokens as a cold run.
3. MLA matches reference tokens and reduces KV cache bytes by **≥ 50%** vs standard GQA/MHA.
4. Image preprocessing matches the reference model processor and a `1024×768` screenshot runs end
   to end through `ayaka serve` and `ayaka bench`.
5. Opt-in KV FP8 stays inside pre-registered first-token KL and long-context quality thresholds.
6. `ayaka plan` prints state, KV, and vision byte budgets; `ayaka bench` includes image and
   long-prompt groups.

## Decisions to Measure

- State checkpoint density (RAM overhead vs suffix recompute time on prefix hits).
- KV FP8 default policy (opt-in capacity mode unless a newer kernel eliminates the `1.28–1.50×`
  batch-1 latency penalty measured on FlashInfer 0.7.4 in #99).
- Vision encoder residency (eager at startup vs lazy load on first image request).
- Flagship coding-agent model for M8; whether a residual checkpoint layer is required.

---

## Execution Nodes (8 Nodes)

### `M8-N01` — Hybrid Cache Spec, Recurrent State Slots & Latent KV Pool
- **Status:** `BLOCKED`
- **Depends On:** `M7-N05`
- **Hardware:** `CPU`
- **Touches:** `python/ayaka/kvcache/`, `python/ayaka/configs/`, `python/ayaka/memory/`, `tests/kvcache/`
- **Deliverable:**
  - Extend KV/cache specs to allocate and track per-request **recurrent state slots** (for GDN/KDA
    layers) and **compressed latent KV pools** (for MLA) alongside standard paged KV blocks, with
    `check_integrity()` coverage.
- **Exit Check:** Unit tests for mixed-layer allocation, preemption/release, and integrity checks
  pass on CPU.

### `M8-N02` — Linear-Attention (GDN/KDA) & MLA Op Contracts + GPU Kernels
- **Status:** `BLOCKED`
- **Depends On:** `M8-N01`
- **Hardware:** `GPU-L4`
- **Touches:** `python/ayaka/ops/`, `python/ayaka/ops/backends/`, `python/ayaka/layers/`, `tests/ops/`
- **Deliverable:**
  - Op contracts, `torch` reference implementations, and FlashInfer/Triton GPU kernels for
    GDN/linear-attention recurrence and MLA compressed-KV attention (prefill & CUDA-graph decode).
- **Exit Check:** Op parity green vs fp64/torch reference; MLA memory footprint verified `≤ 0.5×`
  standard KV; decode replays inside CUDA graphs on L4.

### `M8-N03` — State Checkpointing & Suffix Replay for Prefix-Cache Hits
- **Status:** `BLOCKED`
- **Depends On:** `M8-N02`
- **Hardware:** `GPU-L4`
- **Touches:** `python/ayaka/kvcache/`, `python/ayaka/scheduler/`, `python/ayaka/execution/`, `tests/e2e/`
- **Deliverable:**
  - Block-aligned state checkpointing and suffix replay so hybrid linear-attention models achieve
    exact prefix-cache hits across multi-turn coding sessions.
  - Measure checkpoint density trade-off (RAM vs recompute latency).
- **Exit Check:** Single-chunk prefill == multi-chunk prefill == warm state-prefix-hit tokens
  (exact ID match).

### `M8-N04` — Representative Hybrid Architectures (`Qwen3.5`/`3.8`, `Gemma 4`, `gpt-oss` Class)
- **Status:** `BLOCKED`
- **Depends On:** `M8-N02`, `M8-N03`
- **Hardware:** `GPU-L4`
- **Touches:** `python/ayaka/models/`, `python/ayaka/weights/`, `tests/e2e/`
- **Deliverable:**
  - Model definitions, weight mappers, and registry entries for the pinned M8 representative
    architectures (GDN hybrid, sliding/vision dense, open-weight sink/dense).
- **Exit Check:** Greedy parity against `transformers` passes on tiny & real checkpoints on L4.

### `M8-N05` — Image Preprocessing, Token Expansion & Multimodal Wire Payloads
- **Status:** `BLOCKED`
- **Depends On:** `M8-N01`
- **Hardware:** `CPU`
- **Touches:** `python/ayaka/entrypoints/wire/`, `python/ayaka/tokenizers/`, `python/ayaka/core/`, `tests/entrypoints/`
- **Deliverable:**
  - OpenAI & Anthropic image payload parsing (base64 / validated local inputs; no silent remote URL
    fetches violating `docs/SAFETY.md`), deterministic image preprocessing matching reference
    processors, and multimodal placeholder token expansion.
- **Exit Check:** Preprocessing outputs match reference model processor pixel/patch tensors;
  wire policy & golden-byte tests pass.

### `M8-N06` — Vision Encoder, Projector & `1024×768` Screenshot E2E
- **Status:** `BLOCKED`
- **Depends On:** `M8-N04`, `M8-N05`
- **Hardware:** `GPU-L4`
- **Touches:** `python/ayaka/models/`, `python/ayaka/execution/`, `tests/e2e/`
- **Deliverable:**
  - Vision tower & multimodal projector execution integrated into the runner, measuring eager vs
    lazy vision-encoder loading.
- **Exit Check:** Real `1024×768` IDE/terminal screenshot processed end to end through `ayaka serve`
  with coherent structured output and exact processor parity.

### `M8-N07` — Opt-In KV FP8 Codec & Quality Gate
- **Status:** `BLOCKED`
- **Depends On:** `M8-N01`
- **Hardware:** `GPU-L4`
- **Touches:** `python/ayaka/kvcache/`, `python/ayaka/ops/`, `python/ayaka/configs/`, `benchmarks/e2e/`
- **Deliverable:**
  - Opt-in `float8_e4m3fn` KV cache capacity mode (`--kv-dtype fp8`) halving KV VRAM, paired with
    pre-registered first-token KL and long-context quality gates (building directly on #99).
- **Exit Check:** Quality gates pass on L4; KV capacity doubles in `ayaka plan` and live startup;
  latency trade-off honestly recorded in `docs/EXPERIMENTS.md`.

### `M8-N08` — Hybrid/Vision `plan` & `bench` Support + `v0.2.0-a5`
- **Status:** `BLOCKED`
- **Depends On:** `M8-N04`, `M8-N06`, `M8-N07`
- **Hardware:** `GPU-L4`
- **Touches:** `python/ayaka/entrypoints/plan.py`, `python/ayaka/entrypoints/bench.py`, `docs/EXPERIMENTS.md`, `docs/capabilities.toml`, `docs/CAPABILITIES.md`
- **Deliverable:**
  - `ayaka plan` reports state bytes, latent KV bytes, and vision encoder/activation bytes;
    `ayaka bench` adds image (`1024×768`) and long-prompt groups; tag `v0.2.0-a5`.
- **Exit Check:** All 6 M8 exit conditions green; `tests/test_docs.py` & CI pass; tag `v0.2.0-a5`.

## Risks

- Divergent GDN/KDA recurrence variants across newer checkpoints.
- State checkpoints inflating host RAM under multi-session loads.
- Subtle image resize/normalization drift breaking multimodal token alignment.
