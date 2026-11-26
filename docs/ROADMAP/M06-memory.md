# M6 — Memory (`v0.2.0-a3`)

- **Goal:** Running out of VRAM never stops or corrupts the server, and evicted conversations are
  preserved in host RAM instead of lost.
- **Status:** Implemented & verified on CPU (PR 1 #141 & stacked PR 2); GPU benchmark gates pending CUDA runs on L4 & WSL2 RTX 3050.
- **Target Checkpoint:** `v0.2.0-a3` (internal tag on the road to `0.2.0` GA)
- **Structure:** Two waves — Wave 1 makes the server unkillable under OOM/pressure; Wave 2 adds
  host-RAM KV tiering, atomic pool resize, and recorded session replay.

## Exit Conditions

1. With a deliberately small KV pool and **3 long concurrent requests**, every request finishes and
   `kv_manager.check_integrity()` is clean.
2. Atomic runtime KV pool resize either completes cleanly or refuses with the previous cache intact
   and zero leaked blocks.
3. Returning to a conversation evicted from VRAM to the host tier produces **identical token IDs**
   and beats cold recomputation (`< 0.5×` cold prefill TTFT on prompts `≥ 4K` tokens).
4. An injected CUDA OOM inside a step does not stop the engine thread or drop unrelated streams.
5. `ayaka doctor` prints pass/fail per check with one concrete fix per failure.

## Pre-Registered Measurement Gates

- **Host KV tier promotion vs recompute:** `< 0.5×` cold prefill TTFT on `≥ 4K` prompts (L4
  Llama-3.1-8B & RTX 3050 Llama-3.2-1B) with exact token equivalence.
- **Host KV tier default:** Stays **opt-in** until Linux & WSL2 pinned-RAM ceilings are measured
  and steady-state write-behind latency overhead is `≤ 2%`.

---

## Execution Nodes (10 Nodes)

### Wave 1 — Crash Resilience, Preemption & Diagnostics

#### `M6-N01` — `ayaka doctor` Diagnostic CLI
- **Status:** `DONE (CPU/unit verified; L4/RTX 3050 live CUDA reports pending)`
- **Depends On:** `M5-N12`
- **Hardware:** `CPU` + `GPU-L4` + `GPU-4GB-WSL2`
- **Touches:** `python/ayaka/entrypoints/doctor.py`, `python/ayaka/entrypoints/cli.py`, `python/ayaka/device/`, `python/ayaka/memory/`, `tests/entrypoints/`, `docs/TROUBLESHOOTING.md`
- **Deliverable:**
  - `ayaka doctor` checking CUDA/driver visibility, NVML-vs-CUDA free-memory gap (detecting WSL2
    WDDM oversubscription risk), host pinned-RAM headroom, FlashInfer/`nvcc` availability, cache
    disk space, and startup profile validity — printing pass/fail + 1 actionable fix per item
    (plus `--output json`).
- **Exit Check:** Unit tests with mocked probes + live Linux L4 and WSL2 RTX 3050 doctor reports
  green; `tests/test_docs.py` verifies `ayaka doctor` flags.

#### `M6-N02` — Memory Wire Error Codes & Over-Long Conversation Policy
- **Status:** `DONE`
- **Depends On:** `M6-N01`
- **Hardware:** `CPU`
- **Touches:** `python/ayaka/core/errors.py`, `python/ayaka/entrypoints/wire/`, `python/ayaka/engine/`, `docs/AI_REFERENCE.md`, `tests/entrypoints/`
- **Deliverable:**
  - Add `out_of_memory`, `preempted_timeout`, and `FinishEvent.limit = "memory"` to the closed
    error code table and wire policy table (`tools/generate_wire_reference.py`).
  - Explicit rejection or opt-in safe truncation for conversations exceeding configured KV capacity.
- **Exit Check:** Golden-byte error tests green across OpenAI & Anthropic adapters;
  `tests/test_docs.py` wire-reference drift test passes.

#### `M6-N03` — Pressure Watermarks & Scheduler Preemption
- **Status:** `DONE (CPU/unit verified)`
- **Depends On:** `M6-N02`
- **Hardware:** `CPU` + `GPU-L4`
- **Touches:** `python/ayaka/scheduler/`, `python/ayaka/kvcache/`, `python/ayaka/metrics/`, `tests/scheduler/`, `tests/kvcache/`
- **Deliverable:**
  - KV pressure levels (`normal`, `high`, `critical`), victim selection policy (newest / lowest
    progress), recompute preemption releasing all KV blocks cleanly, and preemption timeout
    accounting.
- **Exit Check:** Scheduler preemption tests pass; `kv_manager.check_integrity()` clean after every
  preemption/resume cycle; Prometheus preemption counters exposed.

#### `M6-N04` — Step-Level CUDA OOM Recovery
- **Status:** `DONE (synthetic OOM/unit & E2E verified on CPU; live CUDA GPU run pending)`
- **Depends On:** `M6-N03`
- **Hardware:** `GPU-L4` + `GPU-4GB-WSL2`
- **Touches:** `python/ayaka/execution/runner.py`, `python/ayaka/engine/`, `python/ayaka/scheduler/`, `tests/engine/`
- **Deliverable:**
  - Catch `torch.OutOfMemoryError` inside prefill/decode steps, clear allocator cache, preempt or
    split active batch, emit `out_of_memory` only if a single minimal request cannot fit, and keep
    the engine loop and healthy concurrent requests alive.
- **Exit Check:** Injected OOM mid-step leaves engine thread healthy, finishes surviving requests,
  and passes `check_integrity()`.

#### `M6-N05` — Wave 1 Fault-Injection Suite
- **Status:** `DONE (CPU fault-injection suite passing; @pytest.mark.gpu smoke test pending GPU)`
- **Depends On:** `M6-N04`
- **Hardware:** `CPU` + `GPU-L4`
- **Touches:** `tests/e2e/test_memory_faults.py`, `tests/engine/`
- **Deliverable:**
  - Reproducible fault-injection suite: small KV pool with **3 long concurrent requests** all
    finishing cleanly, injected step OOM, preemption timeout, and repeated integrity verification.
- **Exit Check:** Entire fault-injection suite passes on CPU and GPU with zero leaked blocks.

---

### Wave 2 — Host-RAM KV Tier, Atomic Resize & Session Replay

#### `M6-N06` — Host Memory Policy & Linux/WSL2 Pinning Probe
- **Status:** `IMPLEMENTED (config/policy/probe verified on CPU; L4 & WSL2 RTX 3050 CUDA measurements pending)`
- **Depends On:** `M6-N01`
- **Hardware:** `GPU-L4` + `GPU-4GB-WSL2`
- **Touches:** `python/ayaka/configs/`, `python/ayaka/memory/`, `benchmarks/memory/`, `docs/EXPERIMENTS.md`
- **Deliverable:**
  - `HostMemoryConfig` (`ram_keep`, host KV byte cap, pinned vs pageable fallback).
  - Pre-registered probe measuring safe `cudaHostAlloc` pinned ceiling and H2D/D2H bandwidth on
    both native Linux (L4) and WSL2 (RTX 3050), plus WDDM shared-RAM spill detection.
- **Exit Check:** Measurements recorded in `docs/EXPERIMENTS.md`; config validation unit tests
  green.

#### `M6-N07` — Host-RAM KV Store, Async Write-Behind, Promotion & Prefix Pinning
- **Status:** `DONE (CPU unit & race tests verified; CUDA transfer timing pending)`
- **Depends On:** `M6-N03`, `M6-N06`
- **Hardware:** `GPU-L4` + `GPU-4GB-WSL2`
- **Touches:** `python/ayaka/kvcache/`, `python/ayaka/memory/`, `python/ayaka/scheduler/`, `tests/kvcache/`
- **Deliverable:**
  - Host-RAM KV tier storing evicted blocks asynchronously (write-behind stream) and promoting
    blocks back into VRAM on conversation return with strict race-free ownership locking.
  - Shared-prefix pinning in VRAM so active coding-agent system prompts resist eviction.
- **Exit Check:** Evicted-and-promoted conversations generate **identical token IDs** to un-evicted
  runs; concurrency race/stress tests pass; `check_integrity()` clean.

#### `M6-N08` — Wait for In-Flight Prefixes & Prewarm
- **Status:** `DONE (CPU scheduler & EngineCore.prewarm verified)`
- **Depends On:** `M6-N07`
- **Hardware:** `CPU` + `GPU-L4`
- **Touches:** `python/ayaka/kvcache/`, `python/ayaka/scheduler/`, `tests/scheduler/`
- **Deliverable:**
  - Coalesce concurrent requests sharing an uncached prefix so the second waits briefly for the
    in-flight prefix instead of duplicating cold prefill compute; optional prewarm hook.
- **Exit Check:** Two simultaneous cold requests with identical 4K prefix compute prefill once and
  both complete with exact tokens and `cached_tokens > 0` on the waiter.

#### `M6-N09` — Atomic Runtime KV Pool Resize
- **Status:** `DONE (grow, shrink compaction, OOM rollback & EngineCore resize verified on CPU)`
- **Depends On:** `M6-N07`
- **Hardware:** `GPU-L4`
- **Touches:** `python/ayaka/kvcache/`, `python/ayaka/engine/`, `tests/kvcache/`
- **Deliverable:**
  - Runtime pool grow/shrink path that migrates live blocks cleanly or rolls back atomically with
    the old pool completely intact when allocation fails.
- **Exit Check:** Success resize + forced-OOM rollback tests pass with zero lost conversations and
  clean `check_integrity()`.

#### `M6-N10` — Session Replay (`entrypoints/replay.py`), Host-Tier Gate & `v0.2.0-a3`
- **Status:** `IMPLEMENTED (ayaka replay & bench_host_kv_tier.py verified on CPU; GPU <0.5x TTFT / <=2% overhead gates & v0.2.0-a3 tag pending real GPU runs)`
- **Depends On:** `M6-N05`, `M6-N08`, `M6-N09`
- **Hardware:** `GPU-L4` + `GPU-4GB-WSL2`
- **Touches:** `python/ayaka/entrypoints/replay.py`, `benchmarks/e2e/`, `docs/EXPERIMENTS.md`, `docs/CAPABILITIES.md`, `docs/ROADMAP/decisions.md`
- **Deliverable:**
  - `entrypoints/replay.py` replaying `--record` token sessions under tight VRAM to verify
    preemption + host-tier exactness.
  - Measure host-tier promotion vs cold prefill (`< 0.5×` on `≥ 4K`), write-behind steady overhead
    (`≤ 2%`), time-budgeted chunking, and overlapped scheduling; tag internal checkpoint `v0.2.0-a3`.
- **Exit Check:** All 5 M6 exit conditions green; numbers recorded in `docs/EXPERIMENTS.md`; tag
  `v0.2.0-a3`.

## Risks

- Eviction/promotion race conditions corrupting block reference counts.
- WSL2 capping `cudaHostAlloc` pinned regions far below physical RAM.
- Failed runtime resize stranding active request block tables.
