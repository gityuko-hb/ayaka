# Memory (`v0.2.0-a2`)

- **Goal:** Running out of VRAM never stops or corrupts the server, and evicted conversations are
  preserved in host RAM instead of lost.
- **Status:** Planned (not filed yet). It starts after the Serving Core v2 exit audit (Serving Core v2 step 12).
- **Target Checkpoint:** `v0.2.0-a2` (internal tag on the road to `0.2.0` GA)
- **Hardware this round:** NVIDIA A100-SXM4-80GB (sm_80) is the development GPU. Gates written for the NVIDIA L4 stay unverified until they run on an NVIDIA L4; no L4 result is claimed in this file.
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

#### memory step 1 — `ayaka doctor` Diagnostic CLI
- **Status:** `BLOCKED`
- **Depends On:** Serving Core v2 step 12
- **Hardware:** `CPU` + `GPU-L4` + `GPU-4GB-WSL2`
- **Touches:** `python/ayaka/entrypoints/doctor.py`, `python/ayaka/entrypoints/cli.py`, `python/ayaka/device/`, `python/ayaka/memory/`, `tests/entrypoints/`, `docs/TROUBLESHOOTING.md`
- **Deliverable:**
  - `ayaka doctor` checking CUDA/driver visibility, NVML-vs-CUDA free-memory gap (detecting WSL2
    WDDM oversubscription risk), host pinned-RAM headroom, FlashInfer/`nvcc` availability, cache
    disk space, and startup profile validity — printing pass/fail + 1 actionable fix per item
    (plus `--output json`).
- **Exit Check:** Unit tests with mocked probes + live Linux L4 and WSL2 RTX 3050 doctor reports
  green; `tests/test_docs.py` verifies `ayaka doctor` flags.

#### memory step 2 — Memory Wire Error Codes & Over-Long Conversation Policy
- **Status:** `BLOCKED`
- **Depends On:** memory step 1
- **Hardware:** `CPU`
- **Touches:** `python/ayaka/core/errors.py`, `python/ayaka/entrypoints/wire/`, `python/ayaka/engine/`, `docs/AI_REFERENCE.md`, `tests/entrypoints/`
- **Deliverable:**
  - Add `out_of_memory`, `preempted_timeout`, and `FinishEvent.limit = "memory"` to the closed
    error code table and wire policy table (`tools/generate_wire_reference.py`).
  - Explicit rejection or opt-in safe truncation for conversations exceeding configured KV capacity.
- **Exit Check:** Golden-byte error tests green across OpenAI & Anthropic adapters;
  `tests/test_docs.py` wire-reference drift test passes.

#### memory step 3 — Pressure Watermarks & Scheduler Preemption
- **Status:** `BLOCKED`
- **Depends On:** memory step 2
- **Hardware:** `CPU` + `GPU-L4`
- **Touches:** `python/ayaka/scheduler/`, `python/ayaka/kvcache/`, `python/ayaka/metrics/`, `tests/scheduler/`, `tests/kvcache/`
- **Deliverable:**
  - KV pressure levels (`normal`, `high`, `critical`), victim selection policy (newest / lowest
    progress), recompute preemption releasing all KV blocks cleanly, and preemption timeout
    accounting.
- **Exit Check:** Scheduler preemption tests pass; `kv_manager.check_integrity()` clean after every
  preemption/resume cycle; Prometheus preemption counters exposed.

#### memory step 4 — Step-Level CUDA OOM Recovery
- **Status:** `BLOCKED`
- **Depends On:** memory step 3
- **Hardware:** `GPU-L4` + `GPU-4GB-WSL2`
- **Touches:** `python/ayaka/execution/runner.py`, `python/ayaka/engine/`, `python/ayaka/scheduler/`, `tests/engine/`
- **Deliverable:**
  - Catch `torch.OutOfMemoryError` inside prefill/decode steps, clear allocator cache, preempt or
    split active batch, emit `out_of_memory` only if a single minimal request cannot fit, and keep
    the engine loop and healthy concurrent requests alive.
- **Exit Check:** Injected OOM mid-step leaves engine thread healthy, finishes surviving requests,
  and passes `check_integrity()`.

#### memory step 5 — Wave 1 Fault-Injection Suite
- **Status:** `BLOCKED`
- **Depends On:** memory step 4
- **Hardware:** `CPU` + `GPU-L4`
- **Touches:** `tests/e2e/test_memory_faults.py`, `tests/engine/`
- **Deliverable:**
  - Reproducible fault-injection suite: small KV pool with **3 long concurrent requests** all
    finishing cleanly, injected step OOM, preemption timeout, and repeated integrity verification.
- **Exit Check:** Entire fault-injection suite passes on CPU and GPU with zero leaked blocks.

---

### Wave 2 — Host-RAM KV Tier, Atomic Resize & Session Replay

#### memory step 6 — Host Memory Policy & Linux/WSL2 Pinning Probe
- **Status:** `BLOCKED`
- **Depends On:** memory step 1
- **Hardware:** `GPU-L4` + `GPU-4GB-WSL2`
- **Touches:** `python/ayaka/configs/`, `python/ayaka/memory/`, `benchmarks/memory/`, `docs/EXPERIMENTS.md`
- **Deliverable:**
  - `HostMemoryConfig` (`ram_keep`, host KV byte cap, pinned vs pageable fallback).
  - Pre-registered probe measuring safe `cudaHostAlloc` pinned ceiling and H2D/D2H bandwidth on
    both native Linux (L4) and WSL2 (RTX 3050), plus WDDM shared-RAM spill detection.
- **Exit Check:** Measurements recorded in `docs/EXPERIMENTS.md`; config validation unit tests
  green.

#### memory step 7 — Host-RAM KV Store, Async Write-Behind, Promotion & Prefix Pinning
- **Status:** `BLOCKED`
- **Depends On:** memory step 3, memory step 6
- **Hardware:** `GPU-L4` + `GPU-4GB-WSL2`
- **Touches:** `python/ayaka/kvcache/`, `python/ayaka/memory/`, `python/ayaka/scheduler/`, `tests/kvcache/`
- **Deliverable:**
  - Host-RAM KV tier storing evicted blocks asynchronously (write-behind stream) and promoting
    blocks back into VRAM on conversation return with strict race-free ownership locking.
  - Shared-prefix pinning in VRAM so active coding-agent system prompts resist eviction.
- **Exit Check:** Evicted-and-promoted conversations generate **identical token IDs** to un-evicted
  runs; concurrency race/stress tests pass; `check_integrity()` clean.

#### memory step 8 — Wait for In-Flight Prefixes & Prewarm
- **Status:** `BLOCKED`
- **Depends On:** memory step 7
- **Hardware:** `CPU` + `GPU-L4`
- **Touches:** `python/ayaka/kvcache/`, `python/ayaka/scheduler/`, `tests/scheduler/`
- **Deliverable:**
  - Coalesce concurrent requests sharing an uncached prefix so the second waits briefly for the
    in-flight prefix instead of duplicating cold prefill compute; optional prewarm hook.
- **Exit Check:** Two simultaneous cold requests with identical 4K prefix compute prefill once and
  both complete with exact tokens and `cached_tokens > 0` on the waiter.

#### memory step 9 — Atomic Runtime KV Pool Resize
- **Status:** `BLOCKED`
- **Depends On:** memory step 7
- **Hardware:** `GPU-L4`
- **Touches:** `python/ayaka/kvcache/`, `python/ayaka/engine/`, `tests/kvcache/`
- **Deliverable:**
  - Runtime pool grow/shrink path that migrates live blocks cleanly or rolls back atomically with
    the old pool completely intact when allocation fails.
- **Exit Check:** Success resize + forced-OOM rollback tests pass with zero lost conversations and
  clean `check_integrity()`.

#### memory step 10 — Session Replay (`entrypoints/replay.py`), Host-Tier Gate & `v0.2.0-a2`
- **Status:** `BLOCKED`
- **Depends On:** memory step 5, memory step 8, memory step 9
- **Hardware:** `GPU-L4` + `GPU-4GB-WSL2`
- **Touches:** `python/ayaka/entrypoints/replay.py`, `benchmarks/e2e/`, `docs/EXPERIMENTS.md`, `docs/CAPABILITIES.md`, `docs/ROADMAP/decisions.md`
- **Deliverable:**
  - `entrypoints/replay.py` replaying `--record` token sessions under tight VRAM to verify
    preemption + host-tier exactness.
  - Measure host-tier promotion vs cold prefill (`< 0.5×` on `≥ 4K`), write-behind steady overhead
    (`≤ 2%`), time-budgeted chunking, and overlapped scheduling; tag internal checkpoint `v0.2.0-a2`.
- **Exit Check:** All 5 memory milestone exit conditions green; numbers recorded in `docs/EXPERIMENTS.md`; tag
  `v0.2.0-a2`.

## Risks

- Eviction/promotion race conditions corrupting block reference counts.
- WSL2 capping `cudaHostAlloc` pinned regions far below physical RAM.
- Failed runtime resize stranding active request block tables.
