# M14 — Multi-Agent & Distributed Prefix Fabric (`0.4.0`)

- **Goal:** Evolve Ayaka into an **Agent Swarm Inference OS** — enabling many concurrent coding
  subagents (planners, coders, reviewers, test runners) working on the same codebase to share a
  single zero-copy prefix, KV, and hybrid-state fabric with strict interactive QoS priority and
  opt-in persistent snapshots across restarts.
- **Status:** Planned (long-horizon, post-`0.3.0`)
- **Target Release:** **`ayaka-llm 0.4.0`**

## Exit Conditions

1. An 8-agent coding swarm reading the same repository shares a single zero-copy prefix/KV/state
   tree with `≥ 85%` prefix reuse across subagents and zero duplicated system/repo KV blocks.
2. Interactive foreground coding turns meet strict TTFT / ITL SLOs while background swarm subagents
   saturate spare GPU capacity.
3. Opt-in persistent KV/session snapshots survive a full server restart and resume warm with exact
   token equivalence and owner-only/encrypted storage.
4. Swarm benchmark numbers (throughput, cache sharing ratio, foreground ITL p99) are recorded in
   `docs/EXPERIMENTS.md` and **`ayaka-llm 0.4.0`** is published.

---

## Execution Nodes (5 Nodes)

### `M14-N01` — Zero-Copy Shared Prefix / KV / State Fabric for Agent Swarms
- **Status:** `BLOCKED`
- **Depends On:** `M13-N06`, `M10-N05`
- **Hardware:** `GPU` / `Multi-GPU`
- **Touches:** `python/ayaka/kvcache/`, `python/ayaka/distributed/`, `python/ayaka/scheduler/`, `tests/kvcache/`
- **Deliverable:**
  - Unified radix/hash-chain prefix & hybrid-state fabric shared across concurrent local subagents
    (and across P/D/multi-GPU workers via M12 KV transfer), preventing duplicated repo-context
    prefill or VRAM copies.
- **Exit Check:** 8 concurrent subagents sharing a 32K repo prefix allocate shared blocks once,
  emit exact outputs, and pass `check_integrity()`.

### `M14-N02` — Cross-Agent Speculative KV Prewarm & Branch-Aware Scheduling
- **Status:** `BLOCKED`
- **Depends On:** `M14-N01`
- **Hardware:** `GPU`
- **Touches:** `python/ayaka/scheduler/`, `python/ayaka/kvcache/`, `python/ayaka/entrypoints/`
- **Deliverable:**
  - Branch-aware scheduling for tree-structured subagent forks (parallel search/edit branches sharing
    a parent prefix) and idle-cycle speculative KV prewarming for newly edited files.
- **Exit Check:** Measured reduction in subagent branch TTFT with zero regression on single-stream
  workloads.

### `M14-N03` — Multi-Tenant QoS, Priority Lanes & Fairness Scheduler
- **Status:** `BLOCKED`
- **Depends On:** `M14-N01`
- **Hardware:** `CPU` + `GPU`
- **Touches:** `python/ayaka/scheduler/`, `python/ayaka/entrypoints/wire/`, `python/ayaka/metrics/`, `tests/scheduler/`
- **Deliverable:**
  - Request priority lanes (`interactive` foreground vs `background` swarm/eval) with token-budget
    fairness and preemption priority so background subagents never stall a human's active turn.
- **Exit Check:** Foreground ITL p99 stays within `≤ 1.15×` solo baseline while 6+ background
  subagents run concurrently.

### `M14-N04` — Opt-In Persistent KV & Session Snapshot Store
- **Status:** `BLOCKED`
- **Depends On:** `M10-N01`, `M14-N01`
- **Hardware:** `CPU` + `NVMe SSD`
- **Touches:** `python/ayaka/kvcache/`, `python/ayaka/memory/`, `docs/SAFETY.md`, `tests/kvcache/`
- **Deliverable:**
  - Explicitly opt-in (`--persist-cache-dir`), owner-only (and optionally encrypted) local KV +
    session snapshot persistence so restarting `ayaka serve` resumes warm repo prefixes without
    cold re-prefill.
- **Exit Check:** Restart-and-resume produces **identical token IDs** to an uninterrupted warm run,
  beats cold prefill TTFT on `≥ 16K` prompts, and enforces strict file-permission checks.

### `M14-N05` — 8-Agent Coding-Swarm Benchmark & Publish `ayaka-llm 0.4.0`
- **Status:** `BLOCKED`
- **Depends On:** `M14-N02`, `M14-N03`, `M14-N04`
- **Hardware:** `GPU` / `Multi-GPU`
- **Touches:** `benchmarks/e2e/`, `docs/EXPERIMENTS.md`, `docs/CAPABILITIES.md`, `python/ayaka/__version__.py`, `CHANGELOG.md`
- **Deliverable:**
  - Reproducible 8-agent coding-swarm benchmark suite measuring aggregate tok/s, shared-cache ratio,
    foreground ITL p99, and warm-restart recovery; publish **`ayaka-llm 0.4.0`**.
- **Exit Check:** All M14 exit conditions verified and recorded; `ayaka-llm 0.4.0` published.

## Risks

- Shared prefix fabric accidentally leaking KV state across distinct trust/tenant boundaries
  (must keep salt isolation strict per #52).
- Persistent disk KV snapshots storing sensitive source code if defaults are not strictly opt-in.
- Priority starvation of background subagents without bounded aging rules.
