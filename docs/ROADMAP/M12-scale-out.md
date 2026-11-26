# M12 — Scale-Out (`v0.3.0-a1`)

- **Goal:** Scale Ayaka beyond a single GPU to multi-GPU machines (Tensor / Pipeline / Expert
  Parallelism), Prefill/Decode (P/D) worker disaggregation, KV transfer across workers/nodes, and
  trillion-parameter-class MoE (`Kimi K3` class).
- **Status:** Planned (not filed yet)
- **Target Checkpoint:** `v0.3.0-a1` (leading into `0.3.0` alongside M13)
- **Representative Architecture Target:** `Kimi K3` class (8 GPUs or 1× 128 GB workstation + SSD).

## Exit Conditions

1. 2-GPU Tensor Parallelism on a dense model emits **identical greedy tokens** to single-GPU
   execution.
2. Split prefill and decode workers transfer KV blocks cleanly and serve concurrent requests end
   to end.
3. A 2-machine distributed benchmark is recorded in `docs/EXPERIMENTS.md`.
4. `Kimi K3`-class checkpoint runs on 8 GPUs or 1× 128 GB machine + SSD.
5. `ayaka plan --distributed` prints a verified TP/PP/EP placement plan.

## Decisions to Measure

- TP vs PP first on consumer dual-GPU rigs without NVLink (e.g., 2× 24 GB over PCIe).
- Whether P/D disaggregation pays at 2 GPUs vs only at higher concurrency/node scale.
- Whether Vulkan or Metal justifies maintenance cost (strictly conditional on real test hardware).

---

## Execution Nodes (6 Nodes)

### `M12-N01` — Distributed Planner (`ayaka plan --distributed`) & Comms Contract
- **Status:** `BLOCKED`
- **Depends On:** `M11-N08`
- **Hardware:** `CPU` + `2+ GPU`
- **Touches:** `python/ayaka/entrypoints/plan.py`, `python/ayaka/distributed/`, `python/ayaka/configs/`, `tests/distributed/`
- **Deliverable:**
  - `ayaka plan --distributed` outputting TP / PP / EP memory, communication volume, and device
    placement plans without downloading weights.
  - Clean `ayaka.distributed` comms abstraction (NCCL / PyTorch distributed process groups, lazily
    imported so single-GPU & CPU installs stay zero-overhead).
- **Exit Check:** Unit tests verify TP/PP/EP memory math and CLI output; `import ayaka` remains
  torch-free.

### `M12-N02` — Intra-Node Tensor Parallelism (TP)
- **Status:** `BLOCKED`
- **Depends On:** `M12-N01`
- **Hardware:** `2+ GPU`
- **Touches:** `python/ayaka/layers/`, `python/ayaka/models/`, `python/ayaka/distributed/`, `tests/e2e/`
- **Deliverable:**
  - Column/row-parallel linear and sharded attention/MLP layers with all-reduce synchronization
    compatible with CUDA graph capture and position-keyed RNG.
- **Exit Check:** 2-GPU TP on a dense model emits **identical greedy tokens** to 1-GPU execution;
  multi-GPU throughput & PCIe/NVLink scaling recorded in `docs/EXPERIMENTS.md`.

### `M12-N03` — KV Transfer Protocol & Prefill/Decode (P/D) Disaggregation
- **Status:** `BLOCKED`
- **Depends On:** `M12-N02`
- **Hardware:** `2+ GPU`
- **Touches:** `python/ayaka/kvcache/`, `python/ayaka/distributed/`, `python/ayaka/engine/`, `benchmarks/e2e/`
- **Deliverable:**
  - Zero-copy/low-latency KV block transfer between dedicated prefill and decode workers (NCCL /
    RDMA / TCP fallback) plus pre-registered 2-GPU P/D vs unified benchmark.
- **Exit Check:** Concurrent P/D serving finishes with exact token parity and zero KV leaks;
  2-GPU payoff verdict recorded in `docs/EXPERIMENTS.md`.

### `M12-N04` — Expert Parallelism (EP) & MoE Load Balancing
- **Status:** `BLOCKED`
- **Depends On:** `M12-N02`
- **Hardware:** `2+ GPU`
- **Touches:** `python/ayaka/layers/moe.py`, `python/ayaka/distributed/`, `python/ayaka/metrics/`
- **Deliverable:**
  - Sharded MoE experts across GPUs with all-to-all token dispatch/combine and per-expert load
    imbalance telemetry.
- **Exit Check:** Multi-GPU EP MoE matches single-GPU `fused` greedy tokens on a tiny MoE and
  scales on a real MoE checkpoint.

### `M12-N05` — Multi-Node Serving & `Kimi K3`-Class Frontier Scale-Out
- **Status:** `BLOCKED`
- **Depends On:** `M12-N03`, `M12-N04`
- **Hardware:** `Multi-Node / 8-GPU or 128 GB + SSD`
- **Touches:** `python/ayaka/models/`, `python/ayaka/distributed/`, `benchmarks/e2e/`, `docs/EXPERIMENTS.md`, `docs/CAPABILITIES.md`
- **Deliverable:**
  - Multi-node coordination & 2-machine benchmark; `Kimi K3`-class model support running on 8 GPUs
    or 1× 128 GB workstation + NVMe SSD.
- **Exit Check:** 2-machine benchmark and `Kimi K3`-class run committed with raw `results.json`.

### `M12-N06` — Conditional Non-CUDA Backends (Vulkan / Metal) & `v0.3.0-a1` *(GATED)*
- **Status:** `GATED`
- **Depends On:** `M12-N05`
- **Hardware:** Conditional (Apple Silicon / Vulkan GPU)
- **Touches:** `python/ayaka/ops/backends/`, `docs/ROADMAP/decisions.md`
- **Deliverable:**
  - Evaluate Vulkan or Metal backend **strictly if real test hardware and maintainer bandwidth
    exist**; otherwise formally record deferment in `decisions.md`. Tag `v0.3.0-a1`.
- **Exit Check:** M12 exit conditions 1–5 verified; tag `v0.3.0-a1`.

## Risks

- Lack of multi-GPU / multi-node hardware for routine regression testing.
- Distributed error handling complicating single-GPU simplicity.
- Premature non-CUDA backends diluting focus from NVIDIA GPUs.
