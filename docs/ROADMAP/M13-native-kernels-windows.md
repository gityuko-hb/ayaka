# M13 — Native Kernel Engine (`kops` / `csrc/`) & Windows Native (`0.3.0`)

- **Goal:** Build Ayaka's native C++/CUDA kernel backend (`kops` in `csrc/`) and deliver full
  **native Windows GPU support without WSL2**, unlocking Blackwell (`sm_100`/`sm_120`) FP4/FP8
  fused kernels, fast KV codecs, deterministic fused sampling, and MoE offload DMA — shipped via
  `cibuildwheel` as **`ayaka-llm 0.3.0`**.
- **Status:** Early spike `M13-N00` scheduled right after M7 (`v0.2.0-a4`); full `csrc/` build
  activates after `0.2.0` GA (or earlier if M9 MoE offload hits a hard FlashInfer/Triton ceiling).
- **Target Release:** **`ayaka-llm 0.3.0`** (combined with M12 Scale-out)

## Exit Conditions

1. Native `kops` backend passes full op-parity suites on both Linux and **native Windows (no WSL2)**.
2. GPU e2e, CUDA graph capture, `ayaka bench`, and `ayaka connect --check` (7/7) pass on native
   Windows.
3. `import ayaka` continues to succeed cleanly on machines without CUDA or `torch` installed.
4. `cibuildwheel` builds Linux and Windows release wheels that install cleanly into fresh
   environments and pass smoke + GPU validation.

---

## Execution Nodes (7 Nodes)

### `M13-N00` — Early Spike: Binding ADR & 2-Kernel CUDA Graph Experiment *(Runs after M7)*
- **Status:** `BLOCKED` (unblocks immediately after `M7-N05` / `v0.2.0-a4`, parallel with M8/M9)
- **Depends On:** `M7-N05`
- **Hardware:** `Linux GPU` + `Windows Native GPU`
- **Touches:** `docs/ROADMAP/decisions.md`, `benchmarks/ops/` *(spike branch / issue ADR — no build-system switch on `main` yet)*
- **Deliverable:**
  - Architecture Decision Record (ADR) filed as a GitHub issue agreeing on:
    1. Binding library (`nanobind` vs `torch.library` custom ops vs `pybind11` / C ABI).
    2. Minimum CUDA toolkit version and supported `sm_` architecture matrix (`sm_86`, `sm_89`,
       `sm_90`, `sm_100`, `sm_120`).
    3. Binary ABI compatibility policy across PyTorch minor releases.
  - Small 2-kernel CUDA graph spike verifying custom CUDA kernels capture and replay cleanly inside
    PyTorch CUDA graphs on both Linux and native Windows.
- **Exit Check:** ADR agreed in issue; 2-kernel graph capture passes parity on Linux & native
  Windows; findings recorded in `docs/EXPERIMENTS.md`.

### `M13-N01` — Build System Migration (`scikit-build-core` + `CMakeLists.txt` + `csrc/`)
- **Status:** `BLOCKED`
- **Depends On:** `M13-N00`, `M11-N08` *(may pull earlier if M9 offload requires native DMA)*
- **Hardware:** `Linux` + `Windows`
- **Touches:** `pyproject.toml`, `CMakeLists.txt`, `csrc/`, `.github/workflows/`, `python/ayaka/ops/backends/kops/`
- **Deliverable:**
  - Switch `[build-system]` to `scikit-build-core`, add root `CMakeLists.txt` and `csrc/` outside
    `python/` without changing the `python/ayaka/` package layout (per `AGENTS.md`).
  - Keep CPU-only build & clean import-time fallback so `import ayaka` always works without CUDA.
- **Exit Check:** CPU CI and CUDA compilation CI both green; `import ayaka` works with no GPU and
  no `torch`.

### `M13-N02` — Native Paged Attention & KV Kernels (`kops`)
- **Status:** `BLOCKED`
- **Depends On:** `M13-N01`
- **Hardware:** `Linux GPU` + `Windows Native GPU`
- **Touches:** `csrc/attention/`, `python/ayaka/ops/backends/kops/`, `tests/ops/`
- **Deliverable:**
  - Native CUDA paged prefill, multi-row verify, and decode attention kernels replacing Linux-only
    FlashInfer on Windows (and benchmarked against FlashInfer on Linux).
- **Exit Check:** Full op-parity suite green vs `torch` reference on Linux and native Windows;
  CUDA graph capture verified.

### `M13-N03` — Native Fused Deterministic Sampler & Fast KV Codecs
- **Status:** `BLOCKED`
- **Depends On:** `M13-N01`
- **Hardware:** `Linux GPU` + `Windows Native GPU`
- **Touches:** `csrc/sampling/`, `csrc/kvcache/`, `python/ayaka/ops/backends/kops/`, `tests/ops/`, `benchmarks/ops/`
- **Deliverable:**
  - Fused deterministic counter-based sampler (external position-keyed RNG + top-k/top-p + grammar
    bitmask in one kernel).
  - Fast FP8/INT8 KV pack/unpack and paged read kernels aimed at removing the `1.28–1.50×` batch-1
    FP8 KV latency overhead observed on FlashInfer 0.7.4 in #99.
- **Exit Check:** Exact token-ID equivalence with `torch` deterministic sampler; FP8 KV latency
  measured against M1 #99 baseline in `docs/EXPERIMENTS.md`.

### `M13-N04` — Native Quantized GEMM (FP8 / W4A16 / Blackwell FP4) & MoE Offload DMA
- **Status:** `BLOCKED`
- **Depends On:** `M13-N01`
- **Hardware:** `Linux GPU` + `Windows Native GPU` (L4 / RTX 3050 / Blackwell)
- **Touches:** `csrc/quant/`, `csrc/moe/`, `python/ayaka/ops/backends/kops/`, `benchmarks/ops/`
- **Deliverable:**
  - Native CUDA quantized GEMM kernels covering FP8 (tensor/row/128×128 block), W4A16, and
    Blackwell (`sm_100`/`sm_120`) MXFP4/NVFP4.
  - Native fused MoE routing + overlapped pinned-RAM/NVMe DMA prefetch kernels accelerating M9/M11
    offload.
- **Exit Check:** Parity + speed benchmarks beat or match M7/M9 Python+Triton baselines on Linux
  and run natively on Windows.

### `M13-N05` — Native Windows GPU E2E & `ayaka connect --check` Without WSL2
- **Status:** `BLOCKED`
- **Depends On:** `M13-N02`, `M13-N03`, `M13-N04`
- **Hardware:** `Windows Native GPU`
- **Touches:** `tests/e2e/`, `docs/INSTALL.md`, `docs/TROUBLESHOOTING.md`, `docs/CAPABILITIES.md`
- **Deliverable:**
  - Full GPU e2e, CUDA graph capture, `ayaka bench`, `ayaka plan` (WDDM-aware), and `ayaka connect
    --check` (7/7) running natively on Windows PowerShell without WSL2.
- **Exit Check:** Native Windows benchmark JSON committed and added to `docs/CAPABILITIES.md`.

### `M13-N06` — `cibuildwheel` Release Wheels & Publish `ayaka-llm 0.3.0`
- **Status:** `BLOCKED`
- **Depends On:** `M13-N05`, `M12-N06`
- **Hardware:** `Linux CI` + `Windows CI` + `GPU validation`
- **Touches:** `.github/workflows/`, `pyproject.toml`, `python/ayaka/__version__.py`, `CHANGELOG.md`
- **Deliverable:**
  - Automated `cibuildwheel` pipelines producing Linux and Windows wheels (with CPU fallback and
    CUDA wheel variants), verified in clean virtual environments.
  - Publish **`ayaka-llm 0.3.0`** (Scale-out + Native Kernel Engine + Windows Native).
- **Exit Check:** Fresh-env wheel installs pass on Linux & Windows; release notes backed by
  measured numbers.

## Risks

- Binding ABI fragility across PyTorch/CUDA upgrades.
- Lack of hosted GPU runners on Windows CI (requires explicit local GPU validation reports per
  `CONTRIBUTING.md`).
- Wheel size limits on PyPI when bundling multiple `sm_` architectures.
