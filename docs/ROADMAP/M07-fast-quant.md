# Fast quantized kernels: FP8 & W4A16 (`v0.2.0-a3`)

- **Goal:** Eliminate the `0.11–0.12×` FP8 decode bottleneck measured in models and quantization milestone (#121) and add fast
  4-bit weight execution (`W4A16`, prioritizing safetensors INT4/AWQ/GPTQ via Marlin/Triton) so 14B
  models run fast on 12–24 GB cards and 7B–8B models fit fast on 4–8 GB laptops — using existing
  GPU kernel libraries without requiring custom `csrc/` compilation yet.
- **Status:** Planned (not filed yet). It starts after the memory exit audit (memory step 10).
- **Target Checkpoint:** `v0.2.0-a3` (also triggers early parallel spike native kernel engine step 0 upon completion)
- **Hardware this round:** NVIDIA A100-SXM4-80GB (sm_80) is the development GPU. sm_80 has no FP8 tensor cores, so on A100 an FP8 checkpoint runs as weight-only FP8 (W8A16) kernels. Native FP8 GEMM remains an sm_89+ path (NVIDIA L4 and newer). The L4 gates in this file stay unverified until they run on an NVIDIA L4, and the RTX 3050 gates until they run on that card. No L4 or RTX 3050 result is claimed here.
- **Reference models this round:** `Qwen/Qwen3-8B` (bf16), with `Qwen/Qwen3-8B-FP8` and `Qwen/Qwen3-8B-AWQ` for quantized kernels and `z-lab/Qwen3-8B-DFlash-b16` for speculative decoding; `Qwen/Qwen3-0.6B` for fast tests.

## Exit Conditions

1. Fast FP8 GEMM runs inside CUDA graphs on NVIDIA L4 (`sm_89`) for both per-tensor/per-row
   (`compressed-tensors`) and 128×128 block-FP8 (`Qwen3-8B-FP8`) checkpoints, preserving slice-dequant
   fallback on older GPUs (`sm < 89`).
2. Fast safetensors W4A16 (INT4/AWQ/GPTQ Marlin/Triton) runs inside CUDA graphs on both L4 (`sm_89`)
   and RTX 3050 (`sm_86`), fitting a 7B–8B model on 6–8 GB (and 3B on 4 GB) with passing JSON +
   tool-call sanity checks.
3. `ayaka plan` accurately predicts weights + KV capacity for FP8 and W4A16 checkpoints.
4. All quality (KL), speed, and `llama.cpp` comparisons are recorded in `docs/EXPERIMENTS.md`,
   `docs/CAPABILITIES.md`, and `docs/TROUBLESHOOTING.md`.

## Pre-Registered Measurement Gates

- **FP8 quality:** First-token KL vs 16-bit `≤ 0.05 nats` (32 prompts, `quant_kl.py`).
- **FP8 speed (L4, 3 alternating rounds, `ayaka bench --quick`):**
  - Llama-3.1-8B-FP8 & Qwen3-8B-FP8 decode after 1K context **`≥ 1.2×`** of bf16 (up from
    `0.11–0.12×`).
  - `phi-4-FP8-dynamic` (14B) **`≥ 10 tok/s`** on L4 (up from `0.9 tok/s`).
- **W4A16 quality & speed:**
  - First-token KL vs 16-bit `≤ 0.10 nats` and `2/2` sanity checks passing.
  - 8B decode on L4 **`≥ 1.8×`** of bf16; runnable end-to-end on Tier 1 (4–8 GB) cards.

---

## Execution Nodes (5 Nodes)

### fast quantized kernels step 1 — Quantized GEMM Op Contracts & `sm_arch` Backend Dispatch
- **Status:** `BLOCKED`
- **Depends On:** memory step 10
- **Hardware:** `CPU`
- **Touches:** `python/ayaka/ops/`, `python/ayaka/layers/quantization/`, `python/ayaka/device/`, `tests/ops/`
- **Deliverable:**
  - Extend `ayaka.ops` linear/quantization contracts for per-tensor FP8, per-row FP8, 128×128
    block-FP8, and W4A16 group-quantized linear ops.
  - Capability-based backend dispatch selecting fast GPU kernels on capable `sm_arch` while keeping
    the existing slice-dequant fallback on CPU / `sm < 89`.
- **Exit Check:** Unit & reference-parity tests green on CPU; lazy-import boundary preserved
  (`import ayaka` loads no torch/triton).

### fast quantized kernels step 2 — Fast Per-Tensor / Per-Row FP8 GEMM (`sm_89`+)
- **Status:** `BLOCKED`
- **Depends On:** fast quantized kernels step 1
- **Hardware:** `GPU-L4`
- **Touches:** `python/ayaka/ops/backends/`, `python/ayaka/layers/quantization/`, `tests/ops/`
- **Deliverable:**
  - Wire `torch._scaled_mm` / FlashInfer FP8 GEMM for per-tensor and per-row FP8 checkpoints
    (`RedHatAI/Meta-Llama-3.1-8B-Instruct-FP8`, `RedHatAI/phi-4-FP8-dynamic`) with dynamic/static
    activation scaling and zero-allocation CUDA graph capture compatibility.
- **Exit Check:** Op parity vs dequant reference passes; CUDA graph capture & replay green on L4;
  Llama-3.1-8B-FP8 clears `≥ 1.2×` bf16 gate and Phi-4 14B clears `≥ 10 tok/s`.

### fast quantized kernels step 3 — Fast 128×128 Block-FP8 GEMM (`sm_89`+)
- **Status:** `BLOCKED`
- **Depends On:** fast quantized kernels step 1
- **Hardware:** `GPU-L4`
- **Touches:** `pyproject.toml`, `python/ayaka/ops/backends/triton/`, `python/ayaka/layers/quantization/`, `tests/ops/`
- **Deliverable:**
  - Add optional `triton` extra (Linux GPU, lazily imported) and/or FlashInfer block-scale FP8 kernel
    for Hugging Face 128×128 block-FP8 checkpoints (`Qwen/Qwen3-8B-FP8`) inside CUDA graphs.
- **Exit Check:** Block-FP8 op parity passes; `Qwen3-8B-FP8` replays CUDA graphs on L4 and clears
  `≥ 1.2×` bf16 with KL `≤ 0.05 nats`.

### fast quantized kernels step 4 — Fast Safetensors W4A16 (INT4 / AWQ / GPTQ Marlin) Loader & Kernel
- **Status:** `BLOCKED`
- **Depends On:** fast quantized kernels step 1
- **Hardware:** `GPU-L4` + `GPU-4GB-WSL2`
- **Touches:** `python/ayaka/weights/`, `python/ayaka/layers/quantization/`, `python/ayaka/ops/backends/`, `python/ayaka/memory/`, `tests/layers/`
- **Deliverable:**
  - Safetensors W4A16 loader (AWQ / GPTQ / compressed-tensors INT4, aligned with Ayaka's
    safetensors-only trust policy; direct GGUF `Q4_K_M` execution kept optional/deferred).
  - Marlin / Triton W4A16 linear kernel capturing cleanly inside CUDA graphs on `sm_86` and `sm_89`,
    with startup repacking outside graph capture and exact `ayaka plan` weight/KV sizing.
- **Exit Check:** Op parity + CUDA graph capture green on L4 (`sm_89`) and RTX 3050 (`sm_86`);
  `ayaka plan` matches startup reconciliation; 8B W4A16 clears `≥ 1.8×` bf16 on L4.

### fast quantized kernels step 5 — Quantization Benchmark Record, `llama.cpp` Comparison & `v0.2.0-a3`
- **Status:** `BLOCKED`
- **Depends On:** fast quantized kernels step 2, fast quantized kernels step 3, fast quantized kernels step 4
- **Hardware:** `GPU-L4` + `GPU-4GB-WSL2`
- **Touches:** `benchmarks/e2e/`, `docs/EXPERIMENTS.md`, `docs/capabilities.toml`, `docs/CAPABILITIES.md`, `docs/TROUBLESHOOTING.md`
- **Deliverable:**
  - 3-round alternating `ayaka bench` + `quant_kl.py` + head-to-head `llama.cpp` (`Q8_0` / `Q4_K_M`)
    runs on L4 and RTX 3050.
  - Update `docs/TROUBLESHOOTING.md` (retire the *"FP8 fits, but is slow"* limitation for `sm_89`+),
    regenerate `docs/CAPABILITIES.md`, tag `v0.2.0-a3`, and unblock parallel early spike native kernel engine step 0.
- **Exit Check:** All fast quantized kernels milestone exit conditions and pre-registered gates verified; `tests/test_docs.py`
  green; tag `v0.2.0-a3`.

## Risks

- 128×128 block-FP8 (`Qwen3-8B-FP8`) needing careful Triton tuning to beat bf16 at batch 1.
- W4A16 weight repacking latency at startup if not cached or kept cleanly off the critical path.
- Keeping `triton` strictly optional so CPU CI and Windows installs never break.
