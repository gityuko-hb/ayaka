# M15 — Omni-Modal Computer-Use & `1.0.0` GA

- **Goal:** Complete Ayaka's long-horizon vision as the reference local inference engine for
  next-generation **computer-use & omni-modal coding agents** (real-time screen/video/audio
  streaming with incremental visual KV reuse), freeze the public Python/Wire/Plugin ABI with
  SemVer guarantees, complete a full security/privacy audit, and ship **`ayaka-llm 1.0.0` GA**.
- **Status:** Planned (long-horizon flagship, post-`0.4.0`)
- **Target Release:** **`ayaka-llm 1.0.0` GA**

## Exit Conditions

1. Streaming screen/video & audio inputs run end to end with incremental visual frame diffing /
   KV reuse and measured latency budgets for real-time computer-use loops.
2. Public Python API (`ayaka`), HTTP/MCP wire contracts, and external `ops`/model plugin ABI are
   frozen, documented, and covered by backward-compatibility contract tests.
3. Full security, privacy, and supply-chain audit (MCP controls, persistent sessions, native
   `csrc/` memory safety, safetensors/GGUF parsers, signed wheel artifacts) completes with zero
   open critical/high findings.
4. Multi-tier `docs/CAPABILITIES.md` covers all 3 hardware tiers (`4–8 GB`, `12–24 GB`,
   `32–64 GB+` & multi-GPU) on both Linux and native Windows, backed 100% by raw `results.json`.
5. **`ayaka-llm 1.0.0` GA** is published to PyPI with reproducible signed wheels.

---

## Execution Nodes (5 Nodes)

### `M15-N01` — Screen / Video & Audio Streaming Inputs with Incremental Visual KV Reuse
- **Status:** `BLOCKED`
- **Depends On:** `M8-N06`, `M14-N05`
- **Hardware:** `GPU`
- **Touches:** `python/ayaka/entrypoints/wire/`, `python/ayaka/models/`, `python/ayaka/kvcache/`, `tests/e2e/`
- **Deliverable:**
  - Streaming multimodal ingestion for continuous IDE/browser screen captures and audio inputs,
    with incremental patch/frame diffing so unchanged screen regions reuse cached visual KV blocks.
- **Exit Check:** Parity with reference multimodal processors verified; incremental frame reuse
  measurably cuts per-frame prefill time on static-background IDE recordings.

### `M15-N02` — Real-Time Computer-Use Action Loop Benchmark
- **Status:** `BLOCKED`
- **Depends On:** `M15-N01`
- **Hardware:** `GPU`
- **Touches:** `benchmarks/e2e/`, `python/ayaka/sampling/structured/`, `docs/EXPERIMENTS.md`
- **Deliverable:**
  - End-to-end computer-use agent benchmark: live screenshot stream -> grammar-constrained GUI /
    tool action emitted inside CUDA graphs, measuring frame-to-action p50/p99 latency.
- **Exit Check:** Reproducible benchmark committed to `docs/EXPERIMENTS.md` with 100% schema-valid
  action output.

### `M15-N03` — Freeze Public Python API, Wire Contracts & Plugin ABI (SemVer)
- **Status:** `BLOCKED`
- **Depends On:** `M14-N05`
- **Hardware:** `CPU`
- **Touches:** `python/ayaka/`, `docs/AI_REFERENCE.md`, `tests/test_public_api.py`
- **Deliverable:**
  - Audit and freeze public Python symbols, HTTP/SSE/MCP wire contracts, and external `ops`/model
    backend plugin registration hooks under an explicit SemVer deprecation policy.
- **Exit Check:** Automated API/wire compatibility snapshot tests pass; zero undocumented public
  exports.

### `M15-N04` — Comprehensive Security, Privacy & Supply-Chain Audit
- **Status:** `BLOCKED`
- **Depends On:** `M15-N03`
- **Hardware:** `CPU` + `GPU`
- **Touches:** `csrc/`, `python/ayaka/weights/`, `python/ayaka/entrypoints/mcp/`, `docs/SAFETY.md`, `.github/workflows/`
- **Deliverable:**
  - Threat-model review and fuzzing of native `csrc/` memory boundaries, safetensors/GGUF parsers,
    MCP auth/loopback isolation, opt-in persistent KV encryption, and signed reproducible wheel
    builds (SLSA / attestations).
- **Exit Check:** Audit report closed with zero open critical/high vulnerabilities; fuzz tests
  integrated in CI.

### `M15-N05` — Multi-Tier Verified `CAPABILITIES.md` & Publish `ayaka-llm 1.0.0` GA
- **Status:** `BLOCKED`
- **Depends On:** `M15-N02`, `M15-N04`
- **Hardware:** `All Tiers (4–8 GB, 12–24 GB, 32–64 GB+, Multi-GPU, Linux & Windows Native)`
- **Touches:** `docs/capabilities.toml`, `docs/CAPABILITIES.md`, `README.md`, `python/ayaka/__version__.py`, `CHANGELOG.md`
- **Deliverable:**
  - Refresh full capability matrix across all hardware tiers and OS targets, verify clean installs,
    and publish **`ayaka-llm 1.0.0` GA** with measured release notes.
- **Exit Check:** All M15 exit conditions green; `ayaka-llm 1.0.0` live on PyPI and GitHub Releases.

## Risks

- High-frequency visual frame streaming competing with text KV cache on Tier 1/2 GPUs.
- Premature ABI freeze locking in an inflexible kernel/plugin interface before real third-party
  adoption.
- Security regressions in native `csrc/` parsers or MCP control endpoints.
