# Engineering, Measurement, Safety, and Release Principles

## Engineering Discipline

1. **Capabilities, not PR counts.** A milestone is done only when every exit condition passes with
   reproducible commands — never simply because its PRs merged.
2. **One milestone at a time.** Avoid splitting focus across unrelated milestones. Parallel work is
   limited to tightly scoped preparatory experiments (such as the 2-kernel CUDA graph spike).
3. **Small PRs with real tests.** One problem per PR; every bug fix adds a test that fails without
   it; docs and capability tables update in the same PR as the code.
4. **Strict layering & lazy imports.**
   - Every package `__init__.py` uses `build_lazy_exports`; `import ayaka` never imports `torch`,
     NVML, Hugging Face Hub, `aiohttp`, `llguidance`, or tokenizer adapters.
   - One defining module owns each public name (`tests/test_public_api.py` enforces this via AST).
   - Entrypoint modules stay bounded in size (<700 lines per #78/#82) behind a frozen `EngineClient`
     server-engine seam (#89).
   - Python 3.13 with `pyright` strict mode across `python/ayaka/`.
5. **Minimal unconditional dependencies.** Only `numpy` and `prometheus-client` sit in
   `[project].dependencies`. PyTorch (`cpu` vs `cu130`), `flashinfer`, `hf`, `tiktoken`, `serve`,
   and `structured` remain explicit extras.

## Measurement & Decision-Gate Discipline

Every performance optimization, default change, or quantization trade-off obeys four rules:

1. **Pre-register before running.** Write the exact workload, commands, machine, alternation order
   (e.g., `A B B A`), and pass/fail threshold into the tracking issue or `docs/EXPERIMENTS.md`
   *before* collecting numbers.
2. **New features are opt-in by default.** A feature becomes a default only in a dedicated PR backed
   by before/after measurements on the same machine showing it clears its pre-registered gate without
   unacceptable regression.
3. **Never hide invalid or missed runs.** Over-committed VRAM paging (#59, #73), unstable eager
   metrics (#73), or missed gates (#95 detokenizer seed end-to-end ratio `1.029` vs `≤ 1.0`, #99
   FP8 KV latency `1.28–1.50×`, #121 FP8 speed `0.11–0.12×`) are recorded openly as findings rather
   than edited out.
4. **Separate compilation, CPU checks, and GPU runs.** Hosted CI has no GPU. Every PR and release
   note reports `passed`, `failed`, and `not run` separately, naming GPU model, driver, CUDA
   version, and revision.

## Safety, Privacy, and Trust Posture

Derived from `docs/SAFETY.md` and non-negotiable across all future milestones (including MCP in M10):

- **Zero telemetry**: no phone-home analytics, crash dumps, or silent update checks.
- **Local-by-default**: `ayaka serve` binds `127.0.0.1:8000`; binding `0.0.0.0` without an API key
  prints a loud warning; `--api-key-file` keeps secrets out of process lists and logs.
- **No accidental content retention**: logs and `/metrics` never store prompt/output text;
  `--record` is opt-in and creates owner-only files with explicit warnings.
- **Strict checkpoint trust**: weights load from safetensors (and verified GGUF readers where
  explicitly enabled) only — pickle formats (`.bin`, `.pt`) are refused; checkpoint repo code
  execution stays `strict` (disabled) unless explicitly enabled by the user.

## Release & Distribution Strategy

- **Package naming**: PyPI distribution `ayaka-llm`, Python import `ayaka`, CLI `ayaka`.
- **Milestone-driven releases**: Release numbers are not separate backlog tasks; a release note is
  written when a milestone's exit conditions pass, citing one measured sentence with machine, model,
  and version.
- **Release Ladder (`0.1.0` -> `0.2.0` -> `0.3.0` -> `0.4.0` -> `1.0.0`)**:
  - **`0.1.0` (Current — Preview)**: Tagged on GitHub / installed from checkout (`uv sync`). Not
    published to PyPI. Proves end-to-end serving, coding-tool integration, dense/MoE models, and
    FP8 correctness (M0–M3).
  - **`0.2.0-a1` .. `0.2.0-a7` (Internal Milestone Checkpoints)**: Internal pre-release tags after
    each milestone passes its exit conditions (`a1` M4 Speculative, `a2` M5 Serving Core v2, `a3`
    M6 Memory, `a4` M7 Fast Quant, `a5` M8 Hybrid/Vision, `a6` M9 MoE Small Machines, `a7` M10
    Agent Platform) to anchor regression tracking and `benchmarks/history/`.
  - **`0.2.0` (First Public PyPI Launch — M4 through M11 Complete)**: Published to PyPI
    (`ayaka-llm`) and announced publicly once **M11 (Frontier MoE)** passes. Delivers the complete
    single-machine coding-agent stack: exact-output speculation, request-scoped render/parsers &
    gRPC, OOM-resilient host KV memory, fast FP8/W4A16, hybrid/state/vision, RAM-offloaded MoE,
    core agent platform/MCP, and Frontier MoE on 8–64 GB.
  - **`0.3.0` (M12 + M13)**: Multi-GPU scale-out (TP/PP/EP, P/D split, `Kimi K3`) + native C++/CUDA
    kernel engine (`kops`/`csrc/`) & native Windows wheels without WSL2.
  - **`0.4.0` (M14)**: Multi-agent & distributed prefix/KV/state fabric, swarm QoS lanes, and
    opt-in persistent session/KV snapshots.
  - **`1.0.0` GA (M15)**: Omni-modal computer-use streaming (screen/video/audio), frozen public
    Python/Wire/Plugin ABI with SemVer guarantees, and full security/privacy audit.
