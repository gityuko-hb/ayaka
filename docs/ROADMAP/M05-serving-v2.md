# Serving Core v2: Render, Parsers, Structured Output (XGrammar) & gRPC (`v0.2.0-a1`)

- **Goal:** Upgrade Ayaka's serving layer from server-static parsing and single-backend JSON
  constraints into a decoupled, request-scoped serving core: a dedicated Jinja-aware **Prompt
  Render Engine**, **request-scoped output interpretation** (`none` & `thinking`), multi-family
  **reasoning & XML/JSON tool-call parsers**, a pluggable **Structured Output v2** layer
  (`llguidance` + **`xgrammar`**, 6 constraint kinds including `STRUCTURAL_TAG`, plus `rollback` /
  `validate_tokens` as the grammar contract that speculative decoding uses later), and **dual gRPC services** (Engine/Inference
  seam + Engine-free Render/Parse) — all in pure Python 3.13 with lazy imports and optional extras.
- **Status:** Next (not filed yet)
- **Target Checkpoint:** `v0.2.0-a1` (internal tag on the road to `0.2.0` GA)
- **Order:** First milestone of the round (Serving Core v2, memory, fast quantized kernels, speculative decoding). It starts on `main`, where the models and quantization milestone is done.
- **Hardware this round:** NVIDIA A100-SXM4-80GB (sm_80) is the development GPU. Gates written for the NVIDIA L4 stay unverified until they run on an NVIDIA L4; no L4 result is claimed in this file.
- **Reference models this round:** `Qwen/Qwen3-8B` (bf16), with `Qwen/Qwen3-8B-FP8` and `Qwen/Qwen3-8B-AWQ` for quantized kernels and `z-lab/Qwen3-8B-DFlash-b16` for speculative decoding; `Qwen/Qwen3-0.6B` for fast tests.
- **Reference Architectures Researched (`2026-11-26`):**
  - SGLang `rust/sglang-processor/src/{render,think,tool_call,parser,openai}`, `rust/sglang-grpc`,
    and `python/sglang/srt/constrained/`
  - vLLM `rust/src/parser/src/{tool,reasoning,unified,output_grammar}`, `rust/src/chat`,
    `rust/src/server/src/grpc`, and `vllm/v1/structured_output/{backend_types,backend_xgrammar}.py`

## Exit Conditions

1. **Request-scoped thinking (`none` & `thinking`)**: On one running `ayaka serve` instance
   (e.g., `Qwen3-8B`), Request A with `reasoning_effort="none"` / `thinking: {type: "disabled"}`
   disables thinking in prompt render, parser (`force_reasoning=False`), and grammar constraint
   start, while Request B with `reasoning_effort="high"` / `thinking: {type: "enabled"}` streams
   separated `reasoning_content` / `thinking_delta` + final content/tool calls cleanly.
2. **Multi-family parser parity**: JSON (`hermes`, `llama3_json`, `qwen3_json`, `deepseek_json`),
   XML/tag (`qwen3_coder`, `mimo`, `glm_xml`, `minimax_m2`/`m3`, `deepseek_dsml`), and reasoning
   (`think`, `deepseek_r1`, `deepseek_v3`, `glm45`, `minimax_m3`) all pass byte/random-split
   invariance + `ToolSchemas` coercion suites, with live L4 verification on `Qwen3-8B` and
   `MiMo-7B-RL`.
3. **Structured Output v2 & XGrammar**:
   - All 6 `ConstraintKind` values (`JSON_SCHEMA`, `JSON_OBJECT`, `REGEX`, `CHOICE`, `GRAMMAR`,
     `STRUCTURAL_TAG`) + `--tool-strict-level` (`auto|function|parameter`) work on both
     `LLGuidanceBackend` (default) and `XGrammarBackend` (`--extra xgrammar`), including XML tool
     calls constrained via `STRUCTURAL_TAG` inside CUDA graphs.
   - `validate_tokens` + `rollback(k)` are delivered with their own tests as the grammar contract for speculative decoding. Wiring them into the speculative verify windows is part of the speculative decoding milestone.
4. **gRPC Engine & Render decoupling**:
   - `ayaka serve --engine-grpc <addr>` drives a remote GPU worker over `GrpcEngineClient`
     (satisfying `EngineClient` in `python/ayaka/engine/seam.py`) with all OpenAI & Anthropic
     endpoints and 7/7 `ayaka connect --check` passing identically to in-process mode.
   - `ayaka render-server` serves `RenderChat` / `ParseOutput` over gRPC/UDS on a CPU-only
     environment without importing `torch`.
5. **Lazy-import & overhead gates**: `import ayaka` loads none of `torch`, `llguidance`,
   `xgrammar`, or `grpcio`; gRPC and XGrammar-vs-LLGuidance benchmarks are recorded in
   `docs/EXPERIMENTS.md`.

## Pre-Registered Measurement Gates

- **XGrammar vs LLGuidance default gate (L4, 5 alternating rounds, Serving Core v2 step 12):**
  - Measure: (1) schema compile latency (cold/warm), (2) per-step CUDA-graph mask overhead at
    batch 1, 8, and 32.
  - Rule: Keep `llguidance` as default (zero-dep, Linux/Windows/macOS). Switch Linux CUDA default
    to `xgrammar` in a dedicated PR **only if** `xgrammar` wins by **`≥ 15%`** on runtime mask /
    constrained-decoding throughput (or uniquely supports the `rollback(k)` contract where `llguidance` cannot)
    **without** unacceptable cold schema-compile regression.
- **gRPC Engine seam overhead (L4, UDS & loopback TCP vs in-process `AsyncLLM`, 5 alternating rounds):**
  - Added per-token streaming latency p50 **`≤ 0.3 ms`**.
  - End-to-end tok/s delta **`≤ 2%`** (ratio `≥ 0.98×`).
- **Python vs Rust decision rule**: Keep Render + Parsers + gRPC in pure Python 3.13 unless
  concurrent serving benchmarks prove CPU parse/render exceeds `5%` of GPU step time (consistent
  with serving and hardening milestone #77/#90).

---

## Execution Nodes (13 Nodes)

### Wave 1 — Prompt Render Engine & Request-Scoped Wire Contracts

#### Serving Core v2 step 1 — Prompt Render Layer & Jinja `ThinkingPolicy` Detection
- **Status:** `BLOCKED`
- **Depends On:** `main` (models and quantization milestone done)
- **Hardware:** `CPU`
- **Touches:** `python/ayaka/entrypoints/render/`, `python/ayaka/entrypoints/conversation.py`, `python/ayaka/tokenizers/`, `tests/entrypoints/render/`
- **Deliverable:**
  - Create dedicated `ayaka.entrypoints.render` package (`thinking.py`, `reasoning.py`, `selection.py`,
    `models/`), decoupled from GPU execution.
  - Inspect Jinja chat templates via lexer/AST (`detect_thinking_policy`, ignoring comments, `raw`
    blocks, and string literals as in SGLang `render/thinking.rs`) to classify templates into:
    `Always`, `TemplateToggle(key="enable_thinking"|"thinking", default_enabled)`,
    `NativeToggle(default_enabled, named_tool_disables)`, `ReasoningEffort`, or `Unknown`.
  - Model-specific render hooks (Qwen3/3.8, MiMo, DeepSeek, GLM, MiniMax) and structured output
    `RenderedChat(prompt_text, token_ids, thinking_enabled, starts_inside_reasoning, render_fingerprint)`.
- **Exit Check:** Unit test matrix covering real Jinja templates (Qwen3, DeepSeek-R1, Llama-3.x,
  MiMo, GLM, MiniMax, comments/raw-block edge cases) passes 100% on CPU.

#### Serving Core v2 step 2 — OpenAI & Anthropic Entrypoints: Request-Scoped Thinking, `chat_template_kwargs` & Prefill
- **Status:** `BLOCKED`
- **Depends On:** Serving Core v2 step 1
- **Hardware:** `CPU`
- **Touches:** `python/ayaka/entrypoints/wire/policy.py`, `python/ayaka/entrypoints/openai/`, `python/ayaka/entrypoints/anthropic/`, `docs/AI_REFERENCE.md`, `tests/entrypoints/`
- **Deliverable:**
  - Promote `reasoning_effort` (`none|minimal|low|medium|high|max`), Responses `reasoning`
    (`effort`/`enabled`), Anthropic `thinking` (`{type: "disabled"|"enabled", budget_tokens}`),
    `chat_template_kwargs`, `continue_final_message`, `developer` role mapping, and historical
    assistant `reasoning_content` turns from `HINT`/ignored to real `READ` inputs resolved by
    `ayaka.entrypoints.render`.
  - Regenerate `docs/AI_REFERENCE.md` (`uv run python tools/generate_wire_reference.py`) and extend
    golden-byte wire locks for OpenAI Chat, OpenAI Responses, and Anthropic Messages.
- **Exit Check:** Golden-byte wire tests and `tests/test_docs.py` wire-reference drift check pass.

#### Serving Core v2 step 3 — Request-Scoped Output Interpretation (`none` + `thinking`) & Grammar Start Sync
- **Status:** `BLOCKED`
- **Depends On:** Serving Core v2 step 1, Serving Core v2 step 2
- **Hardware:** `CPU` + `GPU-L4`
- **Touches:** `python/ayaka/entrypoints/parsers/output.py`, `python/ayaka/entrypoints/parsers/unified.py`, `python/ayaka/entrypoints/serving.py`, `tests/entrypoints/parsers/`
- **Deliverable:**
  - Replace server-static parser instantiation with request-scoped `ChatResponseProcessor` /
    `OutputInterpretation` taking `RenderedChat` metadata:
    - `force_reasoning: bool | None`: `False` = **`none`** (disables reasoning parsing completely
      for this request), `True` = starts inside thinking, `None` = auto/tag-driven.
    - `previous_content` (for `continue_final_message` / assistant prefill) and `stream_reasoning`.
  - Introduce `UnifiedParser` + `CombinedParser` interface emitting transport-neutral semantic
    events (`Role`, `TextDelta`, `ReasoningDelta` with exact token-attribution excluding marker
    tokens, `ToolCall*`, `Usage`, `Finish`) consumed identically by OpenAI, Anthropic, and gRPC.
  - Synchronize `constraint_start()` with request-scoped `thinking_enabled` /
    `starts_inside_reasoning` so constraints start immediately at token 0 whenever thinking is
    disabled (`reasoning_effort="none"`, `thinking.type="disabled"`, or `named_tool_disables`).
- **Exit Check:** Request A (`none`) and Request B (`thinking`) served concurrently on one server
  produce exact separated/unseparated streams and zero grammar deadlocks.

---

### Wave 2 — Parsers (`reasoning`, `think`, `tool_call`) & Structured Output v2 (`XGrammar`)

#### Serving Core v2 step 4 — Enhanced `reasoning_parser` & `think` Detectors (`tool_start` Recovery & Model Families)
- **Status:** `BLOCKED`
- **Depends On:** Serving Core v2 step 3
- **Hardware:** `CPU`
- **Touches:** `python/ayaka/entrypoints/parsers/reasoning.py`, `python/ayaka/entrypoints/parsers/formats.py`, `tests/entrypoints/parsers/`
- **Deliverable:**
  - Enhance `TagConfig` / `ReasoningParser` with `tool_start` / `tool_start_at_line_start`
    recovery (auto-closing unclosed thinking when a tool-call opening tag appears),
    `ReasoningDefault` (`on_unless` vs `off_unless`), `previous_content`, and
    `force_nonempty_content`.
  - Support reasoning formats: `think` (enhanced), `deepseek_r1`, `deepseek_v3`, `glm45`, and
    `minimax_m3` (plus clean `UnifiedParser` extension hook for `gemma4` in hybrid, state and vision milestone and `kimi_k3` in
    scale-out milestone).
- **Exit Check:** Streaming vs one-shot split-invariance test matrix (1-char, random splits,
  multi-byte UTF-8, unclosed `</think>` before `<tool_call>`) passes 100%.

#### Serving Core v2 step 5 — Tool Parser Core: `ToolSchemas` Type Coercion, JSON Family & Grammar Spec Sync
- **Status:** `BLOCKED`
- **Depends On:** Serving Core v2 step 3
- **Hardware:** `CPU`
- **Touches:** `python/ayaka/entrypoints/parsers/parameters.py`, `python/ayaka/entrypoints/parsers/tool_calls.py`, `python/ayaka/entrypoints/parsers/output_grammar.py`, `tests/entrypoints/parsers/`
- **Deliverable:**
  - Add `parameters.py` (`ToolSchemas`, typed conversion for integer/number/boolean/JSON scalar
    strings emitted by XML/tag parsers while preserving valid string fields).
  - Create `output_grammar.py` (`OutputGrammarContext`, `GrammarCoverage`: `FromTokenZero` vs
    `FinalOutputOnly`, `ToolStrictLevel`: `auto|function|parameter`) so every tool parser emits its
    own normalized constraint spec.
  - Complete JSON tool-call family (`hermes`, `llama3_json`, `qwen3_json`, `deepseek_json`) and
    upgrade `suggest_tool_call_parser` auto-detection.
- **Exit Check:** Unit tests for JSON streaming deltas, schema coercion, and parser-to-grammar spec
  parity pass 100%.

#### Serving Core v2 step 6 — XML & Tag-Based Tool Call Parsers (`qwen3_coder`, `mimo`, `glm_xml`, `minimax_m2/m3`, `deepseek_dsml`)
- **Status:** `BLOCKED`
- **Depends On:** Serving Core v2 step 4, Serving Core v2 step 5
- **Hardware:** `CPU` + `GPU-L4`
- **Touches:** `python/ayaka/entrypoints/parsers/tool_calls/`, `python/ayaka/entrypoints/parsers/formats.py`, `tests/entrypoints/parsers/`, `tests/e2e/`
- **Deliverable:**
  - Implement incremental streaming + one-shot parsers for XML/tag families: `qwen3_coder`
    (`<function=name><parameter=k>v</parameter></function>`), `mimo`, `glm_xml`
    (`<arg_key>/<arg_value>`), `minimax_m2`/`m3`, and `deepseek_dsml` (`<｜DSML｜...>`).
- **Exit Check:** All 5 XML/tag families pass split-invariance + `ToolSchemas` coercion suites;
  live L4 tool-call sanity checks pass on `Qwen3-8B` and `MiMo-7B-RL`.

#### Serving Core v2 step 7 — Structured Output v2 Contract (`6 ConstraintKind`, `StructuralTag`, `rollback` & Guided Speculation)
- **Status:** `BLOCKED`
- **Depends On:** Serving Core v2 step 5
- **Hardware:** `CPU` + `GPU-L4`
- **Touches:** `python/ayaka/core/constraints.py`, `python/ayaka/sampling/structured/`, `python/ayaka/execution/runner.py`, `python/ayaka/entrypoints/wire/policy.py`, `docs/AI_REFERENCE.md`, `tests/sampling/`
- **Deliverable:**
  - Expand `ConstraintKind` from 3 to **6 kinds**: `JSON_SCHEMA`, `JSON_OBJECT`, `REGEX`, `CHOICE`,
    `GRAMMAR` (EBNF/Lark, supporting custom tool grammars), and `STRUCTURAL_TAG`.
  - Add `--tool-strict-level` (`auto|function|parameter`) and expose `regex` / `choice` / `grammar`
    / `structural_tag` cleanly through `wire/policy.py`.
  - Introduce pluggable `StructuredOutputBackend` & `StructuredOutputGrammar` contracts
    (`accept_tokens`, `validate_tokens`, `rollback(num_tokens)`, `fill_bitmask`, `is_terminated`,
    `reset`) and refactor existing `llguidance` into `LLGuidanceBackend` (preserving default).
  - Deliver `validate_tokens` + `rollback(k)` as the grammar contract for speculative decoding, with its own unit tests. Wiring it into the speculative verify windows is part of the speculative decoding milestone.
- **Exit Check:** All 6 `ConstraintKind` unit tests pass on `LLGuidanceBackend`; `rollback(k)` restores the grammar state exactly, tested against replayed token sequences.

#### Serving Core v2 step 8 — XGrammar Backend (`--extra xgrammar`) & XML/JSON `StructuralTag` CUDA-Graph Sync
- **Status:** `BLOCKED`
- **Depends On:** Serving Core v2 step 6, Serving Core v2 step 7
- **Hardware:** `CPU` + `GPU-L4`
- **Touches:** `pyproject.toml`, `uv.lock`, `python/ayaka/sampling/structured/xgrammar_backend.py`, `python/ayaka/entrypoints/parsers/output_grammar.py`, `tests/sampling/`
- **Deliverable:**
  - Add optional `xgrammar` extra in `pyproject.toml`, loaded lazily via `require_module`
    (`--structured-backend llguidance|xgrammar`, keeping `llguidance` as default pending Serving Core v2 step 12).
  - Implement `XGrammarBackend` applying the two critical vLLM production patterns:
    1. **Dual compilers**: `json_compiler` (special tokens filtered out for JSON) and
       `full_vocab_compiler` (special tokens enabled for `STRUCTURAL_TAG`).
    2. Accurate `TokenizerInfo` (handling BPE, tiktoken, and SentencePiece prefix-space edge cases),
       `disable_any_whitespace` + explicit `separators`, native `rollback(num_tokens)`, and
       `fill_next_token_bitmask` feeding directly into Ayaka's captured CUDA-graph bitmask op.
  - Connect every JSON & XML tool parser from Serving Core v2 step 5/Serving Core v2 step 6 (`hermes`, `llama3_json`,
    `qwen3_coder`, `mimo`, `glm_xml`, `minimax`, `deepseek_dsml`) to the `STRUCTURAL_TAG` builder so
    XML tool calls are fully grammar-constrained inside CUDA graphs.
- **Exit Check:** All 6 `ConstraintKind` + XML `STRUCTURAL_TAG` + `rollback(k)` tests
  pass on `XGrammarBackend` inside CUDA graphs on L4; `import ayaka` succeeds without `xgrammar`.

---

### Wave 3 — gRPC Transport (Engine Seam & Engine-Free Render)

#### Serving Core v2 step 9 — Protobuf Schemas & Optional `grpc` Extra
- **Status:** `BLOCKED`
- **Depends On:** Serving Core v2 step 3, Serving Core v2 step 7
- **Hardware:** `CPU`
- **Touches:** `proto/ayaka/v1/inference.proto`, `proto/ayaka/v1/render.proto`, `python/ayaka/entrypoints/grpc/`, `pyproject.toml`, `uv.lock`, `tests/entrypoints/grpc/`
- **Deliverable:**
  - Define versioned protobuf contracts:
    - `inference.proto`: `Generate` (server-streaming token-in/token-out carrying full 6-kind
      `StructuredOutput` spec), `Abort`, `AbortAll`, `Health`, `GetEngineInfo`, `WatchEngineState`.
    - `render.proto`: `RenderChat` (returns `input_ids`, `structured_output`,
      `engine_reasoning_gate`, `render_fingerprint`) and `ParseOutput` (streaming token deltas ->
      semantic `ChatEvent`).
  - Default `64 MiB` max message size; add optional `grpc` extra (`grpcio`, `protobuf`) loaded
    lazily via `require_module`.
- **Exit Check:** Protobuf round-trip unit tests pass; `import ayaka` succeeds without `grpcio`
  installed (`tests/test_public_api.py` green).

#### Serving Core v2 step 10 — Engine / Inference gRPC Server & Remote `GrpcEngineClient`
- **Status:** `BLOCKED`
- **Depends On:** Serving Core v2 step 9
- **Hardware:** `CPU` + `GPU-L4`
- **Touches:** `python/ayaka/entrypoints/grpc/inference.py`, `python/ayaka/entrypoints/grpc/client.py`, `python/ayaka/entrypoints/serve.py`, `python/ayaka/entrypoints/cli.py`, `tests/entrypoints/grpc/`
- **Deliverable:**
  - `InferenceService` serving any `EngineClient` over gRPC (TCP loopback or Unix Domain Socket)
    with cancellation propagation, graceful shutdown drain, and `EngineStateSnapshot` watch.
  - `GrpcEngineClient` implementing the exact `EngineClient` Protocol (`python/ayaka/engine/seam.py`),
    enabling `ayaka serve --engine-grpc <addr>` to decouple HTTP gateway from GPU worker with zero
    protocol-adapter changes.
- **Exit Check:** Seam contract & cancellation tests pass over UDS and loopback TCP; full OpenAI +
  Anthropic serving suite passes against remote `GrpcEngineClient`.

#### Serving Core v2 step 11 — Engine-Free `Render` & `Parse` gRPC Service (`ayaka render-server`)
- **Status:** `BLOCKED`
- **Depends On:** Serving Core v2 step 9, Serving Core v2 step 6
- **Hardware:** `CPU`
- **Touches:** `python/ayaka/entrypoints/grpc/render.py`, `python/ayaka/entrypoints/cli.py`, `tests/entrypoints/grpc/`
- **Deliverable:**
  - Standalone `ayaka render-server MODEL` exposing `RenderChat` and `ParseOutput` over gRPC/UDS
    with `x-render-fingerprint` metadata validation — zero GPU and zero `torch` required.
- **Exit Check:** Runs in a no-torch / `--extra cpu --extra grpc --extra hf` environment;
  `RenderChat` output matches in-process `Serving` token IDs and constraints bit-for-bit.

#### Serving Core v2 step 12 — Benchmarks (`XGrammar` vs `LLGuidance` & gRPC Overhead), Live Agent Verification & `v0.2.0-a1`
- **Status:** `BLOCKED`
- **Depends On:** Serving Core v2 step 8, Serving Core v2 step 10, Serving Core v2 step 11
- **Hardware:** `GPU-L4`
- **Touches:** `benchmarks/e2e/`, `docs/EXPERIMENTS.md`, `docs/capabilities.toml`, `docs/CAPABILITIES.md`, `docs/AI_REFERENCE.md`, `docs/ROADMAP/decisions.md`
- **Deliverable:**
  - **Pre-registered Benchmark 1 — `XGrammar` vs `LLGuidance` on L4**: cold/warm compile latency,
    CUDA-graph per-step mask overhead (batch 1/8/32), XML `STRUCTURAL_TAG`; record default-backend verdict in `docs/EXPERIMENTS.md` & `decisions.md`.
  - **Pre-registered Benchmark 2 — gRPC Engine seam overhead on L4**: UDS & loopback TCP vs
    in-process `AsyncLLM` (`≤ 0.3 ms` p50/token, `≤ 2%` tok/s).
  - **Live L4 coding-agent verification**: `Qwen3-8B` with `reasoning_effort="none"` vs `"high"`
    (OpenAI/Codex) and `thinking: disabled` vs `enabled` (Anthropic/Claude Code), `MiMo-7B-RL`
    constrained XML tool calls, and 7/7 `ayaka connect --check` over both in-process and
    `--engine-grpc` modes; tag `v0.2.0-a1`.
- **Exit Check:** All 5 Serving Core v2 milestone exit conditions and both pre-registered gates verified;
  `tests/test_docs.py` & CI green; tag `v0.2.0-a1`.

### Wave 4 — Tool Choice (moved from speculative decoding)

#### Serving Core v2 step 13 — Keep Tools in the Prompt for `tool_choice="none"`
- **Status:** `READY`
- **Depends On:** None
- **Hardware:** `CPU`
- **Touches:** `python/ayaka/entrypoints/`, `python/ayaka/sampling/structured/`, `tests/entrypoints/`
- **Deliverable:**
  - Keep tool schemas rendered in the prompt when `tool_choice="none"` so switching tool choice mid-session preserves prefix-cache blocks, while disallowing tool-call emission cleanly (PR #134, moved here on 2026-11-27).
  - Golden-byte wire test + prefix-cache hit unit test verifying `cached_tokens > 0` across `tool_choice="auto"` -> `"none"`.
- **Exit Check:** Golden-byte and prefix-cache unit tests pass; CI green.

## Risks

- XGrammar special-token handling edge cases between JSON and `STRUCTURAL_TAG` (mitigated upfront
  by vLLM's dual-compiler pattern and token-boundary parity tests).
- Subtle Jinja template quirks across model families causing `detect_thinking_policy` false
  positives/negatives (mitigated by lexer-level inspection and real template fixture tests).
- Grammar state desynchronizing after `rollback(k)` (mitigated by replay tests that compare each rollback with the state before the drafted tokens; speculative verify-window tests live in the speculative decoding milestone).
- `grpcio` asyncio event-loop contention under concurrent bursts (mitigated by L4 overhead gate in
  Serving Core v2 step 12 and UDS transport).
