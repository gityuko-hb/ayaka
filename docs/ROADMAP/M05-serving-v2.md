# M5 — Serving Core v2: Render, Parsers, Structured Output (XGrammar) & gRPC (`v0.2.0-a2`)

- **Goal:** Upgrade Ayaka's serving layer from server-static parsing and single-backend JSON
  constraints into a decoupled, request-scoped serving core: a dedicated Jinja-aware **Prompt
  Render Engine**, **request-scoped output interpretation** (`none` & `thinking`), multi-family
  **reasoning & XML/JSON tool-call parsers**, a pluggable **Structured Output v2** layer
  (`llguidance` + **`xgrammar`**, 6 constraint kinds including `STRUCTURAL_TAG`, plus `rollback` /
  `validate_tokens` for Guided Speculative Decoding), and **dual gRPC services** (Engine/Inference
  seam + Engine-free Render/Parse) — all in pure Python 3.13 with lazy imports and optional extras.
- **Status:** Next after M4 Speculative (not filed yet)
- **Target Checkpoint:** `v0.2.0-a2` (internal tag on the road to `0.2.0` GA)
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
   - `validate_tokens` + `rollback(k)` enable **Guided Speculative Decoding** with exact-output
     equivalence (M4 + M5 working together).
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

- **XGrammar vs LLGuidance default gate (L4, 5 alternating rounds, `M5-N12`):**
  - Measure: (1) schema compile latency (cold/warm), (2) per-step CUDA-graph mask overhead at
    batch 1, 8, and 32, and (3) Guided Speculative Decoding tok/s with rollback.
  - Rule: Keep `llguidance` as default (zero-dep, Linux/Windows/macOS). Switch Linux CUDA default
    to `xgrammar` in a dedicated PR **only if** `xgrammar` wins by **`≥ 15%`** on runtime mask /
    guided-speculative throughput (or uniquely unlocks guided speculation where `llguidance` cannot)
    **without** unacceptable cold schema-compile regression.
- **gRPC Engine seam overhead (L4, UDS & loopback TCP vs in-process `AsyncLLM`, 5 alternating rounds):**
  - Added per-token streaming latency p50 **`≤ 0.3 ms`**.
  - End-to-end tok/s delta **`≤ 2%`** (ratio `≥ 0.98×`).
- **Python vs Rust decision rule**: Keep Render + Parsers + gRPC in pure Python 3.13 unless
  concurrent serving benchmarks prove CPU parse/render exceeds `5%` of GPU step time (consistent
  with M2 #77/#90).

---

## Execution Nodes (12 Nodes)

### Wave 1 — Prompt Render Engine & Request-Scoped Wire Contracts

#### `M5-N01` — Prompt Render Layer & Jinja `ThinkingPolicy` Detection
- **Status:** `BLOCKED`
- **Depends On:** `M4-N06`
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

#### `M5-N02` — OpenAI & Anthropic Entrypoints: Request-Scoped Thinking, `chat_template_kwargs` & Prefill
- **Status:** `BLOCKED`
- **Depends On:** `M5-N01`
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

#### `M5-N03` — Request-Scoped Output Interpretation (`none` + `thinking`) & Grammar Start Sync
- **Status:** `BLOCKED`
- **Depends On:** `M5-N01`, `M5-N02`
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

#### `M5-N04` — Enhanced `reasoning_parser` & `think` Detectors (`tool_start` Recovery & Model Families)
- **Status:** `BLOCKED`
- **Depends On:** `M5-N03`
- **Hardware:** `CPU`
- **Touches:** `python/ayaka/entrypoints/parsers/reasoning.py`, `python/ayaka/entrypoints/parsers/formats.py`, `tests/entrypoints/parsers/`
- **Deliverable:**
  - Enhance `TagConfig` / `ReasoningParser` with `tool_start` / `tool_start_at_line_start`
    recovery (auto-closing unclosed thinking when a tool-call opening tag appears),
    `ReasoningDefault` (`on_unless` vs `off_unless`), `previous_content`, and
    `force_nonempty_content`.
  - Support reasoning formats: `think` (enhanced), `deepseek_r1`, `deepseek_v3`, `glm45`, and
    `minimax_m3` (plus clean `UnifiedParser` extension hook for `gemma4` in M8 and `kimi_k3` in
    M12).
- **Exit Check:** Streaming vs one-shot split-invariance test matrix (1-char, random splits,
  multi-byte UTF-8, unclosed `</think>` before `<tool_call>`) passes 100%.

#### `M5-N05` — Tool Parser Core: `ToolSchemas` Type Coercion, JSON Family & Grammar Spec Sync
- **Status:** `BLOCKED`
- **Depends On:** `M5-N03`
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

#### `M5-N06` — XML & Tag-Based Tool Call Parsers (`qwen3_coder`, `mimo`, `glm_xml`, `minimax_m2/m3`, `deepseek_dsml`)
- **Status:** `BLOCKED`
- **Depends On:** `M5-N04`, `M5-N05`
- **Hardware:** `CPU` + `GPU-L4`
- **Touches:** `python/ayaka/entrypoints/parsers/tool_calls/`, `python/ayaka/entrypoints/parsers/formats.py`, `tests/entrypoints/parsers/`, `tests/e2e/`
- **Deliverable:**
  - Implement incremental streaming + one-shot parsers for XML/tag families: `qwen3_coder`
    (`<function=name><parameter=k>v</parameter></function>`), `mimo`, `glm_xml`
    (`<arg_key>/<arg_value>`), `minimax_m2`/`m3`, and `deepseek_dsml` (`<｜DSML｜...>`).
- **Exit Check:** All 5 XML/tag families pass split-invariance + `ToolSchemas` coercion suites;
  live L4 tool-call sanity checks pass on `Qwen3-8B` and `MiMo-7B-RL`.

#### `M5-N07` — Structured Output v2 Contract (`6 ConstraintKind`, `StructuralTag`, `rollback` & Guided Speculation)
- **Status:** `BLOCKED`
- **Depends On:** `M5-N05`
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
  - Integrate `validate_tokens` + `rollback` with M4's speculative verification windows so
    **Guided Speculative Decoding** runs without turning off speculation on constrained turns.
- **Exit Check:** All 6 `ConstraintKind` unit tests pass on `LLGuidanceBackend`; guided speculative
  decoding produces identical greedy/sampled tokens to non-speculative constrained decoding.

#### `M5-N08` — XGrammar Backend (`--extra xgrammar`) & XML/JSON `StructuralTag` CUDA-Graph Sync
- **Status:** `BLOCKED`
- **Depends On:** `M5-N06`, `M5-N07`
- **Hardware:** `CPU` + `GPU-L4`
- **Touches:** `pyproject.toml`, `uv.lock`, `python/ayaka/sampling/structured/xgrammar_backend.py`, `python/ayaka/entrypoints/parsers/output_grammar.py`, `tests/sampling/`
- **Deliverable:**
  - Add optional `xgrammar` extra in `pyproject.toml`, loaded lazily via `require_module`
    (`--structured-backend llguidance|xgrammar`, keeping `llguidance` as default pending `M5-N12`).
  - Implement `XGrammarBackend` applying the two critical vLLM production patterns:
    1. **Dual compilers**: `json_compiler` (special tokens filtered out for JSON) and
       `full_vocab_compiler` (special tokens enabled for `STRUCTURAL_TAG`).
    2. Accurate `TokenizerInfo` (handling BPE, tiktoken, and SentencePiece prefix-space edge cases),
       `disable_any_whitespace` + explicit `separators`, native `rollback(num_tokens)`, and
       `fill_next_token_bitmask` feeding directly into Ayaka's captured CUDA-graph bitmask op.
  - Connect every JSON & XML tool parser from `M5-N05`/`M5-N06` (`hermes`, `llama3_json`,
    `qwen3_coder`, `mimo`, `glm_xml`, `minimax`, `deepseek_dsml`) to the `STRUCTURAL_TAG` builder so
    XML tool calls are fully grammar-constrained inside CUDA graphs.
- **Exit Check:** All 6 `ConstraintKind` + XML `STRUCTURAL_TAG` + guided-speculative rollback tests
  pass on `XGrammarBackend` inside CUDA graphs on L4; `import ayaka` succeeds without `xgrammar`.

---

### Wave 3 — gRPC Transport (Engine Seam & Engine-Free Render)

#### `M5-N09` — Protobuf Schemas & Optional `grpc` Extra
- **Status:** `BLOCKED`
- **Depends On:** `M5-N03`, `M5-N07`
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

#### `M5-N10` — Engine / Inference gRPC Server & Remote `GrpcEngineClient`
- **Status:** `BLOCKED`
- **Depends On:** `M5-N09`
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

#### `M5-N11` — Engine-Free `Render` & `Parse` gRPC Service (`ayaka render-server`)
- **Status:** `BLOCKED`
- **Depends On:** `M5-N09`, `M5-N06`
- **Hardware:** `CPU`
- **Touches:** `python/ayaka/entrypoints/grpc/render.py`, `python/ayaka/entrypoints/cli.py`, `tests/entrypoints/grpc/`
- **Deliverable:**
  - Standalone `ayaka render-server MODEL` exposing `RenderChat` and `ParseOutput` over gRPC/UDS
    with `x-render-fingerprint` metadata validation — zero GPU and zero `torch` required.
- **Exit Check:** Runs in a no-torch / `--extra cpu --extra grpc --extra hf` environment;
  `RenderChat` output matches in-process `Serving` token IDs and constraints bit-for-bit.

#### `M5-N12` — Benchmarks (`XGrammar` vs `LLGuidance` & gRPC Overhead), Live Agent Verification & `v0.2.0-a2`
- **Status:** `BLOCKED`
- **Depends On:** `M5-N08`, `M5-N10`, `M5-N11`
- **Hardware:** `GPU-L4`
- **Touches:** `benchmarks/e2e/`, `docs/EXPERIMENTS.md`, `docs/capabilities.toml`, `docs/CAPABILITIES.md`, `docs/AI_REFERENCE.md`, `docs/ROADMAP/decisions.md`
- **Deliverable:**
  - **Pre-registered Benchmark 1 — `XGrammar` vs `LLGuidance` on L4**: cold/warm compile latency,
    CUDA-graph per-step mask overhead (batch 1/8/32), XML `STRUCTURAL_TAG`, and Guided Speculative
    Decoding tok/s; record default-backend verdict in `docs/EXPERIMENTS.md` & `decisions.md`.
  - **Pre-registered Benchmark 2 — gRPC Engine seam overhead on L4**: UDS & loopback TCP vs
    in-process `AsyncLLM` (`≤ 0.3 ms` p50/token, `≤ 2%` tok/s).
  - **Live L4 coding-agent verification**: `Qwen3-8B` with `reasoning_effort="none"` vs `"high"`
    (OpenAI/Codex) and `thinking: disabled` vs `enabled` (Anthropic/Claude Code), `MiMo-7B-RL`
    constrained XML tool calls, and 7/7 `ayaka connect --check` over both in-process and
    `--engine-grpc` modes; tag `v0.2.0-a2`.
- **Exit Check:** All 5 M5 exit conditions and both pre-registered gates verified;
  `tests/test_docs.py` & CI green; tag `v0.2.0-a2`.

## Risks

- XGrammar special-token handling edge cases between JSON and `STRUCTURAL_TAG` (mitigated upfront
  by vLLM's dual-compiler pattern and token-boundary parity tests).
- Subtle Jinja template quirks across model families causing `detect_thinking_policy` false
  positives/negatives (mitigated by lexer-level inspection and real template fixture tests).
- Guided speculative rollback desynchronizing grammar state after partial verify acceptance
  (mitigated by adversarial exact-output equivalence tests against non-speculative constrained runs).
- `grpcio` asyncio event-loop contention under concurrent bursts (mitigated by L4 overhead gate in
  `M5-N12` and UDS transport).
