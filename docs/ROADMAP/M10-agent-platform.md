# M10 — Agent Platform (`v0.2.0-a7`)

- **Goal:** Turn Ayaka into a complete local coding-agent backend — split cleanly into **Tier 1:
  Core Agent Runtime** (required for `v0.2.0-a7` and `0.2.0` GA) and **Tier 2: Extended Platform**
  (strictly gated by RFC + value/overhead measurements).
- **Status:** Planned (not filed yet)
- **Target Checkpoint:** `v0.2.0-a7`

## Exit Conditions (Tier 1 Core)

1. `ayaka connect --check` passes **7/7** on at least **4** (target 6) coding harnesses.
2. An MCP client calls `plan`, `bench`, `doctor`, `connect`, `status`, and `stop` against Ayaka's
   MCP server with loopback/auth security controls verified.
3. `previous_response_id` works end to end and a **100-turn** coding-agent session maintains
   `cached_tokens > 0` with zero memory leaks.
4. Context compaction cuts whole turns and passes invariant tests proving it never splits or
   orphans a `tool_use` / `tool_result` pair.
5. Reasoning effort (`low`/`medium`/`high`/`max`) enforces token budgets and always closes the
   reasoning block cleanly before answer/tool output.

## Exit Conditions (Tier 2 Extended — Gated)

- Embeddings/Rerank: `recall@k` verified against reference before shipping.
- OpenTelemetry: opt-in tracing overhead `< 1%` steady-state latency.
- LoRA/Multi-LoRA: identical tokens to manual weight merge; hot-swap leaks zero VRAM.
- Model Router: preserves prefix-cache hit rate over single-model baseline.

---

## Execution Nodes (7 Nodes)

### Tier 1 — Core Agent Runtime

#### `M10-N01` — Session Store & Stateful `previous_response_id`
- **Status:** `BLOCKED`
- **Depends On:** `M9-N07`
- **Hardware:** `CPU` + `GPU-L4`
- **Touches:** `python/ayaka/entrypoints/`, `python/ayaka/engine/`, `python/ayaka/configs/`, `tests/entrypoints/`
- **Deliverable:**
  - In-memory (default, matching `docs/SAFETY.md` zero-retention posture) + opt-in owner-only
    session store backing OpenAI Responses `previous_response_id` so agents advance multi-turn
    loops without resending full histories.
- **Exit Check:** Stateful multi-turn tests pass; 100-turn replay preserves `cached_tokens > 0` with
  bounded memory growth.

#### `M10-N02` — Turn-Safe Context Compaction
- **Status:** `BLOCKED`
- **Depends On:** `M10-N01`
- **Hardware:** `CPU`
- **Touches:** `python/ayaka/entrypoints/`, `python/ayaka/engine/`, `tests/entrypoints/`
- **Deliverable:**
  - Automatic/explicit compaction rules that drop or summarize whole turns cleanly while preserving
    system/tool prefix blocks and strictly refusing to cut inside any `tool_use`/`tool_result`
    exchange.
- **Exit Check:** Property/fuzz tests prove compaction never produces orphaned tool calls or
  invalid protocol histories across OpenAI & Anthropic adapters.

#### `M10-N03` — Reasoning Effort Controller (`low`/`medium`/`high`/`max`)
- **Status:** `BLOCKED`
- **Depends On:** `M10-N01`, `M5-N03`
- **Hardware:** `CPU` + `GPU-L4`
- **Touches:** `python/ayaka/sampling/`, `python/ayaka/entrypoints/`, `tests/sampling/`, `tests/entrypoints/`
- **Deliverable:**
  - Configurable reasoning effort budgets integrated with reasoning parsers (`think` etc.) that
    inject/force a clean closing delimiter (`</think>`) at budget exhaustion before switching into
    answer or grammar-constrained tool output.
- **Exit Check:** Unit & GPU e2e tests confirm all 4 effort levels respect budgets and emit valid
  closed reasoning + tool-call outputs.

#### `M10-N04` — Built-In Local MCP Server (`--mcp`)
- **Status:** `BLOCKED`
- **Depends On:** `M6-N01`
- **Hardware:** `CPU`
- **Touches:** `python/ayaka/entrypoints/mcp/`, `python/ayaka/entrypoints/cli.py`, `docs/SAFETY.md`, `tests/entrypoints/`
- **Deliverable:**
  - Opt-in MCP server exposing `plan`, `bench`, `doctor`, `connect`, `status`, and `stop` tools over
    stdio / loopback HTTP with explicit auth & read/write separation (no unauthenticated remote
    `stop` or `connect --write`).
- **Exit Check:** MCP client integration test calls all 6 tools cleanly; security tests verify
  unauthorized / non-loopback control attempts are rejected.

#### `M10-N05` — Expand `ayaka connect` to 6 Harnesses & 100-Turn Agent Eval
- **Status:** `BLOCKED`
- **Depends On:** `M10-N01`, `M10-N02`, `M10-N03`, `M10-N04`
- **Hardware:** `GPU-L4`
- **Touches:** `python/ayaka/entrypoints/connect/`, `benchmarks/e2e/`, `tests/entrypoints/`, `docs/AI_SETUP.md`
- **Deliverable:**
  - Extend `ayaka connect` (`--write`, `--check`, `--observe`) from 3 harnesses (Claude Code, Codex
    CLI, OpenCode) to 6 coding harnesses.
  - Automated 100-turn agent evaluation harness measuring cache hit rate, tool validity, compaction
    safety, and latency stability.
- **Exit Check:** `≥ 4` (target 6) harnesses pass 7/7 `connect --check`; 100-turn eval passes on L4
  and is recorded in `docs/EXPERIMENTS.md`.

---

### Tier 2 — Extended Platform *(Gated by RFC + Measurement)*

#### `M10-N06` — Embeddings/Rerank & Opt-In OpenTelemetry *(GATED)*
- **Status:** `GATED`
- **Depends On:** `M10-N05`
- **Hardware:** `GPU-L4`
- **Touches:** `python/ayaka/entrypoints/`, `python/ayaka/models/`, `python/ayaka/metrics/`
- **Deliverable:**
  - Built only after an RFC establishes demand and gates pass: `/v1/embeddings` + `/v1/rerank`
    verified on `recall@k`; opt-in local OTLP traces with `< 1%` steady-state overhead.
- **Exit Check:** Pre-registered `recall@k` and `< 1%` overhead gates recorded in
  `docs/EXPERIMENTS.md` (or explicitly deferred with rationale in `decisions.md`).

#### `M10-N07` — LoRA / Multi-LoRA & Prefix-Safe Model Router + `v0.2.0-a7` *(GATED)*
- **Status:** `GATED`
- **Depends On:** `M10-N05`
- **Hardware:** `GPU-L4`
- **Touches:** `python/ayaka/layers/`, `python/ayaka/execution/`, `python/ayaka/entrypoints/`, `docs/EXPERIMENTS.md`
- **Deliverable:**
  - Built only if gated RFC approves: deterministic LoRA / Multi-LoRA matching manual weight merge
    with zero hot-swap VRAM leaks; prefix-aware model router.
  - Tag internal checkpoint `v0.2.0-a7` once Tier 1 (`M10-N01..N05`) and any approved Tier 2 items
    clear exit checks.
- **Exit Check:** Tier 1 exit conditions 100% green; Tier 2 either passes gates or is formally
  deferred; tag `v0.2.0-a7`.

## Risks

- MCP exposing local control verbs (`stop`, `connect --write`) without airtight auth boundaries.
- Session persistence leaking private repository text to disk without explicit opt-in.
- Compaction dropping crucial file paths or constraints mid-session.
- Non-deterministic LoRA split-K or naive routing breaking prefix-cache hits.
