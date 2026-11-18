# Experiments

Measured results, each with the threshold it was judged against written before the run. A
measurement here is reproducible from the described setup; nothing is carried over from an
earlier state of the code without saying so.

## Prompt normalisation: the Claude Code billing header (#57)

- Setup: `ayaka serve` on SmolLM2-135M, CUDA, fp16, hermes tool-call parser, prefix cache on;
  the sanitized recording of Claude Code 2.1.327's first request
  (`tests/entrypoints/fixtures/claude-code-2.1.327/001-anthropic.json`) as the session shape.
  Two sessions A and B differ only in the billing header's values
  (`cc_version=2.1.327.a8f; cc_entrypoint=sdk-cli;` vs `cc_version=2.2.0.1b7;
  cc_entrypoint=terminal;`, two builds of the same client). Order per pair: A B B A, five
  pairs, temperature 0, the cached-token share of the answer's usage.
- Threshold, written before the run: keep `claude-code.billing-header` when the first B of
  each pair improves by at least 25 percentage points with normalisation on, and answers stay
  non-empty text with the same stop reason.

| first cross-session request (B1) | cached-token share |
| --- | --- |
| normalisation off (`--no-normalize`) | 20.5% — one block, the part before the churn |
| normalisation on | 83.1% — every block but the partial last one |

Warm requests read their own session's blocks in both settings (82–98% at this prompt size;
the difference between the two settings' warm numbers is the rounding of a 77–82 token prompt,
not a real difference: with a full session the last partial block costs well under 1%).

- Answer check: temperature 0 answers with and without normalisation were non-empty with the
  same `max_tokens` stop reason. The 135M model writes babble either way, so this setup cannot
  show more than that the request still completes and answers; a model that follows
  instructions is where answer equivalence is worth re-measuring.
- Measured caveat: SmolLM2 has no chat template, so the server's ChatML fallback renders no
  tool section and the measured prompt is 82 tokens where the real session carries ~14 KB of
  tool schemas. The block before the churn costs one block regardless of size, so on a
  templated model the off number for a full session is far below the 20.5% measured here, and
  the on number stays at the warm ceiling.
- Decision: **keep** `claude-code.billing-header` (62.6 percentage points on this shape, above
  the 25-point threshold).

Not added, with the reason recorded: a rule for the Codex CLI's per-session skill paths (they
are what the model is told to open with its tools, so rewriting them would break the calls)
and for OpenCode (its recorded sessions showed no churn in a normalizable field).

### Re-measurement on a templated model with the real session (2026-11-19)

Why again: the run above used the sanitized fixture (every text replaced by a placeholder) on a
model without a chat template, so its prompt was 82 tokens and carried no tool section. Two new
Claude Code 2.1.327 sessions recorded on 2026-11-19 (same folder, same version, two prompts)
also show that the churn is per session, not per build: their headers read
`cc_version=2.1.327.e78` and `cc_version=2.1.327.e0f` (the first recording read `.a8f`), at
character 75 of the system prompt; the tools and the rest of the system prompt were identical.

- Setup, written before the run: `ayaka serve` on unsloth/Llama-3.2-1B-Instruct (its chat
  template renders the 14 tools), CUDA fp16, flashinfer, prefix cache on, `--memory-ratio 0.86`;
  the two raw session requests (not committed: they hold private text), sent as recorded with
  `stream: false`, `max_tokens: 32`, `temperature: 0` and the client's `user-agent`. Each pair k
  gives both sessions a fresh build suffix of three hex digits, so no pair reads another pair's
  blocks; order A B B A, five pairs, once with `--no-normalize` and once with normalisation on.
  The measured value is the cached-token share of the first B of each pair.
- Threshold, written before the run: keep `claude-code.billing-header` when the median share
  of the first B improves by at least 25 percentage points with normalisation on. The answer
  check compares the answers to the same request with and without normalisation at
  temperature 0 and reports whether they are equal; a difference is reported, not hidden.

| first request of a new session (first B of each pair) | cached-token share, median of 5 |
| --- | --- |
| normalisation off (`--no-normalize`) | 0.2% (32 of 13,614 tokens: the block before the header) |
| normalisation on | 84.5% (11,505 of 13,610 tokens: everything up to the session's own question) |

All five pairs gave the same values; the repeats (second B, second A) read 99.9% in both
settings. Answers: 20 of 20 requests answered identically with and without normalisation
(temperature 0, the same `max_tokens` stop reason). The 1B model answers these prompts with a
tool call written as plain JSON, which says nothing about quality, only that the prompt change
does not change the answer.

- Decision: **keep** `claude-code.billing-header` (84.3 percentage points, above the 25-point
  threshold). The value it rewrites changes with every session of the same client version, so
  without the rule every new Claude Code session reads almost nothing from the cache.
