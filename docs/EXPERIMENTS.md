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

## llama.cpp comparison on mains, and an over-committed start-up profile (#59)

- Question: how does Ayaka compare with llama.cpp on the dev GPU on mains power? The first
  comparison (2026-11-18 folder) ran on battery with the GPU in P3.
- Setup, the same as the battery run: Llama-3.2-1B-Instruct, Ayaka fp16 from the safetensors
  (`--cuda-graph --backends flashinfer,torch --memory-ratio 0.86 --kv-reserve-tokens 8192`),
  llama.cpp `edec462` with the f16 GGUF converted from the same file (`-ngl 99 -c 8192 -np 1`);
  prompts of 1,038 and 4,110 tokens, 128 output tokens, six rounds in A B / B A order, one engine
  up at a time. Both power readings `ac`. Commit `f65658d`.

### The first mains run was not a measurement of the engine

Its prompt read at 4K came out at 0.37x llama.cpp (2,025 against 5,429 tok/s), half of what
`ayaka bench` measured on the same machine (#70). Checked before the run was used:

- Not a first-request cost: three different 3,549-token prompts on one server read at 1,862,
  1,936 and 1,982 tok/s.
- In process, the same chunk of about 1,700 tokens took ~950 ms per step with this start's plan
  (1,592 blocks, `max_running` 3) and ~420 ms with `max_running` 1, with `--kv-reserve-tokens
  20480`, or with `--memory-ratio 0.80`. No allocator retries in any case; `cudaMemGetInfo` read 0
  free.
- The saved start-up profile held `runtime_bytes` = 266 MB. A start that fits measures 640 to
  720 MB, which is the NVML/CUDA gap under WSL2 (`NVML 4154458112 bytes before the backend, CUDA
  3459986228 bytes after it`) plus the context. With 266 MB the plan gives the allocator more than
  CUDA's free memory; WDDM accepts it, and, as I read it (not proven: no Windows counter was read),
  pages part of it out. Because free then reads 0, that start measures a low runtime again and
  saves it, so the state repeats.
- Once a fitting start had saved an honest profile, the same flags started 3 times out of 3 and
  read 3,549 tokens at about 3,400 tok/s. With `--memory-ratio 0.82` the honest profile left the
  KV floor 55 MB short on the next start, so a lower ratio is not a way around it either.

The over-committed run is kept as `compare-llama-cpp-f16-oversubscribed.json`, not used for the
table. The planning fault is a separate bug, written up for an issue.

### Result (`benchmarks/results/2026-11-19-rtx-3050-laptop-gpu/compare-llama-cpp-f16.json`)

Same flags, the profile in its honest state (681 MB after the run):

| median of 6 | Ayaka | llama.cpp | Ayaka / llama.cpp (median of paired rounds) |
| --- | --- | --- | --- |
| prompt read, 4,110 tokens | 3,359 tok/s | 4,672 tok/s | 0.71 (rounds 0.71 to 0.73) |
| prompt read, 1,038 tokens | 3,460 tok/s | 3,549 tok/s | 0.97 |
| TTFT cold, 4,110 tokens | 1.22 s | 0.88 s | 1.40 (slower) |
| TTFT warm, 4,110 tokens | 46 ms | 23 ms | 2.01 (slower) |
| generation after 4,110 tokens | 51.0 tok/s | 58.7 tok/s | 0.87 |
| generation after 1,038 tokens | 52.9 tok/s | 60.5 tok/s | 0.88 |
| peak VRAM above idle | 3.46 to 3.53 GB | 2.73 GB | Ayaka fills the pool it plans |

GPU temperature stayed between 57 and 74 °C. Ayaka is slower on every line except the 1K prompt
read, where the two are level. The battery run had generation level (0.98); on mains llama.cpp
gains more from the higher clocks than Ayaka does.

## Slow-client isolation (#50)

- Question: does a client that stops reading its stream slow the other streams down?
- Setup, written before the run: `ayaka serve` on HuggingFaceTB/SmolLM2-135M, CUDA fp16, the
  default `--slow-client-timeout` (30 s); `benchmarks/e2e/slow_client.py` streams 2 chat
  completions of 128 tokens at once, alone (A) and next to a client that opened a stream with a
  1 KiB receive buffer and never reads (B); A B B A, five pairs; the measured value is the median
  time between text frames of the two readers, and the ratio B/A per pair.
- Threshold, written before the run: isolation holds when the median ratio B/A is at most 1.25.

### First run: invalid, the stuck client never blocked the server (2026-11-19)

Run at `8d4ed71` (`slow-client` setup above, 2 s before the readers): median ratio B/A 1.068, within
the threshold, but no request was cancelled for reading too slowly. Checked afterwards:

- The stuck client's receive queue stops at 1,152 bytes after 0.5 s (its `SO_RCVBUF` reads 2,304),
  so it does stop reading.
- With `--slow-client-timeout 10`, the server cancelled it 40.8 s after it opened: the server
  was blocked only from about 31 s on. Linux grows the send buffer to 4 MB (`tcp_wmem` 4096
  16384 4194304) and the server's own write buffer takes more, which at SmolLM2's speed is half a
  minute of output.

Every B phase of that run ended about 3 s after its stuck client opened, so no reader ever ran
next to a blocked write. The ratio is discarded; the run is not committed as a result.

### Second run: setup and threshold, written before the run

- Same server, model and flags, the default `--slow-client-timeout` (30 s).
- A B phase opens the stuck client, waits 40 s, runs the 2 readers, then keeps the stuck client open
  until `/metrics` counts its slow-client cancellation. The server was blocked from
  `cancelled - 30 s`; a B phase counts only if that is before its readers started and the
  cancellation is after they ended. Any phase that fails this makes the run invalid (exit 2).
- A B B A, five pairs; the measured value is unchanged (median gap between the readers' text
  frames, ratio B/A per pair).
- Threshold, unchanged: isolation holds when the median ratio B/A is at most 1.25.

### Second run: result

`benchmarks/results/2026-11-19-rtx-3050-laptop-gpu/slow-client.json`, at `7cb60f9`, on mains.

- Valid: in all 10 B phases the server was blocked from 29.5 to 32.4 s after the stuck client
  opened, the readers ran from 40.0 to 41.2 s, and the cancellations came at 59.5 to 62.4 s; the
  server logged 10 slow-client cancellations.
- Median gap between text frames: A (alone) 6.4 to 7.4 ms, B (beside a blocked stuck client) 7.5 to
  8.9 ms. Ratio B/A per pair: 1.245, 1.206, 1.232, 1.211, 1.173; **median 1.211**, within the
  threshold of 1.25.
- Decision: isolation holds by the threshold written before the run, with little margin.

What the ratio contains, which the setup did not separate: while its writes are blocked, the
stuck request keeps decoding, so a B phase decodes three sequences (CUDA graph bucket 4) where A
decodes two (bucket 2). The 21% is therefore an upper bound on what a stuck client costs the
others; a control with a third reader that keeps up would split it. Also, the unit tests ran on
the same machine during part of this run (CPU only), which may have added noise to some phases.
