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

## Constrained decode inside CUDA graphs (#73)

- Question: with the structural mask of an output constraint applied inside the captured sampling
  path, how much slower is a constrained decode step than a free one, and what do the extra graphs
  cost in memory?
- Setup, written before the run: `benchmarks/e2e/constrained_decode.py` on the RTX 3050 Laptop GPU
  (4 GB), WSL2, fp16, `--backends flashinfer torch`, on SmolLM2-135M and
  unsloth/Llama-3.2-1B-Instruct; graph buckets 1, 2, 4, 8, 16 and 32; batches 1, 8 and 32; 129
  tokens per request, greedy, `ignore_eos`; 5 runs per arm after a warm-up; the value is the median
  decode time per step. The constraint is a JSON schema with one string of at least 2,048
  characters, so the grammar cannot finish and every step carries a mask. Arms: `free_graph` (no
  constraint), `eager_control` (a zero logit bias: the same step run eagerly, which is what a
  constrained step cost before its mask went into the graph) and `constrained`. The script stops
  if `free_graph` or `constrained` replays no graph, or `eager_control` replays one. The torch
  backend's decode attention reads the whole padded block table, so it is not measured.
- Thresholds, written before the run. An earlier run of the same script on an earlier state of
  the code (the mask op was different) had given gaps of 0.4% to 23.1% over its six points, so
  these are not blind; that run is not used below.
  1. At batch 1 and 8, on both models, the median `constrained` step is at most 10% over
     `free_graph`.
  2. At every batch, the GPU work the mask adds (the float32 copy of the logits and
     `apply_token_bitmask`, `gpu_mask_ms`) is at most 5% of the `free_graph` step.
  3. `constrained` replays graphs at every batch (the script enforces it).
- Reported, not judged: batch 32, where the grammar work on the host grows with the rows, and
  memory: the allocator's growth around each captured graph, measured the same way on `main`
  (`a416948`), on the first implementation (`dcf6473`) and on the change below.

### Memory of the graphs, measured before the speed run

The same method on all three trees (`torch.cuda.memory_reserved()` around every graph capture, a
fresh profile cache, buckets 1, 2, 4, 8, 16 and 32, flashinfer and torch backends):

| Tree | Graphs | Llama-3.2-1B | SmolLM2-135M |
| --- | --- | --- | --- |
| `main` | 12 (greedy or sampling, per bucket) | 482 MiB | 344 MiB |
| `dcf6473` | 24 (each of those, with and without a mask) | 772 MiB | 606 MiB |
| this change | 24 | 772 MiB | 606 MiB |

Every graph at a bucket of 16 or fewer grows the allocator by about 20 MiB, masked or not, so the
masked variants add about 20 MiB per graph, two graphs per bucket. Replacing the int64 mask op by
a byte-wise one (below) changes how that growth is split between the graphs of bucket 32 but not
its total: the peak of a captured sampling step is the sampler's full-vocabulary sort, not the
mask's scratch, which the sort reuses.

### First run: invalid, the GPU was over-committed (2026-11-20)

The script on this change, a fresh profile cache and the engine's default `memory_ratio` (0.90).
The start-up log of each engine said the allocator held more than the plan: 979 MiB more on
Llama-3.2-1B (tolerance 257 MiB) and 635 MiB more on SmolLM2-135M (tolerance 239 MiB). The graphs
of six buckets (772 and 606 MiB) do not fit into the 10% headroom (415 MiB). Under WSL2 the driver
serves what exceeds the device from host memory, and the step times show it: on Llama, `free_graph`
at batch 32 took 55 ms per step (26 ms in an earlier run on the same day), `eager_control` 220 to
293 ms (about 38 ms earlier), and even batch 8 was slower than before. The run measured paging, not
the step; it is discarded. On `main` the same six buckets also leave the start-up reserving 4,606
MiB on this 4,096 MiB device (measured above, Llama-3.2-1B, graphs on, default ratio); its own
start-up check was not run.

The script now checks the engine's own reconciliation after start-up and stops with exit 2 when
the allocator holds more than the plan allows, and takes `--memory-ratio`.

### Second run: setup, written before the run

- The same script, models, setup and thresholds, with `--memory-ratio` set to the highest of 0.80,
  0.75, 0.70, 0.65 and 0.60 at which the script's start-up check passes, tried in that order, with
  a fresh profile cache. The value is the same for every arm of a model and is recorded in the
  result. A model for which none passes has no valid run.
- Validity: the start-up check passes (exit 0). Nothing else is filtered; every run that passes
  is reported.

### Second run: Llama-3.2-1B has no valid run at six buckets; third run, written before it

The second run went as set up for SmolLM2-135M (valid at 0.65, all batches). For
Llama-3.2-1B none of the five ratios gave a valid run: 0.80 to 0.65 were refused as over-committed,
and at 0.60 the engine did not start (`CapacityError`: the KV floor needs 2,674,528,256 bytes, 2,492,674,867
are available). By the rule above there is no valid run of that model at batch 32 on this GPU, and
this is reported as a finding, not repaired: the graphs of buckets up to 32 (772 MiB with masks,
482 MiB without, both measured above) do not fit next to the weights of a 1.2 B model in 4 GB.

Batch 1 and 8, where thresholds 1 and 2 are judged, need only the buckets 1, 2, 4 and 8, the
engine's default. The third run, decided before any step time of the second run was read:

- Llama-3.2-1B only, `--batch 1 8`, so the graph buckets are 1, 2, 4 and 8; everything else as
  above, with `--memory-ratio` the highest of 0.90 (the engine's default), 0.85, 0.80, 0.75, 0.70
  and 0.65 whose start-up check passes. The thresholds are unchanged and judged at these two
  batches; at batch 32 the Llama model has no valid run.

### Result

On `dcf6473` plus the working tree that became this change (the byte-wise mask op below, the
benchmark's start-up check). Commands: `uv run python benchmarks/e2e/constrained_decode.py --model
<checkpoint> --backends flashinfer torch --memory-ratio <ratio>`, with `--batch 1 8` for Llama. The
result files are not committed; these tables are the record. Median decode time per step in ms,
five runs per arm.

SmolLM2-135M, `--memory-ratio 0.65` (the first that passed; the allocator held 134 MiB more than the
plan, tolerance 205 MiB), buckets 1 to 32, graphs 606 MiB:

| Batch | `free_graph` | `eager_control` | `constrained` | `constrained` over `free_graph` | `gpu_mask_ms` over `free_graph` |
| --- | --- | --- | --- | --- | --- |
| 1 | 6.06 | 41.53 | 6.07 | +0.2% | 8.0% |
| 8 | 6.81 | 46.96 | 7.31 | +7.3% | 6.9% |
| 32 | 8.66 | 49.48 | 9.36 | +8.1% | 2.4% |

Llama-3.2-1B, batches 1 and 8, `--memory-ratio 0.70` (the first that passed; 174 MiB over the plan,
tolerance 207 MiB), buckets 1, 2, 4, 8, graphs 378 MiB. There is no valid batch 32 run (above).

| Batch | `free_graph` | `eager_control` | `constrained` | `constrained` over `free_graph` | `gpu_mask_ms` over `free_graph` |
| --- | --- | --- | --- | --- | --- |
| 1 | 17.60 | 27.11 | 17.60 | +0.0% | 1.3% |
| 8 | 20.36 | 29.24 | 21.00 | +3.2% | 0.8% |

Every `constrained` and `free_graph` arm replayed graphs (768 replays each) and no `eager_control`
arm did.

Against the thresholds written before the run:

1. Met. At batch 1 and 8 on both models the constrained step is at most 7.3% over the free one.
2. **Not met as written**, on SmolLM2-135M at batch 1 (8.0%) and batch 8 (6.9%); met on Llama (1.3%,
   0.8%) and on SmolLM2 at batch 32 (2.4%).
3. Met.

Threshold 2 uses `gpu_mask_ms`, which times the float32 copy and `apply_token_bitmask` launched
eagerly, so it contains the gaps between kernel launches. It is launch-bound: on SmolLM2 it is
0.48 ms at one row and 0.21 ms at 32. A diagnostic added after these results, so it does not
replace the threshold, captures the same two calls in a CUDA graph and times the replay:

| Vocabulary | Rows | In a graph | Eager |
| --- | --- | --- | --- |
| 49,152 (SmolLM2) | 1 / 8 / 32 | 0.091 / 0.055 / 0.210 ms | 0.118 / 0.129 / 0.246 ms |
| 128,256 (Llama) | 1 / 8 / 32 | 0.064 / 0.125 / 0.531 ms | 0.207 / 0.182 / 0.544 ms |

In a graph the mask part is 1.5%, 0.8% and 2.4% of the SmolLM2 `free_graph` step at 1, 8 and 32
rows, and 0.4% and 0.6% of the Llama step at 1 and 8 rows. A later threshold should use this
metric.

What the graph wins: the same engine runs a step with the mask and no graph (`eager_control`, which
has no grammar cost, so it understates the old cost) in 5.3 to 6.8 times the time on SmolLM2 and 1.4
to 1.5 times on Llama; the constrained step in a graph is within 8.1% of a free step on SmolLM2 and
within 3.2% on Llama. The grammar's own work on the host (`mask()` of llguidance) is 7.7 µs per row
at p50 on SmolLM2 and 31.1 µs on Llama.

### Repeat run: setup, written before the run

The diagnostic above came from a throwaway script outside the repository, so nothing in the
repository reproduces it. The benchmark script now times the same two calls captured in a CUDA
graph (`gpu_mask_graph_ms`). The second and third runs are repeated with it, at the ratios that
passed (SmolLM2-135M at 0.65, batches 1, 8 and 32; Llama-3.2-1B at 0.70, batches 1 and 8; if the
start-up check refuses a ratio, the next lower one of the lists above), everything else unchanged.

- The verdicts above stand; no threshold is edited. One more question is put to the new metric,
  as threshold 2': `gpu_mask_graph_ms` is at most 5% of the `free_graph` step at every batch.
- Everything the repeat measures is reported, whatever the numbers; it also shows how far a second
  run of the same setup is from the first.

### Repeat run: result

The same setup and ratios, valid again (the allocator 134 MiB over the plan on SmolLM2-135M and
174 MiB on Llama-3.2-1B, as before). Median decode time per step in ms, five runs per arm; the last
two columns are the mask part (the float32 copy and `apply_token_bitmask`) launched eagerly
(`gpu_mask_ms`, the metric of threshold 2) and replayed from a graph (`gpu_mask_graph_ms`), with
their share of the `free_graph` step.

| Model | Batch | `free_graph` | `eager_control` | `constrained` | `constrained` over `free_graph` | Mask, eager | Mask, in a graph |
| --- | --- | --- | --- | --- | --- | --- | --- |
| SmolLM2-135M | 1 | 6.01 | 41.42 | 6.17 | +2.7% | 0.123 ms (2.0%) | 0.013 ms (0.2%) |
| SmolLM2-135M | 8 | 6.73 | 43.06 | 6.93 | +3.0% | 0.128 ms (1.9%) | 0.051 ms (0.8%) |
| SmolLM2-135M | 32 | 8.66 | 47.12 | 8.87 | +2.5% | 0.214 ms (2.5%) | 0.200 ms (2.3%) |
| Llama-3.2-1B | 1 | 17.59 | 27.12 | 17.71 | +0.7% | 0.124 ms (0.7%) | 0.065 ms (0.4%) |
| Llama-3.2-1B | 8 | 20.68 | 29.49 | 20.92 | +1.2% | 0.182 ms (0.9%) | 0.125 ms (0.6%) |

Every `constrained` and `free_graph` arm replayed graphs (768 replays), no `eager_control` arm did.

- Threshold 1: met again, at most 3.0% over a free step (8.1% at most in the first run).
- Threshold 2, as written: met in this run (at most 2.5%), after it was not met in the first run
  (8.0% and 6.9%). The verdict of record stays the one of the run it was written for: **not met**.
  The metric is not stable: the same eager measurement on SmolLM2 at one row gave 0.48 ms in the
  first run and 0.12 ms in this one.
- Threshold 2': met, at most 2.3% of the step, in a graph.
- Threshold 3: met.

What the two runs say together: the constrained step in a graph costs between 0% and 8% more than
a free one, and a run of this setup differs from another by up to 5 points of that percentage
(SmolLM2 at batch 8: 7.3% and 3.0%; at batch 32: 8.1% and 2.5%). On this laptop GPU, read a
percentage of a single run with a margin of that size. The mask part in a graph is under 2.5% of
the step in every case (the throwaway script's table above agrees with it for Llama; at one row
of SmolLM2 it gave 0.091 ms where the script gave 0.013 ms, a difference of tens of microseconds
that this timing cannot resolve).

### What the graphs cost in memory

The allocator's growth around the captures, same method as above, at the engine's default buckets
1, 2, 4 and 8:

| Model | `main` | this change |
| --- | --- | --- |
| Llama-3.2-1B | 236 MiB (8 graphs) | 410 MiB (16 graphs) |
| SmolLM2-135M | 212 MiB (8 graphs) | 374 MiB (16 graphs) |

About 75% more, 20 MiB per added graph, for every engine whether or not a request ever carries a
constraint. On a 4 GB GPU that is also what decides which bucket lists start: Llama-3.2-1B with
buckets up to 32 does not start validly (above). Options not taken, each with its cost: apply the
mask in every graph with all-ones rows (two graphs per bucket again, but a float32 copy and the mask
on every free step, 0.06 to 0.5 ms in a graph, and a different float path for free requests);
capture the masked graphs on the first constrained request (an allocation while serving, which can
fail); a setting to leave them out (one more option to explain).

### The mask op

`apply_token_bitmask` of the torch backend turned the mask into int64 and gathered it per token. It
now shifts the bytes of the mask, one byte per token, with identical output (checked on random
masks, including the sign bit and a vocabulary that is not a multiple of 32). Eager, on this GPU,
median of 100 calls:

| Rows | Vocabulary | Before | After | Scratch before | Scratch after |
| --- | --- | --- | --- | --- | --- |
| 1 | 128,256 | 0.358 ms | 0.214 ms | 3.9 MiB | 0.2 MiB |
| 8 | 128,256 | 0.599 ms | 0.169 ms | 18.1 MiB | 2.0 MiB |
| 32 | 128,256 | 2.129 ms | 0.394 ms | 66.9 MiB | 8.3 MiB |
| 32 | 49,152 | 0.867 ms | 0.247 ms | 25.1 MiB | 3.5 MiB |

It does not change the memory of the graphs (see above): the captured sampling step peaks at the
sampler's sort, which reuses the mask's scratch.

## Serving overhead baseline and the two open gates (#77)

Pre-registered setups and thresholds were written to #77 before any benchmark ran:

- **SSE frame coalescing (`sse_coalesce_ms`):** keep `sse_coalesce_ms = 0` when
  `server_cpu_per_frame * frames_per_second` stays below 5% of GPU step time; otherwise record
  engine-side k-step batching instead of adding a sleep on the event loop.
- **Segmented tokenize cache:** build the conditional segmented prompt encoder cache only if
  `render + encode` is at least 5% of cold time to first token on long coding-agent prompts
  (13.6K and 50K tokens); otherwise defer it.

### 1. SSE frame encoding and concurrent serving overhead

- **CPU micro-benchmark (`benchmarks/results/2026-11-20-windows-cpu/sse-encode.json`):**
  Windows 11, Python 3.13.5, Torch `2.14.1+cpu`; 5 rounds x 20,000 iterations of short 1-8 char
  `TextDelta` encoding (`"a"`, `"hi"`, `"token "`, `'quote"'`, `"a\\b"`, `"xin chao"` with UTF-8
  diacritic, `'{"k":1}'`, `"12345678"`).
- **Fake-engine overhead harness
  (`benchmarks/results/2026-11-20-windows-cpu/serving-overhead-fake.json`):** same Windows host,
  `benchmarks/e2e/serving_overhead.py --mode fake --streams 4 --tokens 64 --rounds 5 --step-ms 0.5 --burst --burst-requests 20 --burst-prompt-chars 8000`
  with raw-socket clients in a separate OS process. Note Windows `time.process_time()` updates in
  `15.625 ms` ticks.
- **Live GPU overhead harness
  (`benchmarks/results/2026-11-20-rtx-3050-laptop-gpu/serving-overhead-smollm2.json`):** WSL2
  Ubuntu 24.04 on `NVIDIA GeForce RTX 3050 Laptop GPU` (`sm_86`, 4.0 GiB, driver `591.59`, CUDA
  `13.0`, Torch `2.14.1+cu130`, `flashinfer-python 0.7.4`), `ayaka serve HuggingFaceTB/SmolLM2-135M`
  with `--device cuda --dtype fp16 --backends flashinfer,torch --cuda-graph --memory-ratio 0.75`.
  Memory plan fit cleanly with 5,764 KV blocks (`5,633` allocated, max 11 concurrent requests up to
  8,192 tokens); burst sent 20 x ~509-token prompts concurrently during the 4-stream decode.

Raw and median SSE encoder cost (µs/frame):

| Protocol | Round 1 | Round 2 | Round 3 | Round 4 | Round 5 | Median |
| --- | --- | --- | --- | --- | --- | --- |
| Chat Completions | 4.64 | 5.00 | 4.71 | 4.67 | 4.58 | **4.67 µs/frame** |
| Responses | 4.14 | 4.27 | 4.13 | 4.15 | 4.12 | **4.14 µs/frame** |
| Messages | 3.41 | 3.33 | 3.48 | 3.33 | 3.34 | **3.34 µs/frame** |

Fake-engine vs real RTX 3050 (`SmolLM2-135M`) medians over 5 rounds (256 frames/round steady):

| Run / Phase | FPS | Server CPU/frame | Engine step mean | Step gap mean | Step gap p50 | Step gap p99 | Client ITL p50 | Client ITL p99 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Windows fake (steady) | 3051.3 | 244.1 µs | — | 0.083 ms | 0.067 ms | 0.419 ms | 1.13 ms | 1.53 ms |
| Windows fake (+20 bursts) | 1213.1 | 831.1 µs | — | 0.805 ms | 0.100 ms | **6.750 ms** | 1.21 ms | **11.65 ms** |
| RTX 3050 `SmolLM2-135M` (steady) | 454.1 | 2304.7 µs* | 8.161 ms | 0.077 ms | 0.067 ms | 0.680 ms | 7.28 ms | 9.15 ms |
| RTX 3050 `SmolLM2-135M` (+20 bursts) | 315.0 | 3320.3 µs* | 11.225 ms | 0.371 ms | 0.071 ms | **15.100 ms** | 7.25 ms | **126.82 ms** |

\*On the live GPU process, `/proc/<pid>/stat` counts *all* host CPU threads of `ayaka serve`
(`0.590 s` CPU over `0.564 s` wall time in steady round 1), which is dominated by PyTorch and CUDA
graph launch overhead during the `65` engine steps (`65 * 8.161 ms = 0.530 s`), not SSE frame
formatting. Raw GPU per-round steady `step_gap_p99` values were `[0.68, 0.68, 0.68, 1.54, 1.54] ms`
and burst `step_gap_p99` values were `[14.80, 33.50, 15.10, 15.10, 15.10] ms`.

**SSE coalescing verdict:** **Keep `sse_coalesce_ms = 0`.** At `SmolLM2-135M`'s steady `454.1`
frames/s and `8.161 ms` step time, pure Chat SSE encoding (`4.67 µs/frame * 454.1 fps`) consumes
**`0.21%`** of step time (`0.017 ms` per step), and the engine thread's entire gap between
consecutive active steps (`0.077 ms` mean, `0.067 ms` p50) is **`0.95%`** of step time — well under
the 5% threshold. Even the entire Windows HTTP/asyncio loop (`244 µs/frame` in fake mode, bounded
below by coarse `15.625 ms` ticks) adds under 2.5% per stream at typical GPU decode rates, and
will shrink further with prebuilt SSE byte templates without adding frame coalescing delay. By
contrast, the 20-prompt burst pushed `step_gap_p99` from `0.680 ms` to **`15.100 ms`** (`22.2x`) and
client `itl_p99` from `9.15 ms` to **`126.82 ms`** (`13.9x`), pinpointing prompt admission, lock
contention, and stream overhead as the real burst bottlenecks.

### 2. Long coding-agent prompt render and encode vs cold TTFT

- **Cold prefill baseline (`benchmarks/results/2026-11-20-rtx-3050-laptop-gpu/cold-ttft-llama3.json`):**
  same WSL2 RTX 3050 4.0 GiB card, `ayaka serve unsloth/Llama-3.2-1B-Instruct --device cuda --dtype fp16 --backends flashinfer,torch --cuda-graph --memory-ratio 0.80 --kv-reserve-tokens 2048`
  (248 KV blocks allocated, `3,952` max context tokens). Unique nonce prefixes forced cold prefill on
  alternating 958-token and 2,778-token prompts across 5 rounds (only the fixed 32-token Llama
  system header hit the prefix cache):
  - `958` tokens: `[337.71, 262.17, 255.43, 254.08, 254.57] ms`, median **`255.43 ms`**
  - `2,778` tokens: `[761.64, 750.83, 753.71, 757.93, 759.04] ms`, median **`757.93 ms`**
  - Marginal cold prefill slope: `(757.93 - 255.43) / (2778 - 958) =` **`0.27610 ms/token`**
    (`3,622 tok/s`; direct `2,778`-token ratio `0.27283 ms/token`).
- **Prompt render & encode (`benchmarks/results/2026-11-20-rtx-3050-laptop-gpu/encode-long-llama3.json`):**
  `benchmarks/tokenizers/encode_long.py` using the `unsloth/Llama-3.2-1B-Instruct` HF tokenizer over
  the 23-tool Claude Code fixture (`tests/entrypoints/fixtures/claude-code-2.1.327/004-anthropic.json`)
  padded to 13,779 and 50,069 tokens (5 rounds). Note that 13.6K and 50K prefill times on this
  4 GiB fp16 GPU are linear extrapolations at `0.27610 ms/token` because KV capacity caps at
  `3,952` tokens.

| Prompt size | Serialize median | Render median | Encode median | `render + encode` median | Full uncached service median | Extrapolated cold TTFT | `render + encode` / TTFT | Uncached service / TTFT |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 13,779 tokens | 0.68 ms | 1.53 ms | 28.85 ms | 30.53 ms | 37.29 ms | 3,804.4 ms | **0.80%** | **0.98%** |
| 50,069 tokens | 3.70 ms | 5.63 ms | 104.48 ms | 112.79 ms | 135.04 ms | 13,824.1 ms | **0.82%** | **0.98%** |

Raw 5-round `encode_ms`: `[28.35, 29.21, 27.79, 29.10, 28.85] ms` at `13,779` tokens and
`[104.47, 104.48, 107.16, 104.47, 115.92] ms` at `50,069` tokens. Raw `service_uncached_ms`:
`[36.80, 36.53, 38.84, 41.70, 37.29] ms` and `[135.04, 130.74, 132.96, 138.02, 147.75] ms`.

**Segmented tokenize cache verdict:** **DEFERRED.** Even including JSON serialization and the
special-token guard, uncached tokenization takes **`0.98%`** of cold TTFT (`0.80–0.82%` for
`render + encode` alone), roughly five times below the pre-registered 5% threshold. The existing
session encode cache stays as-is, and segmented caching between special tokens is deferred until
much faster prefill hardware or warm-cache multi-turn regimes push encoding above 5% of TTFT.

## Detokenizer seed and the shared lock (#77, 2026-11-21)

Thresholds, setup and workload were written to #77 before any run
([comment](https://github.com/gityuko-hb/ayaka/issues/77#issuecomment-7069506207)). A is `main` at `5c1d790`, B is
`69532cc` (the bounded seed and the per-detokenizer lock); A B B A order, 5 pairs, each slot a fresh process with
`PYTHONPATH` pointing at that side's `python/`. Machine: Linux 6.6, **Tesla T4** (15 GiB, `sm_75`), driver 580.82.07,
CUDA 13.0, `torch 2.14.1+cu130`, `flashinfer-python 0.7.4`, Python 3.13.16, 2 CPU cores. This is not the RTX 3050 of
the baseline above; the numbers are only compared A with B on this machine. The raw result files were kept on the
machine and not committed; every value they hold is below.

### 1 and 2: creation cost and token delivery during an encode (`benchmarks/tokenizers/detokenizer_seed.py`)

`unsloth/Llama-3.2-1B-Instruct` tokenizer; prompts of 4,096, 16,384 and 50,000 tokens cut from the recorded Claude
Code session (`004-anthropic.json`) repeated to 50,061 tokens. Creation: median of 20 calls of
`TokenizerService.new_detokenizer` per slot. Contention: one-token `update()` calls every millisecond for 3 s while
another thread encodes the 50,061-token prompt in a loop.

Median creation time at 50,000 tokens per slot (ms), slots in run order (A1 B1 B2 A2 per pair):

| Path | A (10 slots) | B (10 slots) |
| --- | --- | --- |
| byte table | 35.07, 66.95, 69.79, 34.42, 31.59, 67.85, 37.26, 61.52, 31.06, 31.39 | 0.0090, 0.0091, 0.0082, 0.0105, 0.0096, 0.0084, 0.0087, 0.0083, 0.0082, 0.0157 |
| `DecodeStream` | 204.59, 98.27, 92.14, 92.16, 91.10, 101.66, 168.88, 91.76, 174.13, 88.79 | 0.1654, 0.1003, 0.1633, 0.0870, 0.1011, 0.0910, 0.0950, 0.0854, 0.0936, 0.1729 |

At 4,096 tokens A took 2.54–5.63 ms (bytes) and 6.75–14.24 ms (stream); B took 0.0081–0.0131 ms and 0.083–0.170 ms,
the same as at 50,000.

p99 of one-token `update()` latency while the other thread encodes (ms):

| Path | A (10 slots) | B (10 slots) |
| --- | --- | --- |
| byte table | 364.1, 699.0, 265.5, 395.7, 387.5, 440.4, 265.5, 441.5, 218.6, 141.7 | 0.060, 0.060, 0.065, 0.064, 0.065, 0.065, 0.062, 0.060, 0.057, 0.049 |
| `DecodeStream` | 247.2, 260.6, 256.4, 469.6, 224.8, 255.4, 144.5, 133.4, 279.7, 405.7 | 0.072, 0.072, 0.068, 0.080, 0.077, 0.078, 0.072, 0.070, 0.068, 0.075 |

Gates: **(1) passed** in every pair: with B the 50,000-token creation is 0.73–1.13× the 4,096-token one (gate ≤ 2×),
and B/A at 50,000 is 0.0002–0.0014 (gate ≤ 0.5). **(2) passed** in every pair: B's update p99 is at most 0.080 ms
(gate ≤ 1 ms), B/A of the p99 is 0.0001–0.0005 (gate ≤ 0.5). With A every update waited for the encoder's lock.

### 3: end to end on the GPU, no-regression gate (`benchmarks/e2e/serving_overhead.py --mode server`)

`ayaka serve unsloth/Llama-3.2-1B-Instruct --device cuda --dtype fp16 --backends flashinfer,torch --cuda-graph
--memory-ratio 0.8 --max-running 6` (19,146 KV blocks), one server per slot; harness `--streams 4 --tokens 128 --rounds
3 --burst --burst-requests 1 --burst-prompt-chars 400000` (the burst prompt is 50,035 tokens). The first burst round
of each slot prefills the whole prompt; rounds 2 and 3 find it in the prefix cache.

Burst-phase client ITL p99 per round (ms), each slot `[round 1, round 2, round 3]`:

| Pair | A1 | B1 | B2 | A2 |
| --- | --- | --- | --- | --- |
| 1 | 1466.6, 25.5, 19.7 | 1528.1, 34.1, 55.6 | 1608.0, 47.6, 66.5 | 1624.5, 58.6, 65.1 |
| 2 | 1617.3, 58.9, 28.7 | 1623.1, 37.0, 60.7 | 1623.2, 56.4, 62.9 | 1568.8, 61.3, 29.1 |
| 3 | 1616.7, 23.9, 38.6 | 1568.0, 56.7, 53.1 | 1606.3, 60.7, 24.0 | 1621.1, 25.1, 31.8 |
| 4 | 1567.5, 56.6, 56.8 | 1621.3, 61.0, 37.0 | 1620.6, 23.2, 52.9 | 1617.3, 25.8, 62.6 |
| 5 | 1627.8, 64.0, 61.2 | 1618.5, 45.2, 32.6 | 1557.6, 23.3, 60.1 | 1612.5, 63.6, 30.9 |

Per pair, mean over the two slots of each side of the per-slot median over rounds:

| Pair | A burst ITL p99 | B burst ITL p99 | B/A | A burst step-gap p99 | B burst step-gap p99 | A steady ITL p99 | B steady ITL p99 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 45.3 ms | 61.0 ms | 1.347 | 5.82 ms | 12.90 ms | 21.34 ms | 22.09 ms |
| 2 | 60.1 ms | 61.8 ms | 1.029 | 4.22 ms | 41.52 ms | 23.09 ms | 22.19 ms |
| 3 | 35.2 ms | 58.7 ms | 1.666 | 4.17 ms | 23.74 ms | 23.76 ms | 22.93 ms |
| 4 | 59.7 ms | 57.0 ms | 0.955 | 4.42 ms | 20.70 ms | 23.49 ms | 22.32 ms |
| 5 | 63.8 ms | 52.6 ms | 0.825 | 4.52 ms | 6.15 ms | 21.74 ms | 24.21 ms |

Gate **(3) missed**: the median of B/A over the pairs is **1.029** (gate ≤ 1.0). The burst-round values are bimodal
(about 24 ms or about 60 ms in both A and B) and the cold first round is about 1.6 s in both, so the ITL comparison is
dominated by that noise. The step-gap p99 of the burst phase, though, is higher with B in all five pairs (6.2–41.5 ms
against 4.2–5.8 ms). Not measured, so only a hypothesis: with the lock gone, the engine thread no longer sleeps while
the 50K prompt is encoded and validated on another thread; it competes with that thread's Python per-token loops for the
GIL instead, in switch-interval slices. Removing those loops is the next item of #77 (admission without
re-validation). Steady-phase ITL p99 is unchanged (21.3–23.8 ms against 22.1–24.2 ms).

**Verdict:** the detokenizer part passes its own gates by three to four orders of magnitude; the end-to-end
no-regression gate is missed. Per the rule written in #77, the change is not merged as a speed-up on this evidence.

## Admission without re-validating the prompt (#77, 2026-11-22)

### Setup and thresholds, written before the run

Not posted to #77 (the work was done locally); this section was committed before any A or B run whose numbers
are reported here. A trial of B alone (3 and 5 repetitions of gate 1, not kept) was run while the change was
written, before these thresholds: it gave about 0.1 ms per admission at every length.

A is `main` at `ca9b06c`; B is `002bf86` (requests prepared on a worker thread, vouched ids taken back without a
second check, id checks at C speed, the KV capacity check without walking the free queue). A B B A order, 5 pairs,
each slot a fresh process with `PYTHONPATH` pointing at that side's `python/`; the harness scripts come from B's
checkout for both sides. Machine: the Tesla T4 of the section above (15 GiB, `sm_75`), driver 580.82.07, CUDA 13.0,
`torch 2.14.1+cu130`, `flashinfer-python 0.7.4`, 2 CPU cores.

1. **Engine-thread admission** (`benchmarks/engine/admission.py --model unsloth/Llama-3.2-1B-Instruct --reps 20`):
   the time the engine thread spends admitting a prompt the server already encoded, at 4,096, 16,384 and 50,000
   tokens. With B, the median at 50,000 is at most **1 ms** and at most **2×** B's median at 4,096; B/A at 50,000 is
   at most **0.1** in every pair.
2. **One long prompt during decoding** (the workload of the section above): `ayaka serve
   unsloth/Llama-3.2-1B-Instruct --device cuda --dtype fp16 --backends flashinfer,torch --cuda-graph --memory-ratio
   0.8 --max-running 6`, harness `serving_overhead.py --mode server --streams 4 --tokens 128 --rounds 3 --burst
   --burst-requests 1 --burst-prompt-chars 400000`. Per slot the median over rounds; per pair the mean of each
   side's two slots. The median over pairs of B/A burst step-gap p99 is at most **0.5**. No regression: the median
   over pairs of B/A burst ITL p99 and of B/A steady ITL p99 are each at most **1.10** (not 1.0: the section above
   found the burst ITL p99 bimodal, with per-pair ratios from 0.83 to 1.67 between two builds).
3. **A burst of short prompts** (the workload of #90): `ayaka serve HuggingFaceTB/SmolLM2-135M --device cuda
   --dtype fp16 --backends flashinfer,torch --cuda-graph --memory-ratio 0.75`, harness `--streams 4 --tokens 64
   --rounds 5 --burst --burst-requests 20 --burst-prompt-chars 4000`. With B, the median over B's slots of burst
   step-gap p99 divided by steady step-gap p99 (same slot) is at most **1.5** (the threshold the roadmap proposed);
   the median over pairs of B/A burst step-gap p99 is at most **0.5**.

A missed threshold is reported as missed; the change is then not described as a speed-up for that workload.

### Result

Raw files: `benchmarks/results/2026-11-22-tesla-t4/admission/` (one JSON per slot, named `<pair>-<side><slot>`). The
first attempt at these runs was discarded before any number was read: the driver named the second A and B slot of
each pair like the first, so they were skipped; gate 2 then stopped at its first slot because `ninja` was not on the
driver's `PATH` (flashinfer compiles its kernels with it). The flashinfer kernels were then compiled once by a
throw-away server and harness run, and the three gates were run again from the start.

**1. Engine-thread admission** (median of 20 admissions per slot, ms; slots A1 B1 B2 A2 per pair):

| Pair | 4,096 tokens | 16,384 tokens | 50,000 tokens |
| --- | --- | --- | --- |
| 1 | 6.021, 0.064, 0.109, 6.573 | 36.023, 0.087, 0.143, 17.953 | 97.095, 0.164, 0.196, 51.725 |
| 2 | 5.970, 0.110, 0.078, 12.287 | 16.871, 0.159, 0.088, 36.536 | 51.262, 0.204, 0.129, 46.417 |
| 3 | 11.276, 0.067, 0.068, 5.945 | 32.397, 0.084, 0.085, 18.559 | 98.970, 0.117, 0.122, 50.057 |
| 4 | 6.489, 0.070, 0.070, 12.487 | 16.772, 0.088, 0.093, 58.967 | 50.875, 0.126, 0.130, 181.731 |
| 5 | 6.123, 0.060, 0.081, 11.640 | 17.743, 0.082, 0.093, 34.649 | 48.902, 0.095, 0.098, 97.633 |

B at 50,000 tokens: 0.095–0.204 ms (threshold ≤ 1 ms: **passed**). B/A at 50,000 per pair: 0.0024, 0.0034, 0.0016,
0.0011, 0.0013 (threshold ≤ 0.1: **passed**). B's 50,000-token median over its 4,096-token median per slot: 2.58,
1.79, 1.86, 1.64, 1.75, 1.81, 1.81, 1.86, 1.57, 1.21 (threshold ≤ 2×: **missed in one slot of ten**, pair 1 B1, where
0.164 ms against 0.064 ms). What still grows with the prompt on the engine thread was not measured further. In A the
time grows with the prompt (the ids checked again) and has a floor of about 5 ms at every length: the KV capacity check
walked the free queue of 19,146 blocks.

**2. One long prompt during decoding** (per pair, mean of each side's two slots of the per-slot median over rounds):

| Pair | burst step-gap p99 A / B (ms) | B/A | burst ITL p99 A / B (ms) | B/A | steady ITL p99 A / B (ms) |
| --- | --- | --- | --- | --- | --- |
| 1 | 6.45 / 3.18 | 0.493 | 54.7 / 47.2 | 0.863 | 22.77 / 22.33 |
| 2 | 25.07 / 1.35 | 0.054 | 57.9 / 52.2 | 0.900 | 21.83 / 20.93 |
| 3 | 25.67 / 1.69 | 0.066 | 54.7 / 81.2 | 1.484 | 21.95 / 21.56 |
| 4 | 6.45 / 1.39 | 0.216 | 53.2 / 56.2 | 1.057 | 22.80 / 21.08 |
| 5 | 39.75 / 2.31 | 0.058 | 58.7 / 49.7 | 0.847 | 20.48 / 22.42 |

Median over pairs: B/A burst step-gap p99 **0.066** (threshold ≤ 0.5: **passed**); B/A burst ITL p99 **0.900** and
B/A steady ITL p99 **0.981** (thresholds ≤ 1.10: **passed**). As in the section above, the first burst round prefills
the 50,035-token prompt (ITL p99 about 1.4–1.6 s on both sides) and the burst ITL p99 of the later rounds is bimodal;
the per-round values are in the raw files. All 560 requests of the 20 server runs ended normally (`length`, from the
server logs).

**3. A burst of short prompts:**

| Pair | burst step-gap p99 A / B (ms) | B/A | burst ITL p99 A / B (ms) | steady step-gap p99 A / B (ms) |
| --- | --- | --- | --- | --- |
| 1 | 68.00 / 1.54 | 0.023 | 21.1 / 20.8 | 5.10 / 0.62 |
| 2 | 67.50 / 3.40 | 0.050 | 21.1 / 20.6 | 5.10 / 0.39 |
| 3 | 111.00 / 2.47 | 0.022 | 20.8 / 21.6 | 3.61 / 0.83 |
| 4 | 109.50 / 3.79 | 0.035 | 73.6 / 20.4 | 3.41 / 1.58 |
| 5 | 67.50 / 3.40 | 0.050 | 74.6 / 58.9 | 2.49 / 1.95 |

Median over pairs of B/A burst step-gap p99 **0.035** (threshold ≤ 0.5: **passed**). With B, burst step-gap p99 over
steady step-gap p99 per slot: 1.72, 4.53, 7.61, 10.0, 2.26, 3.44, 3.46, 1.92, 2.21, 1.44; median **2.85** (threshold
≤ 1.5: **missed**). The step-gap histogram's buckets are coarse at this scale (B's burst values sit on 1.5, 3.4 and
6.8 ms, its steady values well under 1 ms), so the ratio moves in large steps, but the miss is not a rounding effect:
twenty admissions arriving together still cost the engine thread a few milliseconds between two steps. All 2,820 requests of
the 20 server runs ended normally (`length`, from the server logs).

**Verdict:** the engine thread no longer does work that grows with the prompt when a request arrives (gate 1, with
one slot of ten above the flatness bound), and the step gap of a burst drops by 15–30 times on both workloads (gates 2
and 3); gate 2's no-regression bounds on ITL pass. On workload 3, where ITL has no gate, the median over pairs of
B/A steady ITL p99 is 1.053 and of B/A burst ITL p99 0.979. The roadmap's proposed bound of a burst gap within 1.5× of the steady one is missed
(2.85).

## Stream watchdog, linear merge and the slow-client clock (#77, 2026-11-22)

### Setup and thresholds, written before the run

Not posted to #77 (the work was done locally); committed before any run reported here. A is the admission change
of the section above (`e3ac2fc`), B is `264e068` (one watchdog for every stream, the oldest untaken byte as the
slow-client clock, overflow kept in a list). Same Tesla T4 machine; each slot a fresh server with `PYTHONPATH`
pointing at that side's `python/`; harness scripts from B's checkout.

1. **The slow-client clock** (`benchmarks/e2e/slow_client.py`, B only, as the roadmap asks): `ayaka serve
   HuggingFaceTB/SmolLM2-135M --device cuda --dtype fp16 --backends flashinfer,torch --cuda-graph --memory-ratio 0.75
   --slow-client-timeout 10`, harness `--server-timeout 10 --settle 3 --fast 2 --tokens 128 --pairs 5
   --control-reader --threshold 1.25`. The stuck client never reads, so its cancellation delay is measured from when
   it stopped reading. Every stuck phase is cancelled within **12 s** (`slow_client_seconds + 2 s`), and the median
   ratio of the readers' gaps beside the stuck client over alone is at most **1.25** (the threshold of #50). For
   comparison only, not a gate: the same server and harness with A and `--settle 40`, 1 pair.
2. **No cost per frame** (`benchmarks/e2e/serving_overhead.py --mode fake --streams 4 --tokens 64 --rounds 5`, no
   burst, CPU only, A B B A, 5 pairs): the median over pairs of B/A server CPU per frame (per slot the median over
   rounds, per pair the mean of each side's two slots) is at most **1.05**.

The linear merge is checked by tests (a backlog of 1,000 outputs read in order), not timed.

### Result

Raw files: `benchmarks/results/2026-11-22-tesla-t4/stream-watchdog/` (`slow-B.json`, `slow-A.json`, and
`fake-<pair>-<slot>.json`).

**1. The slow-client clock.** With B, the ten stuck phases were cancelled 10.60, 11.04, 10.22, 10.98, 11.03, 11.07,
10.20, 11.00, 10.22 and 11.04 s after the stuck client opened (median 10.99 s; threshold ≤ 12 s: **passed**). Ratio of
the readers' median gap beside the stuck client over alone, per pair: 1.029, 1.038, 1.065, 1.052, 1.025, median
**1.038** (threshold ≤ 1.25: **passed**); the control phase with a third reader that keeps up: 1.011, 1.063, 1.034,
1.067, 1.010, median 1.034. The harness judged every phase valid. For comparison (A, `--settle 40`, one pair, not a
gate): cancelled 44.59 and 44.54 s after opening, of which the 10 s of `--slow-client-timeout` started only once the
kernel's send buffer was full (about 34.5 s); ratio 1.076.

**2. No cost per frame** (`--mode fake`, server CPU per frame in µs, median over 5 rounds; slots A1 B1 B2 A2):

| Pair | A1 | B1 | B2 | A2 | B/A |
| --- | --- | --- | --- | --- | --- |
| 1 | 191.6 | 163.7 | 143.1 | 187.8 | 0.809 |
| 2 | 189.5 | 162.0 | 148.5 | 190.5 | 0.817 |
| 3 | 188.3 | 157.1 | 142.6 | 194.4 | 0.783 |
| 4 | 195.6 | 166.5 | 150.7 | 192.7 | 0.817 |
| 5 | 180.6 | 164.7 | 143.4 | 189.8 | 0.832 |

Median B/A **0.817** (threshold ≤ 1.05: **passed**). The fake engine makes no model calls, so this is the share of the
server's own work per frame; it says nothing about a GPU step.

**Verdict:** a client that stops reading is cancelled `slow_client_seconds` plus at most one watchdog tick after it
stopped, instead of after the kernel's send buffer filled (about 34.5 s here), the other streams are not slowed past
the threshold of #50, and the server spends less CPU per frame.

## Rendering without a JSON round trip, on one preprocess thread (#77, 2026-11-22)

### Setup and thresholds, written before the run

Not posted to #77 (the work was done locally); committed before any run reported here. A is the stream-watchdog
change of the section above (`3e5cf60`), B is `dd44f6b` (the conversation's values rendered as `TemplateData`,
building, rendering and encoding on one `ayaka-preprocess` thread). Same Tesla T4 machine, CPU only.

`benchmarks/serving/loop_delay.py --tokenizer unsloth/Llama-3.2-1B-Instruct --body-kb 200 --requests 20`: an
in-process server with the real tokenizer and an engine that makes no model calls answers 20 Anthropic Messages
requests of 219,983 bytes (the recorded Claude Code session repeated; 29,531 prompt tokens), one at a time; a probe
on the loop sleeps 1 ms over and over and records each overrun. Per slot the median over requests of the largest
overrun and of the total overrun while a request runs; A B B A, 5 pairs; per pair the mean of each side's two slots.

A trial before these thresholds (8 requests per side, not kept) gave A: largest overrun 6.09 ms, total 21.8 ms; B:
5.54 ms and 23.1 ms. Timed on their own, on the loop: `json.loads` of the body 2.1 ms, parsing it into a conversation
5.7 ms more, `template_input()` 2.3 ms (A only); rendering takes 12.5–13.1 ms and encoding 127 ms on the other thread.
The thresholds are kept at the level of the other burst items of #77 rather than fitted to that trial:

1. median over pairs of B/A of the largest overrun at most **0.5**;
2. median over pairs of B/A of the total overrun at most **0.5**;
3. no regression: median over pairs of B/A of the client's round-trip time at most **1.05**.

Rendering from the values equals rendering from the JSON on every recorded request (tests, not timed).

### Result

Raw files: `benchmarks/results/2026-11-22-tesla-t4/loop-delay/` (one JSON per slot). Per-slot medians over the 20
requests, slots A1 B1 B2 A2:

| Pair | largest overrun (ms) | B/A | total overrun (ms) | B/A | round trip (ms) | B/A |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 6.38, 6.02, 5.77, 5.75 | 0.972 | 30.83, 25.44, 24.33, 23.02 | 0.924 | 130.3, 97.8, 97.1, 97.8 | 0.854 |
| 2 | 6.62, 6.78, 5.93, 5.84 | 1.020 | 35.54, 35.87, 25.42, 23.62 | 1.036 | 167.7, 153.9, 99.4, 99.4 | 0.948 |
| 3 | 5.63, 6.75, 6.90, 6.17 | 1.157 | 24.09, 41.10, 45.42, 24.42 | 1.784 | 102.8, 156.0, 163.0, 101.9 | 1.558 |
| 4 | 5.76, 5.69, 6.76, 6.74 | 0.996 | 23.28, 25.73, 32.81, 38.82 | 0.943 | 99.8, 98.7, 157.7, 159.5 | 0.989 |
| 5 | 6.01, 5.92, 5.67, 6.22 | 0.948 | 25.10, 24.23, 24.07, 27.00 | 0.927 | 107.2, 96.5, 97.9, 129.6 | 0.821 |

Medians over pairs: largest overrun B/A **0.996** (threshold ≤ 0.5: **missed**), total overrun B/A **0.943**
(threshold ≤ 0.5: **missed**), round trip B/A **0.948** (threshold ≤ 1.05: **passed**). Whole slots on both sides
sit either near 100 ms or near 155 ms of round trip, and the total overrun moves with them; that machine state, not
the build, decides those pairs.

**Verdict and the decision on the render thread:** removing the JSON round trip and rendering on one fixed thread
does not shorten the time a 200 KB request holds up the event loop. The largest overrun stays at 5.6–6.9 ms on both
sides: what is left on the loop is reading the body and parsing it into a conversation (about 8 ms in all, timed
above), and while the preprocess thread runs Python (the template takes about 13 ms) the loop waits for the GIL in
slices of the 5 ms switch interval, wherever that work runs. Going further would mean parsing the body off the loop
and still leaves the switch interval, which #77 put out of scope, or rendering in another process. The change is
therefore not a measured speed-up of the loop. What it does keep: one render instead of a serialize-and-parse-twice
round trip, render work serialized on one thread in arrival order, and a render that tests show equal to the JSON
path on every recording.

The raw files of the three sections above record the revisions they ran by the hashes those commits had before the
branches were rewritten to correct the commit author (same trees): `c6259e7` is `002bf86`, `4906553` is `e3ac2fc`,
`ac99081` is `264e068`, `25b195c` is `3e5cf60`, `745760b` is `dd44f6b`.

## M1 decision gates and FP8 KV check on NVIDIA L4 24 GB (`sm_89`, native Linux) (#62)

### Setup and thresholds, written before the run

Machine: native Linux (Ubuntu 24.04.4 LTS, kernel `6.6.159+`, `wsl2 = false`), **NVIDIA L4**
(`23,034 MiB`, `sm_89`), driver `580.82.07`, CUDA `13.0`, `nvcc 13.0.88`, Python `3.13.16`,
`torch 2.14.1+cu130`, `flashinfer-python 0.7.4`, 12 CPU cores, 52 GiB RAM. Revision: `main` at
`6c0eb45`. Raw JSON outputs are saved under `benchmarks/results/2026-11-22-nvidia-l4/`. Product
defaults are **not** changed by this run; each gate records what its measurement implies for
defaults and for the 0.2 release statement.

| Gate | Setup | Threshold, written before the run | Implication |
| --- | --- | --- | --- |
| `OPS-B2` #1 (sampling RNG) | Inspect `flashinfer.sampling` (`sampling_from_probs`, `top_k_top_p_sampling_from_logits`) on `flashinfer-python 0.7.4` | Keep sampling on `torch` unless an API accepts external uniform draws `u` matching Ayaka's counter-based inverse-CDF contract | `ops/sampling.py` stays on `torch` |
| `OPS-B2` #2 (decode CUDA graph on `sm_89`) | `BatchDecodeWithPagedKVCacheWrapper(use_cuda_graph=True, use_tensor_cores=True)`, capture once, 100 replays with random lengths (re-planned outside graph), `fp16` and `bf16` | Max absolute error vs `torch` reference ≤ `4 * eps * max(|y|)` on all 100 replays | Decode CUDA graphs stay on `flashinfer` on `sm_89` |
| `OPS-B2` #3 (multi-row ragged causal prefill) | `BatchPrefillWithPagedKVCacheWrapper(causal=True)`, 5 mixed requests `(rows/len: 7/7, 1/9, 4/16, 3/20, 16/40)`, NHD and HND, `fp16` and `bf16` | Max absolute error vs `torch` reference ≤ `4 * eps * max(|y|)` | Chunked prefill and verify windows stay on `flashinfer` on `sm_89` |
| `OPS-B2` #4 (FP8 KV cache on `sm_89`) | `BatchDecodeWithPagedKVCacheWrapper` and `BatchPrefillWithPagedKVCacheWrapper` with `kv_data_type=torch.float8_e4m3fn` (`k_scale`, `v_scale`), head dim 64 and 128 (`Llama-3.2-1B` / `TinyLlama` geometry), context 2048 and 8192 | Runs without error or NaN/Inf on both decode and prefill; relative L2 error vs `fp64` oracle ≤ `5%`; decode latency at batch 1 ctx 8192 ≤ `1.10×` `fp16` | Decides whether an 8-bit KV path is viable on `sm_89` before `kops` (0.2 release statement) |
| Attention per op (`OPS-11`) | `benchmarks/ops/attention.py --dtype fp16` and `--dtype bf16`, warmup 5, repeat 50 | `flashinfer` faster than `torch` on decode and prefill shapes | Confirms `backends=("flashinfer", "torch")` on `sm_89` |
| `fp16` vs `bf16` (`EXE-10`) | `benchmarks/e2e/decode.py` on `HuggingFaceTB/SmolLM2-135M` and `unsloth/Llama-3.2-1B-Instruct`, `--tokens 129 --batch 1 4 --repeat 3` | Keep `fp16` default unless `bf16` with `flashinfer+torch` + CUDA graph is faster by ≥ `5%` at batch 1 and 4 (parity already green in `test_real_llama.py`) | Default compute dtype on `sm_89` |
| `graph` vs `eager` batch 1 (`EXE-3`) | Same `benchmarks/e2e/decode.py` runs, `flashinfer+torch` `fp16` batch 1 | CUDA graph speeds up batch-1 decode by ≥ `1.5×` over eager on both models | Implication for enabling `cuda_graph=True` by default on CUDA |
| `memory_ratio` & `plan_budget` vs ledger (`MEM-2`, `MEM-16`) | Fresh-process `plan_engine` vs `build_engine` (`StartupReport`) on `unsloth/Llama-3.2-1B-Instruct` (`fp16`, `flashinfer+torch`, `--cuda-graph`, default `memory_ratio=0.90`, `kv_reserve_tokens=32768`), cold (no profile) and warm (saved profile) | `reconciliation.matches` is `True` (within 5% / 64 MiB floor) at `memory_ratio=0.90` on native Linux; report NVML vs CUDA free gap and cold-vs-warm KV block delta | Confirms default `memory_ratio=0.90` on native Linux and quantifies first-start plan drift |

### Result

*(Pending run on `m1-l4-gates`.)*
