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
