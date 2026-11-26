# Troubleshooting

Each entry is a symptom, its cause and a fix. The numbers in the examples were seen on the development machine
(an RTX 3050 Laptop GPU with 4 GB under WSL2); yours will differ.

## The model does not fit

**Symptom.** `ayaka plan`, `serve`, `run` or `bench` stops with exit code 5 and a line such as

```text
DOES NOT FIT: KV floor needs 6,454 MiB, 2,948 MiB is available, 3,506 MiB short
```

**Cause.** By default every request may reach the model's whole context, and the KV cache for it must fit next to
the weights. Llama-3.2-1B-Instruct allows 131,072 tokens, which needs 6.4 GB of KV in fp16: more than a 4 GB GPU has.

**Fix.** Ask for the context you need with `--kv-reserve-tokens N`; the cache still takes every block the memory
allows, `N` is only the floor. `ayaka plan MODEL --kv-reserve-tokens N` shows the result without loading anything,
and `ayaka plan MODEL --for claude-code` (or `codex`, `opencode`) says whether a request holds that tool's prompt.

## The weights themselves do not fit: FP8 fits, but is slow

**Symptom.** `ayaka plan` says the weights alone (or the weights and the shortest context you need) do not fit,
for example Phi-4 (14B) in bf16, 29 GB, on a 24 GB GPU.

**Cause.** 16-bit weights take two bytes per parameter.

**Fix.** Use an FP8 checkpoint of the same model: Hugging Face `fp8` (such as `Qwen/Qwen3-8B-FP8`) or
`compressed-tensors` with 8-bit float weights (such as RedHat AI's `-FP8` checkpoints). Ayaka keeps those weights in
8 bit and `ayaka plan` says so in one line. It fits, but it is slow: until there is an FP8 matrix kernel, every
product first dequantizes its weights. On an NVIDIA L4 24 GB, three alternating rounds of `ayaka bench --quick`
generated 1.78 tok/s with Llama-3.1-8B-Instruct-FP8 against 15.6 tok/s in bf16, and 1.77 tok/s with Qwen3-8B-FP8
against 15.1 tok/s (0.11 to 0.12 of the 16-bit speed), with the same first tokens on 32 of 32 fixed prompts
([results](../benchmarks/results/2026-11-24-nvidia-l4/), [record](EXPERIMENTS.md)). The memory it frees goes to the
KV cache: 91,552 tokens per request for Llama-3.1-8B in FP8 against 38,128 in bf16. A model that fits in 16 bit
should stay in 16 bit.

## Not enough headroom for a prefill step

**Symptom.**

```text
headroom for one prefill block: need 507853888 bytes, have 332356649, short 175497239
```

**Cause.** `--memory-ratio` (0.9 by default) is the share of free memory the engine plans with; the rest is headroom
for the activations of a prefill step. After the first start-up on a machine, Ayaka measures the runtime memory and
plans with it, so a ratio that worked once can be refused on the next start.

**Fix.** Lower `--memory-ratio` (0.86 worked for Llama-3.2-1B on the 4 GB GPU after the measurement; 0.95 did not).
`ayaka plan` prints the usable memory and the headroom for a ratio.

## Prompts read at about half speed under WSL2

**Symptom.** Under WSL2, a long prompt takes about twice as long to read as it did on an earlier start with the
same flags, with no error. The start-up log reads, for example,

```text
device free cross-check: NVML 4154458112 bytes before the backend, CUDA 3459986228 bytes after it
startup run: ... runtime outside the allocator 272629760 bytes ...
```

where the runtime is well below the gap between the two free figures (here 266 MB against a gap of about 690 MB).

**Cause.** Under WSL2, NVML reports more free memory than CUDA does. Ayaka plans from the NVML figure and from the
runtime memory it measured on the previous start. When a start measured too little runtime, the plan gives the
allocator more than CUDA's free memory; the driver accepts it, prompt reading slows down, and that start measures
too little again. Seen on the development machine with Llama-3.2-1B-Instruct, `--memory-ratio 0.86
--kv-reserve-tokens 8192`: 1,900 tok/s instead of 3,400 tok/s on a 3,549-token prompt. This is a bug in the
planning, not in your setup.

**Fix.** Start once with a lower `--memory-ratio` (0.80 on the development machine), stop the server, and start
again with your usual ratio: the lower start measures the real runtime, and the next starts plan with it. If the
usual ratio then stops with "KV floor ... short", raise it a little or lower `--kv-reserve-tokens`.

## No GPU is used

**Symptom.** `ayaka info` shows `CUDA none, not usable`, or `--device cuda` fails with a capability error naming
the driver.

**Cause.** The installed PyTorch is the CPU build (`--extra cpu`), or the NVIDIA driver is not visible (on WSL2, the
Windows driver provides it).

**Fix.** `uv sync --extra cu130 ...` instead of `--extra cpu` (never both), and check `nvidia-smi` in the same shell.

## flashinfer does not load

**Symptom.** An error naming flashinfer and the remedy
`install it on Linux or WSL2 with uv sync --extra cu130 --extra flashinfer`.

**Cause.** flashinfer is Linux-only, needs the `cu130` PyTorch, and compiles kernels on first use, which needs the
CUDA toolkit's `nvcc` on `PATH`.

**Fix.** Install the extras above on Linux or WSL2, put the toolkit on `PATH` (for example
`export PATH=/usr/local/cuda/bin:$PATH`), or leave flashinfer out: `--backends torch`.

## The port is taken

**Symptom.**

```text
error while attempting to bind on address ('127.0.0.1', 8000): [errno 98] address already in use
```

**Cause.** Another server (often an earlier `ayaka serve`) listens on the port.

**Fix.** Stop it, or pass `--port N` (`--port 0` picks a free one and prints it).

## 401 from the server

**Symptom.** Every request answers 401.

**Cause.** The server was started with `--api-key` or `--api-key-file` and the client sends no key or another one.

**Fix.** Send the key as `Authorization: Bearer KEY` or `x-api-key: KEY`. `ayaka connect TOOL` writes it where each
tool expects it.

## WSL2 runs out of memory while loading

**Symptom.** The process is killed while a model loads, or the machine swaps.

**Cause.** WSL2 gives Linux a share of the machine's RAM (`free -g` shows it), and loading a model needs room for its
weights in RAM before they move to the GPU.

**Fix.** Close other programs, or give WSL2 more memory in `%UserProfile%\.wslconfig` (`[wsl2]` `memory=...`) and run
`wsl --shutdown`.

## A tokenizer, Hub or serving extra is missing

**Symptom.** An error saying a capability is not available, with a remedy such as `uv sync --extra hf`,
`uv sync --extra serve` or `uv sync --extra structured`.

**Cause.** The extra that provides it is not installed: Hugging Face tokenizers and Hub downloads need `hf`,
`ayaka serve` needs `serve`, JSON answers and forced tool calls need `structured`.

**Fix.** Run the remedy the error names (together with the extras you already use: `uv sync` removes extras that are
not listed).

## A coding tool cannot work with the server

Run `ayaka connect TOOL --check` against the running server. Each of its seven steps that fails names one fix; the
last one says whether one request holds the tool's measured prompt.

## Preflight check reports warnings or failures (`ayaka doctor`)

**Symptom.** `ayaka doctor` (or `ayaka doctor --output json`) exits with status 1 or prints `[WARN]` / `[FAIL]` lines.

**Cause.** The host environment has an issue before serving: missing CUDA PyTorch build, an NVML vs CUDA free-memory gap
exceeding 20% under WSL2, insufficient `cudaHostAlloc` pinned RAM ceiling, missing `nvcc` when `flashinfer` is
installed, low free disk space in the model cache directory, or stale runtime startup profiles.

**Fix.** Follow the `fix:` line printed under each check: for example `uv sync --extra cu130` for CPU-only torch,
lowering `--memory-ratio` (around `0.80` on WSL2) or resetting `~/.cache/ayaka/startup_profiles`, freeing cache disk
space, adding the CUDA toolkit `bin` directory to `PATH` or selecting `--backends torch`, or running
`ayaka plan MODEL` to refresh startup profiles.

## Out-of-memory recovery, preemption timeouts, and over-long inputs

**Symptom.** Requests receive `503 out_of_memory` (`Retry-After`), `408 preempted_timeout`, or stop early with
`finish_reason: "length"` even below the model's trained context limit.

**Cause.** When KV cache occupancy reaches `pressure_high` / `pressure_critical`, new waiting admissions pause and
excess requests are preempted by releasing their KV blocks (and recomputing them once space returns). If a step throws
a device out-of-memory error, the runner drops stale compiled graphs/caches, frees memory, and the scheduler preempts a
victim or halves the prefill chunk size instead of crashing the engine. A request that waits longer than
`preemption_timeout` without room fails with `408 preempted_timeout`; a single minimal request that still exhausts
device memory fails with `503 out_of_memory`. Prompts that exceed the machine context/KV limit fail with `400` by
default.

**Fix.** Reduce `--memory-ratio` slightly or lower concurrency (`--max-running`), retry `503 out_of_memory` /
`408 preempted_timeout` responses after backoff, or opt in to oldest-turn truncation on the server with
`--overlong-policy truncate` (or `truncation: "auto"` on `POST /v1/responses`).
