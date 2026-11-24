# Changelog

## Unreleased

### Added

- Qwen2 and Qwen2.5 checkpoints (`Qwen2ForCausalLM`). A model whose sliding window is shorter than its trained context
  is served up to the window, because windowed attention is not implemented; `ayaka plan` shows the shorter context.
- Qwen3 checkpoints (`Qwen3ForCausalLM`), whose `<think>` reasoning and `<tool_call>` calls go through the `think` and
  `hermes` parsers.
- Phi-3 and Phi-4 checkpoints (`Phi3ForCausalLM`). `longrope` scaling and a partial rotary embedding (Phi-3.5-mini,
  Phi-4-mini) are refused with the reason. Phi-3-mini-4k is served up to its 2,047-token sliding window.
- When the first listed backend has no attention kernels for a model's head width (flashinfer: 64, 128 and 256 only),
  attention comes from the next backend in `--backends`, with a log line, instead of failing on the first request.
- FP8 checkpoints of the Llama, Qwen2, Qwen3 and Phi-3 architectures: Hugging Face `fp8` (e4m3, `weight_block_size`
  blocks, as in Qwen3-8B-FP8) and `compressed-tensors` 8-bit float weights with per-tensor, per-channel or block scales
  (as in RedHatAI's FP8 checkpoints). The weights stay 8 bit in memory and each linear layer dequantizes a bounded slice
  of rows before its product, which is slower than 16-bit weights; `ayaka plan` says so in one line. Activation scales
  are not used. Any other `quantization_config` is refused with the reason.
- `ayaka plan <file>.gguf` plans a `llama` GGUF file from its header: the weights are the sum of the tensor sizes the
  header gives (any mix of GGML quantization types), and the KV cache follows the metadata. The engine does not run
  GGUF tensors and refuses such a file with the reason.
- A mixture-of-experts op family, `moe` (`route`: float32 softmax and top-k, ties to the lower expert id;
  `experts`: the SiLU-gated expert MLPs), and `MoEBlock`, which holds every expert in device memory. The torch backend
  picks one of three exact strategies by the number of rows; CUDA graphs capture MoE steps of up to 32 rows.
- OLMoE (`OlmoeForCausalLM`, such as OLMoE-1B-7B) and Qwen3-MoE (`Qwen3MoeForCausalLM`, such as Qwen3-30B-A3B)
  checkpoints, with every expert in device memory. `clip_qkv` and quantized MoE checkpoints are refused with the
  reason. `ModelConfig` reads the expert fields of `config.json`.

### Fixed

- The torch backend's decode attention no longer runs out of GPU memory on long-context models (#122). It gathered
  every request's whole block table, sized for the model's full context, into float32 at once: for Llama-3.2-1B
  (131,072 tokens) with 4 requests, a decode step peaked at 3,201 MiB, which the memory plan does not count. It now
  walks the table in chunks of at most 64 MiB of keys with an online softmax, and the same step peaks at 232 MiB. The
  flashinfer backend was not affected.

- The start-up memory check no longer fails for an engine built in a process that already used the GPU (#123). The
  start-up run read the allocator while it still cached the run's freed scratch and counted that cache as workspace;
  capturing CUDA graphs then gave it back, so the allocator held less than the ledger. On an NVIDIA L4 the sixth
  engine built in one test process was 402 MiB short against a 336 MiB tolerance. The cache is now released before
  the reading, as the capture does.

## 0.1.0 (not published on PyPI)

On an NVIDIA L4 24 GB, Ayaka reads a 4,096-token prompt of Llama-3.1-8B-Instruct in fp16 at 2,393 tok/s and
generates 15.1 tok/s after it, 0.93× the generation speed of llama.cpp on the same card and model in f16
([bench result](benchmarks/results/2026-11-23-nvidia-l4/llama-3.1-8b-instruct/results.json),
[comparison](benchmarks/results/2026-11-23-nvidia-l4/compare-llama-cpp-f16-llama-3.1-8b.json); both measured at commit
`46437b4`, before the SentencePiece grammar fix of #106, which leaves byte-level tokenizers such as Llama 3's unchanged).

The distribution is named `ayaka-llm`, because the name `ayaka` is taken on PyPI. The package is still imported as
`ayaka` and the command is still `ayaka`. This version is installed from a checkout ([Install](docs/INSTALL.md)).

### What 0.1.0 does

- Runs Llama-architecture checkpoints (and GPT-2 on the CPU) in fp16 or bf16 on an NVIDIA GPU through flashinfer and
  CUDA graphs, with a hash-chain prefix cache, chunked prefill and several sequences per request.
- `ayaka serve` answers OpenAI (Chat Completions, Completions, Responses) and Anthropic (Messages) clients, with
  streaming, tool calls (`hermes` and `llama3_json`), reasoning sections and, with the `structured` extra, JSON answers
  and forced tool calls checked by a grammar.
- `ayaka plan` prints whether a model fits before anything is loaded; `ayaka info` and `ayaka config show` print the
  machine and the settings.
- `ayaka connect` writes, checks and observes the configuration of Claude Code, Codex and OpenCode.
- `ayaka bench` measures prompt read, generation, the first token, a short agent session and peak memory, with two
  sanity checks, and writes every raw run to `results.json`.

What runs where, and the result file behind every number, is in [Capabilities](docs/CAPABILITIES.md).

### Limits

- Only Llama-architecture models run on the GPU. Other families are not supported yet.
- Linux and WSL2 only; there is no native Windows build.
- There are no native (C++/CUDA) extensions of Ayaka's own yet; the GPU kernels come from flashinfer and PyTorch.
- There is no preemption yet. A prompt longer than the KV cache is refused; when running requests together run out of
  KV cache memory, the lowest-ranked one ends early with `length` and a note that memory was the limit.
