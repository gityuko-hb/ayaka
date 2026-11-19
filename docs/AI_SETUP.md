# Setting Ayaka up for someone (for agents)

A sequence an assistant can follow to install Ayaka on a user's machine and point a coding tool at it. Every step has
a check; stop at the first check that fails and use [TROUBLESHOOTING.md](TROUBLESHOOTING.md). Do not skip a check,
and do not report a step as done without its check's output.

## Prerequisites to confirm with the user

- The operating system: Linux, WSL2 on Windows, or another one (then: CPU only, no flashinfer).
- Whether there is an NVIDIA GPU (`nvidia-smi` answers).
- The model to serve (a Hugging Face repo id or a local directory with `config.json` and safetensors files), and
  whether downloading it is allowed.
- Which coding tool: `claude-code`, `codex` or `opencode` (others get the generic setup).

## Steps

1. Get the code and install (see [INSTALL.md](INSTALL.md) for the extras):

   ```sh
   git clone https://github.com/gityuko-hb/ayaka.git && cd ayaka
   uv sync --extra cu130 --extra flashinfer --extra serve --extra structured --extra hf
   ```

   Without a GPU use `--extra cpu` instead of `--extra cu130 --extra flashinfer`.
   **Check:** `uv run ayaka info` lists the expected device and extras.

2. Make sure the model fits, for the tool:

   ```sh
   uv run ayaka plan MODEL --device cuda --dtype fp16 --kv-reserve-tokens 32768 --for TOOL
   ```

   **Check:** exit code 0 and a `for` line that says `FITS`. Exit code 5 prints what is short; lower
   `--kv-reserve-tokens`, change `--memory-ratio`, or choose a smaller model. A repo id must be in the Hub cache for
   `plan`; download it first with `ayaka run MODEL -p hi` if needed.

3. Start the server with a key, keeping it on this machine:

   ```sh
   head -c 24 /dev/urandom | base64 > ~/.ayaka-key && chmod 600 ~/.ayaka-key
   uv run ayaka serve MODEL --device cuda --dtype fp16 --backends flashinfer,torch --cuda-graph \
       --kv-reserve-tokens 32768 --tool-call-parser hermes --served-model-name local \
       --api-key-file ~/.ayaka-key
   ```

   Use the same memory flags as in step 2, and the tool-call parser of the model's format (`hermes`, or
   `llama3_json` for Llama 3.x). **Check:** the server prints `ready: local on http://127.0.0.1:8000`, and
   `curl -s localhost:8000/health` answers `{"status": "ready"}`.

4. Check the tool's protocol against the server (from another terminal):

   ```sh
   uv run ayaka connect TOOL --check --url http://127.0.0.1:8000 --api-key-file ~/.ayaka-key
   ```

   **Check:** exit code 0. A failed step names its fix; `--output json` gives the same report as data.

5. Show the user the change to the tool's configuration, then write it:

   ```sh
   uv run ayaka connect TOOL --url http://127.0.0.1:8000 --api-key-file ~/.ayaka-key --write
   ```

   It prints a diff (the key redacted), backs the old file up next to it and asks before writing; `--yes` writes
   without asking and is only for when the user has seen the diff. **Check:** the line `wrote PATH` (with the backup's path when a file was replaced).

6. Start the tool as the last line of step 5 says (for Codex and OpenCode, with `AYAKA_API_KEY` set to the key).

## What not to do

- Do not put the key on the command line of the server (`--api-key`) on a shared machine: use `--api-key-file`.
- Do not bind the server to `0.0.0.0` unless the user asks; without a key it would serve anyone on the network.
- Do not edit the tool's configuration by hand when `connect --write` can do it with a diff and a backup.
