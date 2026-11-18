# Safety and privacy

Four questions, and what the code does about each.

## Does Ayaka send telemetry?

No. Ayaka has no telemetry, crash reporting or update check.

## When does it use the network?

- **Downloading a model or tokenizer** from the Hugging Face Hub, when the model is a repo id that is not in the local
  cache (through `huggingface_hub`, the `hf` extra). `--offline` forbids it: only the local cache is read, and a
  missing file is an error. `ayaka plan` never downloads.
- **Serving**: `ayaka serve` listens on `127.0.0.1:8000` by default, reachable from this machine only.
- **`ayaka connect`** talks to the server URL it is given (`--url`, by default the one in the server settings), and
  `--observe` runs a proxy on `127.0.0.1` unless `--listen` says otherwise.
- Building the flashinfer kernels or installing extras with `uv` downloads packages; that is uv's network use, not
  Ayaka's.

## What stays on the machine?

Everything a request contains: prompts, answers and token ids are kept in memory while the request runs and are not
written anywhere unless you ask:

- The server log carries request ids, paths, status codes, token counts, times, error codes and the names of fields
  or tools it ignored; it does not write prompt or answer text.
- `--record DIR` (on `serve` and `bench`) writes every request's prompt and output **as token ids** to a session file
  in `DIR`. It is off by default; the file is created owner-only on Linux and macOS. Token ids are the content: anyone
  with the model's tokenizer can read them.
- `ayaka connect --observe` keeps token **counts** only; `--record DIR` there keeps whole requests, credentials
  removed, and warns that prompts are in them.
- `/metrics` exposes counters and timings, no content.

## How is a checkpoint trusted?

- Weights are read from safetensors only. A pickle checkpoint (`.bin`, `.pt`) is refused, because loading one can run
  arbitrary code. The model's Hub download fetches `config.json` and the safetensors files, nothing else; the
  tokenizer adapter fetches the tokenizer's own files.
- The model's code is Ayaka's own implementation of the architecture named in `config.json`; an architecture it does
  not implement is refused, not loaded from the repository. The trust policy for code shipped in a checkpoint
  repository is `strict` by default; allowing it is an explicit setting for repositories you trust.
- A revision can be pinned to a full commit SHA, and the `reproducible` setting refuses anything else.

## API key and reachability

- `--api-key` or `--api-key-file FILE` makes every endpoint but `/health` require the key, as
  `Authorization: Bearer KEY` or `x-api-key: KEY`. `--api-key-file` keeps the key out of the process list.
- Binding to an address other than loopback (`--host 0.0.0.0`) without a key prints a warning at start-up: anyone on
  the network could then use the model.
- The key is never printed by `serve`'s start-up lines, by `ayaka connect` (its diffs and reports redact it) or in
  `--observe` recordings.
