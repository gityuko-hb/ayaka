# Ayaka reference (for agents)

What the server and the commands do, as the code does it. **Not stated here means unknown, not unsupported**: check
the code or ask before assuming either way.

## Endpoints

`ayaka serve` (the `serve` extra) listens on `127.0.0.1:8000` by default.

| Method and path | Protocol | Notes |
| --- | --- | --- |
| `GET /health` | | `{"status": "ready"}` with 200; `starting`, `draining`, `stalled` or `dead` with 503. `stalled`: the engine has had work and no progress for `--stall-timeout` seconds (60); it goes back to `ready` once a step finishes, and the server keeps serving meanwhile. No key needed. |
| `GET /metrics` | Prometheus text | counters, gauges and histograms; no per-request labels. `ayaka_engine_progress_age_seconds` is the seconds since the engine last made progress while it had work (0 when idle) |
| `GET /v1/models` | OpenAI | each served name with `max_model_len`, `max_context_tokens` (what one request can hold on this machine), `prefix_cache` and, when detected from the chat template, `suggested_tool_call_parser` |
| `POST /v1/chat/completions` | OpenAI Chat Completions | streaming, tools, `n`, logprobs, `response_format` |
| `POST /v1/completions` | OpenAI Completions | streaming; `echo` and `suffix` are refused |
| `POST /v1/responses` | OpenAI Responses | stateless: `previous_response_id` and `conversation` are refused; `n` must be 1 |
| `POST /v1/messages` | Anthropic Messages | streaming, tools, `system` messages inside `messages`; `n` must be 1 |
| `POST /v1/messages/count_tokens` | Anthropic | the prompt length of a Messages request |

With `--api-key` or `--api-key-file`, every path but `/health` needs `Authorization: Bearer KEY` or `x-api-key: KEY`.

## Streaming: the last events

| Protocol | The stream ends with |
| --- | --- |
| Chat Completions, Completions | a chunk with `finish_reason`, then `data: [DONE]` (a usage chunk before it with `stream_options.include_usage`) |
| Responses | `event: response.completed`, or `response.incomplete` when `max_output_tokens` cut it |
| Messages | `event: message_stop` |

`ayaka connect TOOL --check` checks these on a running server, in the tool's protocol.

Usage and the prefix cache: Chat Completions, Completions and Responses report the prompt tokens read from the
prefix cache as `cached_tokens`; Responses also reports the prompt tokens this request registered in it as
`cache_write_tokens`. Messages splits the prompt three ways that add up to it: `cache_read_input_tokens` (read
from the cache), `cache_creation_input_tokens` (computed and registered, whole cache blocks only) and
`input_tokens` (the rest). A streamed Messages answer reports `input_tokens` alone in `message_start` and the whole
usage, cumulatively, in `message_delta`, so the SDK's final message has the same usage as an unstreamed answer.

## Errors

| Cause | Status | Notes |
| --- | --- | --- |
| an invalid field | 400 | names the field |
| a feature this server does not have | 400 | names the feature |
| a prompt that leaves no room in the model's context for one output token | 400 | code `context_length_exceeded`; the message gives the prompt tokens, the requested output tokens and the model's limit, and starts with `the prompt is too long` |
| a prompt that needs more KV than one request can get | 400 | |
| no key or a wrong key | 401 | |
| too many requests waiting | 429 | `Retry-After` |
| a request that timed out | 408 | |
| not ready, draining, or the engine stopped | 503 | `Retry-After`; no internal detail, the request id only |
| anything else | 500 | logged with its traceback |

Shapes: OpenAI endpoints answer `{"error": {"message", "type", "param", "code"}}`; Anthropic endpoints answer
`{"type": "error", "error": {"type", "message"}, "request_id"}` with the same id in a `request-id` header. A
Responses stream that fails ends with `response.failed`, whose `error.code` is `context_length_exceeded`,
`rate_limit_exceeded`, `invalid_prompt` or `server_error`. Every code comes from one fixed table; a failure with no
row in it is answered as `internal_error`. A request that asks for more output than its context leaves is not an
error: `max_tokens` is capped to the room. Unknown request fields are ignored (with a debug log line naming
them); a known field that would change the answer and is not available is refused with 400.

## Tool calls and structured output

- Tool calls need `--tool-call-parser` (`hermes`, `llama3_json`). Reasoning blocks need
  `--reasoning-parser` (`think`, `deepseek_r1`). With `think` the block counts only when it is the first thing
  the model writes; a `<think>` after answer text is text. With `deepseek_r1` the prompt opened the block.
  A grammar starts where the parser starts reading: after the block, or at once when the model does not open one.
- `llama3_json` reads the Llama 3.x layout: a bare JSON object `{"name": ..., "parameters": {...}}` at the start of
  the answer, and more calls separated by `;`. A string `function` next to `"type": "function"` counts as the name,
  because Llama-3.2-1B writes that when the tools are shown wrapped that way. With no marker to wait for, a forced
  call (`tool_choice`) is constrained from the first token of the answer; strict arguments of an optional call
  cannot be constrained and are refused with 400. It has been checked
  against recorded answers of Llama-3.2-1B-Instruct (`tests/entrypoints/parsers/fixtures/`), written outside
  `ayaka serve`; no Llama model has driven a coding tool through this parser yet.
- With the `structured` extra: JSON answers (`response_format`, `text.format`, `output_format`), forced tool calls
  (`tool_choice`) and strict arguments are enforced by a grammar; a request that offers tools gets its calls checked
  while the rest of the answer stays free. Without the extra, a request that needs a grammar is refused.

## Commands

| Command | What it does | Exit codes |
| --- | --- | --- |
| `ayaka run MODEL -p TEXT` | generate for one prompt, print the text | the common codes below |
| `ayaka serve MODEL` | the HTTP server above | the common codes below |
| `ayaka plan MODEL [--for TOOL]` | the memory plan, without loading weights | 0 fits, 5 does not fit (or one request cannot hold TOOL's prompt) |
| `ayaka info` | version, devices, backends, extras, directories | 0 |
| `ayaka config show` | the server settings and where each comes from | 0 |
| `ayaka connect TOOL [--write\|--check\|--observe]` | configure and check a coding tool | `--check`: 0 no step failed, 1 a step failed, 2 could not run |
| `ayaka bench MODEL` | measure prompt read, generation, first token, an agent session, sanity | 0 done, 1 the benchmark failed, 2 a sanity check failed (results written, marked invalid) |

The common exit codes: 0 done; 1 an unexpected error (a bug; `-v` prints the traceback); 2 wrong arguments or a
wrong request; 3 the model or the tokenizer cannot be loaded; 4 something needed is not installed or available (torch,
a tokenizer package, a GPU); 5 the memory plan does not fit; 6 the engine failed while generating; 130 interrupted.
A problem the user can fix is one line, `ayaka: error: ...`, never a traceback.

Flags shared by `run`, `serve`, `plan` and `bench`: `--device`, `--dtype`, `--backends`, `--cuda-graph`,
`--memory-ratio`, `--kv-reserve-tokens`, `--no-prefix-cache`, `--revision`, `--tokenizer`, `--offline`.
`serve` and `bench` also take `--record DIR` (token ids of every request, owner-only). `ayaka COMMAND --help` lists
every flag with its default.

## Coding tools

`ayaka connect` knows `claude-code` (Anthropic Messages), `codex` (OpenAI Responses) and `opencode` (OpenAI Chat
Completions), each verified against one installed version (in the record's `verified` field). Any other tool gets the
generic OpenAI and Anthropic settings. Each record carries the tool's measured fixed prompt in tokens, with the
version and tokenizer that measured it; `--check` step 7 and `ayaka plan --for` compare it with the context one
request can hold.
