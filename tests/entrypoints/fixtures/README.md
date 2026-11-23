# Recorded client requests

One sanitized recording per real request a coding tool sent to `ayaka serve`, stored as
`{"path", "headers", "body"}`. The replay tests live in `tests/entrypoints/test_replay.py`
(streaming, unknown fields, refusals) and `tests/e2e/test_serving.py` (the answer parsed by the
tool's own client library).

## What was removed

A recording keeps the tool's whole request: its system prompt, the project files it carries and
the conversation. That is private, so a fixture keeps only the request's *shape*:

- headers: only the ones that say how the tool talks (`user-agent`, the feature flags), never
  the session ids, the credentials or the machine;
- bodies: every free text (system prompt, instructions, user and developer messages, tool
  descriptions, including inside the tool schemas, tool-call arguments and tool outputs,
  reasoning summaries) is replaced with a `<redacted: ...>` placeholder; arguments sent as JSON
  text become `{"redacted": true}`, so the server still parses them; ids of messages, sessions
  and tool calls are replaced; field names, roles, block types,
  sampling fields and the tool schemas' structure stay as the tool sent them.

`tests/_support/sanitize_recording.py` does this and refuses to write a file in which a marker
from `PRIVATE_MARKERS` survives; `TestReplay::TestFixtureHygiene` re-checks every fixture.

## The recordings

| Fixture | Tool (version) | Request |
| --- | --- | --- |
| `claude-code-2.1.327/001` | Claude Code 2.1.327 (Messages) | the first request of a session: `thinking` adaptive with `display`, `context_management`, `metadata`, `output_config`, 14 tools |
| `claude-code-2.1.327/003` | Claude Code 2.1.327 (Messages) | the same request with `thinking` without `display` |
| `claude-code-2.1.327/004` | Claude Code 2.1.327 (Messages) | a later session with 23 tools |
| `claude-code-2.1.330/002` | Claude Code 2.1.330 (Messages) | a tool round trip after executing `Read`: assistant `tool_use` and user `tool_result` turns, recorded live against `unsloth/Llama-3.1-8B-Instruct` with `--tool-call-parser llama3_json` |
| `codex-0.163.0/001` | Codex CLI 0.163.0 (Responses) | the first request of a session: `reasoning`, `include`, `client_metadata`, `prompt_cache_key`, a `namespace` tool and a `web_search` tool among the function tools |
| `codex-0.171.0/002` | Codex CLI 0.171.0 (Responses) | a tool round trip: a `function_call` item (`exec_command`) and its `function_call_output`, recorded live against `unsloth/Llama-3.1-8B-Instruct` with `--tool-call-parser llama3_json`. Codex did not run the command: the model wrote numbers and booleans as strings (`"10000"`, `"true"`), and the output Codex sent back is its argument error |
| `opencode-2.3.2/001` | OpenCode 2.3.2 (Chat Completions) | an agent-shaped request: the tool's full system prompt, `max_completion_tokens`, `stream_options` and 13 tools |
| `opencode-2.3.2/002` | OpenCode 2.3.2 (Chat Completions) | the session-title request that opens a run |

The raw recordings were taken on the dev machine with
`ayaka connect <tool> --observe --record <dir>` while the tool ran against `ayaka serve`
(Claude Code and Codex on SmolLM2-135M; OpenCode on the same server). They stay outside the
repository; only these sanitized fixtures are committed.

## Redoing a recording

1. Serve a model: `ayaka serve <model> --served-model-name smollm2 --tool-call-parser hermes`.
2. Start the proxy: `ayaka connect <tool> --observe --record <dir> --tokenizer <model>`; it
   prints the configuration to point the tool at the proxy. Wait for the tool's first requests.
3. Sanitize into this directory and replace the fixtures:
   `uv run python tests/_support/sanitize_recording.py <dir>/*.json --to <this directory>`.
   The hygiene test must pass, and the replay tests must stay green.
4. If the tool's request shape changed in a way the server refuses or mis-handles, fix the
   protocol adapter and add what the new shape needs to the replay tests.

## Golden wire bytes

`golden/` is not a recording: it holds the exact bytes the server writes, compared byte for byte
by `tests/entrypoints/test_golden_frames.py` (every stream encoder, whole answer and error shape on
fixed events) and `tests/entrypoints/test_replay.py` (`golden/replay/`, the streamed answer to each
recording above with the request id and clock fixed). Rewrite them with
`AYAKA_UPDATE_GOLDEN=1 uv run pytest tests/entrypoints/test_golden_frames.py tests/entrypoints/test_replay.py`
and review the diff: a changed file is a changed wire.
