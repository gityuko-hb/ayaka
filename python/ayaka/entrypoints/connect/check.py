"""``ayaka connect --check``: seven steps against a running server, in the tool's own protocol.

Each step is a request the tool would send; a step that fails says how to fix it. The API key
is never written to a report.
"""

from __future__ import annotations

import http.client
import json
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Final, cast

from ayaka.__version__ import __version__
from ayaka.entrypoints.connect.tools import (
    CONTEXT_MARGIN_TOKENS,
    Protocol,
    ServerInfo,
    ToolRecord,
    json_object,
    redact,
)

if TYPE_CHECKING:
    from collections.abc import Mapping


__all__ = ["CachedShareThreshold", "CheckReport", "CheckStep", "check_lines", "run_check"]


#: Step 6 passes when the repeated request reads at least this share of its prompt from the cache.
#: Written before any run: a repeated identical request should reuse all but its last block.
CachedShareThreshold: Final = 0.5


@dataclass(frozen=True, slots=True)
class CheckStep:
    """One step of ``--check``: ``ok`` is ``None`` when the step could not be measured."""

    id: int
    name: str
    ok: bool | None
    detail: str
    fix: str | None = None


@dataclass(slots=True)
class CheckReport:
    """The result of ``--check``, as ``--output json`` prints it."""

    tool: str
    version: str | None
    protocol: str
    steps: list[CheckStep] = field(default_factory=list[CheckStep])

    @property
    def ok(self) -> bool:
        """Whether no step failed (a step that could not be measured is not a pass, nor a fail)."""
        return all(step.ok is not False for step in self.steps)

    def as_json(self) -> dict[str, object]:
        """The report as JSON-ready data."""
        return {
            "tool": self.tool,
            "version": self.version,
            "protocol": self.protocol,
            "ok": self.ok,
            "ayaka": __version__,
            "steps": [asdict(step) for step in self.steps],
        }


class _Client:
    """A minimal HTTP client for the checks: standard library, no SDK, and it never raises.

    Every call returns ``(status, body)``; a connection that fails is status ``0`` with the error
    as the body, so a step fails with what happened instead of stopping the check.
    """

    def __init__(self, server: ServerInfo, timeout: float) -> None:
        self.server = server
        self.timeout = timeout

    def _headers(self, protocol: Protocol) -> dict[str, str]:
        headers = {"content-type": "application/json"}
        if self.server.api_key:
            headers["authorization"] = f"Bearer {self.server.api_key}"
        if protocol == "anthropic":
            headers["anthropic-version"] = "2023-06-01"
        return headers

    def _open(self, request: urllib.request.Request) -> tuple[int, bytes]:
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:  # noqa: S310
                return response.status, response.read()
        except urllib.error.HTTPError as exc:
            return exc.code, exc.read()
        except (
            OSError,
            ValueError,
            http.client.HTTPException,
        ) as exc:  # refused, timed out, bad URL
            return 0, redact(f"{type(exc).__name__}: {exc}", self.server).encode()

    def get(self, path: str) -> tuple[int, bytes]:
        return self._open(
            urllib.request.Request(  # noqa: S310  # the server's own URL
                self.server.base_url.rstrip("/") + path, headers=self._headers("openai-chat")
            )
        )

    def post(self, path: str, body: Mapping[str, object], protocol: Protocol) -> tuple[int, bytes]:
        return self._open(
            urllib.request.Request(  # noqa: S310  # the server's own URL
                self.server.base_url.rstrip("/") + path,
                data=json.dumps(body).encode(),
                headers=self._headers(protocol),
                method="POST",
            )
        )


def _answered(status: int, body: bytes) -> str:
    """How the server answered, for a step's detail: the status and its error message."""
    if status == 0:
        return f"nothing ({body.decode(errors='replace')[:200]})"
    if status == 200:
        return "200"
    error = (json_object(body) or {}).get("error")
    message = cast("dict[str, object]", error).get("message") if isinstance(error, dict) else None
    text = str(message) if message else body.decode(errors="replace")
    return f"{status} ({text[:200]})"


_CHECK_TOOL: Final = {
    "name": "read_file",
    "description": "Read a file of the project",
    "schema": {
        "type": "object",
        "properties": {"path": {"type": "string", "maxLength": 40}},
        "required": ["path"],
        "additionalProperties": False,
    },
}


_PATHS: Final = {"anthropic": "/v1/messages", "openai-responses": "/v1/responses"}


def _path(protocol: Protocol) -> str:
    return _PATHS.get(protocol, "/v1/chat/completions")


def _text_request(protocol: Protocol, model: str, *, stream: bool) -> dict[str, object]:
    prompt = "Say hello in one short sentence."
    if protocol == "anthropic":
        return {
            "model": model,
            "max_tokens": 16,
            "temperature": 0,
            "messages": [{"role": "user", "content": prompt}],
            "stream": stream,
        }
    if protocol == "openai-responses":
        return {
            "model": model,
            "input": prompt,
            "max_output_tokens": 16,
            "temperature": 0,
            "stream": stream,
            "store": False,
        }
    return {
        "model": model,
        "max_tokens": 16,
        "temperature": 0,
        "messages": [{"role": "user", "content": prompt}],
        "stream": stream,
    }


def _tool_request(protocol: Protocol, model: str) -> dict[str, object]:
    """A request in the tool's own shape that forces one call of ``read_file``."""
    # A few blocks of prompt, so the repeat in step 6 has something to read from the cache.
    prompt = "Read the file README.md of this project. " + "Context line. " * 16
    name, description, schema = (
        _CHECK_TOOL["name"],
        _CHECK_TOOL["description"],
        _CHECK_TOOL["schema"],
    )
    if protocol == "anthropic":
        return {
            "model": model,
            "max_tokens": 128,
            "temperature": 0,
            "messages": [{"role": "user", "content": prompt}],
            "tools": [
                {"name": name, "description": description, "input_schema": schema, "strict": True}
            ],
            "tool_choice": {"type": "tool", "name": name, "disable_parallel_tool_use": True},
        }
    if protocol == "openai-responses":
        return {
            "model": model,
            "input": prompt,
            "max_output_tokens": 128,
            "temperature": 0,
            "store": False,
            "tools": [
                {
                    "type": "function",
                    "name": name,
                    "description": description,
                    "parameters": schema,
                    "strict": True,
                }
            ],
            "tool_choice": {"type": "function", "name": name},
            "parallel_tool_calls": False,
        }
    return {
        "model": model,
        "max_tokens": 128,
        "temperature": 0,
        "messages": [{"role": "user", "content": prompt}],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": name,
                    "description": description,
                    "parameters": schema,
                    "strict": True,
                },
            }
        ],
        "tool_choice": {"type": "function", "function": {"name": name}},
        "parallel_tool_calls": False,
    }


def _objects(value: object) -> list[dict[str, object]]:
    """The JSON objects of a list (anything else in it, or a non-list, is skipped)."""
    if not isinstance(value, list):
        return []
    items = cast("list[object]", value)
    return [cast("dict[str, object]", item) for item in items if isinstance(item, dict)]


def _first_message(answer: Mapping[str, object]) -> dict[str, object]:
    choices = _objects(answer.get("choices"))
    message = choices[0].get("message") if choices else None
    return cast("dict[str, object]", message) if isinstance(message, dict) else {}


def _text_of(protocol: Protocol, answer: Mapping[str, object]) -> str:
    if protocol == "anthropic":
        blocks = _objects(answer.get("content"))
        return "".join(str(b.get("text") or "") for b in blocks if b.get("type") == "text")
    if protocol == "openai-responses":
        texts: list[str] = []
        for item in _objects(answer.get("output")):
            if item.get("type") == "message":
                texts.extend(str(part.get("text") or "") for part in _objects(item.get("content")))
        return "".join(texts)
    content = _first_message(answer).get("content")
    return content if isinstance(content, str) else ""


def _call_of(protocol: Protocol, answer: Mapping[str, object]) -> tuple[str, object] | None:
    """``(name, arguments)`` of the first call, arguments decoded from JSON when they are text."""
    if protocol == "anthropic":
        for block in _objects(answer.get("content")):
            if block.get("type") == "tool_use":
                return str(block.get("name")), block.get("input")
        return None
    if protocol == "openai-responses":
        for item in _objects(answer.get("output")):
            if item.get("type") == "function_call":
                return str(item.get("name")), _decoded(item.get("arguments"))
        return None
    calls = _objects(_first_message(answer).get("tool_calls"))
    function = calls[0].get("function") if calls else None
    if not isinstance(function, dict):
        return None
    values = cast("dict[str, object]", function)
    return str(values.get("name")), _decoded(values.get("arguments"))


def _decoded(value: object) -> object:
    if isinstance(value, str):
        try:
            return json.loads(value)
        except ValueError:
            return value
    return value


def _number(value: object) -> int:
    return value if isinstance(value, int) and not isinstance(value, bool) else 0


def _cache_share(protocol: Protocol, answer: Mapping[str, object]) -> tuple[int, int]:
    """``(cached, prompt)`` tokens from the usage of an answer."""
    usage = answer.get("usage")
    values = cast("dict[str, object]", usage) if isinstance(usage, dict) else {}
    if protocol == "anthropic":
        cached = _number(values.get("cache_read_input_tokens"))
        return cached, cached + _number(values.get("input_tokens"))
    key, total = (
        ("input_tokens_details", "input_tokens")
        if protocol == "openai-responses"
        else ("prompt_tokens_details", "prompt_tokens")
    )
    details = values.get(key)
    found = cast("dict[str, object]", details) if isinstance(details, dict) else {}
    return _number(found.get("cached_tokens")), _number(values.get(total))


def _sse_lines(body: bytes, field: str) -> list[str]:
    prefix = f"{field}:"
    return [
        line.removeprefix(prefix).strip()
        for line in body.decode("utf-8", errors="replace").splitlines()
        if line.startswith(prefix)
    ]


def _stream_terminated(protocol: Protocol, body: bytes) -> tuple[bool, str]:
    """Whether the stream's last event is the protocol's terminal one, and its name."""
    events = _sse_lines(body, "event")
    if protocol == "anthropic":
        return bool(events) and events[-1] == "message_stop", "message_stop"
    if protocol == "openai-responses":
        # A stream cut by max_output_tokens ends with response.incomplete, also a terminal event.
        ended = bool(events) and events[-1] in ("response.completed", "response.incomplete")
        return ended, "response.completed or response.incomplete"
    data = _sse_lines(body, "data")
    finished = False
    for chunk in data[:-1]:
        for choice in _objects((json_object(chunk.encode()) or {}).get("choices")):
            finished = finished or bool(choice.get("finish_reason"))
    return bool(data) and data[-1] == "[DONE]" and finished, "a finish_reason, then [DONE]"


def _valid_arguments(arguments: object) -> bool:
    """Whether ``arguments`` follow the check tool's schema (written out, no validator library)."""
    if not isinstance(arguments, dict):
        return False
    values = cast("dict[str, object]", arguments)
    path = values.get("path")
    return set(values) == {"path"} and isinstance(path, str) and len(path) <= 40


def _wait_ready(client: _Client, timeout: float) -> tuple[bool, int, bytes]:
    deadline = time.monotonic() + timeout
    while True:
        status, body = client.get("/health")
        if status == 200 and (json_object(body) or {}).get("status") == "ready":
            return True, status, body
        if time.monotonic() >= deadline:
            return False, status, body
        time.sleep(0.5)


def run_check(
    record: ToolRecord,
    server: ServerInfo,
    *,
    timeout: float = 60.0,
    version: str | None = None,
) -> CheckReport:
    """The seven steps of ``--check`` against a running server, in the tool's protocol.

    A step that cannot get an answer fails with what happened; a step whose subject the server
    does not have (a prefix cache, a measured floor) is reported as not measured, never as a pass.
    """
    client = _Client(server, timeout)
    report = CheckReport(tool=record.name, version=version, protocol=record.protocol)
    steps = report.steps

    ready, status, body = _wait_ready(client, timeout)
    steps.append(
        CheckStep(
            1,
            "server ready",
            ready,
            f"/health answered {_answered(status, body)}"
            + (f": {body[:80].decode(errors='replace')}" if status == 200 else ""),
            None
            if ready
            else f"start the server (ayaka serve <model>) at {server.base_url}, or give it longer "
            "to load (--timeout)",
        )
    )
    if not ready:
        return report

    status, body = client.get("/v1/models")
    entries = _objects((json_object(body) or {}).get("data")) if status == 200 else []
    names = {str(entry.get("id")) for entry in entries}
    aliases = record.aliases(server)
    missing = [alias for alias in aliases if alias not in names]
    listed = bool(entries) and not missing
    steps.append(
        CheckStep(
            2,
            "model and aliases",
            listed,
            f"/v1/models answered {_answered(status, body)}; the server serves {sorted(names)}, "
            f"the tool sends {list(aliases)}",
            None
            if listed
            else "serve every name the tool sends: --served-model-name "
            + " --served-model-name ".join(missing or aliases),
        )
    )
    none: dict[str, object] = {}
    entry = next((e for e in entries if str(e.get("id")) == server.model), none)

    path = _path(record.protocol)
    status, body = client.post(
        path, _text_request(record.protocol, server.model, stream=False), record.protocol
    )
    text = _text_of(record.protocol, json_object(body) or {}) if status == 200 else ""
    steps.append(
        CheckStep(
            3,
            "text",
            bool(text.strip()),
            f"{path} answered {_answered(status, body)} with {len(text)} characters of text",
            None if text.strip() else "the server log names this request's error",
        )
    )

    status, body = client.post(
        path, _text_request(record.protocol, server.model, stream=True), record.protocol
    )
    ended, terminal = _stream_terminated(record.protocol, body)
    steps.append(
        CheckStep(
            4,
            "streaming",
            status == 200 and ended,
            f"the stream answered {_answered(status, body)}; it "
            f"{'ends' if ended else 'does not end'} with {terminal}",
            None
            if status == 200 and ended
            else (
                "a proxy between the tool and the server may buffer or cut server-sent events"
                if status == 200
                else "the server log names this request's error"
            ),
        )
    )

    tool_request = _tool_request(record.protocol, server.model)
    status, body = client.post(path, tool_request, record.protocol)
    answer = json_object(body) if status == 200 else None
    call = _call_of(record.protocol, answer) if answer is not None else None
    valid = call is not None and call[0] == _CHECK_TOOL["name"] and _valid_arguments(call[1])
    steps.append(
        CheckStep(
            5,
            "tool call",
            valid,
            f"{path} answered {_answered(status, body)}; call: {call!r}"[:300],
            None
            if valid
            else "serve with the model's --tool-call-parser and the structured extra "
            "(uv sync --extra structured), so a forced call follows the schema",
        )
    )

    if entry.get("prefix_cache") is False:
        steps.append(
            CheckStep(
                6,
                "prefix cache",
                None,
                "not measured: the server runs without a prefix cache (--no-prefix-cache)",
            )
        )
    else:
        status, body = client.post(path, tool_request, record.protocol)
        repeat = json_object(body) if status == 200 else None
        cached, prompt = _cache_share(record.protocol, repeat) if repeat is not None else (0, 0)
        share = cached / prompt if prompt else 0.0
        hit = status == 200 and share >= CachedShareThreshold
        detail = (
            f"the repeated request answered {_answered(status, body)} and read {cached} of "
            f"{prompt} prompt tokens from the cache ({share:.0%}; threshold "
            f"{CachedShareThreshold:.0%})"
        )
        if status == 200 and not hit:
            # Both prompts are the same request, so they agree everywhere: the cache stopped
            # matching at the first token it did not keep.
            detail += f"; the identical prompts diverge in the cache at token {cached}"
        steps.append(
            CheckStep(
                6,
                "prefix cache",
                hit,
                detail,
                None
                if hit
                else "serve with the prefix cache on (no --no-prefix-cache) and enough KV blocks "
                "to keep a prompt between two requests (ayaka plan)",
            )
        )

    steps.append(_context_step(record, server, entry))
    return report


def _context_step(record: ToolRecord, server: ServerInfo, entry: Mapping[str, object]) -> CheckStep:
    context = server.context_tokens
    value = entry.get("max_context_tokens") or entry.get("max_model_len")
    if isinstance(value, int):
        context = value
    floor = record.min_context_tokens
    if floor is None or context is None:
        reason = (
            f"no recorded fixed-prompt size for {record.name} "
            f"(ayaka connect {record.name} --observe measures it)"
            if floor is None
            else "the server does not say how long one request may be"
        )
        given = f"; the server gives {context} tokens per request" if context else ""
        return CheckStep(7, "context", None, f"not measured: {reason}{given}")
    needed = floor + CONTEXT_MARGIN_TOKENS
    enough = context >= needed
    return CheckStep(
        7,
        "context",
        enough,
        f"{context} tokens per request; {record.name} needs {floor} for its fixed prompt "
        f"({record.measured}) plus {CONTEXT_MARGIN_TOKENS} to work, {needed} in all",
        None
        if enough
        else "a model with a longer context, or more memory for the KV cache: "
        f"ayaka plan <model> --for {record.name}",
    )


def check_lines(report: CheckReport) -> list[str]:
    """The report as the table ``--check`` prints."""
    marks = {True: "pass", False: "FAIL", None: "n/a "}
    lines = [
        f"{report.tool} ({report.version or 'version unknown'}), {report.protocol}",
    ]
    for step in report.steps:
        lines.append(f"  {step.id}. {marks[step.ok]}  {step.name}: {step.detail}")
        if step.fix and step.ok is False:
            lines.append(f"           fix: {step.fix}")
    lines.append("result: " + ("all steps passed or not measurable" if report.ok else "FAILED"))
    return lines
