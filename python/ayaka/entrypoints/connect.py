"""``ayaka connect``: point a coding tool at a running ``ayaka serve`` and check that it works.

Four jobs, one record per tool:

* print the configuration the tool needs (the default);
* ``--write``: show a diff of the tool's configuration file, back the file up, ask, then write it
  through a temporary file and an atomic replace;
* ``--check``: seven steps against the running server, in the tool's own protocol, each with a
  fix when it fails;
* ``--observe``: a small proxy between the tool and the server that counts the tokens of the first
  request by part (system prompt, tool definitions, conversation) and stores only the counts; with
  ``--record DIR`` it also keeps each request (credentials removed) to build replay fixtures.

A tool is a :class:`ToolRecord`: data, not code paths. The records follow the tools' documentation
and an installed version (see each record); a client changing its format is a one-record change.
The API key is never printed, never written to a report and never kept in a recording.
"""

from __future__ import annotations

import difflib
import http.client
import http.server
import json
import logging
import os
import shutil
import tempfile
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Final, Literal, cast

from ayaka.__version__ import __version__

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

__all__ = [
    "CONTEXT_MARGIN_TOKENS",
    "TOOLS",
    "CachedShareThreshold",
    "CheckReport",
    "CheckStep",
    "ObservedPrompt",
    "ServerInfo",
    "ToolRecord",
    "count_parts",
    "redact_headers",
    "render_config",
    "run_check",
    "tool_record",
    "write_config",
]

Protocol = Literal["anthropic", "openai-responses", "openai-chat"]

_LOG = logging.getLogger(__name__)

#: Tokens a tool needs beyond its fixed prompt to do any work: the user's files and the answers.
CONTEXT_MARGIN_TOKENS: Final = 8192

#: Step 6 passes when the repeated request reads at least this share of its prompt from the cache.
#: Written before any run: a repeated identical request should reuse all but its last block.
CachedShareThreshold: Final = 0.5

_KEY_PLACEHOLDER: Final = "ayaka-local"
_SECRET_HEADERS: Final = frozenset({"authorization", "x-api-key", "cookie", "proxy-authorization"})


@dataclass(frozen=True, slots=True)
class ServerInfo:
    """Where the server is and what to call the model.

    Attributes:
        base_url: The server's root, such as ``http://127.0.0.1:8000``.
        api_key: The key the server requires, or ``None``. Never printed.
        model: The name the tool sends.
        small_model: The name for a tool's background work; the same model when ``None``.
        context_tokens: The longest context the server gives one request, when known.
    """

    base_url: str
    api_key: str | None
    model: str
    small_model: str | None = None
    context_tokens: int | None = None

    @property
    def openai_base(self) -> str:
        """The base URL OpenAI clients take: the root plus ``/v1``."""
        return self.base_url.rstrip("/") + "/v1"


@dataclass(frozen=True, slots=True)
class ToolRecord:
    """Everything ``ayaka connect`` knows about one coding tool.

    Attributes:
        name: The name on the command line.
        protocol: The API the tool speaks.
        binary: The executable, for ``--version``.
        config_home: The tool's configuration folder (the same path on every platform).
        config_home_env: The variable the tool reads to move that folder, if any.
        config_name: The file in that folder.
        config_format: ``"json"`` or ``"toml"``.
        render: Builds the new file content from the server info and the current content.
        min_context_tokens: The tool's fixed prompt (system and tools) in tokens, as measured by
            ``--observe`` on a recorded version; ``None`` until measured, never guessed.
        verified: The tool version and documentation the record was checked against.
        usage: How to start the tool once configured.
    """

    name: str
    protocol: Protocol
    binary: str
    config_home: str
    config_home_env: str | None
    config_name: str
    config_format: Literal["json", "toml"]
    render: Callable[[ServerInfo, str], str]
    min_context_tokens: int | None
    verified: str
    usage: str

    def config_path(self, environ: Mapping[str, str] | None = None) -> Path:
        """The configuration file, in the folder the tool's variable names when it is set."""
        env = os.environ if environ is None else environ
        home = env.get(self.config_home_env) if self.config_home_env else None
        return Path(home or self.config_home).expanduser() / self.config_name

    def aliases(self, server: ServerInfo) -> tuple[str, ...]:
        """Every model name the tool will send with this configuration."""
        names = [server.model]
        if server.small_model and server.small_model != server.model:
            names.append(server.small_model)
        return tuple(names)


def _load_json(current: str) -> dict[str, object]:
    if not current.strip():
        return {}
    value = json.loads(current)
    if not isinstance(value, dict):
        raise ValueError("the existing file is not a JSON object")
    return cast("dict[str, object]", value)


def _render_claude_code(server: ServerInfo, current: str) -> str:
    """``~/.claude/settings.json``: the gateway variables in its ``env`` table, merged."""
    settings = _load_json(current)
    env = settings.get("env")
    merged: dict[str, object] = (
        dict(cast("dict[str, object]", env)) if isinstance(env, dict) else {}
    )
    merged.update(
        {
            "ANTHROPIC_BASE_URL": server.base_url.rstrip("/"),
            # ANTHROPIC_AUTH_TOKEN is sent as a Bearer token; the server accepts either header.
            "ANTHROPIC_AUTH_TOKEN": server.api_key or _KEY_PLACEHOLDER,
            "ANTHROPIC_MODEL": server.model,
            "ANTHROPIC_DEFAULT_HAIKU_MODEL": server.small_model or server.model,
            "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
        }
    )
    settings["env"] = merged
    return json.dumps(settings, indent=2) + "\n"


def _toml_string(value: str) -> str:
    return json.dumps(value)  # a JSON string is a valid TOML basic string


def _render_codex(server: ServerInfo, current: str) -> str:
    """``~/.codex/ayaka.config.toml``: a profile file (Codex 0.134+), selected with ``--profile``.

    The whole file belongs to Ayaka, so it is written whole; ``current`` is only diffed.
    """
    del current
    lines = [
        "# Written by `ayaka connect codex`. Use it with: codex --profile ayaka",
        f"model = {_toml_string(server.model)}",
        'model_provider = "ayaka"',
    ]
    if server.context_tokens:
        lines.append(f"model_context_window = {server.context_tokens}")
    lines += [
        "",
        "[model_providers.ayaka]",
        'name = "Ayaka"',
        f"base_url = {_toml_string(server.openai_base)}",
        'wire_api = "responses"',
        'env_key = "AYAKA_API_KEY"',
    ]
    return "\n".join(lines) + "\n"


def _render_opencode(server: ServerInfo, current: str) -> str:
    """``opencode.json``: an OpenAI-compatible provider ``ayaka`` and the default model, merged."""
    config = _load_json(current)
    config.setdefault("$schema", "https://opencode.ai/config.json")
    providers = config.get("providers")
    table: dict[str, object] = (
        dict(cast("dict[str, object]", providers)) if isinstance(providers, dict) else {}
    )
    limit: dict[str, object] = {"context": server.context_tokens} if server.context_tokens else {}
    table["ayaka"] = {
        "name": "Ayaka",
        # The credential: opencode reads the variable for a custom provider itself.
        "env": ["AYAKA_API_KEY"],
        "package": "@opencode/ai/providers/openai-compatible",
        "settings": {"baseURL": server.openai_base, "apiKey": "{env:AYAKA_API_KEY}"},
        "models": {
            server.model: {
                "modelID": server.model,
                "capabilities": {"tools": True, "input": ["text"], "output": ["text"]},
                **({"limit": limit} if limit else {}),
            }
        },
    }
    config["providers"] = table
    config["model"] = f"ayaka/{server.model}"
    return json.dumps(config, indent=2) + "\n"


#: The tools ``ayaka connect`` knows. ``min_context_tokens`` stays ``None`` until ``--observe``
#: measured a version and a reviewed data change records it.
TOOLS: Final[Mapping[str, ToolRecord]] = {
    "claude-code": ToolRecord(
        name="claude-code",
        protocol="anthropic",
        binary="claude",
        config_home="~/.claude",
        config_home_env="CLAUDE_CONFIG_DIR",
        config_name="settings.json",
        config_format="json",
        render=_render_claude_code,
        min_context_tokens=None,
        verified="Claude Code 2.1.327; code.claude.com/docs llm-gateway-connect, env-vars",
        usage="claude   (or: claude -p 'prompt')",
    ),
    "codex": ToolRecord(
        name="codex",
        protocol="openai-responses",
        binary="codex",
        config_home="~/.codex",
        config_home_env="CODEX_HOME",
        config_name="ayaka.config.toml",
        config_format="toml",
        render=_render_codex,
        min_context_tokens=None,
        verified="codex-cli 0.163.0; config-reference (profiles as <name>.config.toml)",
        usage="AYAKA_API_KEY=<key> codex --profile ayaka",
    ),
    "opencode": ToolRecord(
        name="opencode",
        protocol="openai-chat",
        binary="opencode",
        config_home="~/.config/opencode",
        config_home_env=None,
        config_name="opencode.json",
        config_format="json",
        render=_render_opencode,
        min_context_tokens=None,
        verified="opencode 2.3.2; opencode.ai/docs providers, config, models",
        usage="AYAKA_API_KEY=<key> opencode   (or: opencode run --model ayaka/<model> 'prompt')",
    ),
}


def tool_record(name: str) -> ToolRecord | None:
    """The record of a tool, or ``None`` for a tool without one."""
    return TOOLS.get(name)


def render_config(record: ToolRecord, server: ServerInfo, current: str = "") -> str:
    """The tool's configuration file with the server filled in, merged into ``current``."""
    return record.render(server, current)


def generic_setup(server: ServerInfo) -> list[str]:
    """What to set in a tool without a record: the two protocols' usual variables."""
    return [
        "No record for this tool; most tools take one of these:",
        f"  OpenAI-compatible:    base URL {server.openai_base}, API key <your key>, "
        f"model {server.model}",
        f"  Anthropic-compatible: ANTHROPIC_BASE_URL={server.base_url.rstrip('/')} "
        f"ANTHROPIC_AUTH_TOKEN=<your key> ANTHROPIC_MODEL={server.model}",
    ]


def redact(text: str, server: ServerInfo) -> str:
    """``text`` with the API key replaced, for anything shown or stored."""
    if server.api_key:
        return text.replace(server.api_key, "<api key>")
    return text


# -- write ------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class WriteResult:
    """What ``--write`` did."""

    path: Path
    changed: bool
    backup: Path | None
    written: bool


def write_config(
    path: Path,
    content: str,
    *,
    confirm: Callable[[str], bool],
    create_dirs: bool = False,
    show: Callable[[str], None] = print,
    redactor: Callable[[str], str] = lambda text: text,
) -> WriteResult:
    """Show a diff, back the old file up, ask, then write atomically.

    Args:
        path: The file.
        content: Its new content.
        confirm: Asked with the question; the file is written only on ``True``.
        create_dirs: Whether a missing parent directory may be created.
        show: Where the diff goes.
        redactor: Applied to the diff before it is shown, so a key never appears.

    Raises:
        FileNotFoundError: If the parent directory is missing and ``create_dirs`` is false.
    """
    old = path.read_text(encoding="utf-8") if path.exists() else ""
    if old == content:
        show(f"{path} already has this configuration")
        return WriteResult(path, changed=False, backup=None, written=False)
    diff = difflib.unified_diff(
        old.splitlines(keepends=True),
        content.splitlines(keepends=True),
        fromfile=str(path),
        tofile=f"{path} (new)",
    )
    show(redactor("".join(diff)))
    if not path.parent.exists() and not create_dirs:
        raise FileNotFoundError(f"{path.parent} does not exist; pass --create-dirs to create it")
    if not confirm(f"write {path}?"):
        show("nothing written")
        return WriteResult(path, changed=True, backup=None, written=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    backup: Path | None = None
    if path.exists():
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        backup = path.with_name(f"{path.name}.{stamp}.bak")
        shutil.copy2(path, backup)
        _private(backup)
    handle, temporary = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    try:
        with os.fdopen(handle, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(content)
        _private(Path(temporary))
        Path(temporary).replace(path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise
    show(f"wrote {path}" + (f" (backup: {backup})" if backup else ""))
    return WriteResult(path, changed=True, backup=backup, written=True)


def _private(path: Path) -> None:
    """Owner-only permissions on POSIX; Windows keeps the user profile's ACL."""
    if os.name == "posix":
        path.chmod(0o600)


# -- check ------------------------------------------------------------------------------------


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
    """A minimal HTTP client for the checks: standard library, no SDK."""

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

    def get(self, path: str) -> tuple[int, bytes]:
        request = urllib.request.Request(  # noqa: S310  # the server's own URL
            self.server.base_url.rstrip("/") + path, headers=self._headers("openai-chat")
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:  # noqa: S310
                return response.status, response.read()
        except urllib.error.HTTPError as exc:
            return exc.code, exc.read()

    def post(self, path: str, body: Mapping[str, object], protocol: Protocol) -> tuple[int, bytes]:
        request = urllib.request.Request(  # noqa: S310  # the server's own URL
            self.server.base_url.rstrip("/") + path,
            data=json.dumps(body).encode(),
            headers=self._headers(protocol),
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:  # noqa: S310
                return response.status, response.read()
        except urllib.error.HTTPError as exc:
            return exc.code, exc.read()


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


def _text_of(protocol: Protocol, answer: Mapping[str, object]) -> str:
    if protocol == "anthropic":
        blocks = cast("list[dict[str, object]]", answer.get("content") or [])
        return "".join(str(b.get("text", "")) for b in blocks if b.get("type") == "text")
    if protocol == "openai-responses":
        items = cast("list[dict[str, object]]", answer.get("output") or [])
        texts: list[str] = []
        for item in items:
            for part in cast("list[dict[str, object]]", item.get("content") or []):
                texts.append(str(part.get("text", "")))
        return "".join(texts)
    choices = cast("list[dict[str, object]]", answer.get("choices") or [{}])
    message = cast("dict[str, object]", choices[0].get("message") or {})
    return str(message.get("content") or "")


def _call_of(protocol: Protocol, answer: Mapping[str, object]) -> tuple[str, object] | None:
    """``(name, arguments)`` of the first call, arguments decoded from JSON when they are text."""
    if protocol == "anthropic":
        for block in cast("list[dict[str, object]]", answer.get("content") or []):
            if block.get("type") == "tool_use":
                return str(block.get("name")), block.get("input")
        return None
    if protocol == "openai-responses":
        for item in cast("list[dict[str, object]]", answer.get("output") or []):
            if item.get("type") == "function_call":
                return str(item.get("name")), _decoded(item.get("arguments"))
        return None
    choices = cast("list[dict[str, object]]", answer.get("choices") or [{}])
    message = cast("dict[str, object]", choices[0].get("message") or {})
    calls = cast("list[dict[str, object]]", message.get("tool_calls") or [])
    if not calls:
        return None
    function = cast("dict[str, object]", calls[0].get("function") or {})
    return str(function.get("name")), _decoded(function.get("arguments"))


def _decoded(value: object) -> object:
    if isinstance(value, str):
        try:
            return json.loads(value)
        except ValueError:
            return value
    return value


def _cache_share(protocol: Protocol, answer: Mapping[str, object]) -> tuple[int, int]:
    """``(cached, prompt)`` tokens from the usage of an answer."""
    usage = cast("dict[str, object]", answer.get("usage") or {})
    if protocol == "anthropic":
        cached = int(cast("int", usage.get("cache_read_input_tokens") or 0))
        fresh = int(cast("int", usage.get("input_tokens") or 0))
        return cached, cached + fresh
    if protocol == "openai-responses":
        details = cast("dict[str, object]", usage.get("input_tokens_details") or {})
        return int(cast("int", details.get("cached_tokens") or 0)), int(
            cast("int", usage.get("input_tokens") or 0)
        )
    details = cast("dict[str, object]", usage.get("prompt_tokens_details") or {})
    return int(cast("int", details.get("cached_tokens") or 0)), int(
        cast("int", usage.get("prompt_tokens") or 0)
    )


def _stream_terminated(protocol: Protocol, body: bytes) -> tuple[bool, str]:
    text = body.decode("utf-8", errors="replace")
    if protocol == "anthropic":
        return "event: message_stop" in text, "message_stop"
    if protocol == "openai-responses":
        # A stream cut by max_output_tokens ends with response.incomplete, also a terminal event.
        ended = "event: response.completed" in text or "event: response.incomplete" in text
        return ended, "response.completed or response.incomplete"
    return "data: [DONE]" in text and '"finish_reason"' in text, "a finish_reason, then [DONE]"


def _valid_arguments(arguments: object) -> bool:
    """Whether ``arguments`` follow the check tool's schema (written out, no validator library)."""
    if not isinstance(arguments, dict):
        return False
    values = cast("dict[str, object]", arguments)
    path = values.get("path")
    return set(values) == {"path"} and isinstance(path, str) and len(path) <= 40


def run_check(
    record: ToolRecord,
    server: ServerInfo,
    *,
    timeout: float = 60.0,
    version: str | None = None,
) -> CheckReport:
    """The seven steps of ``--check`` against a running server, in the tool's protocol."""
    client = _Client(server, timeout)
    report = CheckReport(tool=record.name, version=version, protocol=record.protocol)
    steps = report.steps

    deadline = time.monotonic() + timeout
    status, body = 0, b""
    while time.monotonic() < deadline:
        try:
            status, body = client.get("/health")
        except OSError as exc:
            body = str(exc).encode()
        if status == 200:
            break
        time.sleep(0.5)
    ready = status == 200
    steps.append(
        CheckStep(
            1,
            "server ready",
            ready,
            f"/health answered {status}: {body[:120].decode(errors='replace')}",
            None if ready else f"start it: ayaka serve <model> (listening at {server.base_url})",
        )
    )
    if not ready:
        return report

    status, body = client.get("/v1/models")
    listing = cast("dict[str, object]", json.loads(body or b"{}")) if status == 200 else {}
    entries = cast("list[dict[str, object]]", listing.get("data") or [])
    names = {str(entry.get("id")) for entry in entries}
    missing = [alias for alias in record.aliases(server) if alias not in names]
    steps.append(
        CheckStep(
            2,
            "model and aliases",
            status == 200 and not missing,
            f"the server serves {sorted(names)}; the tool sends {list(record.aliases(server))}",
            None
            if not missing
            else "start the server with --served-model-name "
            + " --served-model-name ".join(missing),
        )
    )

    path = _path(record.protocol)
    status, body = client.post(
        path, _text_request(record.protocol, server.model, stream=False), record.protocol
    )
    text = _text_of(record.protocol, json.loads(body)) if status == 200 else ""
    steps.append(
        CheckStep(
            3,
            "text",
            bool(text.strip()),
            f"{path} answered {status} with {len(text)} characters",
            None if text.strip() else "look at the server log for this request's error",
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
            f"stream answered {status}; expected {terminal}",
            None if ended else "a proxy may be buffering or cutting server-sent events",
        )
    )

    tool_request = _tool_request(record.protocol, server.model)
    status, body = client.post(path, tool_request, record.protocol)
    answer = cast("dict[str, object]", json.loads(body)) if status == 200 else {}
    call = _call_of(record.protocol, answer) if answer else None
    valid = call is not None and call[0] == _CHECK_TOOL["name"] and _valid_arguments(call[1])
    steps.append(
        CheckStep(
            5,
            "tool call",
            valid,
            f"answered {status}; call: {call!r}"[:300],
            None
            if valid
            else "start the server with --tool-call-parser for this model and the structured "
            "extra (uv sync --extra structured)",
        )
    )

    status, body = client.post(path, tool_request, record.protocol)
    repeat = cast("dict[str, object]", json.loads(body)) if status == 200 else {}
    cached, prompt = _cache_share(record.protocol, repeat) if repeat else (0, 0)
    share = cached / prompt if prompt else 0.0
    hit = share >= CachedShareThreshold
    steps.append(
        CheckStep(
            6,
            "prefix cache",
            hit if status == 200 else False,
            f"the repeated request read {cached} of {prompt} prompt tokens from the cache "
            f"({share:.0%}; threshold {CachedShareThreshold:.0%})",
            None
            if hit
            else "the identical request did not hit: start the server without --no-prefix-cache",
        )
    )

    context = server.context_tokens
    for entry in entries:
        if str(entry.get("id")) == server.model:
            value = entry.get("max_context_tokens") or entry.get("max_model_len")
            if isinstance(value, int):
                context = value
    floor = record.min_context_tokens
    if floor is None or context is None:
        steps.append(
            CheckStep(
                7,
                "context",
                None,
                f"not measured: the tool's fixed prompt has no recorded size "
                f"(run ayaka connect {record.name} --observe); the server gives {context} tokens",
            )
        )
    else:
        needed = floor + CONTEXT_MARGIN_TOKENS
        enough = context >= needed
        steps.append(
            CheckStep(
                7,
                "context",
                enough,
                f"{context} tokens per request; the tool needs {floor} plus "
                f"{CONTEXT_MARGIN_TOKENS} to work",
                None if enough else "a smaller model or more memory (see ayaka plan)",
            )
        )
    return report


# -- observe and record -----------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ObservedPrompt:
    """The fixed prompt of a tool's first request, as token counts. No content is kept."""

    tool: str
    version: str | None
    protocol: str
    system_tokens: int
    tools_tokens: int
    conversation_tokens: int
    tool_count: int
    at: str

    @property
    def fixed_tokens(self) -> int:
        """What every request of the tool carries before the user's work: system and tools."""
        return self.system_tokens + self.tools_tokens


def _protocol_of(path: str) -> Protocol | None:
    if path.startswith("/v1/messages"):
        return "anthropic"
    if path.startswith("/v1/responses"):
        return "openai-responses"
    if path.startswith("/v1/chat/completions"):
        return "openai-chat"
    return None


def _flatten(content: object) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in cast("list[object]", content):
            if isinstance(item, dict):
                block = cast("dict[str, object]", item)
                parts.append(str(block.get("text") or block.get("content") or ""))
            else:
                parts.append(str(item))
        return "\n".join(parts)
    return "" if content is None else json.dumps(content)


def count_parts(
    protocol: Protocol, body: Mapping[str, object], count: Callable[[str], int]
) -> tuple[int, int, int, int]:
    """``(system, tools, conversation, number of tools)`` tokens of a request body."""
    tools = cast("list[object]", body.get("tools") or [])
    tools_text = json.dumps(tools, ensure_ascii=False) if tools else ""
    system = ""
    conversation: list[str] = []
    if protocol == "anthropic":
        system = _flatten(body.get("system"))
        for message in cast("list[dict[str, object]]", body.get("messages") or []):
            conversation.append(_flatten(message.get("content")))
    elif protocol == "openai-responses":
        system = str(body.get("instructions") or "")
        items = body.get("input")
        for item in cast("list[dict[str, object]]", items if isinstance(items, list) else []):
            if item.get("role") in ("system", "developer"):
                system += "\n" + _flatten(item.get("content"))
            else:
                conversation.append(_flatten(item.get("content") or item.get("output")))
        if isinstance(items, str):
            conversation.append(items)
    else:
        for message in cast("list[dict[str, object]]", body.get("messages") or []):
            text = _flatten(message.get("content"))
            if message.get("role") in ("system", "developer"):
                system += "\n" + text
            else:
                conversation.append(text)
    return (
        count(system) if system.strip() else 0,
        count(tools_text) if tools_text else 0,
        sum(count(text) for text in conversation if text),
        len(tools),
    )


def redact_headers(headers: Mapping[str, str]) -> dict[str, str]:
    """The headers without credentials, for a recording."""
    return {key: value for key, value in headers.items() if key.lower() not in _SECRET_HEADERS}


class _Proxy(http.server.ThreadingHTTPServer):
    upstream: str
    on_request: Callable[[str, dict[str, str], bytes], None]


class _Handler(http.server.BaseHTTPRequestHandler):
    server: _Proxy  # pyright: ignore[reportIncompatibleVariableOverride]  # our subclass

    def log_message(self, format: str, *args: object) -> None:  # noqa: A002  # base signature
        del format, args  # no request lines with content on stderr

    def _forward(self, method: str) -> None:
        length = int(self.headers.get("content-length") or 0)
        body = self.rfile.read(length) if length else b""
        headers = dict(self.headers.items())
        try:
            self.server.on_request(self.path, headers, body)
        except Exception:  # an observer must never break the tool's request
            _LOG.debug("connect: the observer failed on %s", self.path, exc_info=True)
        target = self.server.upstream.rstrip("/") + self.path
        parsed = urllib.parse.urlsplit(target)
        connection = http.client.HTTPConnection(parsed.hostname or "127.0.0.1", parsed.port or 80)
        forward = {
            key: value
            for key, value in headers.items()
            if key.lower() not in ("host", "content-length", "connection")
        }
        connection.request(
            method, parsed.path + (f"?{parsed.query}" if parsed.query else ""), body, forward
        )
        response = connection.getresponse()
        self.send_response(response.status)
        for key, value in response.getheaders():
            if key.lower() not in ("transfer-encoding", "connection", "content-length"):
                self.send_header(key, value)
        self.send_header("connection", "close")
        self.end_headers()
        while chunk := response.read1(65536):
            self.wfile.write(chunk)
            self.wfile.flush()
        connection.close()

    def do_GET(self) -> None:  # http.server naming
        self._forward("GET")

    def do_POST(self) -> None:
        self._forward("POST")

    def do_HEAD(self) -> None:
        self._forward("HEAD")


def serve_proxy(
    port: int,
    upstream: str,
    on_request: Callable[[str, dict[str, str], bytes], None],
    *,
    stop: threading.Event,
    host: str = "127.0.0.1",
) -> int:
    """Forward every request to ``upstream`` until ``stop`` is set; return the port used."""
    proxy = _Proxy((host, port), _Handler)
    proxy.upstream = upstream
    proxy.on_request = on_request
    thread = threading.Thread(
        target=proxy.serve_forever, kwargs={"poll_interval": 0.2}, daemon=True
    )
    thread.start()
    bound = int(proxy.server_address[1])

    def close() -> None:
        stop.wait()
        proxy.shutdown()
        proxy.server_close()

    threading.Thread(target=close, daemon=True).start()
    return bound


def recorder(
    directory: Path, *, limit: int | None = None
) -> Callable[[str, dict[str, str], bytes], bool]:
    """Keep each model request as JSON in ``directory`` (owner-only), credentials removed.

    Returns a callable that stores one request and says whether the limit is reached.
    """
    directory.mkdir(parents=True, exist_ok=True)
    seen = [0]

    def keep(path: str, headers: dict[str, str], body: bytes) -> bool:
        protocol = _protocol_of(path)
        if protocol is None:
            return False
        seen[0] += 1
        try:
            payload: object = json.loads(body) if body else None
        except ValueError:
            payload = body.decode("utf-8", errors="replace")
        target = directory / f"{seen[0]:03d}-{protocol}.json"
        target.write_text(
            json.dumps(
                {"path": path, "headers": redact_headers(headers), "body": payload},
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        _private(target)
        return limit is not None and seen[0] >= limit

    return keep


def observer(
    tool: str, version: str | None, count: Callable[[str], int], found: list[ObservedPrompt]
) -> Callable[[str, dict[str, str], bytes], None]:
    """Count the parts of the first model request; keep only the counts in ``found``."""

    def observe(path: str, headers: dict[str, str], body: bytes) -> None:
        del headers
        protocol = _protocol_of(path)
        if protocol is None or found or path.endswith("count_tokens"):
            return
        parsed = cast("dict[str, object]", json.loads(body))
        system, tools, conversation, tool_count = count_parts(protocol, parsed, count)
        found.append(
            ObservedPrompt(
                tool=tool,
                version=version,
                protocol=protocol,
                system_tokens=system,
                tools_tokens=tools,
                conversation_tokens=conversation,
                tool_count=tool_count,
                at=datetime.now(UTC).isoformat(timespec="seconds"),
            )
        )

    return observe


def tool_version(record: ToolRecord) -> str | None:
    """The installed tool's ``--version``, or ``None`` when it is not installed."""
    import subprocess

    binary = shutil.which(record.binary)
    if binary is None:
        return None
    try:
        result = subprocess.run(  # noqa: S603  # a fixed argv
            [binary, "--version"], capture_output=True, text=True, timeout=20, check=False
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    line = (result.stdout or result.stderr).strip().splitlines()
    return line[0] if line else None


def save_observation(observed: ObservedPrompt, directory: Path) -> Path:
    """Append the counts to ``directory/observations.jsonl`` (owner-only); never content."""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "observations.jsonl"
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(asdict(observed) | {"fixed_tokens": observed.fixed_tokens}) + "\n")
    _private(path)
    return path


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


def known_tools() -> Sequence[str]:
    """The names of the tools with a record."""
    return tuple(TOOLS)
