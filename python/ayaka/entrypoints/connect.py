"""``ayaka connect``: point a coding tool at a running ``ayaka serve`` and check that it works.

Four jobs, one record per tool:

* print the configuration the tool needs (the default);
* ``--write``: show a diff of the tool's configuration file, back the file up, ask, then write it
  through a temporary file and an atomic replace;
* ``--check``: seven steps against the running server, in the tool's own protocol, each with a
  fix when it fails;
* ``--observe``: a small proxy between the tool and the server that counts the tokens of each
  model request by part (system prompt, tool definitions, conversation) and stores only the counts
  of the largest one, the tool's floor (a tool's first request is not always its agent request:
  OpenCode first asks for a session title); with ``--record DIR`` it also keeps each request
  (credentials removed) to build replay fixtures.

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
    "WriteResult",
    "agent_version",
    "check_lines",
    "count_parts",
    "generic_setup",
    "known_tools",
    "largest",
    "observer",
    "recorder",
    "redact",
    "redact_headers",
    "render_config",
    "run_check",
    "save_observation",
    "serve_proxy",
    "tool_record",
    "tool_version",
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
        measured: Which tool version and which tokenizer gave ``min_context_tokens`` (another
            model's tokenizer counts the same text differently).
        verified: The tool version and documentation the record was checked against.
        usage: How to start the tool once configured.
        agent: The product name of the tool's ``User-Agent``, to recognise its requests.
        alternates: Other files the tool reads in the same folder instead of ``config_name``
            (such as JSON with comments); ``--write`` will not overwrite one, since a rewrite
            would drop its comments.
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
    measured: str | None
    verified: str
    usage: str
    agent: str
    alternates: tuple[str, ...] = ()

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
            # /model opus, sonnet or haiku in the tool picks these names: all go to the server.
            "ANTHROPIC_DEFAULT_OPUS_MODEL": server.model,
            "ANTHROPIC_DEFAULT_SONNET_MODEL": server.model,
            "ANTHROPIC_DEFAULT_HAIKU_MODEL": server.small_model or server.model,
            "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
        }
    )
    if server.context_tokens:
        # Claude Code does not know a gateway's model and assumes a 1M window, so it would not
        # compact before the server refuses; checked on 2.1.327 (its warning goes away).
        merged["CLAUDE_CODE_MAX_CONTEXT_TOKENS"] = str(server.context_tokens)
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
    ]
    if server.api_key:
        # Codex refuses to start when the named variable is unset, so it is named only when the
        # server needs a key.
        lines.append('env_key = "AYAKA_API_KEY"')
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


#: The tools ``ayaka connect`` knows. ``min_context_tokens`` comes from ``--observe`` on a real
#: session of the named version and changes only through a reviewed data change: the tokens of
#: the system text and the tool definitions as the request sends them, counted with one tokenizer
#: (another model's tokenizer counts the same text a little differently, which the margin of
#: step 7 absorbs).
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
        min_context_tokens=18281,
        measured="Claude Code 2.1.327, the largest of 4 recorded requests (23 tools), "
        "SmolLM2-135M tokenizer, 2026-11-18",
        verified="Claude Code 2.1.327; code.claude.com/docs llm-gateway-connect, env-vars",
        usage="claude   (or: claude -p 'prompt')",
        agent="claude-cli",
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
        min_context_tokens=13348,
        measured="codex-cli 0.163.0, 1 recorded request (9 tools), SmolLM2-135M tokenizer, "
        "2026-11-18",
        verified="codex-cli 0.163.0; config-reference (profiles as <name>.config.toml)",
        usage="codex --profile ayaka   (with AYAKA_API_KEY=<key> when the server has a key)",
        agent="codex_exec",
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
        min_context_tokens=11260,
        measured="opencode 2.3.2, the largest of 4 recorded requests (13 tools), "
        "SmolLM2-135M tokenizer, 2026-11-18",
        verified="opencode 2.3.2; opencode.ai/docs providers, config, models",
        usage="AYAKA_API_KEY=<key> opencode   (or: opencode run --model ayaka/<model> 'prompt')",
        agent="opencode",
        alternates=("opencode.jsonc",),
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
        serial = 1
        while backup.exists():  # two writes in one second keep both backups
            backup = path.with_name(f"{path.name}.{stamp}-{serial}.bak")
            serial += 1
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


def _json_object(body: bytes) -> dict[str, object] | None:
    """The body as a JSON object, or ``None`` when it is not one."""
    try:
        value: object = json.loads(body)
    except ValueError:
        return None
    return cast("dict[str, object]", value) if isinstance(value, dict) else None


def _answered(status: int, body: bytes) -> str:
    """How the server answered, for a step's detail: the status and its error message."""
    if status == 0:
        return f"nothing ({body.decode(errors='replace')[:200]})"
    if status == 200:
        return "200"
    error = (_json_object(body) or {}).get("error")
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
        for choice in _objects((_json_object(chunk.encode()) or {}).get("choices")):
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
        if status == 200 and (_json_object(body) or {}).get("status") == "ready":
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
    entries = _objects((_json_object(body) or {}).get("data")) if status == 200 else []
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
    text = _text_of(record.protocol, _json_object(body) or {}) if status == 200 else ""
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
    answer = _json_object(body) if status == 200 else None
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
        repeat = _json_object(body) if status == 200 else None
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


# -- observe and record -----------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ObservedPrompt:
    """The fixed prompt of one request of a tool, as token counts. No content is kept.

    Attributes:
        tool: The record's name.
        version: The tool's version, from its ``User-Agent`` (else the installed binary).
        protocol: The API of the request.
        tokenizer: The model whose tokenizer counted.
        system_tokens: System and developer text.
        tools_tokens: The tool definitions, as the request sends them (JSON).
        conversation_tokens: User, assistant and tool-result text.
        tool_count: How many tools the request offers.
        at: When, in UTC.
    """

    tool: str
    version: str | None
    protocol: str
    tokenizer: str
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
    active: int
    idle: threading.Condition


class _Handler(http.server.BaseHTTPRequestHandler):
    server: _Proxy  # pyright: ignore[reportIncompatibleVariableOverride]  # our subclass

    def log_message(self, format: str, *args: object) -> None:  # noqa: A002  # base signature
        del format, args  # no request lines with content on stderr

    def _body(self) -> bytes:
        if "chunked" in (self.headers.get("transfer-encoding") or "").lower():
            chunks: list[bytes] = []
            while (size := int(self.rfile.readline().split(b";")[0].strip() or b"0", 16)) > 0:
                chunks.append(self.rfile.read(size))
                self.rfile.readline()  # the CRLF after the chunk
            while self.rfile.readline() not in (b"\r\n", b"\n", b""):  # trailers
                pass
            return b"".join(chunks)
        length = int(self.headers.get("content-length") or 0)
        return self.rfile.read(length) if length else b""

    def _forward(self, method: str) -> None:
        with self.server.idle:
            self.server.active += 1
        try:
            self._relay(method)
        finally:
            with self.server.idle:
                self.server.active -= 1
                self.server.idle.notify_all()

    def _relay(self, method: str) -> None:
        body = self._body()
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
            if key.lower() not in ("host", "content-length", "connection", "transfer-encoding")
        }
        try:
            connection.request(
                method, parsed.path + (f"?{parsed.query}" if parsed.query else ""), body, forward
            )
            response = connection.getresponse()
        except (OSError, http.client.HTTPException) as exc:
            connection.close()
            message = json.dumps({"error": {"message": f"the server did not answer: {exc}"}})
            self.send_response(502)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(message)))
            self.send_header("connection", "close")
            self.end_headers()
            self.wfile.write(message.encode())
            return
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
    closed: threading.Event | None = None,
    drain_timeout: float = 900.0,
) -> int:
    """Forward every request to ``upstream`` until ``stop`` is set; return the port used.

    Once ``stop`` is set the proxy takes no new request, lets the ones in flight finish (the
    request that set it included, up to ``drain_timeout`` seconds), closes, then sets ``closed``.
    """
    proxy = _Proxy((host, port), _Handler)
    proxy.upstream = upstream
    proxy.on_request = on_request
    proxy.active = 0
    proxy.idle = threading.Condition()
    thread = threading.Thread(
        target=proxy.serve_forever, kwargs={"poll_interval": 0.2}, daemon=True
    )
    thread.start()
    bound = int(proxy.server_address[1])

    def close() -> None:
        stop.wait()
        proxy.shutdown()  # no new request; the handlers in flight keep running
        with proxy.idle:
            proxy.idle.wait_for(lambda: proxy.active == 0, timeout=drain_timeout)
        proxy.server_close()
        if closed is not None:
            closed.set()

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


def agent_version(headers: Mapping[str, str], agent: str) -> str | None:
    """The version in a ``User-Agent`` whose product is ``agent``, such as ``claude-cli/2.1.3``.

    OpenCode writes ``opencode/latest/2.3.2/cli``: the first part that starts with a digit is the
    version.
    """
    value = next((v for k, v in headers.items() if k.lower() == "user-agent"), "")
    product = value.split(" ", 1)[0]
    parts = product.split("/")
    if not parts or parts[0] != agent:
        return None
    return next((part for part in parts[1:] if part[:1].isdigit()), None)


def observer(
    record: ToolRecord,
    tokenizer: str,
    count: Callable[[str], int],
    found: list[ObservedPrompt],
    *,
    installed: str | None = None,
) -> Callable[[str, dict[str, str], bytes], None]:
    """Count the parts of each model request into ``found``; keep only the counts.

    A tool's first request is not always its agent request (OpenCode first asks for a session
    title, with no tools), so every request is counted and :func:`largest` picks the floor.
    """

    def observe(path: str, headers: dict[str, str], body: bytes) -> None:
        protocol = _protocol_of(path)
        if protocol is None or "count_tokens" in path:
            return
        parsed = _json_object(body)
        if parsed is None:
            return
        system, tools, conversation, tool_count = count_parts(protocol, parsed, count)
        found.append(
            ObservedPrompt(
                tool=record.name,
                version=agent_version(headers, record.agent) or installed,
                protocol=protocol,
                tokenizer=tokenizer,
                system_tokens=system,
                tools_tokens=tools,
                conversation_tokens=conversation,
                tool_count=tool_count,
                at=datetime.now(UTC).isoformat(timespec="seconds"),
            )
        )

    return observe


def largest(found: Sequence[ObservedPrompt]) -> ObservedPrompt | None:
    """The request with the largest fixed prompt: the one the tool's context must hold."""
    return max(found, key=lambda observed: observed.fixed_tokens, default=None)


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
