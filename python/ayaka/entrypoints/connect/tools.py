"""The tool records of ``ayaka connect``: where each tool keeps its settings and what to put there.

A tool is a :class:`ToolRecord`: data, not code paths. The records follow the tools' documentation
and an installed version (see each record); a client changing its format is a one-record change.
:func:`render_config` merges what Ayaka adds into the tool's current file; :func:`redact` takes
the API key out of anything printed.
"""

from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Final, Literal, cast

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

__all__ = [
    "CONTEXT_MARGIN_TOKENS",
    "TOOLS",
    "Protocol",
    "ServerInfo",
    "ToolRecord",
    "generic_setup",
    "json_object",
    "known_tools",
    "redact",
    "render_config",
    "tool_record",
    "tool_version",
]

Protocol = Literal["anthropic", "openai-responses", "openai-chat"]

#: Tokens a tool needs beyond its fixed prompt to do any work: the user's files and the answers.
CONTEXT_MARGIN_TOKENS: Final = 8192


_KEY_PLACEHOLDER: Final = "ayaka-local"


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


# -- check ------------------------------------------------------------------------------------


# -- observe and record -----------------------------------------------------------------------


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


def known_tools() -> Sequence[str]:
    """The names of the tools with a record."""
    return tuple(TOOLS)


def json_object(body: bytes) -> dict[str, object] | None:
    """The body as a JSON object, or ``None`` when it is not one."""
    try:
        value: object = json.loads(body)
    except ValueError:
        return None
    return cast("dict[str, object]", value) if isinstance(value, dict) else None
