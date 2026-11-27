"""The server settings of ``serve`` and ``config show``: flags, then a TOML file, then defaults.

The settings never come from the environment. :func:`server_config` merges the flags over the
``[server]`` table of ``--config FILE`` (or the user's configuration file when it exists) over the
defaults, and checks the parser names before anything loads; :func:`describe_server` is what
``serve`` prints before it starts, the API key only as set or not.
"""

from __future__ import annotations

import tomllib
from typing import TYPE_CHECKING, Final, cast

from ayaka.configs.server import ServerConfig, user_config_path
from ayaka.core.errors import ConfigError, InvalidRequestError
from ayaka.entrypoints.parsers.formats import reasoning_format, tool_call_format

if TYPE_CHECKING:
    import argparse
    from pathlib import Path

    from ayaka.configs.engine import EngineConfig


__all__ = [
    "SERVER_FLAGS",
    "config_file_path",
    "describe_server",
    "flag_values",
    "read_config_file",
    "read_key",
    "server_config",
]


SERVER_FLAGS: Final = (
    ("host", "host"),
    ("port", "port"),
    ("cors_origin", "cors_origins"),
    ("served_model_name", "served_model_names"),
    ("reasoning_parser", "reasoning_parser"),
    ("tool_call_parser", "tool_call_parser"),
    ("tool_strict_level", "tool_strict_level"),
    ("engine_grpc", "engine_grpc"),
    ("max_body_bytes", "max_body_bytes"),
    ("shutdown_timeout", "shutdown_timeout"),
    ("sse_coalesce_ms", "sse_coalesce_ms"),
    ("keepalive", "keepalive_seconds"),
    ("slow_client_timeout", "slow_client_seconds"),
    ("stall_timeout", "stall_seconds"),
    ("normalize_prompts", "normalize_prompts"),
    ("overlong_policy", "overlong_policy"),
)


def server_config(args: argparse.Namespace) -> ServerConfig:
    """The server settings: the flags, then ``--config``'s ``[server]`` table, then defaults.

    Raises:
        ConfigError: If the file or a value is invalid.
        InvalidRequestError: If the key file cannot be read.
    """
    values: dict[str, object] = {}
    file = config_file_path(args)
    if file is not None:
        values.update(read_config_file(file))
    for flag, key in SERVER_FLAGS:
        value: object = getattr(args, flag)
        if isinstance(value, list):
            values[key] = tuple(cast("list[object]", value))
        elif value is not None:
            values[key] = value
    if args.api_key is not None and args.api_key_file is not None:
        raise InvalidRequestError("give --api-key or --api-key-file, not both")
    if args.api_key is not None:
        values["api_key"] = args.api_key
    elif args.api_key_file is not None:
        values["api_key"] = read_key(args.api_key_file)
    config = ServerConfig.from_mapping(values)
    # Unknown parser names fail here, before anything loads.
    if config.reasoning_parser is not None:
        reasoning_format(config.reasoning_parser)
    if config.tool_call_parser is not None:
        tool_call_format(config.tool_call_parser)
    return config


def config_file_path(args: argparse.Namespace) -> Path | None:
    """The configuration file to read: ``--config``, else the user's file when it exists."""
    if args.config is not None:
        return cast("Path", args.config)
    default = user_config_path()
    return default if default.is_file() else None


def flag_values(args: argparse.Namespace) -> dict[str, object]:
    """The server settings given as flags, by setting name; the key reported only as set."""
    values: dict[str, object] = {}
    for flag, key in SERVER_FLAGS:
        value: object = getattr(args, flag)
        if isinstance(value, list):
            values[key] = tuple(cast("list[object]", value))
        elif value is not None:
            values[key] = value
    if args.api_key is not None or args.api_key_file is not None:
        values["api_key"] = "set"
    return values


def read_config_file(path: Path) -> dict[str, object]:
    try:
        with path.open("rb") as handle:
            document = tomllib.load(handle)
    except OSError as exc:
        raise ConfigError("config", "UNREADABLE", f"cannot read {path}: {exc}") from exc
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError("config", "INVALID", f"{path} is not valid TOML: {exc}") from exc
    unknown = sorted(set(document) - {"server"})
    if unknown:
        raise ConfigError("config", "UNKNOWN_KEY", f"{path}: unknown tables {unknown}")
    table = document.get("server", {})
    if not isinstance(table, dict):
        raise ConfigError("config.server", "INVALID", "must be a table")
    return cast("dict[str, object]", table)


def read_key(path: Path) -> str:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise InvalidRequestError(f"cannot read the API key file {path}: {exc}") from exc
    key = lines[0].strip() if lines else ""
    if not key:
        raise InvalidRequestError(f"the API key file {path} is empty")
    return key


def describe_server(config: ServerConfig, engine: EngineConfig, model: str) -> list[str]:
    """The lines ``serve`` prints before it starts: the effective settings, never the key."""
    device = engine.device
    graphs = ", decode in CUDA graphs" if device.cuda_graph else ""
    names = ", ".join(config.served_model_names) or model
    lines = [
        f"serving {model} as {names} on {device.device} in {device.dtype.label} "
        f"(ops backends: {', '.join(device.backends)}{graphs})",
        f"reasoning parser: {config.reasoning_parser or 'none'}; "
        f"tool-call parser: {config.tool_call_parser or 'none (requests with tools get 400)'}; "
        f"API key: {'required' if config.api_key else 'none'}; "
        f"CORS: {', '.join(config.cors_origins) or 'off'}",
    ]
    if not config.is_loopback and config.api_key is None:
        lines.append(
            f"warning: {config.host} is reachable from other machines and no API key is set: "
            "anyone who can reach this port can use the model; set --api-key-file"
        )
    return lines
