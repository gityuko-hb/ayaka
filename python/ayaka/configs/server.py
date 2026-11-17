"""How the HTTP server listens, who may call it and how it streams.

``ServerConfig`` is a frozen value validated at construction. Its values come from the command
line, then from the ``[server]`` table of a configuration file, then from the defaults here, in
that order and from nowhere else: no environment variable changes the server.

The API key is kept out of :meth:`~ayaka.configs.base.ConfigMixin.to_dict` and the fingerprint, so
printing or hashing the configuration never reveals it.
"""

from __future__ import annotations

import ipaddress
import math
from dataclasses import dataclass, field, fields
from typing import TYPE_CHECKING, Final, cast

from ayaka.configs.base import ConfigMixin
from ayaka.core.errors import ConfigError

if TYPE_CHECKING:
    from collections.abc import Mapping

__all__ = ["DEFAULT_PORT", "ServerConfig"]

#: The port ``ayaka serve`` listens on unless told otherwise.
DEFAULT_PORT: Final = 8000

_MIB: Final = 1024 * 1024


def _is_loopback(host: str) -> bool:
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


@dataclass(frozen=True, slots=True)
class ServerConfig(ConfigMixin):
    """The settings of ``ayaka serve``.

    Attributes:
        host: Address to bind. The default answers this machine only.
        port: TCP port, ``0`` for any free port.
        api_key: When set, every request but ``/health`` must carry it, as
            ``Authorization: Bearer <key>`` or ``x-api-key: <key>``.
        cors_origins: Origins a browser may call from; ``*`` allows any. Empty sends no CORS
            headers, so browsers refuse cross-origin calls.
        max_body_bytes: Largest request body accepted; a larger one gets 413.
        shutdown_timeout: Seconds requests in flight get to finish after SIGINT or SIGTERM
            before they are cancelled at the next step boundary.
        sse_coalesce_ms: Milliseconds a stream waits after a token for more tokens to put in the
            same frame. ``0`` writes what is ready at once (provisional default; see ``OUT-5``).
        keepalive_seconds: Idle seconds after which a stream sends a keepalive frame, so proxies do
            not cut a connection while a long prompt is being read.
        slow_client_seconds: Seconds a write to a client may block before the request is
            cancelled and the connection closed. The engine never waits for a client.
        served_model_names: Names clients may use for the model, the first one reported in
            responses. Empty: the name the model was loaded by.
        reasoning_parser: Name of the model's reasoning format, or ``None``.
        tool_call_parser: Name of the model's tool-call format, or ``None``; without one,
            requests that offer tools are refused.

    Raises:
        ConfigError: If a value is out of range.
    """

    host: str = "127.0.0.1"
    port: int = DEFAULT_PORT
    api_key: str | None = field(default=None, repr=False)
    cors_origins: tuple[str, ...] = ()
    max_body_bytes: int = 32 * _MIB
    shutdown_timeout: float = 30.0
    sse_coalesce_ms: int = 0
    keepalive_seconds: float = 15.0
    slow_client_seconds: float = 30.0
    served_model_names: tuple[str, ...] = ()
    reasoning_parser: str | None = None
    tool_call_parser: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.host, str) or not self.host:  # pyright: ignore[reportUnnecessaryIsInstance]
            raise ConfigError("server.host", "INVALID", "must be a non-empty host name or address")
        _int(self.port, "server.port", 0, 65535)
        if self.api_key is not None and (not isinstance(self.api_key, str) or not self.api_key):  # pyright: ignore[reportUnnecessaryIsInstance]
            raise ConfigError("server.api_key", "INVALID", "must be a non-empty string")
        _strings(self.cors_origins, "server.cors_origins")
        _int(self.max_body_bytes, "server.max_body_bytes", 1024, None)
        _seconds(self.shutdown_timeout, "server.shutdown_timeout", allow_zero=True)
        _int(self.sse_coalesce_ms, "server.sse_coalesce_ms", 0, 1000)
        _seconds(self.keepalive_seconds, "server.keepalive_seconds", allow_zero=False)
        _seconds(self.slow_client_seconds, "server.slow_client_seconds", allow_zero=False)
        _strings(self.served_model_names, "server.served_model_names")
        if len(set(self.served_model_names)) != len(self.served_model_names):
            raise ConfigError("server.served_model_names", "DUPLICATE", "names must be unique")

    @property
    def is_loopback(self) -> bool:
        """Whether only this machine can reach the server."""
        return _is_loopback(self.host)

    @classmethod
    def from_mapping(cls, values: Mapping[str, object], *, path: str = "server") -> ServerConfig:
        """Build from a ``[server]`` table of a configuration file.

        Lists become tuples. A key this class does not have is an error, so a typo cannot be
        silently ignored.

        Raises:
            ConfigError: For an unknown key or a bad value.
        """
        known = {item.name for item in fields(cls)}
        unknown = sorted(set(values) - known)
        if unknown:
            raise ConfigError(path, "UNKNOWN_KEY", f"unknown keys: {unknown}")
        converted: dict[str, object] = {}
        for key, value in values.items():
            converted[key] = (
                tuple(cast("list[object]", value)) if isinstance(value, list) else value
            )
        try:
            return cls(**converted)  # type: ignore[arg-type]  # each field is checked in __post_init__
        except TypeError as exc:
            raise ConfigError(path, "INVALID", str(exc)) from exc


def _int(value: object, path: str, minimum: int, maximum: int | None) -> None:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ConfigError(path, "INVALID_TYPE", "must be an integer")
    if value < minimum or (maximum is not None and value > maximum):
        bound = f"in [{minimum}, {maximum}]" if maximum is not None else f">= {minimum}"
        raise ConfigError(path, "OUT_OF_RANGE", f"must be {bound}, got {value}")


def _seconds(value: object, path: str, *, allow_zero: bool) -> None:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ConfigError(path, "INVALID_TYPE", "must be a number of seconds")
    if not math.isfinite(value) or value < 0 or (value == 0 and not allow_zero):
        raise ConfigError(path, "OUT_OF_RANGE", f"must be {'>=' if allow_zero else '>'} 0")


def _strings(value: object, path: str) -> None:
    if not isinstance(value, tuple):
        raise ConfigError(path, "INVALID_TYPE", "must be a tuple of strings")
    for item in cast("tuple[object, ...]", value):
        if not isinstance(item, str) or not item:
            raise ConfigError(path, "INVALID", "every entry must be a non-empty string")
