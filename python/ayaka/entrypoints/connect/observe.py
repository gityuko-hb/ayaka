"""``ayaka connect --observe``: a proxy between a tool and the server that counts the tool's prompt.

The proxy forwards every request, counts the tokens of each model request by part (system
prompt, tool definitions, conversation) and keeps only the counts of the largest one, the
tool's floor. With ``--record DIR`` it also keeps each request, credentials removed, to build
replay fixtures.
"""

from __future__ import annotations

import http.client
import http.server
import json
import logging
import threading
import urllib.parse
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Final, cast

from ayaka.entrypoints.connect.tools import Protocol, ToolRecord, json_object
from ayaka.entrypoints.connect.write import make_private

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence
    from pathlib import Path


_LOG = logging.getLogger(__name__)

__all__ = [
    "ObservedPrompt",
    "agent_version",
    "count_parts",
    "largest",
    "observer",
    "recorder",
    "redact_headers",
    "save_observation",
    "serve_proxy",
]


_SECRET_HEADERS: Final = frozenset({"authorization", "x-api-key", "cookie", "proxy-authorization"})


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
        make_private(target)
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
        parsed = json_object(body)
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


def save_observation(observed: ObservedPrompt, directory: Path) -> Path:
    """Append the counts to ``directory/observations.jsonl`` (owner-only); never content."""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "observations.jsonl"
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(asdict(observed) | {"fixed_tokens": observed.fixed_tokens}) + "\n")
    make_private(path)
    return path
