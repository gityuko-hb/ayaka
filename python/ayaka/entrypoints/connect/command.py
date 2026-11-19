"""``ayaka connect``: the command, from its flags to the server, the tool and the exit code.

The server a tool should reach comes from the flags, then the server settings, then the server
itself (``/v1/models`` names the served model and the context one request may hold). The tool
records, the writer, the check and the observing proxy live in the other modules of this
package; this one only runs them.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING, cast

from ayaka.configs.model_source import ModelSourceConfig
from ayaka.configs.server import ServerConfig, user_config_path
from ayaka.core.errors import InvalidRequestError
from ayaka.entrypoints.console import EXIT_BUG, EXIT_OK, EXIT_USAGE, say
from ayaka.entrypoints.settings import read_config_file, read_key

if TYPE_CHECKING:
    import argparse
    from collections.abc import Mapping, Sequence

    from ayaka.entrypoints.connect.tools import ServerInfo, ToolRecord


__all__ = ["connect_command"]


def _connect_server(args: argparse.Namespace) -> ServerInfo:
    """The server a tool should reach: flags, then the server settings, then the server itself.

    A running server is asked for ``/v1/models``: it names the served model and says how many
    tokens one request may hold, which ``--write`` puts into the tool's configuration
    (``model_context_window`` for Codex, ``limit.context`` for OpenCode).
    """
    from ayaka.entrypoints.connect.tools import ServerInfo

    file = args.config if args.config is not None else user_config_path()
    values = read_config_file(file) if file is not None and file.is_file() else {}
    settings = ServerConfig.from_mapping(values)
    host = "127.0.0.1" if settings.host in ("0.0.0.0", "::") else settings.host  # noqa: S104
    url = args.url or f"http://{host}:{settings.port}"
    key = read_key(args.api_key_file) if args.api_key_file is not None else settings.api_key
    model = args.model or (settings.served_model_names[0] if settings.served_model_names else None)
    entries = _models(url, key)
    if model is None and entries:
        model = str(entries[0].get("id"))
    return ServerInfo(
        base_url=url,
        api_key=key,
        model=model or "<model>",
        small_model=args.small_model,
        context_tokens=_context_of(entries, model),
    )


def _models(url: str, key: str | None) -> list[dict[str, object]]:
    """The entries a running server reports on ``/v1/models``, or ``[]`` when it does not answer."""
    import json
    import urllib.request

    headers = {"authorization": f"Bearer {key}"} if key else {}
    request = urllib.request.Request(url.rstrip("/") + "/v1/models", headers=headers)  # noqa: S310
    try:
        # A local server answers at once; a second keeps a stopped one from stalling the print.
        with urllib.request.urlopen(request, timeout=1) as response:  # noqa: S310
            listing: object = json.loads(response.read())
    except (OSError, ValueError):
        return []
    data = cast("dict[str, object]", listing).get("data") if isinstance(listing, dict) else None
    if not isinstance(data, list):
        return []
    entries = cast("list[object]", data)
    if not all(isinstance(entry, dict) for entry in entries):
        return []
    return cast("list[dict[str, object]]", entries)


def _context_of(entries: Sequence[Mapping[str, object]], model: str | None) -> int | None:
    """The context one request may hold: the tool's model's entry, else the first entry."""
    for entry in entries:
        if str(entry.get("id")) == model:
            entries = [entry]
            break
    for entry in entries:
        value = entry.get("max_context_tokens") or entry.get("max_model_len")
        if isinstance(value, int):
            return value
    return None


def connect_command(args: argparse.Namespace) -> int:
    """Print, write, check or observe a tool's configuration."""
    import json

    from ayaka.entrypoints.connect.check import check_lines, run_check
    from ayaka.entrypoints.connect.tools import (
        generic_setup,
        redact,
        render_config,
        tool_record,
        tool_version,
    )
    from ayaka.entrypoints.connect.write import write_config

    server = _connect_server(args)
    record = tool_record(args.tool) if args.tool else None
    if record is None:
        if args.tool:
            say(f"no record for {args.tool!r}; known: claude-code, codex, opencode")
        for line in generic_setup(server):
            print(line)
        return EXIT_USAGE if (args.write or args.check or args.observe) else EXIT_OK
    if args.check:
        report = run_check(record, server, timeout=args.timeout, version=tool_version(record))
        if args.output == "json":
            print(json.dumps(report.as_json(), indent=2))
        else:
            for line in check_lines(report):
                print(line)
        return EXIT_OK if report.ok else EXIT_BUG
    if args.observe:
        return _observe(args, record, server)
    path = record.config_path()
    if not args.write:
        # Only what Ayaka adds: the rest of the user's file is theirs and is not printed.
        print(f"# merge into {path} (ayaka connect {record.name} --write does it)")
        print(redact(render_config(record, server, ""), server), end="")
        print(f"# then: {record.usage}")
        return EXIT_OK
    for name in record.alternates:
        if (path.parent / name).exists() and not path.exists():
            raise InvalidRequestError(
                f"{path.parent / name} holds this tool's settings and may carry comments a "
                "rewrite would drop; merge the printed configuration into it by hand "
                f"(ayaka connect {record.name})"
            )
    current = path.read_text(encoding="utf-8") if path.exists() else ""
    try:
        content = render_config(record, server, current)
    except ValueError as exc:
        raise InvalidRequestError(
            f"{path} is not a JSON object Ayaka can merge into ({exc}); fix it or move it away"
        ) from exc
    interactive = sys.stdin is not None and sys.stdin.isatty()
    if not args.yes and not interactive:
        raise InvalidRequestError("--write without a terminal needs --yes")

    def confirm(question: str) -> bool:
        if args.yes:
            return True
        return input(f"{question} [y/N] ").strip().lower() in ("y", "yes")

    try:
        write_config(
            path,
            content,
            confirm=confirm,
            create_dirs=args.create_dirs,
            redactor=lambda text: redact(text, server),
        )
    except FileNotFoundError as exc:
        raise InvalidRequestError(str(exc)) from exc
    print(f"then: {record.usage}")
    return EXIT_OK


def _observe(args: argparse.Namespace, record: ToolRecord, server: ServerInfo) -> int:
    """Run the proxy until the tool's first requests arrived; print and keep only the counts."""
    import threading
    from dataclasses import replace

    from ayaka.configs.tokenizer import tokenizer_config_from_source
    from ayaka.entrypoints.connect.observe import (
        ObservedPrompt,
        largest,
        observer,
        recorder,
        save_observation,
        serve_proxy,
    )
    from ayaka.entrypoints.connect.tools import redact, render_config, tool_version
    from ayaka.tokenizers.factory import DefaultTokenizerFactory

    if not args.tokenizer:
        raise InvalidRequestError("--observe needs --tokenizer MODEL to count tokens")
    loaded = DefaultTokenizerFactory.from_config(
        tokenizer_config_from_source(ModelSourceConfig(args.tokenizer, offline=True))
    )
    tokenizer = loaded.tokenizer
    if tokenizer is None:
        raise InvalidRequestError(f"{args.tokenizer} has no tokenizer to count with")

    def count(text: str) -> int:
        return len(tokenizer.encode(text, add_special_tokens=False))

    found: list[ObservedPrompt] = []
    stop = threading.Event()
    keep = recorder(args.record) if args.record is not None else None
    observe = observer(record, args.tokenizer, count, found, installed=tool_version(record))

    def on_request(path: str, headers: dict[str, str], body: bytes) -> None:
        before = len(found)
        observe(path, headers, body)
        if keep is not None:
            keep(path, headers, body)
        if len(found) > before:
            latest = found[-1]
            say(
                f"request {len(found)}: {latest.tool_count} tools, fixed prompt "
                f"{latest.fixed_tokens} tokens"
            )
        if len(found) >= args.count:
            stop.set()

    closed = threading.Event()
    port = serve_proxy(
        args.port, server.base_url, on_request, stop=stop, host=args.listen, closed=closed
    )
    shown = "127.0.0.1" if args.listen in ("0.0.0.0", "::") else args.listen  # noqa: S104
    proxied = replace(server, base_url=f"http://{shown}:{port}")
    say(f"listening on {args.listen}:{port}, forwarding to {server.base_url}")
    say(
        f"point {record.name} at the proxy with this configuration, then send one prompt "
        f"(the proxy stops after {args.count} model requests, or on Ctrl-C):"
    )
    print(redact(render_config(record, proxied, ""), server), end="")
    if args.record is not None:
        say(
            f"warning: the requests are kept in {args.record}, prompts included; "
            "remove personal and project content before sharing them"
        )
    try:
        stop.wait()
        closed.wait()  # the last request reaches the tool before the proxy goes away
    except KeyboardInterrupt:
        stop.set()
    observed = largest(found)
    if observed is None:
        say("no model request arrived")
        return EXIT_USAGE
    out = args.out or Path.cwd() / "ayaka-observations"
    saved = save_observation(observed, out)
    print(
        f"{record.name} {observed.version or '(version unknown)'}: the largest of "
        f"{len(found)} requests has system {observed.system_tokens} tokens and "
        f"{observed.tool_count} tools {observed.tools_tokens} tokens: fixed prompt "
        f"{observed.fixed_tokens} tokens with the {args.tokenizer} tokenizer "
        f"(counts only, in {saved})"
    )
    return EXIT_OK
