"""The ``ayaka`` command line.

::

    ayaka run MODEL -p "Once upon a time" --max-tokens 64
    ayaka serve MODEL --port 8000 --tool-call-parser hermes
    ayaka plan MODEL --device cuda --dtype fp16
    ayaka info
    ayaka config show
    ayaka connect claude-code [--write | --check | --observe]

``MODEL`` is a local directory or a Hugging Face repo id. ``run`` writes the generated text to
standard output as it is produced; everything else (the line that says what was started,
warnings, errors) goes to standard error, so ``ayaka run ... > out.txt`` holds only the text. The
line that says what was started names the values really used, which come from the flags and never
from the environment.

``serve`` answers OpenAI and Anthropic clients over HTTP (it needs the ``serve`` extra). Its
settings come from the flags, then from the ``[server]`` table of ``--config FILE`` (TOML; without
the flag, the user's configuration file when it exists, see ``ayaka info``), then from the
defaults; never from the environment. ``config show`` prints those settings and where each came
from.

``plan`` prints whether the model fits this machine and how the memory would be divided, from the
same planner a start-up runs, without loading a weight or downloading anything; ``--output json``
gives the same as data. ``info`` prints the version, devices, backends, installed extras and the
directories Ayaka uses.

``connect TOOL`` prints the configuration a coding tool needs for a running server; ``--write``
puts it in the tool's file after a diff, a backup and a confirmation; ``--check`` runs seven steps
against the server in the tool's protocol and exits 1 when one fails; ``--observe`` stands between
the tool and the server and counts the tokens of the tool's fixed prompt (``--record DIR`` keeps
the requests themselves, credentials removed). It prints where it listens and when it is ready,
and stops on SIGINT or SIGTERM: new requests get 503, requests in flight get
``--shutdown-timeout`` seconds, and a second signal exits at once.

Exit codes are the contract of the command::

    0    the text was generated, the server stopped cleanly, or the plan fits
    1    an unexpected error: a bug; ``-v`` prints its traceback
    2    the arguments or the request are wrong (a flag, a prompt too long, an unsupported option)
    3    the model or the tokenizer cannot be loaded
    4    something the run needs is not installed or not available (torch, a tokenizer package,
         a GPU)
    5    the memory plan does not fit this machine (``plan`` prints three ways out)
    6    the engine failed while generating
    130  interrupted

A problem the user can fix is one line, ``ayaka: error: ...``, never a traceback.

Importing this module imports neither torch nor aiohttp: ``--help`` and every argument error are
answered without them; torch loads once an engine is built, aiohttp once ``serve`` starts.
"""

from __future__ import annotations

import argparse
import dataclasses
import logging
import sys
import tomllib
from pathlib import Path
from typing import TYPE_CHECKING, Final, TextIO, cast

from ayaka.__version__ import __version__
from ayaka.configs.cache import CacheConfig
from ayaka.configs.device import DeviceConfig
from ayaka.configs.engine import EngineConfig
from ayaka.configs.model_source import ModelSourceConfig
from ayaka.configs.scheduler import SchedulerConfig
from ayaka.configs.server import DEFAULT_PORT, ServerConfig, user_config_path
from ayaka.core.errors import (
    AyakaError,
    CapabilityError,
    CapacityError,
    ConfigError,
    EngineError,
    InvalidRequestError,
    ModelLoadError,
)
from ayaka.core.params import SamplingParams
from ayaka.core.types import FinishReason
from ayaka.engine.llm import LLM
from ayaka.entrypoints.parsers.reasoning import reasoning_format
from ayaka.entrypoints.parsers.tool_calls import tool_call_format
from ayaka.tokenizers.contracts import ChatMessage
from ayaka.utils.dtypes import COMPUTE_DTYPES
from ayaka.utils.logging import init_logger

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping, Sequence

    from ayaka.core.outputs import RequestOutput
    from ayaka.entrypoints.connect import ServerInfo, ToolRecord

__all__ = ["main"]

EXIT_OK: Final = 0
EXIT_BUG: Final = 1
EXIT_USAGE: Final = 2
EXIT_LOAD: Final = 3
EXIT_MISSING: Final = 4
EXIT_MEMORY: Final = 5
EXIT_ENGINE: Final = 6
EXIT_INTERRUPTED: Final = 130

_PROG: Final = "ayaka"
_DTYPES: Final = {dtype.label: dtype for dtype in COMPUTE_DTYPES}


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command line and return its exit code.

    Args:
        argv: The arguments after the program name; ``None`` reads ``sys.argv``.

    Returns:
        The exit code described in this module's docstring. Argument errors and ``--help`` exit
        through ``SystemExit`` like any ``argparse`` program, with code 2 and 0.
    """
    parser = _parser()
    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_usage(sys.stderr)
        _say("error: no command given; try 'ayaka run --help' or 'ayaka serve --help'")
        return EXIT_USAGE
    if args.command == "config" and args.action is None:
        _say("error: say what to do: ayaka config show")
        return EXIT_USAGE
    if getattr(args, "verbose", 0):
        init_logger("ayaka", level=logging.DEBUG if args.verbose > 1 else logging.INFO)
    elif args.command == "serve":
        # Request lines (ids, outcomes, token counts; never text) are part of serving.
        init_logger("ayaka", level=logging.INFO)
    try:
        handler = {
            "serve": _serve,
            "run": _run,
            "plan": _plan,
            "info": _info,
            "config": _config_show,
            "connect": _connect,
        }[args.command]
        return handler(args)
    except KeyboardInterrupt:
        _say("interrupted")
        return EXIT_INTERRUPTED
    except AyakaError as exc:
        return _report(exc)
    except Exception as exc:
        if getattr(args, "verbose", 0):
            raise
        _say(f"error: unexpected {type(exc).__name__}: {exc} (run with -v for the traceback)")
        return EXIT_BUG


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog=_PROG, description="Ayaka: an LLM inference engine.")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    commands = parser.add_subparsers(dest="command", metavar="command")

    run = commands.add_parser(
        "run",
        help="generate text for a prompt and print it",
        description="Load a model, generate text for one prompt and print it to standard output.",
    )
    run.add_argument("model", help="a local model directory or a Hugging Face repo id (owner/name)")
    run.add_argument(
        "-p",
        "--prompt",
        help="the prompt; when absent it is read from standard input",
    )
    run.add_argument(
        "--chat",
        action="store_true",
        help="send the prompt as one user message, rendered with the model's chat template",
    )
    sampling = run.add_argument_group("sampling")
    sampling.add_argument(
        "--max-tokens", type=int, default=128, metavar="N", help="most tokens to generate (128)"
    )
    sampling.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        metavar="T",
        help="softmax temperature; 0 is greedy decoding (0)",
    )
    sampling.add_argument("--top-k", type=int, default=-1, metavar="K", help="top-k; -1 is off")
    sampling.add_argument("--top-p", type=float, default=1.0, metavar="P", help="top-p (1)")
    sampling.add_argument(
        "--seed",
        type=int,
        default=None,
        metavar="S",
        help="seed of the random numbers; the same seed gives the same text",
    )
    sampling.add_argument(
        "--stop",
        action="append",
        default=[],
        metavar="TEXT",
        help="stop when this text is generated and cut it from the output (repeatable)",
    )
    _engine_flags(run)
    run.add_argument("-q", "--quiet", action="store_true", help="do not print the start-up line")
    run.add_argument(
        "-v", "--verbose", action="count", default=0, help="log progress; twice for debug"
    )
    _serve_parser(commands)
    _plan_parser(commands)
    info = commands.add_parser(
        "info",
        help="print the version, devices, backends, extras and directories",
        description="Print what this installation has: the version, the devices, the ops backends "
        "that run here, the installed extras and the directories Ayaka uses, with their sizes.",
    )
    info.add_argument("--output", choices=("text", "json"), default="text", help="text or json")
    info.add_argument("-v", "--verbose", action="count", default=0, help="log more")
    config = commands.add_parser(
        "config",
        help="print the settings a server would use",
        description="Inspect the configuration.",
    )
    actions = config.add_subparsers(dest="action", metavar="action")
    show = actions.add_parser(
        "show",
        help="print each server setting and where it comes from",
        description="Print the server settings `ayaka serve` would use with the same flags, each "
        "with its source: a flag, the configuration file or the default.",
    )
    _server_flags(show)
    show.add_argument("-v", "--verbose", action="count", default=0, help="log more")
    _connect_parser(commands)
    return parser


def _connect_parser(commands: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:  # pyright: ignore[reportPrivateUsage]
    connect = commands.add_parser(
        "connect",
        help="configure a coding tool for a running server, and check it",
        description="Print, write or check the configuration of a coding tool "
        "(claude-code, codex, opencode) for a running `ayaka serve`. Exit codes of --check: "
        "0 every step passed or could not be measured, 1 a step failed, 2 the check could not run.",
    )
    connect.add_argument("tool", nargs="?", help="claude-code, codex or opencode")
    target = connect.add_argument_group("the server")
    target.add_argument("--url", help="the server's root URL (from the server settings)")
    target.add_argument("--model", help="the model name the tool sends (the server's first name)")
    target.add_argument("--small-model", help="the name for the tool's background work")
    target.add_argument("--api-key-file", type=Path, metavar="FILE", help="the server's key")
    target.add_argument("--config", type=Path, metavar="FILE", help="a TOML file with [server]")
    mode = connect.add_mutually_exclusive_group()
    mode.add_argument("--write", action="store_true", help="write the tool's configuration file")
    mode.add_argument("--check", action="store_true", help="run the seven steps")
    mode.add_argument("--observe", action="store_true", help="count the tool's fixed prompt")
    connect.add_argument("--yes", action="store_true", help="write without asking (--write)")
    connect.add_argument(
        "--create-dirs", action="store_true", help="create the file's folder if missing (--write)"
    )
    connect.add_argument(
        "--timeout", type=float, default=60.0, metavar="S", help="seconds per step (--check)"
    )
    connect.add_argument(
        "--tokenizer", metavar="MODEL", help="the model whose tokenizer counts (--observe)"
    )
    connect.add_argument("--port", type=int, default=8100, help="the proxy's port (--observe)")
    connect.add_argument(
        "--listen", default="127.0.0.1", metavar="HOST", help="the proxy's address (--observe)"
    )
    connect.add_argument(
        "--out", type=Path, metavar="DIR", help="where the counts go (--observe; owner-only)"
    )
    connect.add_argument(
        "--record", type=Path, metavar="DIR", help="also keep each request in DIR (--observe)"
    )
    connect.add_argument(
        "--count",
        type=int,
        default=3,
        metavar="N",
        help="model requests to wait for; the largest is the tool's floor (--observe)",
    )
    connect.add_argument("--output", choices=("text", "json"), default="text", help="text or json")
    connect.add_argument("-v", "--verbose", action="count", default=0, help="log more")


def _engine_flags(command: argparse.ArgumentParser) -> None:
    """The flags that say where and how the model runs, shared by ``run`` and ``serve``."""
    where = command.add_argument_group("where and how it runs")
    where.add_argument(
        "--device", default="cpu", metavar="DEVICE", help="cpu, cuda or cuda:N (cpu)"
    )
    where.add_argument(
        "--dtype",
        choices=sorted(_DTYPES),
        default="fp32",
        help="compute dtype of the model (fp32)",
    )
    where.add_argument(
        "--backends",
        default="torch",
        metavar="LIST",
        help="ops backends, most preferred first, comma separated; each op family comes from the "
        "first that serves it, e.g. flashinfer,torch on Linux or WSL2 (torch)",
    )
    where.add_argument(
        "--cuda-graph",
        action="store_true",
        help="capture decode steps in CUDA graphs at start-up and replay them (GPU only)",
    )
    where.add_argument(
        "--memory-ratio",
        type=float,
        default=None,
        metavar="R",
        help="share of the free memory the engine may use, in (0, 1] (0.9)",
    )
    where.add_argument(
        "--kv-reserve-tokens",
        type=int,
        default=None,
        metavar="N",
        help="tokens of KV one request must be able to hold; the model context when absent",
    )
    where.add_argument(
        "--no-prefix-cache",
        dest="prefix_cache",
        action="store_false",
        help="do not share the KV of a prompt prefix between requests",
    )
    where.add_argument("--revision", default="", help="repo branch, tag or commit; default: latest")
    where.add_argument("--tokenizer", default="", help="tokenizer directory or repo id")
    where.add_argument(
        "--offline", action="store_true", help="never use the network; read the local cache only"
    )


def _serve_parser(commands: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:  # pyright: ignore[reportPrivateUsage]
    serve = commands.add_parser(
        "serve",
        help="answer OpenAI and Anthropic clients over HTTP",
        description="Load a model and answer OpenAI Chat Completions, Completions and Responses "
        "and Anthropic Messages requests over HTTP. Needs the serve extra.",
    )
    serve.add_argument("model", help="a local model directory or a Hugging Face repo id")
    _server_flags(serve)
    serve.add_argument(
        "--max-running", type=int, metavar="N", help="requests run at once (from the memory plan)"
    )
    _engine_flags(serve)
    serve.add_argument(
        "-v", "--verbose", action="count", default=0, help="log more; twice for debug"
    )


def _plan_parser(commands: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:  # pyright: ignore[reportPrivateUsage]
    plan = commands.add_parser(
        "plan",
        help="print whether a model fits this machine and how memory would be divided",
        description="Plan the memory of a model on this machine with the planner a start-up "
        "runs: weights, KV cache, context per request, requests at once. Reads config.json only: "
        "no weight is loaded and nothing is downloaded. Exits 5 when it does not fit.",
    )
    plan.add_argument("model", help="a local model directory or a cached Hugging Face repo id")
    plan.add_argument(
        "--max-running", type=int, metavar="N", help="requests run at once (from the memory plan)"
    )
    plan.add_argument("--output", choices=("text", "json"), default="text", help="text or json")
    plan.add_argument(
        "--for",
        dest="for_tool",
        choices=("claude-code", "codex", "opencode"),
        help="also say whether one request holds this coding tool's measured prompt "
        "(exit 5 when not)",
    )
    _engine_flags(plan)
    plan.add_argument("-v", "--verbose", action="count", default=0, help="log more")


def _server_flags(command: argparse.ArgumentParser) -> None:
    """The flags of the server settings, shared by ``serve`` and ``config show``."""
    server = command.add_argument_group(
        "server (flags win over --config, which wins over defaults)"
    )
    server.add_argument("--config", type=Path, metavar="FILE", help="a TOML file with [server]")
    server.add_argument("--host", help="address to bind (127.0.0.1: this machine only)")
    server.add_argument("--port", type=int, help=f"TCP port ({DEFAULT_PORT}); 0 picks a free one")
    server.add_argument("--api-key", help="require this key (Authorization: Bearer, or x-api-key)")
    server.add_argument(
        "--api-key-file",
        type=Path,
        metavar="FILE",
        help="read the key from the first line of FILE (keeps it out of the process list)",
    )
    server.add_argument(
        "--cors-origin",
        action="append",
        metavar="ORIGIN",
        help="let browsers call from ORIGIN; * for any (repeatable)",
    )
    server.add_argument(
        "--served-model-name",
        action="append",
        metavar="NAME",
        help="a name clients may use for the model; the first is reported (repeatable)",
    )
    server.add_argument(
        "--reasoning-parser",
        metavar="NAME",
        help="the model's reasoning format: think, deepseek_r1",
    )
    server.add_argument(
        "--tool-call-parser", metavar="NAME", help="the model's tool-call format: hermes"
    )
    server.add_argument("--max-body-bytes", type=int, metavar="N", help="largest request body")
    server.add_argument(
        "--shutdown-timeout",
        type=float,
        metavar="S",
        help="seconds requests in flight get after SIGINT/SIGTERM (30)",
    )
    server.add_argument(
        "--sse-coalesce-ms", type=int, metavar="MS", help="wait this long to batch tokens (0)"
    )
    server.add_argument(
        "--keepalive", type=float, metavar="S", help="idle seconds between keepalive frames (15)"
    )
    server.add_argument(
        "--slow-client-timeout",
        type=float,
        metavar="S",
        help="cancel a request whose client blocks a write this long (30)",
    )


def _run(args: argparse.Namespace) -> int:
    """Build the engine from the flags, generate and print; raise typed errors for the caller."""
    prompt = _read_prompt(args.prompt)
    sampling = SamplingParams(
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        seed=args.seed,
        max_tokens=args.max_tokens,
        stop=tuple(args.stop),
    )
    config = _engine_config(args)
    if not args.quiet:
        graphs = ", decode in CUDA graphs" if config.device.cuda_graph else ""
        _say(
            f"running {args.model} on {config.device.device} in {config.device.dtype.label} "
            f"(ops backends: {', '.join(config.device.backends)}{graphs})"
        )
    with LLM.from_config(config) as llm:
        chunks = (
            llm.stream_chat([ChatMessage(role="user", content=prompt)], sampling)
            if args.chat
            else llm.stream(prompt, sampling)
        )
        return _print(chunks, sys.stdout)


def _engine_config(args: argparse.Namespace, *, max_running: int | None = 1) -> EngineConfig:
    """The engine configuration the flags describe; ``run`` runs one request at a time."""
    cache = CacheConfig(kv_reserve_tokens=args.kv_reserve_tokens, prefix_cache=args.prefix_cache)
    if args.memory_ratio is not None:
        cache = dataclasses.replace(cache, memory_ratio=args.memory_ratio)
    return EngineConfig(
        source=ModelSourceConfig(
            args.model, revision=args.revision, tokenizer=args.tokenizer, offline=args.offline
        ),
        device=DeviceConfig(
            device=args.device,
            dtype=_DTYPES[args.dtype],
            backends=tuple(name.strip() for name in args.backends.split(",") if name.strip()),
            cuda_graph=args.cuda_graph,
        ),
        cache=cache,
        scheduler=SchedulerConfig(max_running=max_running),
    )


_SERVER_FLAGS: Final = (
    ("host", "host"),
    ("port", "port"),
    ("cors_origin", "cors_origins"),
    ("served_model_name", "served_model_names"),
    ("reasoning_parser", "reasoning_parser"),
    ("tool_call_parser", "tool_call_parser"),
    ("max_body_bytes", "max_body_bytes"),
    ("shutdown_timeout", "shutdown_timeout"),
    ("sse_coalesce_ms", "sse_coalesce_ms"),
    ("keepalive", "keepalive_seconds"),
    ("slow_client_timeout", "slow_client_seconds"),
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
        values.update(_config_file(file))
    for flag, key in _SERVER_FLAGS:
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
        values["api_key"] = _read_key(args.api_key_file)
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


def _flag_values(args: argparse.Namespace) -> dict[str, object]:
    """The server settings given as flags, by setting name; the key reported only as set."""
    values: dict[str, object] = {}
    for flag, key in _SERVER_FLAGS:
        value: object = getattr(args, flag)
        if isinstance(value, list):
            values[key] = tuple(cast("list[object]", value))
        elif value is not None:
            values[key] = value
    if args.api_key is not None or args.api_key_file is not None:
        values["api_key"] = "set"
    return values


def _config_file(path: Path) -> dict[str, object]:
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


def _read_key(path: Path) -> str:
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


def _serve(args: argparse.Namespace) -> int:
    """Start the server and run it until a signal stops it."""
    import asyncio

    from ayaka.utils.import_utils import require_module

    config = server_config(args)
    engine = _engine_config(args, max_running=args.max_running)
    require_module("aiohttp", capability="ayaka serve", remedy="uv sync --extra serve")
    from ayaka.entrypoints.serve import serve

    for line in describe_server(config, engine, args.model):
        _say(line)
    asyncio.run(serve(config, engine, model_name=args.model, say=_say))
    return EXIT_OK


def _plan(args: argparse.Namespace) -> int:
    """Print the memory plan of the model on this machine; exit 5 when it does not fit."""
    import json
    from dataclasses import asdict, replace

    from ayaka.engine.build import plan_engine
    from ayaka.entrypoints.connect import tool_record
    from ayaka.entrypoints.plan import (
        plan_lines,
        plan_report,
        shortfall_lines,
        shortfall_of,
        tool_fit,
        tool_fit_lines,
    )

    config = _engine_config(args, max_running=args.max_running)
    # A plan never downloads: a repo id must already be in the Hub cache.
    config = replace(config, source=replace(config.source, offline=True))
    try:
        plan = plan_engine(config)
    except CapacityError as exc:
        shortfall = shortfall_of(exc)
        if args.output == "json":
            report = {"fits": False, "model": args.model, "shortfall": asdict(shortfall)}
            print(json.dumps(report, indent=2))
        else:
            for line in shortfall_lines(shortfall):
                print(line)
        return EXIT_MEMORY
    record = tool_record(args.for_tool) if args.for_tool else None
    fit = tool_fit(plan, record) if record is not None else None
    if args.output == "json":
        report = plan_report(plan, args.model)
        if fit is not None:
            report["for"] = asdict(fit) | {"needed_tokens": fit.needed_tokens, "fits": fit.fits}
        print(json.dumps(report, indent=2))
    else:
        for line in plan_lines(plan, args.model) + (tool_fit_lines(fit) if fit else []):
            print(line)
    return EXIT_MEMORY if fit is not None and fit.fits is False else EXIT_OK


def _info(args: argparse.Namespace) -> int:
    """Print what this installation has."""
    import json

    from ayaka.entrypoints.info import info_lines, info_report

    report = info_report()
    if args.output == "json":
        print(json.dumps(report.as_json(), indent=2))
    else:
        for line in info_lines(report):
            print(line)
    return EXIT_OK


def _config_show(args: argparse.Namespace) -> int:
    """Print each server setting, its value and its source."""
    from ayaka.entrypoints.info import config_rows

    server_config(args)  # the same checks serve makes: a bad value fails here too
    file = config_file_path(args)
    rows = config_rows(_flag_values(args), _config_file(file) if file else {}, file)
    width = max(len(name) for name, _, _ in rows)
    print(f"configuration file: {file if file is not None else 'none'}")
    for name, value, source in rows:
        print(f"{name:<{width}}  {value}  [{source}]")
    return EXIT_OK


def _connect_server(args: argparse.Namespace) -> ServerInfo:
    """The server a tool should reach: flags, then the server settings, then the server itself.

    A running server is asked for ``/v1/models``: it names the served model and says how many
    tokens one request may hold, which ``--write`` puts into the tool's configuration
    (``model_context_window`` for Codex, ``limit.context`` for OpenCode).
    """
    from ayaka.entrypoints.connect import ServerInfo

    file = args.config if args.config is not None else user_config_path()
    values = _config_file(file) if file is not None and file.is_file() else {}
    settings = ServerConfig.from_mapping(values)
    host = "127.0.0.1" if settings.host in ("0.0.0.0", "::") else settings.host  # noqa: S104
    url = args.url or f"http://{host}:{settings.port}"
    key = _read_key(args.api_key_file) if args.api_key_file is not None else settings.api_key
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


def _connect(args: argparse.Namespace) -> int:
    """Print, write, check or observe a tool's configuration."""
    import json

    from ayaka.entrypoints.connect import (
        check_lines,
        generic_setup,
        redact,
        render_config,
        run_check,
        tool_record,
        tool_version,
        write_config,
    )

    server = _connect_server(args)
    record = tool_record(args.tool) if args.tool else None
    if record is None:
        if args.tool:
            _say(f"no record for {args.tool!r}; known: claude-code, codex, opencode")
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
    from ayaka.entrypoints.connect import (
        ObservedPrompt,
        largest,
        observer,
        recorder,
        redact,
        render_config,
        save_observation,
        serve_proxy,
        tool_version,
    )
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
            _say(
                f"request {len(found)}: {latest.tool_count} tools, fixed prompt "
                f"{latest.fixed_tokens} tokens"
            )
        if len(found) >= args.count:
            stop.set()

    port = serve_proxy(args.port, server.base_url, on_request, stop=stop, host=args.listen)
    shown = "127.0.0.1" if args.listen in ("0.0.0.0", "::") else args.listen  # noqa: S104
    proxied = replace(server, base_url=f"http://{shown}:{port}")
    _say(f"listening on {args.listen}:{port}, forwarding to {server.base_url}")
    _say(
        f"point {record.name} at the proxy with this configuration, then send one prompt "
        f"(the proxy stops after {args.count} model requests, or on Ctrl-C):"
    )
    print(redact(render_config(record, proxied, ""), server), end="")
    if args.record is not None:
        _say(
            f"warning: the requests are kept in {args.record}, prompts included; "
            "remove personal and project content before sharing them"
        )
    try:
        stop.wait()
    except KeyboardInterrupt:
        stop.set()
    observed = largest(found)
    if observed is None:
        _say("no model request arrived")
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


def _read_prompt(prompt: str | None) -> str:
    if prompt is not None:
        return prompt
    if sys.stdin is None or sys.stdin.isatty():
        raise InvalidRequestError("no prompt: pass one with -p, or pipe it to standard input")
    text = sys.stdin.read()
    if not text.strip():
        raise InvalidRequestError("the prompt on standard input is empty")
    return text.rstrip("\n")


def _print(chunks: Iterator[RequestOutput], out: TextIO) -> int:
    """Write the text of each chunk as it arrives; return the exit code of the last one."""
    last: RequestOutput | None = None
    wrote = ""
    for chunk in chunks:
        last = chunk
        text = chunk.outputs[0].text
        if text:
            out.write(text)
            out.flush()
            wrote = text
    if wrote and not wrote.endswith("\n"):
        out.write("\n")
        out.flush()
    if last is None:
        raise EngineError("the engine produced no output")
    reason = last.outputs[0].finish_reason
    if reason in (FinishReason.ERROR, FinishReason.TIMEOUT):
        detail = last.error.message if last.error is not None else str(reason)
        _say(f"error: generation failed: {detail}")
        return EXIT_ENGINE
    if last.error is not None:
        _say(f"warning: {last.error.message}")
    return EXIT_OK


def _report(error: AyakaError) -> int:
    """Print a typed error as one line (and its fix when it has one); return its exit code."""
    if isinstance(error, CapabilityError):
        _say(f"error: {error.capability} is not available: {error.detail}")
        _say(f"fix: {error.remedy}")
        return EXIT_MISSING
    _say(f"error: {error.message if not isinstance(error, ConfigError) else error}")
    if isinstance(error, ModelLoadError):
        return EXIT_LOAD
    if isinstance(error, CapacityError):
        return EXIT_MEMORY
    if isinstance(error, (InvalidRequestError, ConfigError)):
        return EXIT_USAGE
    return EXIT_ENGINE


def _say(message: str) -> None:
    print(f"{_PROG}: {message}", file=sys.stderr)
