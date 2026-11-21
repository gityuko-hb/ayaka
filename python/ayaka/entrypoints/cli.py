"""The ``ayaka`` command line.

::

    ayaka run MODEL -p "Once upon a time" --max-tokens 64
    ayaka serve MODEL --port 8000 --tool-call-parser hermes
    ayaka plan MODEL --device cuda --dtype fp16
    ayaka info
    ayaka config show
    ayaka connect claude-code [--write | --check | --observe]
    ayaka bench MODEL [--quick] [--out DIR] [--record DIR]

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

Exit codes are the contract of the command; :mod:`ayaka.entrypoints.console` lists them (0 done,
1 a bug, 2 a wrong argument or request, 3 a model that does not load, 4 something missing,
5 a plan that does not fit, 6 an engine failure, 130 interrupted).

A problem the user can fix is one line, ``ayaka: error: ...``, never a traceback.

Importing this module imports neither torch nor aiohttp: ``--help`` and every argument error are
answered without them; torch loads once an engine is built, aiohttp once ``serve`` starts.
"""

from __future__ import annotations

import argparse
import dataclasses
import logging
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Final, TextIO

from ayaka.__version__ import __version__
from ayaka.configs.cache import CacheConfig
from ayaka.configs.device import DeviceConfig
from ayaka.configs.engine import EngineConfig
from ayaka.configs.model_source import ModelSourceConfig
from ayaka.configs.scheduler import SchedulerConfig
from ayaka.configs.server import DEFAULT_PORT
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
from ayaka.entrypoints.console import (
    EXIT_BUG,
    EXIT_ENGINE,
    EXIT_INTERRUPTED,
    EXIT_LOAD,
    EXIT_MEMORY,
    EXIT_MISSING,
    EXIT_OK,
    EXIT_USAGE,
    PROG,
    say,
)
from ayaka.entrypoints.settings import (
    config_file_path,
    describe_server,
    flag_values,
    read_config_file,
    server_config,
)
from ayaka.tokenizers.contracts import ChatMessage
from ayaka.utils.dtypes import COMPUTE_DTYPES
from ayaka.utils.logging import init_logger

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

    from ayaka.core.outputs import RequestOutput

__all__ = ["main"]


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
        say("error: no command given; try 'ayaka run --help' or 'ayaka serve --help'")
        return EXIT_USAGE
    if args.command == "config" and args.action is None:
        say("error: say what to do: ayaka config show")
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
            "bench": _bench,
        }[args.command]
        return handler(args)
    except KeyboardInterrupt:
        say("interrupted")
        return EXIT_INTERRUPTED
    except AyakaError as exc:
        return _report(exc)
    except Exception as exc:
        if getattr(args, "verbose", 0):
            raise
        say(f"error: unexpected {type(exc).__name__}: {exc} (run with -v for the traceback)")
        return EXIT_BUG


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog=PROG, description="Ayaka: an LLM inference engine.")
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
    _bench_parser(commands)
    return parser


def _bench_parser(commands: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:  # pyright: ignore[reportPrivateUsage]
    bench = commands.add_parser(
        "bench",
        help="measure a model on this machine: prompt read, generation, first token, sanity",
        description="Load a model in-process and measure prompt read (cold), generation, the "
        "first token (cold and warm), a short agent session, peak memory, and two sanity checks "
        "(a forced tool call, a JSON answer). Writes results.json with every raw run. Exit codes: "
        "0 done, 1 the benchmark failed, 2 a sanity check failed (the result is written and "
        "marked invalid).",
    )
    bench.add_argument("model", help="a local model directory or a Hugging Face repo id")
    bench.add_argument(
        "--out",
        type=Path,
        metavar="DIR",
        help="where results.json goes (benchmarks/results/<date>-<machine>/)",
    )
    bench.add_argument(
        "--history",
        type=Path,
        metavar="DIR",
        help="append the summary line to DIR/bench.jsonl (benchmarks/history when it exists)",
    )
    bench.add_argument("--quick", action="store_true", help="1K and 4K only, two repetitions")
    bench.add_argument("--seed", type=int, default=0, help="the run's seed (sampling is greedy)")
    bench.add_argument(
        "--record",
        type=Path,
        metavar="DIR",
        help="write every benchmark request as token ids to DIR (owner-only)",
    )
    bench.add_argument("--json", action="store_true", help="print results.json, not the summary")
    bench.add_argument(
        "--max-running", type=int, metavar="N", help="requests run at once (from the memory plan)"
    )
    _engine_flags(bench)
    bench.add_argument("-v", "--verbose", action="count", default=0, help="log more")


def _bench(args: argparse.Namespace) -> int:
    """Run the benchmark and write its results; exit 2 when a sanity check failed."""
    from ayaka.entrypoints.bench import bench_command

    return bench_command(args, _engine_config(args, max_running=args.max_running))


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
        "--record",
        type=Path,
        metavar="DIR",
        help="write every request's prompt and output token ids to DIR (owner-only; off by "
        "default; the files hold the session's content)",
    )
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
        "--tool-call-parser",
        metavar="NAME",
        help="the model's tool-call format: hermes, llama3_json",
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
    server.add_argument(
        "--stall-timeout",
        type=float,
        metavar="S",
        help="/health answers stalled once the engine has work but no progress for S (60)",
    )
    server.add_argument(
        "--no-normalize",
        dest="normalize_prompts",
        action="store_false",
        default=None,
        help="send every prompt to the tokenizer unchanged, with no client churn rules",
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
        say(
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


def _serve(args: argparse.Namespace) -> int:
    """Start the server and run it until a signal stops it."""
    import asyncio

    from ayaka.utils.import_utils import require_module

    config = server_config(args)
    engine = _engine_config(args, max_running=args.max_running)
    require_module("aiohttp", capability="ayaka serve", remedy="uv sync --extra serve")
    from ayaka.entrypoints.serve import serve

    for line in describe_server(config, engine, args.model):
        say(line)
    asyncio.run(serve(config, engine, model_name=args.model, say=say, record=args.record))
    return EXIT_OK


def _plan(args: argparse.Namespace) -> int:
    """Print the memory plan of the model on this machine; exit 5 when it does not fit."""
    import json
    from dataclasses import asdict, replace

    from ayaka.engine.build import plan_engine
    from ayaka.entrypoints.connect.tools import tool_record
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
    rows = config_rows(flag_values(args), read_config_file(file) if file else {}, file)
    width = max(len(name) for name, _, _ in rows)
    print(f"configuration file: {file if file is not None else 'none'}")
    for name, value, source in rows:
        print(f"{name:<{width}}  {value}  [{source}]")
    return EXIT_OK


def _connect(args: argparse.Namespace) -> int:
    """Print, write, check or observe a tool's configuration (``ayaka.entrypoints.connect``)."""
    from ayaka.entrypoints.connect.command import connect_command

    return connect_command(args)


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
        say(f"error: generation failed: {detail}")
        return EXIT_ENGINE
    if last.error is not None:
        say(f"warning: {last.error.message}")
    return EXIT_OK


def _report(error: AyakaError) -> int:
    """Print a typed error as one line (and its fix when it has one); return its exit code."""
    if isinstance(error, CapabilityError):
        say(f"error: {error.capability} is not available: {error.detail}")
        say(f"fix: {error.remedy}")
        return EXIT_MISSING
    say(f"error: {error.message if not isinstance(error, ConfigError) else error}")
    if isinstance(error, ModelLoadError):
        return EXIT_LOAD
    if isinstance(error, CapacityError):
        return EXIT_MEMORY
    if isinstance(error, (InvalidRequestError, ConfigError)):
        return EXIT_USAGE
    return EXIT_ENGINE
