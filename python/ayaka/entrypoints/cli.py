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

import dataclasses
import logging
import sys
from typing import TYPE_CHECKING, Final, TextIO

from ayaka.configs.cache import CacheConfig
from ayaka.configs.device import DeviceConfig
from ayaka.configs.engine import EngineConfig
from ayaka.configs.model_source import ModelSourceConfig
from ayaka.configs.scheduler import SchedulerConfig
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
from ayaka.entrypoints._cli_parsers import build_parser
from ayaka.entrypoints.console import (
    EXIT_BUG,
    EXIT_ENGINE,
    EXIT_INTERRUPTED,
    EXIT_LOAD,
    EXIT_MEMORY,
    EXIT_MISSING,
    EXIT_OK,
    EXIT_USAGE,
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
    import argparse
    from collections.abc import Iterator, Sequence

    from ayaka.core.outputs import RequestOutput

__all__ = ["main"]


_DTYPES: Final = {dtype.label: dtype for dtype in COMPUTE_DTYPES}


def _parser() -> argparse.ArgumentParser:
    return build_parser()


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


def _bench(args: argparse.Namespace) -> int:
    """Run the benchmark and write its results; exit 2 when a sanity check failed."""
    from ayaka.entrypoints.bench import bench_command

    return bench_command(args, _engine_config(args, max_running=args.max_running))


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
    cache = CacheConfig(
        kv_reserve_tokens=args.kv_reserve_tokens,
        prefix_cache=args.prefix_cache,
        kv_dtype=args.kv_dtype,
        kv_fp8_scale=args.kv_fp8_scale,
        state_checkpoint_stride=args.state_checkpoint_stride,
        state_checkpoint_max_bytes=args.state_checkpoint_max_mb * 1024 * 1024,
    )
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
            vision_load=args.vision_load,
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
