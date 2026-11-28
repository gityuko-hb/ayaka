"""The ``ayaka`` command line.

::

    ayaka run MODEL -p "Once upon a time" --max-tokens 64
    ayaka serve MODEL --port 8000 --tool-call-parser hermes
    ayaka plan MODEL --device cuda --dtype fp16
    ayaka info
    ayaka config show
    ayaka connect claude-code [--write | --check | --observe]
    ayaka bench MODEL [--quick] [--out DIR] [--record DIR]

Importing this module imports neither torch nor aiohttp: ``--help`` and every argument error are
answered without them.
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
from ayaka.configs.speculative import (
    SpeculativeConfig,
)
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
    """Run the command line and return its exit code."""
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
            "engine-server": _engine_server,
            "render-server": _render_server,
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
    from ayaka.entrypoints.bench import bench_command

    return bench_command(args, _engine_config(args, max_running=args.max_running))


def _run(args: argparse.Namespace) -> int:
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
        spec = f", speculative={config.speculative.method}" if config.speculative.enabled else ""
        say(
            f"running {args.model} on {config.device.device} in {config.device.dtype.label} "
            f"(ops backends: {', '.join(config.device.backends)}{graphs}{spec})"
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
    speculative = SpeculativeConfig(
        enabled=args.speculative != "off",
        method="lookup" if args.speculative == "off" else args.speculative,
        max_draft_tokens=args.spec_max_draft_tokens,
        controller=args.spec_controller,
        fixed_k=args.spec_fixed_k,
    )
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
        structured_backend=args.structured_backend,
        speculative=speculative,
    )


def _serve(args: argparse.Namespace) -> int:
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
    from ayaka.entrypoints.info import config_rows

    server_config(args)
    file = config_file_path(args)
    rows = config_rows(flag_values(args), read_config_file(file) if file else {}, file)
    width = max(len(name) for name, _, _ in rows)
    print(f"configuration file: {file if file is not None else 'none'}")
    for name, value, source in rows:
        print(f"{name:<{width}}  {value}  [{source}]")
    return EXIT_OK


def _connect(args: argparse.Namespace) -> int:
    from ayaka.entrypoints.connect.command import connect_command

    return connect_command(args)


def _engine_server(args: argparse.Namespace) -> int:
    import asyncio

    from ayaka.entrypoints.grpc.runners import serve_engine_grpc
    from ayaka.utils.import_utils import require_module

    require_module("grpc", capability="ayaka engine-server", remedy="uv sync --extra grpc")
    engine = _engine_config(args, max_running=args.max_running)
    asyncio.run(serve_engine_grpc(engine, listen=args.listen, say=say))
    return EXIT_OK


def _render_server(args: argparse.Namespace) -> int:
    import asyncio

    from ayaka.configs.tokenizer import tokenizer_config_from_source
    from ayaka.entrypoints.grpc.runners import serve_render_grpc
    from ayaka.utils.import_utils import require_module

    require_module("grpc", capability="ayaka render-server", remedy="uv sync --extra grpc")
    cfg = server_config(args)
    source = ModelSourceConfig(
        args.model, revision=args.revision, tokenizer=args.tokenizer, offline=args.offline
    )
    tok_cfg = tokenizer_config_from_source(source)
    asyncio.run(
        serve_render_grpc(
            source,
            tok_cfg,
            cfg,
            model_name=args.model,
            listen=args.listen,
            max_model_len=args.max_model_len,
            say=say,
        )
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
