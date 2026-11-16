"""The ``ayaka`` command line.

::

    ayaka run MODEL -p "Once upon a time" --max-tokens 64

``MODEL`` is a local directory or a Hugging Face repo id. The generated text goes to standard
output as it is produced; everything else (the line that says what was started, warnings,
errors) goes to standard error, so ``ayaka run ... > out.txt`` holds only the text. The line that
says what was started names the values really used, which come from the flags and never from the
environment.

Exit codes are the contract of the command::

    0    the text was generated
    1    an unexpected error: a bug; ``-v`` prints its traceback
    2    the arguments or the request are wrong (a flag, a prompt too long, an unsupported option)
    3    the model or the tokenizer cannot be loaded
    4    something the run needs is not installed or not available (torch, a tokenizer package,
         a GPU)
    5    the memory plan does not fit this machine
    6    the engine failed while generating
    130  interrupted

A problem the user can fix is one line, ``ayaka: error: ...``, never a traceback.

Importing this module does not import torch: ``--help`` and every argument error are answered
without it, and torch loads only once an engine is built.
"""

from __future__ import annotations

import argparse
import logging
import sys
from typing import TYPE_CHECKING, Final, TextIO

from ayaka.__version__ import __version__
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
from ayaka.tokenizers.contracts import ChatMessage
from ayaka.utils.dtypes import COMPUTE_DTYPES
from ayaka.utils.logging import init_logger

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

    from ayaka.core.outputs import RequestOutput

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
        _say("error: no command given; try 'ayaka run --help'")
        return EXIT_USAGE
    if args.verbose:
        init_logger("ayaka", level=logging.DEBUG if args.verbose > 1 else logging.INFO)
    try:
        return _run(args)
    except KeyboardInterrupt:
        _say("interrupted")
        return EXIT_INTERRUPTED
    except AyakaError as exc:
        return _report(exc)
    except Exception as exc:
        if args.verbose:
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
    where = run.add_argument_group("where and how it runs")
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
        "--memory-ratio",
        type=float,
        default=None,
        metavar="R",
        help="share of the free memory the engine may use, in (0, 1] (0.9)",
    )
    where.add_argument("--revision", default="", help="repo branch, tag or commit; default: latest")
    where.add_argument("--tokenizer", default="", help="tokenizer directory or repo id")
    where.add_argument(
        "--offline", action="store_true", help="never use the network; read the local cache only"
    )
    run.add_argument("-q", "--quiet", action="store_true", help="do not print the start-up line")
    run.add_argument(
        "-v", "--verbose", action="count", default=0, help="log progress; twice for debug"
    )
    return parser


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
        _say(
            f"running {args.model} on {config.device.device} in {config.device.dtype.label} "
            f"(ops backends: {', '.join(config.device.backends)})"
        )
    with LLM.from_config(config) as llm:
        chunks = (
            llm.stream_chat([ChatMessage(role="user", content=prompt)], sampling)
            if args.chat
            else llm.stream(prompt, sampling)
        )
        return _print(chunks, sys.stdout)


def _engine_config(args: argparse.Namespace) -> EngineConfig:
    """The engine configuration the flags describe, for one request."""
    cache = (
        CacheConfig() if args.memory_ratio is None else CacheConfig(memory_ratio=args.memory_ratio)
    )
    return EngineConfig(
        source=ModelSourceConfig(
            args.model, revision=args.revision, tokenizer=args.tokenizer, offline=args.offline
        ),
        device=DeviceConfig(device=args.device, dtype=_DTYPES[args.dtype]),
        cache=cache,
        # One request is all this command ever runs, so the cache is built for one.
        scheduler=SchedulerConfig(max_running=1),
    )


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
