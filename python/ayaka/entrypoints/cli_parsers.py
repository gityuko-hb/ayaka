"""Argument parser definitions for every ``ayaka`` subcommand.

Kept separate from :mod:`ayaka.entrypoints.cli` so the command dispatcher and
individual subcommand parsers stay concise and under the module size limit.
Nothing here imports ``torch`` or ``aiohttp``.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Final

from ayaka.__version__ import __version__
from ayaka.configs.server import DEFAULT_PORT
from ayaka.entrypoints.console import PROG
from ayaka.utils.dtypes import COMPUTE_DTYPES

__all__ = ["build_parser"]

_DTYPES: Final = {dtype.label: dtype for dtype in COMPUTE_DTYPES}


def build_parser() -> argparse.ArgumentParser:
    """Build the top-level ``ayaka`` CLI argument parser."""
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
    _info_parser(commands)
    _doctor_parser(commands)
    _config_parser(commands)
    _connect_parser(commands)
    _bench_parser(commands)
    return parser


def _info_parser(commands: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:  # pyright: ignore[reportPrivateUsage]
    info = commands.add_parser(
        "info",
        help="print the version, devices, backends, extras and directories",
        description="Print what this installation has: the version, the devices, the ops backends "
        "that run here, the installed extras and the directories Ayaka uses, with their sizes.",
    )
    info.add_argument("--output", choices=("text", "json"), default="text", help="text or json")
    info.add_argument("-v", "--verbose", action="count", default=0, help="log more")


def _doctor_parser(commands: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:  # pyright: ignore[reportPrivateUsage]
    doctor = commands.add_parser(
        "doctor",
        help="check CUDA, memory, toolchain, disk space and startup profiles",
        description="Run environment and memory-health diagnostics: CUDA driver visibility, "
        "NVML-vs-CUDA free-memory gap, host pinned-RAM headroom, FlashInfer/nvcc availability, "
        "cache disk space and startup profile validity. Prints pass/warn/fail with one concrete "
        "fix per issue. Exits 1 when any check fails.",
    )
    doctor.add_argument("--output", choices=("text", "json"), default="text", help="text or json")
    doctor.add_argument(
        "--min-disk-gib",
        type=float,
        default=5.0,
        metavar="GIB",
        help="minimum free disk space on cache directories in GiB (5.0)",
    )
    doctor.add_argument("-v", "--verbose", action="count", default=0, help="log more")


def _config_parser(commands: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:  # pyright: ignore[reportPrivateUsage]
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
        "runs: weights, KV cache, context per request, requests at once. Reads config.json or a "
        "GGUF header only: no weight is loaded, nothing is downloaded. Exits 5 if it does not fit.",
    )
    plan.add_argument("model", help="a model directory, a .gguf file or a cached Hub repo id")
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
    server.add_argument(
        "--overlong-policy",
        choices=("reject", "truncate"),
        default=None,
        help="what to do when a conversation exceeds context or KV capacity (reject)",
    )
