"""How long a large request holds up the server's event loop.

An in-process server (the real HTTP application and :class:`~ayaka.entrypoints.serving.Serving`,
a real tokenizer, an engine that makes no model calls) answers Anthropic Messages requests of
about ``--body-kb`` KB, built from the recorded Claude Code session, one at a time from a client
in a separate process. A probe on the server's loop sleeps 1 ms over and over and records by how
much each sleep overran: that overrun is time the loop could not run anything else, such as
another stream's frames.

Per request the result keeps the largest overrun and the total overrun (both in ms) and the
client's round-trip time. One run measures whatever ``ayaka`` is importable, so a before and
after pair runs the same script with ``PYTHONPATH`` pointing at each checkout's ``python/``::

    PYTHONPATH=<checkout>/python uv run python benchmarks/serving/loop_delay.py \
        --tokenizer unsloth/Llama-3.2-1B-Instruct --label B --revision <sha> --out out.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Final

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "e2e"))

SCHEMA: Final = "ayaka.loop-delay/1"
_FIXTURE: Final = (
    Path(__file__).resolve().parents[2]
    / "tests"
    / "entrypoints"
    / "fixtures"
    / "claude-code-2.1.327"
    / "004-anthropic.json"
)
_PROBE_SECONDS: Final = 0.001


def _body(kilobytes: int) -> bytes:
    """A Messages request of at least ``kilobytes`` KB: the recorded session's turns, repeated."""
    record = json.loads(_FIXTURE.read_text(encoding="utf-8"))["body"]
    turns = list(record["messages"])
    first_role = turns[0]["role"]
    messages = list(turns)
    while len(json.dumps(messages).encode()) < kilobytes * 1024:
        messages.extend(turns)
    # Roles must alternate where the repeats meet; a user turn bridges them when they do not.
    fixed: list[Any] = []
    for message in messages:
        if fixed and fixed[-1]["role"] == message["role"]:
            other = "assistant" if message["role"] == "user" else "user"
            fixed.append({"role": other, "content": "continue"})
        fixed.append(message)
    if fixed[-1]["role"] != "user":
        fixed.append({"role": "user", "content": "continue"})
    del first_role
    body = {
        "model": "local",
        "max_tokens": 1,
        "system": record.get("system"),
        "tools": record.get("tools", []),
        "messages": fixed,
    }
    return json.dumps({k: v for k, v in body.items() if v is not None}).encode()


def _client(url: str, kilobytes: int, requests: int, gap: float) -> list[float]:
    """Post the body ``requests`` times, one at a time; the round-trip time of each, in ms."""
    body = _body(kilobytes)
    times: list[float] = []
    for _ in range(requests):
        request = urllib.request.Request(  # noqa: S310  # the server under test
            url + "/v1/messages",
            data=body,
            headers={"content-type": "application/json", "anthropic-version": "2023-06-01"},
        )
        started = time.perf_counter()
        try:
            with urllib.request.urlopen(request, timeout=600) as response:  # noqa: S310
                response.read()
        except urllib.error.HTTPError as exc:
            raise RuntimeError(f"status {exc.code}: {exc.read()[:500]!r}") from None
        times.append((time.perf_counter() - started) * 1e3)
        time.sleep(gap)
    return times


async def _serve(args: argparse.Namespace) -> dict[str, Any]:
    from aiohttp import web
    from serving_overhead import _FakeCore  # pyright: ignore[reportMissingImports, reportUnknownVariableType]  # a script beside this one

    from ayaka.configs.server import ServerConfig
    from ayaka.configs.tokenizer import TokenizerConfig
    from ayaka.engine.async_llm import AsyncLLM
    from ayaka.entrypoints.http import create_app
    from ayaka.entrypoints.parsers.output import OutputFormat
    from ayaka.entrypoints.serving import Serving
    from ayaka.tokenizers.factory import DefaultTokenizerFactory
    from ayaka.tokenizers.service import TokenizerService

    config = TokenizerConfig(args.tokenizer, mode="hf")
    loaded = DefaultTokenizerFactory.from_config(config)
    assert loaded.tokenizer is not None  # noqa: S101
    service = TokenizerService(
        config,
        max_model_len=1 << 20,
        model_vocab_size=max(loaded.tokenizer.valid_token_ids) + 1,
        loaded=loaded,
    )
    serving = Serving(
        ServerConfig(host="127.0.0.1", port=0, max_body_bytes=64 << 20), model_name="local"
    )
    llm = AsyncLLM(_FakeCore(), registry=serving.metrics.registry)  # type: ignore[arg-type]
    llm.start()
    loop = asyncio.get_running_loop()
    windows: list[tuple[float, float]] = []

    @web.middleware
    async def timed(request: web.Request, handler: Any) -> web.StreamResponse:
        started = loop.time()
        try:
            return await handler(request)
        finally:
            windows.append((started, loop.time()))

    app = create_app(serving)
    app.middlewares.insert(0, timed)
    runner = web.AppRunner(app, handler_cancellation=True, access_log=None)
    await runner.setup()
    overruns: list[tuple[float, float]] = []
    probing = True

    async def probe() -> None:
        while probing:
            started = loop.time()
            await asyncio.sleep(_PROBE_SECONDS)
            overruns.append((started, loop.time() - started - _PROBE_SECONDS))

    try:
        site = web.TCPSite(runner, "127.0.0.1", 0)
        await site.start()
        port = int(runner.addresses[0][1])
        # The recorded requests offer tools: the server needs a tool-call format to take them.
        output_format = OutputFormat.resolve(tool_calls="llama3_json", tokenizer=loaded.tokenizer)
        serving.attach(llm, service, output_format=output_format)  # type: ignore[arg-type]
        task = asyncio.create_task(probe())
        client = await asyncio.create_subprocess_exec(
            sys.executable,
            __file__,
            "--client",
            f"http://127.0.0.1:{port}",
            "--body-kb",
            str(args.body_kb),
            "--requests",
            str(args.requests + 1),
            stdout=subprocess.PIPE,
        )
        out, _ = await client.communicate()
        probing = False
        await task
    finally:
        await llm.shutdown(2.0)
        await runner.cleanup()
        service.close()
    times = json.loads(out)
    return {"round_trip_ms": times[1:], "rows": _per_request(overruns, windows[1:])}


def _per_request(
    overruns: list[tuple[float, float]], windows: list[tuple[float, float]]
) -> list[dict[str, float]]:
    """The worst and the total overrun of the probe sleeps that began while each request ran."""
    rows: list[dict[str, float]] = []
    for started, ended in windows:
        inside = [overrun for at, overrun in overruns if started <= at <= ended]
        rows.append(
            {
                "max_ms": max(inside, default=0.0) * 1e3,
                "total_ms": sum(inside) * 1e3,
                "server_ms": (ended - started) * 1e3,
            }
        )
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Event-loop delay caused by a large request.")
    parser.add_argument("--client", help=argparse.SUPPRESS)
    parser.add_argument("--tokenizer", default="unsloth/Llama-3.2-1B-Instruct")
    parser.add_argument("--body-kb", type=int, default=200)
    parser.add_argument("--requests", type=int, default=20)
    parser.add_argument("--gap", type=float, default=0.3)
    parser.add_argument("--label", default="")
    parser.add_argument("--revision", default="")
    parser.add_argument("--out")
    args = parser.parse_args(argv)
    if args.client:
        print(json.dumps(_client(args.client, args.body_kb, args.requests, args.gap)))
        return 0

    from _machine import write

    import ayaka

    measured = asyncio.run(_serve(args))
    rows = measured["rows"]
    write(
        {
            "schema": SCHEMA,
            "label": args.label,
            "revision": args.revision,
            "ayaka_path": str(Path(ayaka.__file__).resolve().parent),
            "body_bytes": len(_body(args.body_kb)),
            "requests": args.requests,
            "round_trip_ms": measured["round_trip_ms"],
            "round_trip_median_ms": statistics.median(measured["round_trip_ms"]),
            "per_request": rows,
            "max_overrun_median_ms": statistics.median(r["max_ms"] for r in rows),
            "total_overrun_median_ms": statistics.median(r["total_ms"] for r in rows),
        },
        args.out,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
