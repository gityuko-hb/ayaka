"""Does a client that stops reading slow the other streams down? Measured on a running server.

Each round streams ``--fast`` chat completions at once and records the time between their text
frames, twice: alone, and next to a stuck client. The stuck client opens a streaming request with a
1 KiB receive buffer and then never reads. Rounds run in A B B A order (A alone, B with the stuck
client) and the result keeps every inter-token time, the medians and the ratio B/A of each pair.

A stuck client does not block the server at once: the kernel's send buffer (it grows to 4 MB on
Linux) and the server's own write buffer take about half a minute of output first. A B phase
therefore waits ``--settle`` seconds before its readers start, keeps the stuck client open until
the server cancels it, and reads the time of that cancellation from ``/metrics``. The server was
blocked from ``cancelled - --server-timeout`` on, so a phase counts only when that moment is
before its readers started and the cancellation after they ended; a run with any phase that
does not is invalid and exits 2.

    uv run python benchmarks/e2e/slow_client.py --server-timeout 10 --threshold 1.25 --out r.json

``--server-timeout`` must be the server's ``--slow-client-timeout``.
"""

from __future__ import annotations

import argparse
import itertools
import json
import re
import socket
import statistics
import sys
import threading
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

SCHEMA: Final = "ayaka.slow-client/1"


def stream_gaps(url: str, model: str, tokens: int, key: str | None) -> list[float]:
    """Seconds between the text frames of one streamed chat completion."""
    body = {
        "model": model,
        "messages": [{"role": "user", "content": "Write a long story about a lighthouse."}],
        "max_tokens": tokens,
        "temperature": 0,
        "ignore_eos": True,
        "stream": True,
    }
    headers = {"content-type": "application/json"}
    if key:
        headers["authorization"] = f"Bearer {key}"
    request = urllib.request.Request(  # noqa: S310  # the server under test
        url.rstrip("/") + "/v1/chat/completions", data=json.dumps(body).encode(), headers=headers
    )
    stamps: list[float] = []
    with urllib.request.urlopen(request, timeout=600) as response:  # noqa: S310
        for raw in response:
            if raw.startswith(b"data:") and b'"content"' in raw:
                stamps.append(time.perf_counter())
    return [b - a for a, b in itertools.pairwise(stamps)]


def slow_client_cancels(url: str, key: str | None) -> int:
    """How many requests the server has cancelled because their client read too slowly."""
    headers = {"authorization": f"Bearer {key}"} if key else {}
    request = urllib.request.Request(url.rstrip("/") + "/metrics", headers=headers)  # noqa: S310
    with urllib.request.urlopen(request, timeout=30) as response:  # noqa: S310
        text = response.read().decode("utf-8")
    match = re.search(r'^ayaka_requests_finished_total\{outcome="slow_client"\} (\S+)$', text, re.M)
    return int(float(match.group(1))) if match else 0


class StuckClient:
    """A streaming request whose reader never drains its socket."""

    def __init__(self, url: str, model: str, key: str | None) -> None:
        host, port = url.removeprefix("http://").split("/", 1)[0].split(":")
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1024)
        self.sock.connect((host, int(port)))
        body = json.dumps(
            {
                "model": model,
                "messages": [{"role": "user", "content": "Tell me everything you know."}],
                "max_tokens": 100_000,
                "temperature": 0,
                "ignore_eos": True,
                "stream": True,
            }
        ).encode()
        auth = f"authorization: Bearer {key}\r\n" if key else ""
        head = (
            f"POST /v1/chat/completions HTTP/1.1\r\nhost: {host}\r\n"
            f"content-type: application/json\r\ncontent-length: {len(body)}\r\n{auth}\r\n"
        )
        self.sock.sendall(head.encode() + body)

    def close(self) -> None:
        self.sock.close()


@dataclass
class Phase:
    """One phase: the readers' gaps and, beside a stuck client, when the server was blocked."""

    gaps: list[float]
    readers_started: float
    readers_ended: float
    cancelled_at: float | None = None  # seconds after the stuck client opened, from /metrics
    stuck_opened: float = 0.0


def phase(
    url: str,
    model: str,
    fast: int,
    tokens: int,
    key: str | None,
    *,
    stuck: bool,
    settle: float,
    server_timeout: float,
) -> Phase:
    """The inter-token times of ``fast`` concurrent streams, with or without a stuck client."""
    cancels = slow_client_cancels(url, key) if stuck else 0
    opened = time.monotonic()
    blocker = StuckClient(url, model, key) if stuck else None
    cancelled_at: float | None = None
    watch_stop = threading.Event()
    deadline = opened + settle + server_timeout + 60

    def _watch_cancel() -> None:
        nonlocal cancelled_at
        while not watch_stop.is_set() and time.monotonic() < deadline:
            if slow_client_cancels(url, key) > cancels:
                cancelled_at = time.monotonic() - opened
                return
            watch_stop.wait(0.2)

    watcher = threading.Thread(target=_watch_cancel, daemon=True) if blocker is not None else None
    if watcher is not None:
        watcher.start()
    if blocker is not None:
        time.sleep(settle)  # let the kernel's and the server's buffers fill
    gaps: list[list[float]] = [[] for _ in range(fast)]

    def run(index: int) -> None:
        gaps[index] = stream_gaps(url, model, tokens, key)

    threads = [threading.Thread(target=run, args=(i,)) for i in range(fast)]
    started = time.monotonic()
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    ended = time.monotonic()
    result = Phase([gap for each in gaps for gap in each], started - opened, ended - opened)
    if blocker is not None and watcher is not None:
        watcher.join(timeout=max(0.0, deadline - time.monotonic()))
        watch_stop.set()
        result.cancelled_at = cancelled_at
        blocker.close()
    return result


def compute_pair_ratios(
    rows: list[dict[str, Any]], pairs: int, *, kind: str = "stuck"
) -> list[float]:
    """Per-pair median gap ratio ``target / alone`` for ``kind`` (``'stuck'`` or ``'control'``)."""
    ratios: list[float] = []
    for pair in range(pairs):
        alone = [
            r["median_s"]
            for r in rows
            if r["pair"] == pair and r.get("kind", "stuck" if r["stuck"] else "alone") == "alone"
        ]
        other = [
            r["median_s"]
            for r in rows
            if r["pair"] == pair and r.get("kind", "stuck" if r["stuck"] else "alone") == kind
        ]
        ratios.append(statistics.median(other) / statistics.median(alone))
    return ratios


def build_result(
    *,
    rows: list[dict[str, Any]],
    pairs: int,
    threshold: float,
    settings: dict[str, Any],
    machine_info: dict[str, Any],
    power: tuple[object, object],
    control_reader: bool = False,
) -> dict[str, Any]:
    """Assemble the schema payload, pair ratios and validity verdict."""
    ratios = compute_pair_ratios(rows, pairs, kind="stuck")
    ratio = statistics.median(ratios)
    cancel_delays = [
        r["cancel_delay_seconds"]
        for r in rows
        if r.get("stuck") and r.get("cancel_delay_seconds") is not None
    ]
    valid = all(row["blocked_during_readers"] for row in rows if row["stuck"])
    result: dict[str, Any] = {
        "schema": SCHEMA,
        "machine": machine_info,
        "power": list(power),
        "settings": settings,
        "threshold": threshold,
        "rows": rows,
        "ratios": ratios,
        "median_ratio": ratio,
        "cancel_delays_s": cancel_delays,
        "median_cancel_delay_seconds": statistics.median(cancel_delays) if cancel_delays else None,
        # Without a blocked server the B phases measured nothing; the ratio then means nothing.
        "valid": valid,
        "passed": valid and ratio <= threshold,
    }
    if control_reader:
        control_ratios = compute_pair_ratios(rows, pairs, kind="control")
        result["control_ratios"] = control_ratios
        result["median_control_ratio"] = statistics.median(control_ratios)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Slow-client isolation of a running server.")
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", default="model")
    parser.add_argument("--api-key")
    parser.add_argument("--fast", type=int, default=2, help="concurrent readers that keep up")
    parser.add_argument("--tokens", type=int, default=128)
    parser.add_argument("--pairs", type=int, default=5, help="A B B A pairs")
    parser.add_argument(
        "--settle",
        type=float,
        default=40.0,
        help="seconds beside the stuck client before the readers",
    )
    parser.add_argument(
        "--server-timeout", type=float, required=True, help="the server's --slow-client-timeout"
    )
    parser.add_argument("--threshold", type=float, required=True, help="written before the run")
    parser.add_argument(
        "--control-reader",
        action="store_true",
        help="add a control phase with an extra concurrent reader that keeps up",
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    from _machine import machine

    from ayaka.entrypoints.bench_machine import power_source

    power_start = power_source()
    rows: list[dict[str, Any]] = []

    def _record_row(pair: int, kind: str, reader_count: int, *, stuck: bool) -> None:
        measured = phase(
            args.url,
            args.model,
            reader_count,
            args.tokens,
            args.api_key,
            stuck=stuck,
            settle=args.settle if stuck else 0.0,
            server_timeout=args.server_timeout,
        )
        row: dict[str, Any] = {
            "pair": pair,
            "kind": kind,
            "stuck": stuck,
            "gaps_s": measured.gaps,
            "median_s": statistics.median(measured.gaps),
            "readers_s": [measured.readers_started, measured.readers_ended],
        }
        if stuck:
            cancelled = measured.cancelled_at
            blocked_from = None if cancelled is None else cancelled - args.server_timeout
            row["cancelled_s"] = cancelled
            row["cancel_delay_seconds"] = cancelled
            row["blocked_from_s"] = blocked_from
            # The server must be blocked on the stuck client from before the readers start
            # until after they end: cancelled too early, the phase ends without it.
            row["blocked_during_readers"] = (
                cancelled is not None
                and blocked_from is not None
                and blocked_from <= measured.readers_started
                and cancelled >= measured.readers_ended
            )
        rows.append(row)
        note = (
            f", cancel delay {row['cancel_delay_seconds']}s, server blocked from "
            f"{row['blocked_from_s']}s, readers at {measured.readers_started:.1f}s"
            if stuck
            else ""
        )
        print(f"pair {pair} {kind}: median gap {row['median_s'] * 1e3:.1f} ms{note}")

    for pair in range(args.pairs):
        for stuck in (False, True, True, False):
            _record_row(pair, "stuck" if stuck else "alone", args.fast, stuck=stuck)
        if args.control_reader:
            _record_row(pair, "control", args.fast + 1, stuck=False)

    result = build_result(
        rows=rows,
        pairs=args.pairs,
        threshold=args.threshold,
        settings={
            "url": args.url,
            "fast": args.fast,
            "tokens": args.tokens,
            "pairs": args.pairs,
            "settle_s": args.settle,
            "server_timeout_s": args.server_timeout,
            "control_reader": args.control_reader,
        },
        machine_info=machine(),
        power=(power_start, power_source()),
        control_reader=args.control_reader,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    if not result["valid"]:
        print("INVALID: in at least one B phase the server was not blocked when the readers ran")
        return 2
    verdict = "within" if result["passed"] else "ABOVE"
    ratio = result["median_ratio"]
    print(f"median ratio stuck/alone {ratio:.3f}, {verdict} the threshold {args.threshold}")
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
