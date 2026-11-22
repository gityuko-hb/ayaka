"""Serving overhead harness: server CPU/frame, step gaps, concurrent ITL and prompt bursts.

Drives N concurrent SSE streams from raw sockets in a **separate OS process** so HTTP/JSON
parsing on the client side never shares the server's GIL. Supports both ``--mode fake``
(an in-process HTTP server backed by ``AsyncLLM`` over a zero-weight fake core) and
``--mode server`` (a live ``ayaka serve`` process, reading Linux ``/proc/<pid>/stat`` CPU
counters when ``--server-pid`` is given).

    uv run python benchmarks/e2e/serving_overhead.py --mode fake --streams 4 --tokens 64 \
        --rounds 5 --burst --out out.json
"""

from __future__ import annotations

import argparse
import asyncio
import itertools
import json
import os
import re
import socket
import statistics
import subprocess
import sys
import threading
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Final

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

if TYPE_CHECKING:
    from collections.abc import Sequence

    from ayaka.core.outputs import RequestOutput
    from ayaka.core.request import GenerateRequest
    from ayaka.core.types import RequestId
    from ayaka.engine.seam import EngineInfo
    from ayaka.tokenizers.ports import DecodeStreamLike

SCHEMA: Final = "ayaka.serving-overhead/1"

_STEP_BUCKET_RE = re.compile(
    r'^ayaka_engine_step_gap_seconds_bucket\{le="([^"]+)"\}\s+(\S+)$', re.M
)
_STEP_SUM_RE = re.compile(r"^ayaka_engine_step_gap_seconds_sum\s+(\S+)$", re.M)
_STEP_COUNT_RE = re.compile(r"^ayaka_engine_step_gap_seconds_count\s+(\S+)$", re.M)
_STEP_DUR_SUM_RE = re.compile(r"^ayaka_engine_step_seconds_sum\s+(\S+)$", re.M)
_STEP_DUR_COUNT_RE = re.compile(r"^ayaka_engine_step_seconds_count\s+(\S+)$", re.M)


# -- raw-socket client worker (runs in its own OS process) -------------------------------------


def _http_post_raw(
    host: str, port: int, path: str, body: bytes, key: str | None
) -> tuple[list[float], float]:
    """Send one HTTP/1.1 POST over a raw socket and return content-frame stamps and request end."""
    sock = socket.create_connection((host, port), timeout=120)
    try:
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        auth = f"authorization: Bearer {key}\r\n" if key else ""
        head = (
            f"POST {path} HTTP/1.1\r\n"
            f"host: {host}:{port}\r\n"
            f"content-type: application/json\r\n"
            f"content-length: {len(body)}\r\n"
            f"connection: close\r\n"
            f"{auth}\r\n"
        ).encode("ascii")
        sock.sendall(head + body)
        buf = bytearray()
        stamps: list[float] = []
        headers_done = False
        while True:
            chunk = sock.recv(65536)
            now = time.perf_counter()
            if not chunk:
                break
            buf.extend(chunk)
            if not headers_done:
                idx = buf.find(b"\r\n\r\n")
                if idx == -1:
                    continue
                status_line = bytes(buf[:idx]).split(b"\r\n", 1)[0]
                if b" 200 " not in status_line:
                    raise RuntimeError(
                        f"unexpected HTTP status from {path}: "
                        f"{status_line.decode('ascii', errors='replace')}"
                    )
                del buf[: idx + 4]
                headers_done = True
            while True:
                sep = buf.find(b"\n\n")
                if sep == -1:
                    break
                frame = bytes(buf[:sep])
                del buf[: sep + 2]
                if (
                    b'"content":"' in frame or b'"content": "' in frame
                ) and b'"role":' not in frame:
                    stamps.append(now)
                if b"[DONE]" in frame:
                    return stamps, now
        return stamps, time.perf_counter()
    finally:
        sock.close()


def _run_worker(job: dict[str, Any]) -> dict[str, Any]:
    """Open ``streams`` concurrent SSE connections (plus optional long-prompt burst) in one run."""
    host: str = job["host"]
    port: int = job["port"]
    model: str = job["model"]
    streams: int = job["streams"]
    tokens: int = job["tokens"]
    key: str | None = job.get("api_key")
    burst_requests: int = int(job.get("burst_requests", 0))
    burst_prompt_chars: int = int(job.get("burst_prompt_chars", 8000))

    stream_body = json.dumps(
        {
            "model": model,
            "messages": [{"role": "user", "content": "Write a detailed systems walkthrough."}],
            "max_tokens": tokens,
            "temperature": 0,
            "ignore_eos": True,
            "stream": True,
        }
    ).encode("utf-8")
    stream_results: list[tuple[list[float], float]] = [([], 0.0) for _ in range(streams)]
    barrier = threading.Barrier(streams + (1 if burst_requests > 0 else 0))

    def _stream_thread(i: int) -> None:
        barrier.wait()
        stream_results[i] = _http_post_raw(host, port, "/v1/chat/completions", stream_body, key)

    def _burst_thread() -> None:
        barrier.wait()
        time.sleep(0.02)
        long_text = " ".join(["context"] * max(1, burst_prompt_chars // 8))
        burst_body = json.dumps(
            {
                "model": model,
                "messages": [{"role": "user", "content": long_text}],
                "max_tokens": 1,
                "temperature": 0,
                "stream": False,
            }
        ).encode("utf-8")
        # Fire burst prompts concurrently to stress admission & tokenization during decode.
        inner = [
            threading.Thread(
                target=_http_post_raw,
                args=(host, port, "/v1/chat/completions", burst_body, key),
            )
            for _ in range(burst_requests)
        ]
        for t in inner:
            t.start()
        for t in inner:
            t.join()

    threads = [threading.Thread(target=_stream_thread, args=(i,)) for i in range(streams)]
    if burst_requests > 0:
        threads.append(threading.Thread(target=_burst_thread))
    t0 = time.perf_counter()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    wall_s = time.perf_counter() - t0

    itls_s: list[float] = []
    frames = 0
    for stamps, _ in stream_results:
        frames += len(stamps)
        itls_s.extend(b - a for a, b in itertools.pairwise(stamps))
    return {"wall_s": wall_s, "frames": frames, "itls_s": itls_s}


def _invoke_worker(job: dict[str, Any]) -> dict[str, Any]:
    proc = subprocess.run(  # noqa: S603  # spawns this script's own --worker entrypoint
        [sys.executable, str(Path(__file__).resolve()), "--worker"],
        input=json.dumps(job),
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(proc.stdout)


# -- server metrics & CPU sampling -------------------------------------------------------------


def _fetch_metrics(url: str, key: str | None) -> str:
    headers = {"authorization": f"Bearer {key}"} if key else {}
    req = urllib.request.Request(url.rstrip("/") + "/metrics", headers=headers)  # noqa: S310
    with urllib.request.urlopen(req, timeout=30) as resp:  # noqa: S310
        return resp.read().decode("utf-8")


@dataclass(frozen=True, slots=True)
class _StepGapSnapshot:
    count: float
    total_s: float
    buckets: tuple[tuple[float, float], ...]
    step_count: float = 0.0
    step_total_s: float = 0.0


def _parse_step_gap(text: str) -> _StepGapSnapshot:
    buckets: list[tuple[float, float]] = []
    for bound, val in _STEP_BUCKET_RE.findall(text):
        upper = float("inf") if bound == "+Inf" else float(bound)
        buckets.append((upper, float(val)))
    count_m = _STEP_COUNT_RE.search(text)
    sum_m = _STEP_SUM_RE.search(text)
    step_c_m = _STEP_DUR_COUNT_RE.search(text)
    step_s_m = _STEP_DUR_SUM_RE.search(text)
    return _StepGapSnapshot(
        count=float(count_m.group(1)) if count_m else 0.0,
        total_s=float(sum_m.group(1)) if sum_m else 0.0,
        buckets=tuple(buckets),
        step_count=float(step_c_m.group(1)) if step_c_m else 0.0,
        step_total_s=float(step_s_m.group(1)) if step_s_m else 0.0,
    )


def _histogram_quantile(
    before: _StepGapSnapshot, after: _StepGapSnapshot, quantile: float
) -> float | None:
    """Linear-interpolated quantile over a Prometheus histogram bucket diff."""
    total = after.count - before.count
    if total <= 0 or len(before.buckets) != len(after.buckets):
        return None
    target = quantile * total
    prev_bound = 0.0
    prev_cum = 0.0
    for (bound, a_val), (_, b_val) in zip(after.buckets, before.buckets, strict=True):
        cum = a_val - b_val
        if cum >= target:
            if bound == float("inf"):
                return prev_bound
            span = max(1e-9, cum - prev_cum)
            frac = max(0.0, min(1.0, (target - prev_cum) / span))
            return prev_bound + frac * (bound - prev_bound)
        prev_bound = bound
        prev_cum = cum
    return prev_bound


def _read_server_cpu_seconds(server_pid: int | None) -> float | None:
    """Read server process user+kernel CPU seconds; supports current process or Linux ``/proc``."""
    if server_pid is None or server_pid == os.getpid():
        return time.process_time()
    stat_path = Path(f"/proc/{server_pid}/stat")
    if not stat_path.exists():
        return None
    fields = stat_path.read_text(encoding="utf-8").rsplit(")", 1)[1].split()
    utime = int(fields[11])
    stime = int(fields[12])
    sysconf = getattr(os, "sysconf", None)
    ticks = int(sysconf("SC_CLK_TCK")) if sysconf is not None else 100
    return (utime + stime) / ticks


def _percentile(values: Sequence[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, round(q * (len(ordered) - 1))))
    return ordered[idx]


# -- self-contained fake engine core for --mode fake -------------------------------------------


class _FakeByteTokenizer:
    """Minimal byte-level tokenizer so ``--mode fake`` needs no checkpoints or torch."""

    vocab_size = 260
    max_token_id = 259
    valid_token_ids: frozenset[int] = frozenset(range(260))
    bos_token_id: int | None = None
    eos_token_ids: frozenset[int] = frozenset({257})
    all_special_tokens: tuple[str, ...] = ("<s>", "</s>")
    special_tokens_map: ClassVar[dict[str, str]] = {"bos_token": "<s>", "eos_token": "</s>"}

    def fingerprint(self) -> str:
        return "fake-byte-v1"

    def encode(self, text: str, *, add_special_tokens: bool = True) -> list[int]:
        del add_special_tokens
        return list(text.encode("utf-8"))

    def encode_batch(
        self, texts: Sequence[str], *, add_special_tokens: bool = True
    ) -> list[list[int]]:
        return [self.encode(t, add_special_tokens=add_special_tokens) for t in texts]

    def decode(
        self,
        ids: Sequence[int] | int,
        *,
        skip_special_tokens: bool = False,
        spaces_between_special_tokens: bool = True,
    ) -> str:
        del skip_special_tokens, spaces_between_special_tokens
        seq = [ids] if isinstance(ids, int) else list(ids)
        return bytes(i & 0xFF for i in seq).decode("utf-8", errors="replace")

    def apply_chat_template(
        self,
        messages: Sequence[dict[str, Any]],
        *,
        tools: Sequence[dict[str, Any]] | None = None,
        add_generation_prompt: bool = True,
        chat_template: str | None = None,
    ) -> str:
        del tools, chat_template
        joined = "\n".join(f"{m.get('role')}:{m.get('content', '')}" for m in messages)
        return joined + ("\nassistant:" if add_generation_prompt else "")

    def new_decode_stream(self, *, skip_special_tokens: bool = False) -> DecodeStreamLike | None:
        del skip_special_tokens
        return None

    def get_vocab(self) -> dict[str, int]:
        return {chr(i): i for i in range(256)}

    def get_decoded_vocab(self) -> tuple[bytes, ...]:
        return tuple(bytes([i]) for i in range(260))

    def token_bytes_table(self) -> tuple[bytes, ...]:
        return tuple(bytes([i]) for i in range(260))


@dataclass
class _FakeRequest:
    request_id: RequestId
    remaining: int
    prompt_len: int


class _FakeCore:
    """Zero-model engine core exercised via real ``AsyncLLM`` so step-gap histograms populate."""

    def __init__(self, step_sleep_s: float = 0.0) -> None:
        from ayaka.engine.seam import EngineInfo

        self._info = EngineInfo(
            max_model_len=131072, max_context_tokens=131072, prefix_caching=False
        )
        self._active: list[_FakeRequest] = []
        self._step_sleep_s = step_sleep_s

    @property
    def info(self) -> EngineInfo:
        return self._info

    @property
    def has_unfinished(self) -> bool:
        return bool(self._active)

    def prepare(self, request: GenerateRequest) -> GenerateRequest:
        return request

    def admit(self, request: GenerateRequest, *, stream: bool = False) -> None:
        del stream
        limit = request.sampling.max_tokens if request.sampling.max_tokens is not None else 16
        prompt_len = len(request.prompt) if isinstance(request.prompt, tuple) else 8
        self._active.append(_FakeRequest(request.request_id, max(1, limit), prompt_len))

    def add_request(self, request: GenerateRequest, *, stream: bool = False) -> None:
        self.admit(request, stream=stream)

    def abort(self, request_id: RequestId) -> bool:
        before = len(self._active)
        self._active = [req for req in self._active if req.request_id != request_id]
        return len(self._active) != before

    def step(self) -> list[RequestOutput]:
        from ayaka.core.outputs import CompletionOutput, RequestOutput, Usage
        from ayaka.core.types import FinishReason

        if self._step_sleep_s > 0:
            time.sleep(self._step_sleep_s)
        outputs: list[RequestOutput] = []
        kept: list[_FakeRequest] = []
        for req in self._active:
            req.remaining -= 1
            done = req.remaining <= 0
            outputs.append(
                RequestOutput(
                    request_id=req.request_id,
                    outputs=(
                        CompletionOutput(
                            index=0,
                            token_ids=(97,),
                            text="a",
                            finish_reason=FinishReason.LENGTH if done else None,
                        ),
                    ),
                    finished=done,
                    usage=Usage(prompt_tokens=req.prompt_len, completion_tokens=1),
                )
            )
            if not done:
                kept.append(req)
        self._active = kept
        return outputs

    def shutdown(self) -> None:
        self._active.clear()


# -- round orchestration & summary -------------------------------------------------------------


def _measure_round(
    *,
    host: str,
    port: int,
    base_url: str,
    model: str,
    api_key: str | None,
    server_pid: int | None,
    streams: int,
    tokens: int,
    burst_requests: int,
    burst_prompt_chars: int,
) -> dict[str, Any]:
    snap_before = _parse_step_gap(_fetch_metrics(base_url, api_key))
    cpu_before = _read_server_cpu_seconds(server_pid)
    worker = _invoke_worker(
        {
            "host": host,
            "port": port,
            "model": model,
            "streams": streams,
            "tokens": tokens,
            "api_key": api_key,
            "burst_requests": burst_requests,
            "burst_prompt_chars": burst_prompt_chars,
        }
    )
    cpu_after = _read_server_cpu_seconds(server_pid)
    snap_after = _parse_step_gap(_fetch_metrics(base_url, api_key))

    frames = int(worker["frames"])
    wall_s = float(worker["wall_s"])
    itls_ms = [v * 1e3 for v in worker["itls_s"]]
    cpu_s = (
        max(0.0, cpu_after - cpu_before)
        if cpu_before is not None and cpu_after is not None
        else None
    )
    cpu_per_frame_us = (cpu_s * 1e6 / frames) if (cpu_s is not None and frames > 0) else None
    fps = frames / wall_s if wall_s > 0 else 0.0

    gap_count = round(snap_after.count - snap_before.count)
    gap_sum_s = max(0.0, snap_after.total_s - snap_before.total_s)
    gap_mean_ms = (gap_sum_s * 1e3 / gap_count) if gap_count > 0 else None
    gap_p50_s = _histogram_quantile(snap_before, snap_after, 0.50)
    gap_p99_s = _histogram_quantile(snap_before, snap_after, 0.99)
    step_count = round(snap_after.step_count - snap_before.step_count)
    step_sum_s = max(0.0, snap_after.step_total_s - snap_before.step_total_s)
    engine_step_mean_ms = (step_sum_s * 1e3 / step_count) if step_count > 0 else None

    return {
        "frames": frames,
        "wall_s": wall_s,
        "frames_per_sec": fps,
        "server_cpu_s": cpu_s,
        "server_cpu_per_frame_us": cpu_per_frame_us,
        "engine_step_count": step_count,
        "engine_step_mean_ms": engine_step_mean_ms,
        "step_gap_ms": {
            "count": gap_count,
            "mean": gap_mean_ms,
            "p50": gap_p50_s * 1e3 if gap_p50_s is not None else None,
            "p99": gap_p99_s * 1e3 if gap_p99_s is not None else None,
        },
        "client_itl_ms": {
            "count": len(itls_ms),
            "p50": _percentile(itls_ms, 0.50),
            "p99": _percentile(itls_ms, 0.99),
        },
    }


def _summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    def _med(values: list[float | None]) -> float | None:
        clean = [v for v in values if v is not None]
        return statistics.median(clean) if clean else None

    return {
        "rounds": len(rows),
        "frames_per_sec_median": _med([r["frames_per_sec"] for r in rows]),
        "server_cpu_per_frame_us_median": _med([r["server_cpu_per_frame_us"] for r in rows]),
        "engine_step_mean_ms_median": _med([r["engine_step_mean_ms"] for r in rows]),
        "step_gap_ms": {
            "mean_median": _med([r["step_gap_ms"]["mean"] for r in rows]),
            "p50_median": _med([r["step_gap_ms"]["p50"] for r in rows]),
            "p99_median": _med([r["step_gap_ms"]["p99"] for r in rows]),
        },
        "client_itl_ms": {
            "p50_median": _med([r["client_itl_ms"]["p50"] for r in rows]),
            "p99_median": _med([r["client_itl_ms"]["p99"] for r in rows]),
        },
    }


async def _run_fake_server(args: argparse.Namespace) -> dict[str, Any]:
    from aiohttp import web

    from ayaka.configs.server import ServerConfig
    from ayaka.configs.tokenizer import TokenizerConfig
    from ayaka.engine.async_llm import AsyncLLM
    from ayaka.entrypoints.http import create_app
    from ayaka.entrypoints.serving import Serving
    from ayaka.tokenizers.contracts import LoadedTokenizer, TokenizerCapabilities
    from ayaka.tokenizers.service import TokenizerService

    loaded = LoadedTokenizer(
        _FakeByteTokenizer(),  # type: ignore[arg-type]
        TokenizerCapabilities(
            "hf",
            text_encode=True,
            text_decode=True,
            chat=True,
            decode_stream=False,
            byte_decode=False,
        ),
    )
    service = TokenizerService(
        TokenizerConfig("fake", mode="hf"),
        max_model_len=131072,
        model_vocab_size=260,
        loaded=loaded,
    )
    serving = Serving(ServerConfig(host="127.0.0.1", port=0), model_name=args.model)
    core = _FakeCore(step_sleep_s=args.step_ms / 1e3)
    llm = AsyncLLM(core, registry=serving.metrics.registry)  # type: ignore[arg-type]
    llm.start()
    runner = web.AppRunner(create_app(serving), handler_cancellation=True, access_log=None)
    await runner.setup()
    try:
        site = web.TCPSite(runner, "127.0.0.1", 0)
        await site.start()
        port = int(runner.addresses[0][1])
        serving.attach(llm, service)  # type: ignore[arg-type]
        base_url = f"http://127.0.0.1:{port}"
        return await asyncio.to_thread(
            _execute_rounds,
            host="127.0.0.1",
            port=port,
            base_url=base_url,
            server_pid=os.getpid(),
            args=args,
        )
    finally:
        await llm.shutdown(2.0)
        await runner.cleanup()
        service.close()


def _execute_rounds(
    *, host: str, port: int, base_url: str, server_pid: int | None, args: argparse.Namespace
) -> dict[str, Any]:
    from _machine import machine

    # One warmup call before timing.
    _invoke_worker(
        {
            "host": host,
            "port": port,
            "model": args.model,
            "streams": 1,
            "tokens": min(4, args.tokens),
            "api_key": args.api_key,
            "burst_requests": 0,
            "burst_prompt_chars": args.burst_prompt_chars,
        }
    )

    steady_rows: list[dict[str, Any]] = []
    burst_rows: list[dict[str, Any]] = []
    for r in range(args.rounds):
        steady = _measure_round(
            host=host,
            port=port,
            base_url=base_url,
            model=args.model,
            api_key=args.api_key,
            server_pid=server_pid,
            streams=args.streams,
            tokens=args.tokens,
            burst_requests=0,
            burst_prompt_chars=args.burst_prompt_chars,
        )
        steady["round"] = r
        steady_rows.append(steady)
        if args.burst:
            burst = _measure_round(
                host=host,
                port=port,
                base_url=base_url,
                model=args.model,
                api_key=args.api_key,
                server_pid=server_pid,
                streams=args.streams,
                tokens=args.tokens,
                burst_requests=args.burst_requests,
                burst_prompt_chars=args.burst_prompt_chars,
            )
            burst["round"] = r
            burst_rows.append(burst)

    payload: dict[str, Any] = {
        "schema": SCHEMA,
        "mode": args.mode,
        "machine": machine(),
        "settings": {
            "model": args.model,
            "streams": args.streams,
            "tokens": args.tokens,
            "rounds": args.rounds,
            "step_ms": args.step_ms if args.mode == "fake" else None,
            "burst": args.burst,
            "burst_requests": args.burst_requests if args.burst else 0,
            "burst_prompt_chars": args.burst_prompt_chars if args.burst else 0,
        },
        "rounds": steady_rows,
        "steady_summary": _summarize(steady_rows),
    }
    if args.burst:
        payload["burst_rounds"] = burst_rows
        payload["burst_summary"] = _summarize(burst_rows)
    return payload


def _parse_server_host_port(url: str) -> tuple[str, int]:
    stripped = url.removeprefix("http://").split("/", 1)[0]
    host, port_text = stripped.rsplit(":", 1)
    return host.strip("[]"), int(port_text)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Measure concurrent SSE serving overhead.")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--mode", choices=("fake", "server"), default="fake")
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", default="local")
    parser.add_argument("--api-key")
    parser.add_argument("--server-pid", type=int, default=None)
    parser.add_argument("--streams", type=int, default=4)
    parser.add_argument("--tokens", type=int, default=64)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--step-ms", type=float, default=0.5)
    parser.add_argument("--burst", action="store_true")
    parser.add_argument("--burst-requests", type=int, default=20)
    parser.add_argument("--burst-prompt-chars", type=int, default=8000)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)

    if args.worker:
        job = json.loads(sys.stdin.read())
        sys.stdout.write(json.dumps(_run_worker(job)) + "\n")
        return 0

    if args.mode == "fake":
        result = asyncio.run(_run_fake_server(args))
    else:
        host, port = _parse_server_host_port(args.url)
        result = _execute_rounds(
            host=host,
            port=port,
            base_url=args.url.rstrip("/"),
            server_pid=args.server_pid,
            args=args,
        )

    steady = result["steady_summary"]
    print(
        f"steady ({args.mode}): fps={steady['frames_per_sec_median']:.1f}, "
        f"cpu/frame={steady['server_cpu_per_frame_us_median']} us, "
        f"step_gap_p99={steady['step_gap_ms']['p99_median']} ms, "
        f"itl_p99={steady['client_itl_ms']['p99_median']} ms"
    )
    if args.burst:
        burst = result["burst_summary"]
        print(
            f"burst  ({args.mode}): fps={burst['frames_per_sec_median']:.1f}, "
            f"step_gap_p99={burst['step_gap_ms']['p99_median']} ms, "
            f"itl_p99={burst['client_itl_ms']['p99_median']} ms"
        )
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
