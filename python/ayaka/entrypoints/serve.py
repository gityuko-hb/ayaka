"""The life of ``ayaka serve``: listen, load, warm up, serve, drain, stop.

::

    listening  ->  loading the model (503 "starting")  ->  one warm-up token  ->  ready (200)
    first SIGINT/SIGTERM  ->  draining: new requests get 503; requests in flight get
                              ``shutdown_timeout`` seconds, then end CANCELLED at a step boundary
    second signal         ->  the process exits at once

The server listens before the model is loaded, so a health check can tell "starting" from "down".
Nothing here is imported by ``import ayaka`` or by ``ayaka run``: it loads aiohttp.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import signal
from typing import TYPE_CHECKING, Final

from aiohttp import web

from ayaka.core.errors import EngineError
from ayaka.core.params import SamplingParams
from ayaka.core.request import GenerateRequest
from ayaka.core.types import FinishReason, RequestId
from ayaka.engine.async_llm import AsyncLLM
from ayaka.engine.recorder import SessionRecorder
from ayaka.entrypoints.http import create_app
from ayaka.entrypoints.parsers.output import OutputFormat
from ayaka.entrypoints.serving import Serving
from ayaka.tokenizers.contracts import EncodeOptions, TextInput

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path
    from types import FrameType

    from prometheus_client import CollectorRegistry

    from ayaka.configs.engine import EngineConfig
    from ayaka.configs.server import ServerConfig
    from ayaka.engine.core import EngineCore
    from ayaka.tokenizers.service import TokenizerService

__all__ = ["EXIT_SECOND_SIGNAL", "serve"]

#: The exit code when a second signal ends the process without waiting.
EXIT_SECOND_SIGNAL: Final = 130


class _Signals:
    """SIGINT and SIGTERM: the first sets ``stop``, the second exits the process."""

    def __init__(self, loop: asyncio.AbstractEventLoop, stop: asyncio.Event) -> None:
        self._loop = loop
        self._stop = stop
        self._count = 0
        self._previous: dict[int, object] = {}
        for number in (signal.SIGINT, signal.SIGTERM):
            self._previous[number] = signal.signal(number, self._handle)

    def _handle(self, number: int, frame: FrameType | None) -> None:
        del number, frame
        self._count += 1
        if self._count > 1:
            os._exit(EXIT_SECOND_SIGNAL)
        self._loop.call_soon_threadsafe(self._stop.set)

    def restore(self) -> None:
        for number, previous in self._previous.items():
            signal.signal(number, previous)  # type: ignore[arg-type]  # what signal.signal returned


async def _warm_up(llm: AsyncLLM, service: TokenizerService) -> None:
    """Generate one token, so the first client does not pay for the first step."""
    encoded = await asyncio.to_thread(
        service.encode, TextInput("Hello"), EncodeOptions(max_output_tokens=1)
    )
    request = GenerateRequest(
        request_id=RequestId("ayaka-warmup"),
        prompt=encoded.token_ids,
        sampling=SamplingParams(temperature=0.0, max_tokens=1),
    )
    last = None
    async for output in llm.generate(request):
        last = output
    reason = last.outputs[0].finish_reason if last is not None and last.outputs else None
    if reason not in (FinishReason.STOP, FinishReason.LENGTH):
        detail = last.error.message if last is not None and last.error else str(reason)
        raise EngineError(f"the warm-up request failed: {detail}")


async def serve(
    config: ServerConfig,
    engine: EngineConfig | Callable[[SessionRecorder | None], EngineCore],
    *,
    model_name: str,
    say: Callable[[str], None] = print,
    stop: asyncio.Event | None = None,
    handle_signals: bool = True,
    on_ready: Callable[[str, int], None] | None = None,
    record: Path | None = None,
) -> None:
    """Run the server until stopped, then drain and shut down.

    Args:
        config: The server settings.
        engine: What to load, or a function that builds the engine from the recorder it must
            record to (tests pass one).
        model_name: The name the model was given on the command line.
        say: Where the status lines go.
        stop: Set it to stop the server; a fresh event when ``None``.
        handle_signals: Let SIGINT and SIGTERM stop the server (main thread only).
        on_ready: Called with the bound host and port once the server accepts requests.
        record: A directory for the session recorder (every request as token ids), or ``None``.

    Raises:
        ModelLoadError, ConfigError, CapabilityError, CapacityError: If the engine cannot be
            built; the server stops listening first.
        EngineError: If the warm-up request fails.
    """
    loop = asyncio.get_running_loop()
    stop = stop if stop is not None else asyncio.Event()
    serving = Serving(config, model_name=model_name)
    runner = web.AppRunner(create_app(serving), handler_cancellation=True, access_log=None)
    await runner.setup()
    signals = _Signals(loop, stop) if handle_signals else None
    llm: AsyncLLM | None = None
    recorder: SessionRecorder | None = None
    try:
        site = web.TCPSite(runner, config.host, config.port)
        await site.start()
        host, port = runner.addresses[0][:2]
        say(f"listening on http://{_format_host(host)}:{port}; loading {model_name}")
        if record is not None:
            recorder = SessionRecorder(record)
            recorder.recording = False  # the warm-up is not a request of the session
        build = engine if callable(engine) else _builder(engine, serving.metrics.registry)
        core = await asyncio.to_thread(build, recorder)
        llm = AsyncLLM(core, registry=serving.metrics.registry)
        llm.start()
        output_format = OutputFormat.resolve(
            reasoning=config.reasoning_parser,
            tool_calls=config.tool_call_parser,
            tokenizer=core.tokenizer.tokenizer,
        )
        await _warm_up(llm, core.tokenizer)
        if stop.is_set():
            return
        if recorder is not None:
            recorder.recording = True
            say(
                f"recording every request to {recorder.path}: prompts and answers as token "
                "ids, readable by anyone who has the model's tokenizer"
            )
        serving.attach(llm, core.tokenizer, output_format=output_format)
        say(f"ready: {', '.join(serving.model_names)} on http://{_format_host(host)}:{port}")
        if on_ready is not None:
            on_ready(host, port)
        await stop.wait()
        say(
            "stopping: new requests get 503; requests in flight have "
            f"{config.shutdown_timeout:g} s to finish (a second signal exits at once)"
        )
        serving.begin_draining()
        await llm.shutdown(config.shutdown_timeout)
    finally:
        if llm is not None:
            with contextlib.suppress(Exception):
                await llm.shutdown(0.0)
        if recorder is not None:
            recorder.close()  # the engine closed it already, unless it was never built
        await runner.cleanup()
        if signals is not None:
            signals.restore()
    say("stopped")


def _builder(
    config: EngineConfig, registry: CollectorRegistry
) -> Callable[[SessionRecorder | None], EngineCore]:
    def build(recorder: SessionRecorder | None) -> EngineCore:
        from ayaka.engine.build import build_engine
        from ayaka.metrics.loggers import PrometheusStatLogger

        return build_engine(
            config, stat_loggers=(PrometheusStatLogger(registry=registry),), recorder=recorder
        )

    return build


def _format_host(host: str) -> str:
    return f"[{host}]" if ":" in host else host
