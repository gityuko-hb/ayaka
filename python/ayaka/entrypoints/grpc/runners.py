"""Standalone async entrypoints for ``ayaka engine-server`` and ``ayaka render-server``."""

from __future__ import annotations

import asyncio
import signal
from typing import TYPE_CHECKING, Any

from ayaka.entrypoints.grpc.inference import InferenceGrpcServer
from ayaka.entrypoints.grpc.render import RenderGrpcServer
from ayaka.tokenizers.factory import DefaultTokenizerFactory
from ayaka.tokenizers.service import TokenizerService

if TYPE_CHECKING:
    from collections.abc import Callable

    from ayaka.configs.engine import EngineConfig
    from ayaka.configs.model_source import ModelSourceConfig
    from ayaka.configs.server import ServerConfig
    from ayaka.configs.tokenizer import TokenizerConfig

__all__ = ["serve_engine_grpc", "serve_render_grpc"]


def _build_tokenizer_service(
    tok_config: TokenizerConfig, *, max_model_len: int
) -> TokenizerService:
    loaded = DefaultTokenizerFactory.from_config(tok_config)
    vocab_size = 1
    if loaded.tokenizer is not None:
        valid = loaded.tokenizer.valid_token_ids
        vocab_size = max(loaded.tokenizer.vocab_size, (max(valid) + 1) if valid else 1)
    return TokenizerService(
        tok_config,
        max_model_len=max(1, max_model_len),
        model_vocab_size=vocab_size,
        loaded=loaded,
    )


async def serve_engine_grpc(
    engine_config: EngineConfig,
    *,
    listen: str,
    say: Callable[[str], None] = print,
    stop: asyncio.Event | None = None,
) -> None:
    """Build the local engine and serve :class:`InferenceGrpcServer` until stopped."""
    from ayaka.engine.async_llm import AsyncLLM
    from ayaka.engine.build import build_engine

    stop_event = stop or asyncio.Event()
    loop = asyncio.get_running_loop()

    def _on_signal(_signum: int, _frame: Any) -> None:
        loop.call_soon_threadsafe(stop_event.set)

    prev_int = signal.signal(signal.SIGINT, _on_signal)
    prev_term = signal.signal(signal.SIGTERM, _on_signal)
    llm: AsyncLLM | None = None
    server: InferenceGrpcServer | None = None
    try:
        core = await asyncio.to_thread(build_engine, engine_config)
        llm = AsyncLLM(core)
        llm.start()
        server = InferenceGrpcServer(llm, target=listen)
        bound = await server.start()
        info = core.info
        where = f" ({info.device} {info.dtype})" if info.device and info.dtype else ""
        say(f"gRPC inference server ready on {bound}{where}")
        await stop_event.wait()
    finally:
        if server is not None:
            await server.shutdown(grace=5.0)
        if llm is not None:
            await llm.shutdown(0.0)
        signal.signal(signal.SIGINT, prev_int)
        signal.signal(signal.SIGTERM, prev_term)


async def serve_render_grpc(
    source: ModelSourceConfig,
    tok_config: TokenizerConfig,
    server_config: ServerConfig,
    *,
    model_name: str,
    listen: str,
    max_model_len: int = 32768,
    say: Callable[[str], None] = print,
    stop: asyncio.Event | None = None,
) -> None:
    """Serve engine-free :class:`RenderGrpcServer` without importing ``torch``."""
    _ = source
    stop_event = stop or asyncio.Event()
    loop = asyncio.get_running_loop()

    def _on_signal(_signum: int, _frame: Any) -> None:
        loop.call_soon_threadsafe(stop_event.set)

    prev_int = signal.signal(signal.SIGINT, _on_signal)
    prev_term = signal.signal(signal.SIGTERM, _on_signal)
    service = await asyncio.to_thread(
        _build_tokenizer_service, tok_config, max_model_len=max_model_len
    )
    server = RenderGrpcServer(
        service,
        model_name=model_name,
        reasoning_parser=server_config.reasoning_parser,
        tool_call_parser=server_config.tool_call_parser,
        tool_strict_level=server_config.tool_strict_level,
        target=listen,
    )
    try:
        bound = await server.start()
        say(f"gRPC render server ready on {bound} (fingerprint: {server.fingerprint})")
        await stop_event.wait()
    finally:
        await server.shutdown(grace=5.0)
        service.close()
        signal.signal(signal.SIGINT, prev_int)
        signal.signal(signal.SIGTERM, prev_term)
