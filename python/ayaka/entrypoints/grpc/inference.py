"""gRPC server exposing an :class:`~ayaka.engine.seam.EngineClient` over TCP or Unix sockets."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from typing import TYPE_CHECKING, Any, Final

from ayaka.core.errors import AyakaError, CapabilityError, UnsupportedFeatureError
from ayaka.core.types import RequestId
from ayaka.entrypoints.grpc.errors import abort_with_error
from ayaka.entrypoints.grpc.messages import (
    MAX_GRPC_MESSAGE_BYTES,
    decode_generate_request,
    deserialize_proto,
    encode_engine_health,
    encode_engine_info,
    encode_request_output,
    grpc_options,
    serialize_proto,
)
from ayaka.utils.import_utils import require_module

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from ayaka.engine.seam import EngineClient

__all__ = ["INFERENCE_SERVICE_NAME", "InferenceGrpcServer"]

INFERENCE_SERVICE_NAME: Final = "ayaka.v1.InferenceService"
_LOG = logging.getLogger("ayaka.grpc.inference")


def _grpc() -> Any:
    try:
        return require_module(
            "grpc",
            capability="grpc",
            remedy="install the grpc extra: uv sync --extra grpc",
        )
    except CapabilityError as exc:
        raise UnsupportedFeatureError(
            "gRPC server needs the grpc extra (grpcio), which is not installed",
            context={"features": ("grpc",)},
        ) from exc


class InferenceGrpcServer:
    """Async gRPC server bridging :data:`INFERENCE_SERVICE_NAME` to an :class:`EngineClient`."""

    def __init__(
        self,
        engine: EngineClient,
        *,
        target: str = "127.0.0.1:0",
        max_message_bytes: int = MAX_GRPC_MESSAGE_BYTES,
    ) -> None:
        self._engine = engine
        self._target = target
        self._max_bytes = max_message_bytes
        self._server: Any = None
        self._bound_target = target
        self._port = 0

    @property
    def target(self) -> str:
        """Bound gRPC address (resolves port ``0`` once :meth:`start` runs)."""
        return self._bound_target

    @property
    def port(self) -> int:
        """Bound TCP port, or ``0`` for Unix Domain Sockets / before :meth:`start`."""
        return self._port

    async def start(self) -> str:
        """Bind and start the gRPC server; return the reachable target address."""
        grpc = _grpc()
        _ = require_module("grpc.aio", capability="grpc", remedy="uv sync --extra grpc")
        server = grpc.aio.server(options=grpc_options(self._max_bytes))
        handlers = {
            "Generate": grpc.unary_stream_rpc_method_handler(
                self._generate,
                request_deserializer=deserialize_proto,
                response_serializer=serialize_proto,
            ),
            "Abort": grpc.unary_unary_rpc_method_handler(
                self._abort,
                request_deserializer=deserialize_proto,
                response_serializer=serialize_proto,
            ),
            "AbortAll": grpc.unary_unary_rpc_method_handler(
                self._abort_all,
                request_deserializer=deserialize_proto,
                response_serializer=serialize_proto,
            ),
            "Health": grpc.unary_unary_rpc_method_handler(
                self._health,
                request_deserializer=deserialize_proto,
                response_serializer=serialize_proto,
            ),
            "GetEngineInfo": grpc.unary_unary_rpc_method_handler(
                self._get_info,
                request_deserializer=deserialize_proto,
                response_serializer=serialize_proto,
            ),
            "WatchEngineState": grpc.unary_stream_rpc_method_handler(
                self._watch_state,
                request_deserializer=deserialize_proto,
                response_serializer=serialize_proto,
            ),
        }
        server.add_generic_rpc_handlers(
            (grpc.method_handlers_generic_handler(INFERENCE_SERVICE_NAME, handlers),)
        )
        bound = int(server.add_insecure_port(self._target))
        if bound == 0 and not self._target.startswith("unix:"):
            raise RuntimeError(f"failed to bind gRPC inference server to {self._target!r}")
        await server.start()
        self._server = server
        if self._target.startswith("unix:"):
            self._port = 0
            self._bound_target = self._target
        else:
            host = self._target.rsplit(":", 1)[0]
            self._port = bound
            self._bound_target = f"{host}:{bound}"
        _LOG.info("gRPC inference server listening on %s", self._bound_target)
        return self._bound_target

    async def shutdown(self, grace: float = 0.0) -> None:
        """Stop accepting new RPCs and drain in-flight calls up to ``grace`` seconds."""
        if self._server is not None:
            await self._server.stop(grace=grace)
            self._server = None

    async def _generate(
        self, payload: dict[str, object], context: Any
    ) -> AsyncIterator[dict[str, object]]:
        try:
            request = decode_generate_request(payload)
            stream = await self._engine.add(request)
        except Exception as exc:
            if not isinstance(exc, AyakaError):
                _LOG.exception("unexpected error admitting gRPC request")
            await abort_with_error(context, exc)
            return

        aborted_by_disconnect = False

        def _on_done(_: Any) -> None:
            nonlocal aborted_by_disconnect
            if not stream.finished and context.cancelled():
                aborted_by_disconnect = True
                self._engine.abort(request.request_id)

        context.add_done_callback(_on_done)
        await context.send_initial_metadata((("x-ayaka-admitted", "1"),))
        try:
            async for output in stream:
                yield encode_request_output(output)
        except asyncio.CancelledError:
            if not stream.finished and not aborted_by_disconnect:
                self._engine.abort(request.request_id)
            raise
        finally:
            if not stream.finished and context.cancelled() and not aborted_by_disconnect:
                self._engine.abort(request.request_id)

    async def _abort(self, payload: dict[str, object], context: Any) -> dict[str, object]:
        _ = context
        raw_id = payload.get("request_id")
        if isinstance(raw_id, str) and raw_id:
            self._engine.abort(RequestId(raw_id))
        return {}

    async def _abort_all(self, payload: dict[str, object], context: Any) -> dict[str, object]:
        _ = payload, context
        self._engine.abort_all()
        return {}

    async def _health(self, payload: dict[str, object], context: Any) -> dict[str, object]:
        _ = payload, context
        return encode_engine_health(self._engine.health())

    async def _get_info(self, payload: dict[str, object], context: Any) -> dict[str, object]:
        _ = payload, context
        return encode_engine_info(self._engine.info)

    async def _watch_state(
        self, payload: dict[str, object], context: Any
    ) -> AsyncIterator[dict[str, object]]:
        interval_raw = payload.get("poll_interval_seconds", 0.25)
        interval = float(interval_raw) if isinstance(interval_raw, (int, float)) else 0.25
        interval = max(0.01, min(interval, 5.0))
        last: tuple[object, object] | None = None
        while not context.cancelled():
            health = self._engine.health()
            info = self._engine.info
            snap_key = (health, info)
            if snap_key != last:
                last = snap_key
                yield {
                    "health": encode_engine_health(health),
                    "info": encode_engine_info(info),
                    "observed_at": time.monotonic(),
                }
            with contextlib.suppress(TimeoutError):
                await asyncio.sleep(interval)
