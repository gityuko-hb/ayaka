"""Remote :class:`~ayaka.engine.seam.EngineClient` backed by :class:`InferenceGrpcServer`."""

from __future__ import annotations

import asyncio
import contextlib
import time
from typing import TYPE_CHECKING, Any, cast

from ayaka.core.errors import CapabilityError, EngineDeadError, EngineError, UnsupportedFeatureError
from ayaka.engine.async_llm import RequestStream
from ayaka.engine.seam import EngineHealth, EngineInfo, EngineState
from ayaka.entrypoints.grpc.errors import error_from_rpc
from ayaka.entrypoints.grpc.inference import INFERENCE_SERVICE_NAME
from ayaka.entrypoints.grpc.messages import (
    MAX_GRPC_MESSAGE_BYTES,
    decode_engine_health,
    decode_engine_info,
    decode_request_output,
    deserialize_proto,
    encode_generate_request,
    grpc_options,
    serialize_proto,
)
from ayaka.utils.import_utils import require_module

if TYPE_CHECKING:
    from ayaka.core.request import GenerateRequest
    from ayaka.core.types import RequestId

__all__ = ["GrpcEngineClient"]


def _grpc() -> Any:
    try:
        return require_module(
            "grpc",
            capability="grpc",
            remedy="install the grpc extra: uv sync --extra grpc",
        )
    except CapabilityError as exc:
        raise UnsupportedFeatureError(
            "remote engine client needs the grpc extra (grpcio), which is not installed",
            context={"features": ("grpc",)},
        ) from exc


class GrpcEngineClient:
    """Implements :class:`~ayaka.engine.seam.EngineClient` over gRPC with local health tracking."""

    def __init__(
        self,
        target: str,
        *,
        max_message_bytes: int = MAX_GRPC_MESSAGE_BYTES,
        watch_interval_seconds: float = 0.2,
    ) -> None:
        self._target = target
        self._max_bytes = max_message_bytes
        self._watch_interval = watch_interval_seconds
        self._channel: Any = None
        self._generate_rpc: Any = None
        self._abort_rpc: Any = None
        self._abort_all_rpc: Any = None
        self._health_rpc: Any = None
        self._info_rpc: Any = None
        self._watch_rpc: Any = None
        self._info: EngineInfo | None = None
        self._health = EngineHealth(state=EngineState.STOPPED, progress_at=0.0, busy=False)
        self._streams: dict[RequestId, tuple[RequestStream, asyncio.Task[None], Any]] = {}
        self._watch_task: asyncio.Task[None] | None = None
        self._bg_tasks: set[asyncio.Task[None]] = set()
        self._closed = False

    async def connect(self) -> GrpcEngineClient:
        """Open the async gRPC channel and synchronize initial info and health."""
        grpc = _grpc()
        _ = require_module("grpc.aio", capability="grpc", remedy="uv sync --extra grpc")
        channel = grpc.aio.insecure_channel(self._target, options=grpc_options(self._max_bytes))
        svc = INFERENCE_SERVICE_NAME
        self._channel = channel
        self._generate_rpc = channel.unary_stream(
            f"/{svc}/Generate",
            request_serializer=serialize_proto,
            response_deserializer=deserialize_proto,
        )
        self._abort_rpc = channel.unary_unary(
            f"/{svc}/Abort",
            request_serializer=serialize_proto,
            response_deserializer=deserialize_proto,
        )
        self._abort_all_rpc = channel.unary_unary(
            f"/{svc}/AbortAll",
            request_serializer=serialize_proto,
            response_deserializer=deserialize_proto,
        )
        self._health_rpc = channel.unary_unary(
            f"/{svc}/Health",
            request_serializer=serialize_proto,
            response_deserializer=deserialize_proto,
        )
        self._info_rpc = channel.unary_unary(
            f"/{svc}/GetEngineInfo",
            request_serializer=serialize_proto,
            response_deserializer=deserialize_proto,
        )
        self._watch_rpc = channel.unary_stream(
            f"/{svc}/WatchEngineState",
            request_serializer=serialize_proto,
            response_deserializer=deserialize_proto,
        )
        try:
            self._info = decode_engine_info(await self._info_rpc({}))
            self._health = decode_engine_health(await self._health_rpc({}))
        except grpc.aio.AioRpcError as exc:
            await channel.close()
            raise error_from_rpc(exc) from None
        self._closed = False
        self._watch_task = asyncio.create_task(self._watch_loop())
        return self

    @property
    def info(self) -> EngineInfo:
        """Fixed engine facts fetched when connecting."""
        if self._info is None:
            raise EngineError("GrpcEngineClient is not connected")
        return self._info

    def health(self) -> EngineHealth:
        """Most recent engine health snapshot."""
        return self._health

    async def fetch_health(self) -> EngineHealth:
        """Query engine health immediately over RPC and update the cached snapshot."""
        if self._health_rpc is None:
            raise EngineError("GrpcEngineClient is not connected")
        self._health = decode_engine_health(await self._health_rpc({}))
        return self._health

    async def add(self, request: GenerateRequest) -> RequestStream:
        """Submit ``request`` remotely and return a :class:`RequestStream` of its outputs."""
        if self._closed or self._generate_rpc is None:
            raise EngineError("remote engine client is shut down")
        if self._health.state is EngineState.DEAD:
            raise EngineDeadError("remote engine is dead")
        if self._health.state is not EngineState.RUNNING:
            raise EngineError(f"remote engine is {self._health.state.value}")
        grpc = _grpc()
        call = self._generate_rpc(encode_generate_request(request))
        try:
            md = dict(await call.initial_metadata())
            if md.get("x-ayaka-admitted") != "1":
                await call.read()
                raise EngineError("remote engine ended Generate call before admission")
        except grpc.aio.AioRpcError as exc:
            call.cancel()
            raise error_from_rpc(exc) from None

        stream = RequestStream(request.request_id)
        task = asyncio.create_task(self._pump(request.request_id, stream, call))
        self._streams[request.request_id] = (stream, task, call)
        return stream

    def abort(self, request_id: RequestId) -> None:
        """Close local stream immediately and cancel the request on the remote engine."""
        entry = self._streams.pop(request_id, None)
        if entry is not None:
            stream, task, call = entry
            stream.close()
            call.cancel()
            task.cancel()
        if self._abort_rpc is not None and not self._closed:
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                return
            t = loop.create_task(self._send_abort(request_id))
            self._bg_tasks.add(t)
            t.add_done_callback(self._bg_tasks.discard)

    def abort_all(self) -> None:
        """Ask the remote engine to cancel every active request at the next step boundary."""
        if self._abort_all_rpc is not None and not self._closed:
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                return
            t = loop.create_task(self._send_abort_all())
            self._bg_tasks.add(t)
            t.add_done_callback(self._bg_tasks.discard)

    async def shutdown(self, grace: float = 0.0) -> None:
        """Close active client streams, stop watching health, and close the channel."""
        self._closed = True
        if grace > 0 and self._streams:
            self.abort_all()
            pending = [task for _, task, _ in self._streams.values()]
            if pending:
                await asyncio.wait(pending, timeout=grace)
        for request_id in list(self._streams):
            stream, task, call = self._streams.pop(request_id)
            stream.close()
            call.cancel()
            task.cancel()
        if self._watch_task is not None:
            self._watch_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._watch_task
            self._watch_task = None
        for bg in list(self._bg_tasks):
            bg.cancel()
        if self._channel is not None:
            await self._channel.close()
            self._channel = None
        self._health = EngineHealth(
            state=EngineState.STOPPED, progress_at=time.monotonic(), busy=False
        )

    async def _pump(self, request_id: RequestId, stream: RequestStream, call: Any) -> None:
        grpc = _grpc()
        try:
            async for payload in call:
                stream.put(decode_request_output(payload))
        except grpc.aio.AioRpcError as exc:
            if not stream.finished:
                mapped = error_from_rpc(exc)
                if isinstance(mapped, EngineDeadError):
                    self._health = EngineHealth(
                        state=EngineState.DEAD, progress_at=time.monotonic(), busy=False
                    )
                stream.close()
        except asyncio.CancelledError:
            stream.close()
            raise
        finally:
            self._streams.pop(request_id, None)

    async def _send_abort(self, request_id: RequestId) -> None:
        with contextlib.suppress(Exception):
            await self._abort_rpc({"request_id": str(request_id)})

    async def _send_abort_all(self) -> None:
        with contextlib.suppress(Exception):
            await self._abort_all_rpc({})

    async def _watch_loop(self) -> None:
        if self._watch_rpc is None:
            return
        try:
            async for snap in self._watch_rpc({"poll_interval_seconds": self._watch_interval}):
                h_raw = snap.get("health")
                if isinstance(h_raw, dict):
                    self._health = decode_engine_health(cast("dict[str, object]", h_raw))
                i_raw = snap.get("info")
                if isinstance(i_raw, dict):
                    self._info = decode_engine_info(cast("dict[str, object]", i_raw))
        except asyncio.CancelledError:
            raise
        except Exception:
            if not self._closed:
                self._health = EngineHealth(
                    state=EngineState.DEAD, progress_at=time.monotonic(), busy=False
                )
