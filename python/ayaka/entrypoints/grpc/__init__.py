"""Optional gRPC transport over the engine seam and engine-free rendering service."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ayaka.utils.lazy_exports import build_lazy_exports

if TYPE_CHECKING:
    from ayaka.entrypoints.grpc.client import GrpcEngineClient
    from ayaka.entrypoints.grpc.errors import (
        METADATA_ERROR_KEY,
        abort_with_error,
        error_from_rpc,
        error_from_wire,
        error_to_wire,
    )
    from ayaka.entrypoints.grpc.inference import InferenceGrpcServer
    from ayaka.entrypoints.grpc.messages import (
        MAX_GRPC_MESSAGE_BYTES,
        grpc_options,
    )
    from ayaka.entrypoints.grpc.render import RenderGrpcClient, RenderGrpcServer

__all__ = [
    "MAX_GRPC_MESSAGE_BYTES",
    "METADATA_ERROR_KEY",
    "GrpcEngineClient",
    "InferenceGrpcServer",
    "RenderGrpcClient",
    "RenderGrpcServer",
    "abort_with_error",
    "error_from_rpc",
    "error_from_wire",
    "error_to_wire",
    "grpc_options",
]

_EXPORTS = {
    "client": ("GrpcEngineClient",),
    "errors": (
        "METADATA_ERROR_KEY",
        "abort_with_error",
        "error_from_rpc",
        "error_from_wire",
        "error_to_wire",
    ),
    "inference": ("InferenceGrpcServer",),
    "messages": ("MAX_GRPC_MESSAGE_BYTES", "grpc_options"),
    "render": ("RenderGrpcClient", "RenderGrpcServer"),
}

__getattr__, __dir__ = build_lazy_exports(__name__, _EXPORTS, globals())
