"""Reading request bodies: the field reader and the table of what each field means.

``JsonObject`` reads one JSON object field by field; ``ENDPOINTS`` and ``apply_policy`` say, per
endpoint, which fields are read, hints, ignored or refused. Nothing here imports aiohttp.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ayaka.utils.lazy_exports import build_lazy_exports

if TYPE_CHECKING:
    from ayaka.entrypoints.wire.policy import (
        ENDPOINTS,
        EndpointPolicy,
        FieldRule,
        Rule,
        apply_policy,
        wire_reference,
    )
    from ayaka.entrypoints.wire.reader import JsonObject, sampling_params

__all__ = [
    "ENDPOINTS",
    "EndpointPolicy",
    "FieldRule",
    "JsonObject",
    "Rule",
    "apply_policy",
    "sampling_params",
    "wire_reference",
]

_EXPORTS = {
    "policy": (
        "ENDPOINTS",
        "EndpointPolicy",
        "FieldRule",
        "Rule",
        "apply_policy",
        "wire_reference",
    ),
    "reader": ("JsonObject", "sampling_params"),
}

__getattr__, __dir__ = build_lazy_exports(__name__, _EXPORTS, globals())
