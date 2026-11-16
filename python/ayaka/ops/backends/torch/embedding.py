"""Reference embedding ops; implements :class:`ayaka.ops.embedding.EmbeddingOps`."""

from __future__ import annotations

import torch
from torch.nn import functional

from ayaka.core.errors import BackendError
from ayaka.ops.backends.torch._checks import require_compute_dtype, require_like
from ayaka.utils.torch_utils import dtype_name

__all__ = ["lookup"]


def lookup(token_ids: torch.Tensor, weight: torch.Tensor, out: torch.Tensor) -> None:
    op = "embedding.lookup"
    if token_ids.dtype not in (torch.int32, torch.int64):
        raise BackendError(
            f"{op}: token_ids must be int32 or int64, got {dtype_name(token_ids.dtype)}",
            context={"op": op, "argument": "token_ids"},
        )
    require_compute_dtype(op, "weight", weight)
    if weight.ndim != 2:
        raise BackendError(
            f"{op}: weight must be [vocab, hidden], got {list(weight.shape)}",
            context={"op": op, "argument": "weight"},
        )
    require_like(op, "out", out, (*token_ids.shape, weight.shape[1]), weight.dtype)
    out.copy_(functional.embedding(token_ids, weight))
