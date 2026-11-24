"""Reference MoE ops; implements :class:`ayaka.ops.moe.MoeOps`.

:func:`experts` picks one of three exact strategies by the number of rows alone, so a row count
always takes the same path, eager or inside a CUDA graph:

* up to :data:`GATHER_MAX_ROWS` rows (decode): the chosen experts' weights are gathered and
  multiplied in one batched product, which reads only the ``N * k`` experts used;
* up to :data:`DENSE_MAX_ROWS` rows (the larger decode batches): every expert runs over every
  row and the experts a row did not choose get weight zero. No step depends on which experts
  were chosen, so it can be captured;
* more rows (prefill): rows are grouped by expert and each active expert multiplies its rows
  once. The groups are read on the host, so this path cannot be captured, and a capture with
  more than :data:`DENSE_MAX_ROWS` rows is refused.
"""

from __future__ import annotations

from typing import Final, cast

import torch
from torch.nn import functional

from ayaka.core.errors import BackendError
from ayaka.ops.backends.torch._checks import capturing, require_compute_dtype, require_like

__all__ = ["DENSE_MAX_ROWS", "GATHER_MAX_ROWS", "experts", "route"]

#: Rows up to which the chosen experts are gathered.
GATHER_MAX_ROWS: Final = 4
#: Rows up to which every expert runs over every row; above, rows are grouped by expert.
DENSE_MAX_ROWS: Final = 32


def route(
    logits: torch.Tensor, weights: torch.Tensor, ids: torch.Tensor, *, renormalize: bool
) -> None:
    op = "moe.route"
    require_compute_dtype(op, "logits", logits)
    if logits.ndim != 2:
        raise BackendError(
            f"{op}: logits must be [rows, experts], got {list(logits.shape)}",
            context={"op": op, "argument": "logits"},
        )
    rows, experts_count = logits.shape
    top_k = weights.shape[-1] if weights.ndim == 2 else 0
    if not 1 <= top_k <= experts_count:
        raise BackendError(
            f"{op}: weights must be [rows, k] with 1 <= k <= {experts_count}, "
            f"got {list(weights.shape)}",
            context={"op": op, "argument": "weights"},
        )
    require_like(op, "weights", weights, (rows, top_k), torch.float32)
    require_like(op, "ids", ids, (rows, top_k), torch.int32)
    probabilities = torch.softmax(logits.to(torch.float32), dim=-1)
    # A stable descending sort breaks ties by the lower expert id, on every device.
    ordered, order = torch.sort(probabilities, dim=-1, descending=True, stable=True)
    chosen = ordered[:, :top_k]
    if renormalize:
        chosen = chosen / chosen.sum(dim=-1, keepdim=True)
    weights.copy_(chosen)
    ids.copy_(order[:, :top_k])


def experts(
    x: torch.Tensor,
    gate_up: torch.Tensor,
    down: torch.Tensor,
    weights: torch.Tensor,
    ids: torch.Tensor,
    out: torch.Tensor,
) -> None:
    op = "moe.experts"
    require_compute_dtype(op, "x", x)
    if x.ndim != 2 or gate_up.ndim != 3 or down.ndim != 3:
        raise BackendError(
            f"{op}: x must be [rows, hidden], gate_up [experts, 2 * inner, hidden] and down "
            f"[experts, hidden, inner]; got {list(x.shape)}, {list(gate_up.shape)}, "
            f"{list(down.shape)}",
            context={"op": op},
        )
    rows, hidden = x.shape
    count, double_inner, _ = gate_up.shape
    inner = double_inner // 2
    require_like(op, "gate_up", gate_up, (count, 2 * inner, hidden), x.dtype)
    require_like(op, "down", down, (count, hidden, inner), x.dtype)
    if weights.ndim != 2 or weights.shape[0] != rows:
        raise BackendError(
            f"{op}: weights must be [{rows}, k], got {list(weights.shape)}",
            context={"op": op, "argument": "weights"},
        )
    top_k = weights.shape[1]
    require_like(op, "weights", weights, (rows, top_k), torch.float32)
    require_like(op, "ids", ids, (rows, top_k), torch.int32)
    require_like(op, "out", out, (rows, hidden), x.dtype)
    if rows == 0:
        return
    if rows <= GATHER_MAX_ROWS:
        result = _gathered(x, gate_up, down, weights, ids)
    elif rows <= DENSE_MAX_ROWS:
        result = _dense(x, gate_up, down, weights, ids)
    else:
        if capturing(x):
            raise BackendError(
                f"{op}: {rows} rows cannot be captured in a CUDA graph; the torch backend "
                f"captures MoE steps of at most {DENSE_MAX_ROWS} rows",
                context={"op": op, "rows": rows},
            )
        result = _grouped(x, gate_up, down, weights, ids)
    out.copy_(result)


def _gated(hidden: torch.Tensor, inner: int) -> torch.Tensor:
    gate, up = hidden[..., :inner], hidden[..., inner:]
    return functional.silu(gate) * up


def _gathered(
    x: torch.Tensor,
    gate_up: torch.Tensor,
    down: torch.Tensor,
    weights: torch.Tensor,
    ids: torch.Tensor,
) -> torch.Tensor:
    rows, top_k = ids.shape
    inner = down.shape[2]
    chosen = ids.reshape(-1).long()
    repeated = x.repeat_interleave(top_k, dim=0).unsqueeze(-1)  # [N*k, H, 1]
    hidden = torch.bmm(gate_up[chosen], repeated).squeeze(-1)  # [N*k, 2I]
    act = _gated(hidden, inner).unsqueeze(-1)  # [N*k, I, 1]
    each = torch.bmm(down[chosen], act).squeeze(-1)  # [N*k, H]
    return (each.to(torch.float32).view(rows, top_k, -1) * weights.unsqueeze(-1)).sum(dim=1)


def _dense(
    x: torch.Tensor,
    gate_up: torch.Tensor,
    down: torch.Tensor,
    weights: torch.Tensor,
    ids: torch.Tensor,
) -> torch.Tensor:
    rows = x.shape[0]
    count, _, inner = down.shape
    hidden = torch.matmul(gate_up, x.t())  # [E, 2I, N]
    act = _gated(hidden.transpose(1, 2), inner)  # [E, N, I]
    each = torch.matmul(act, down.transpose(1, 2))  # [E, N, H]
    dense = torch.zeros(rows, count, dtype=torch.float32, device=x.device)
    dense.scatter_add_(1, ids.long(), weights)
    return torch.einsum("ne,enh->nh", dense, each.to(torch.float32))


def _grouped(
    x: torch.Tensor,
    gate_up: torch.Tensor,
    down: torch.Tensor,
    weights: torch.Tensor,
    ids: torch.Tensor,
) -> torch.Tensor:
    rows, top_k = ids.shape
    inner = down.shape[2]
    flat = ids.reshape(-1).long()
    order = torch.argsort(flat, stable=True)
    per_expert = torch.bincount(flat, minlength=gate_up.shape[0])
    # A host read: this path is never captured (``experts`` refuses a capture this large).
    counts = cast("list[int]", per_expert.tolist())  # pyright: ignore[reportUnknownMemberType]
    sources = order // top_k
    scales = weights.reshape(-1)[order]
    total = torch.zeros(rows, x.shape[1], dtype=torch.float32, device=x.device)
    start = 0
    for expert, size in enumerate(counts):
        if not size:
            continue
        taken = sources[start : start + size]
        hidden = functional.linear(x[taken], gate_up[expert])
        each = functional.linear(_gated(hidden, inner), down[expert])
        total.index_add_(0, taken, each.to(torch.float32) * scales[start : start + size, None])
        start += size
    return total
