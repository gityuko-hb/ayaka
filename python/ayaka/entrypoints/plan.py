"""``ayaka plan``: whether a model fits this machine, and how the memory would be divided.

The numbers come from :func:`~ayaka.engine.build.plan_engine`, the same code a start-up runs before
it loads a weight, so the plan printed here and the plan ``serve`` starts with are one plan. The
model is built on torch's meta device: ``config.json`` and the model's code give the weights' size
and the KV geometry, nothing is read from the weight files, nothing is allocated, and nothing is
downloaded (a repo id must already be in the Hub cache).

When the plan does not fit, the command says what is short, by how much, and three ways out with
their numbers. A speed is printed only from a measurement; there is none yet in this version.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

from ayaka.utils.math_utils import div_ceil

if TYPE_CHECKING:
    from collections.abc import Mapping

    from ayaka.core.errors import CapacityError
    from ayaka.engine.build import EnginePlan
    from ayaka.entrypoints.connect.tools import ToolRecord

__all__ = [
    "PlanShortfall",
    "ToolFit",
    "plan_lines",
    "plan_report",
    "shortfall_lines",
    "shortfall_of",
    "tool_fit",
    "tool_fit_lines",
]

_MIB: Final = 2**20


def _mib(value: int) -> str:
    return f"{value / _MIB:,.0f} MiB"


def plan_report(plan: EnginePlan, model: str) -> dict[str, object]:
    """The plan as JSON-ready data, for ``--output json``."""
    budget = plan.budget
    spec = plan.group.spec
    return {
        "fits": True,
        "model": model,
        "architecture": plan.model_config.architecture,
        "directory": str(plan.directory),
        "device": {
            "name": plan.device.name,
            "kind": plan.device.kind,
            "description": plan.capabilities.describe(),
        },
        "memory": {
            "free_bytes": plan.baseline_bytes,
            "usable_bytes": budget.usable_bytes,
            "headroom_bytes": budget.headroom_bytes,
            "weights_bytes": plan.weights_bytes,
            "kv_bytes": plan.kv_bytes,
            "graph_bytes": budget.graph_bytes,
        },
        "kv": {
            "blocks": plan.kv_blocks,
            "fit_blocks": budget.kv_blocks[0],
            "block_size": plan.block_size,
            "dtype": spec.dtype.label,
            "bytes_per_token": spec.bytes_per_token(),
        },
        "context": {
            "max_tokens_per_request": plan.max_context_tokens,
            "model_context": plan.model_config.max_position_embeddings,
            "reserve_tokens": plan.reserve_tokens,
        },
        "quantization": plan.quantization,
        "max_running": plan.max_running,
        "chunk_tokens": plan.chunk_tokens,
        "ops": dict(plan.providers),
        "profiled": plan.profiled,
        "speed": None,
    }


def plan_lines(plan: EnginePlan, model: str) -> list[str]:
    """The plan as the table ``ayaka plan`` prints."""
    budget = plan.budget
    spec = plan.group.spec
    config = plan.model_config
    context = config.max_position_embeddings
    limit = "memory" if plan.max_context_tokens < context else "the model"
    reserve = budget.usable_bytes - plan.weights_bytes - plan.kv_bytes
    ops = ", ".join(f"{family}: {name}" for family, name in sorted(plan.providers.items()))
    weights = (
        f"  weights  {_mib(plan.weights_bytes)} ({spec.dtype.label})"
        if plan.quantization is None
        else f"  weights  {_mib(plan.weights_bytes)} ({plan.quantization} projections, the rest "
        f"{spec.dtype.label})"
    )
    lines = [
        f"machine    {plan.capabilities.describe()}",
        f"model      {model}: {config.architecture}",
        f"files      {plan.directory} (config.json read; no weight read, nothing downloaded)",
        f"memory     free {_mib(plan.baseline_bytes)}, usable {_mib(budget.usable_bytes)}"
        f", headroom {_mib(budget.headroom_bytes)}",
        weights,
        f"  KV       {_mib(plan.kv_bytes)} = {plan.kv_blocks - 1:,} blocks of {plan.block_size}"
        f" tokens ({spec.dtype.label}, {spec.bytes_per_token():,} bytes per token;"
        f" {budget.kv_blocks[0] - 1:,} would fit)",
        f"  rest     {_mib(max(reserve, 0))} of the usable memory left for the runtime",
        f"context    up to {plan.max_context_tokens:,} tokens per request "
        f"(the model allows {context:,}; the limit is {limit})",
        f"running    {plan.max_running} requests at once, prefill steps of up to "
        f"{plan.chunk_tokens:,} tokens",
        f"ops        {ops}",
        "profile    "
        + (
            "start-up measured on this machine before: runtime, activations and graphs included"
            if plan.profiled
            else "not measured yet; the first start-up measures the runtime memory and saves it"
        ),
        "verdict    FITS.  Speed: not measured on this machine (run `ayaka bench`).",
    ]
    if plan.quantization is not None:
        lines.insert(
            -1,
            f"quantized  {plan.quantization}: weights stay 8 bit and are dequantized in slices "
            "before each product, slower than 16-bit weights (no FP8 matrix kernel yet)",
        )
    return lines


@dataclass(frozen=True, slots=True)
class PlanShortfall:
    """What a plan that does not fit is missing, read from the planner's error."""

    what: str
    need: int
    have: int
    short: int
    bytes_per_token: int | None
    weights_bytes: int | None
    reserve_tokens: int | None
    model_context: int | None


def shortfall_of(error: CapacityError) -> PlanShortfall:
    """The numbers of a capacity error from :func:`~ayaka.engine.build.plan_engine`."""
    context: Mapping[str, object] = error.context

    def number(key: str) -> int | None:
        value = context.get(key)
        return value if isinstance(value, int) and not isinstance(value, bool) else None

    return PlanShortfall(
        what=str(context.get("what", "memory")),
        need=number("need") or 0,
        have=number("have") or 0,
        short=number("short") or 0,
        bytes_per_token=number("bytes_per_token"),
        weights_bytes=number("weights_bytes"),
        reserve_tokens=number("kv_reserve_tokens"),
        model_context=number("model_context"),
    )


def shortfall_lines(shortfall: PlanShortfall) -> list[str]:
    """What is missing and three ways out, with their numbers."""
    lines = [
        f"verdict    DOES NOT FIT: {shortfall.what} needs {_mib(shortfall.need)}, "
        f"{_mib(shortfall.have)} is available, {_mib(shortfall.short)} short",
    ]
    per_token = shortfall.bytes_per_token
    reserve = shortfall.reserve_tokens
    if per_token and reserve:
        fewer = reserve - div_ceil(shortfall.short, per_token)
        if fewer > 0:
            lines.append(
                f"  fix      a shorter context: --kv-reserve-tokens {fewer} "
                f"(each token of KV costs {per_token:,} bytes)"
            )
    lines.append(
        "  fix      a smaller model: its weights and its KV per token both shrink"
        + (
            f" (these weights take {_mib(shortfall.weights_bytes)})"
            if shortfall.weights_bytes
            else ""
        )
    )
    lines.append(
        "  fix      more memory: --memory-ratio closer to 1, or close what else uses the device; "
        "8-bit weights are not supported by this version yet"
    )
    return lines


@dataclass(frozen=True, slots=True)
class ToolFit:
    """Whether one request on this plan holds a coding tool's fixed prompt and room to work.

    Attributes:
        tool: The tool.
        context_tokens: The tokens one request gets on this plan.
        floor_tokens: The tool's measured fixed prompt, or ``None`` when it was never measured.
        margin_tokens: What the tool needs beyond it (``ayaka connect`` step 7).
        measured: Where the floor comes from.
    """

    tool: str
    context_tokens: int
    floor_tokens: int | None
    margin_tokens: int
    measured: str | None

    @property
    def needed_tokens(self) -> int | None:
        """The tokens one request must hold, or ``None`` without a measured floor."""
        return None if self.floor_tokens is None else self.floor_tokens + self.margin_tokens

    @property
    def fits(self) -> bool | None:
        """Whether it fits; ``None`` when the floor was never measured."""
        needed = self.needed_tokens
        return None if needed is None else self.context_tokens >= needed


def tool_fit(plan: EnginePlan, record: ToolRecord) -> ToolFit:
    """The plan's context against the tool's measured floor, as ``ayaka connect --check`` does."""
    from ayaka.entrypoints.connect.tools import CONTEXT_MARGIN_TOKENS

    return ToolFit(
        tool=record.name,
        context_tokens=plan.max_context_tokens,
        floor_tokens=record.min_context_tokens,
        margin_tokens=CONTEXT_MARGIN_TOKENS,
        measured=record.measured,
    )


def tool_fit_lines(fit: ToolFit) -> list[str]:
    """What ``ayaka plan --for TOOL`` adds to the plan."""
    needed = fit.needed_tokens
    if needed is None:
        return [f"for        {fit.tool}: not measured (ayaka connect {fit.tool} --observe)"]
    if fit.fits:
        return [
            f"for        {fit.tool}: FITS. {fit.context_tokens:,} tokens per request hold its "
            f"{fit.floor_tokens:,}-token fixed prompt and {fit.margin_tokens:,} to work "
            f"({fit.measured})"
        ]
    short = needed - fit.context_tokens
    return [
        f"for        {fit.tool}: DOES NOT FIT. It needs {needed:,} tokens per request "
        f"({fit.floor_tokens:,} fixed prompt + {fit.margin_tokens:,} to work; {fit.measured}); "
        f"this plan gives {fit.context_tokens:,}, {short:,} short",
        "           ways out: a model with a longer context; more memory for the KV cache "
        "(--memory-ratio, a smaller --max-running, a smaller dtype)",
    ]
